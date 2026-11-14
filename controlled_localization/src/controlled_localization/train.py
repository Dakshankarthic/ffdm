"""Training loop, optimizer parity, gradient accumulation, and checkpointing for Controlled Localization v1.1."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast
import yaml

from .models.detector import ControlledLocalizationDetector
from .losses.segmentation import compute_supervised_segmentation_loss
from .losses.consistency import compute_directed_consistency_loss, compute_outside_consistency_loss
from .utils import seed_all, resolve_device


class CosineWarmupScheduler:
    """Linear warm-up followed by cosine decay to min_lr_ratio * base_lr."""

    def __init__(self, optimizer: AdamW, warmup_steps: int = 100, max_steps: int = 5000, min_lr_ratio: float = 0.1) -> None:
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.min_lr_ratio = min_lr_ratio
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]

    def step(self, current_step: int) -> None:
        if current_step < self.warmup_steps:
            alpha = float(current_step) / float(max(1, self.warmup_steps))
            factor = alpha
        else:
            progress = float(current_step - self.warmup_steps) / float(max(1, self.max_steps - self.warmup_steps))
            factor = self.min_lr_ratio + 0.5 * (1.0 - self.min_lr_ratio) * (1.0 + math.cos(math.pi * min(progress, 1.0)))

        for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            group["lr"] = base_lr * factor


def get_loss_ramp(step: int, ramp_steps: int = 500) -> float:
    """Ramp factor in [0, 1] for auxiliary and consistency losses."""
    return min(1.0, float(step) / float(max(1, ramp_steps)))


def build_optimizer(detector: ControlledLocalizationDetector, rgb_lr: float = 1e-4, new_lr: float = 3e-4, weight_decay: float = 1e-4) -> AdamW:
    """
    Two parameter groups:
      - RGB backbone convolutions: rgb_lr (1e-4)
      - Residual encoder, FPN decoder, and classification heads: new_lr (3e-4)
    Excludes frozen BatchNorm parameters.
    """
    rgb_params = []
    for name, param in detector.rgb_encoder.named_parameters():
        if param.requires_grad:
            rgb_params.append(param)

    new_params = []
    if detector.res_encoder is not None:
        for param in detector.res_encoder.parameters():
            if param.requires_grad:
                new_params.append(param)

    for param in detector.decoder.parameters():
        if param.requires_grad:
            new_params.append(param)

    return AdamW(
        [
            {"params": rgb_params, "lr": rgb_lr},
            {"params": new_params, "lr": new_lr},
        ],
        weight_decay=weight_decay,
    )


def train_step_group(
    detector: ControlledLocalizationDetector,
    group_tensors: Dict[str, torch.Tensor],
    variant: str,
    ramp: float,
    use_amp: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    Execute forward and compute losses for ONE four-view group (A, B, C, D).
    Returns dictionary of float scalar losses.
    """
    img_a = group_tensors["img_a"]  # [1, 3, H, W]
    img_b = group_tensors["img_b"]  # [1, 3, H, W]
    img_c = group_tensors["img_c"]  # [1, 3, H, W]
    img_d = group_tensors["img_d"]  # [1, 3, H, W]

    mask_c = group_tensors["mask_c"]  # [1, 1, H, W]
    dilated_mask_c = group_tensors.get("dilated_mask_c", mask_c)
    valid_mask = group_tensors.get("valid_mask", torch.ones_like(mask_c))

    empty_mask = torch.zeros_like(mask_c)

    with autocast(enabled=use_amp):
        if variant == "T0":
            # T0: A, A, C, C (segmentation only, matched forward count)
            out_a1 = detector(img_a)
            out_a2 = detector(img_a)
            out_c1 = detector(img_c)
            out_c2 = detector(img_c)

            l_seg_a1, _, _ = compute_supervised_segmentation_loss(out_a1["mask_logits"], empty_mask, valid_mask)
            l_seg_a2, _, _ = compute_supervised_segmentation_loss(out_a2["mask_logits"], empty_mask, valid_mask)
            l_seg_c1, _, _ = compute_supervised_segmentation_loss(out_c1["mask_logits"], mask_c, valid_mask)
            l_seg_c2, _, _ = compute_supervised_segmentation_loss(out_c2["mask_logits"], mask_c, valid_mask)

            l_seg = 0.25 * (l_seg_a1 + l_seg_a2 + l_seg_c1 + l_seg_c2)
            l_view = torch.tensor(0.0, device=img_a.device)
            l_out = torch.tensor(0.0, device=img_a.device)
            l_proc = torch.tensor(0.0, device=img_a.device)
            total = l_seg

        else:
            # T1, T2, T2O, T3: A, B, C, D
            out_a = detector(img_a)
            out_b = detector(img_b)
            out_c = detector(img_c)
            out_d = detector(img_d)

            l_seg_a, _, _ = compute_supervised_segmentation_loss(out_a["mask_logits"], empty_mask, valid_mask)
            l_seg_b, _, _ = compute_supervised_segmentation_loss(out_b["mask_logits"], empty_mask, valid_mask)
            l_seg_c, _, _ = compute_supervised_segmentation_loss(out_c["mask_logits"], mask_c, valid_mask)
            l_seg_d, _, _ = compute_supervised_segmentation_loss(out_d["mask_logits"], mask_c, valid_mask)

            l_seg = 0.25 * (l_seg_a + l_seg_b + l_seg_c + l_seg_d)

            # View consistency (T2, T2O, T3)
            if variant in ("T2", "T2O", "T3"):
                l_view_ab = compute_directed_consistency_loss(
                    student_logits=out_b["mask_logits"],
                    teacher_logits=out_a["mask_logits"],
                    gt_mask=empty_mask,
                    valid_mask=valid_mask,
                )
                l_view_cd = compute_directed_consistency_loss(
                    student_logits=out_d["mask_logits"],
                    teacher_logits=out_c["mask_logits"],
                    gt_mask=mask_c,
                    valid_mask=valid_mask,
                )
                l_view = 0.5 * (l_view_ab + l_view_cd)
            else:
                l_view = torch.tensor(0.0, device=img_a.device)

            # Outside consistency (T2O, T3)
            if variant in ("T2O", "T3") and group_tensors.get("outside_eligible", True):
                l_out_ac = compute_outside_consistency_loss(
                    edited_logits=out_c["mask_logits"],
                    unedited_logits=out_a["mask_logits"],
                    dilated_mask=dilated_mask_c,
                    valid_mask=valid_mask,
                )
                l_out_bd = compute_outside_consistency_loss(
                    edited_logits=out_d["mask_logits"],
                    unedited_logits=out_b["mask_logits"],
                    dilated_mask=dilated_mask_c,
                    valid_mask=valid_mask,
                )
                l_out = 0.5 * (l_out_ac + l_out_bd)
            else:
                l_out = torch.tensor(0.0, device=img_a.device)

            # Optional T3 processing loss
            l_proc = torch.tensor(0.0, device=img_a.device)
            if variant == "T3" and "processing_logits" in out_b and "recipe_label" in group_tensors:
                target_recipe = group_tensors["recipe_label"]
                l_proc = nn.functional.cross_entropy(out_b["processing_logits"], target_recipe)

            total = l_seg + ramp * 0.1 * l_view + ramp * 0.1 * l_out + 0.05 * l_proc

    return {
        "total": total,
        "l_seg": l_seg,
        "l_view": l_view,
        "l_out": l_out,
        "l_proc": l_proc,
    }


def save_checkpoint(
    path: str | Path,
    detector: ControlledLocalizationDetector,
    optimizer: AdamW,
    scheduler: CosineWarmupScheduler,
    scaler: GradScaler,
    step: int,
    config: Dict[str, Any],
) -> None:
    """Save full training checkpoint with all scheduler and scaler states."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    temp_path = p.with_suffix(".tmp")

    ckpt = {
        "step": step,
        "model_state_dict": detector.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state": {
            "warmup_steps": scheduler.warmup_steps,
            "max_steps": scheduler.max_steps,
            "base_lrs": scheduler.base_lrs,
        },
        "scaler_state_dict": scaler.state_dict(),
        "config": config,
    }
    torch.save(ckpt, temp_path)
    # Atomic replace
    temp_path.replace(p)


def load_checkpoint(
    path: str | Path,
    detector: ControlledLocalizationDetector,
    optimizer: Optional[AdamW] = None,
    scheduler: Optional[CosineWarmupScheduler] = None,
    scaler: Optional[GradScaler] = None,
    device: torch.device = torch.device("cpu"),
) -> int:
    """Load checkpoint and restore training states."""
    ckpt = torch.load(path, map_location=device)
    detector.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scaler is not None and "scaler_state_dict" in ckpt:
        scaler.load_state_dict(ckpt["scaler_state_dict"])
    return int(ckpt.get("step", 0))


def run_training_experiment(
    config: Dict[str, Any],
    run_dir: Path,
    manifest_path: Optional[str | Path] = None,
    schedule_path: Optional[str | Path] = None,
    resume_path: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """Execute complete training experiment with JSONL logging, checkpointing, and evaluation."""
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = run_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    seed_all(config.get("seed", 42))
    device = resolve_device()

    variant = config.get("variant", "T2")
    train_cfg = config.get("train", {})
    num_steps = train_cfg.get("steps", 100)
    val_interval = train_cfg.get("val_interval", 25)
    save_interval = train_cfg.get("save_interval", 50)
    rgb_lr = float(train_cfg.get("rgb_lr", 1e-4))
    new_lr = float(train_cfg.get("new_lr", 3e-4))
    weight_decay = float(train_cfg.get("weight_decay", 1e-4))
    use_amp = bool(train_cfg.get("use_amp", True)) and torch.cuda.is_available()

    # Research runs must consume explicit on-disk inputs, never synthetic fallback.
    if config.get('synthetic_smoke', False):
        if manifest_path or schedule_path:
            raise ValueError('Synthetic smoke mode cannot accept research inputs')
        from .overfit_check import create_overfit_dataset
        groups, manifest_records = create_overfit_dataset(run_dir / 'dataset', num_groups=8)
    else:
        if not manifest_path or not schedule_path or not config.get('validation_manifest'):
            raise ValueError('Research training requires manifest, schedule and validation_manifest')
        if variant not in ('T0', 'T1', 'T2'):
            raise ValueError('Research loader currently supports T0/T1/T2 only')
        from .data.research_loader import load_research_inputs
        groups, manifest_records = load_research_inputs(manifest_path, schedule_path, config['validation_manifest'])
        if len(groups) < 2*num_steps:
            raise ValueError('Schedule requires two crop entries per optimizer step')

    # Initialize model
    model_cfg = config.get("model", {})
    detector = ControlledLocalizationDetector(
        pretrained_rgb=bool(model_cfg.get("pretrained_rgb", True)),
        weights_name=model_cfg.get("weights_name", "IMAGENET1K_V1"),
        num_processing_classes=model_cfg.get("num_processing_classes", None),
        rgb_only=bool(model_cfg.get("rgb_only", False)),
    ).to(device)

    optimizer = build_optimizer(detector, rgb_lr=rgb_lr, new_lr=new_lr, weight_decay=weight_decay)
    scheduler = CosineWarmupScheduler(
        optimizer,
        warmup_steps=train_cfg.get("warmup_steps", 10),
        max_steps=num_steps,
        min_lr_ratio=0.1,
    )
    scaler = GradScaler(enabled=use_amp)

    start_step = 0
    if resume_path:
        start_step = load_checkpoint(resume_path, detector, optimizer, scheduler, scaler, device)
        print(f"Resumed from step {start_step} from {resume_path}")

    # Save initial run assets
    with open(run_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.dump(config, f, sort_keys=False)

    with open(run_dir / "environment.txt", "w", encoding="utf-8") as f:
        f.write(f"Device: {device}\nGPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}\nPyTorch: {torch.__version__}\n")

    train_log_path = run_dir / "train_log.jsonl"
    log_file = open(train_log_path, "a", encoding="utf-8")

    start_train_time = time.perf_counter()
    print(f"Starting {variant} training on {device} ({num_steps} steps, 2 groups/step)...")

    detector.train()

    for step in range(start_step, num_steps):
        step_start = time.perf_counter()
        scheduler.step(step)
        ramp = get_loss_ramp(step, ramp_steps=train_cfg.get("ramp_steps", 50))
        optimizer.zero_grad()

        # Step accumulates over 2 groups
        g_idx1 = (step * 2) % len(groups)
        g_idx2 = (step * 2 + 1) % len(groups)

        total_loss_accum = 0.0
        l_seg_accum = 0.0
        l_view_accum = 0.0
        l_out_accum = 0.0

        for g_idx in (g_idx1, g_idx2):
            g = groups[g_idx]
            g_dev = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in g.items()}
            losses = train_step_group(detector, g_dev, variant=variant, ramp=ramp, use_amp=use_amp)
            if not torch.isfinite(losses["total"]):
                print(f"Warning: Non-finite loss at step {step} for group {g_idx}. Skipping group.")
                continue

            # Divide by 2 for accumulation over 2 groups per step
            acc_loss = losses["total"] / 2.0
            scaler.scale(acc_loss).backward()

            total_loss_accum += losses["total"].item() / 2.0
            l_seg_accum += losses["l_seg"].item() / 2.0
            l_view_accum += losses["l_view"].item() / 2.0
            l_out_accum += losses["l_out"].item() / 2.0

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(detector.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        step_sec = time.perf_counter() - step_start
        lr_current = optimizer.param_groups[0]["lr"]

        log_entry = {
            "step": step,
            "total_loss": total_loss_accum,
            "l_seg": l_seg_accum,
            "l_view": l_view_accum,
            "l_out": l_out_accum,
            "lr": lr_current,
            "ramp": ramp,
            "step_seconds": step_sec,
        }
        log_file.write(json.dumps(log_entry) + "\n")
        log_file.flush()

        if (step + 1) % 10 == 0 or step == 0:
            print(f"Step {step+1:4d}/{num_steps} | Loss: {total_loss_accum:.4f} (seg: {l_seg_accum:.4f}, view: {l_view_accum:.4f}, out: {l_out_accum:.4f}) | lr: {lr_current:.2e} | {step_sec*1000:.1f}ms")

        if (step + 1) % save_interval == 0:
            save_checkpoint(
                checkpoints_dir / f"checkpoint_step_{step+1:05d}.pt",
                detector, optimizer, scheduler, scaler, step=step+1, config=config,
            )

    log_file.close()
    total_train_sec = time.perf_counter() - start_train_time

    # Save final checkpoint
    final_ckpt = run_dir / "checkpoint_final.pt"
    save_checkpoint(final_ckpt, detector, optimizer, scheduler, scaler, step=num_steps, config=config)

    # Run evaluation
    from .policy import InferencePolicy
    from .evaluate import evaluate_dataset

    policy = InferencePolicy(
        policy_name=f"{variant.lower()}_eval_policy",
        t_mask=0.5,
        t_image=0.5,
        score_direction="higher_is_fake",
    )
    eval_dir = run_dir / "eval_results"
    eval_metrics = evaluate_dataset(manifest_records, detector, policy, device, eval_dir)

    summary = {
        "variant": variant,
        "device": str(device),
        "steps_completed": num_steps,
        "total_train_runtime_seconds": total_train_sec,
        "final_loss": total_loss_accum,
        "validation_macro_iou": eval_metrics["localization"]["macro_iou"],
        "validation_pixel_f1": eval_metrics["localization"]["macro_f1"],
        "detection_auroc": eval_metrics["detection"]["auroc"],
        "clean_fp_pct": eval_metrics["localization"]["clean_false_positive_pixel_pct"],
        "checkpoint_final": str(final_ckpt.resolve()),
    }
    with open(run_dir / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Run {variant} finished in {total_train_sec:.2f}s! Validation Macro IoU: {summary['validation_macro_iou']:.4f}, Det AUROC: {summary['detection_auroc']:.4f}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Controlled Localization v1.1 Training CLI")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML training config")
    parser.add_argument("--run-dir", type=str, required=True, help="Path to output run directory")
    parser.add_argument("--manifest", type=str, default=None, help="Optional path to manifest")
    parser.add_argument("--schedule", type=str, default=None, help="Optional path to schedule.jsonl")
    parser.add_argument("--resume", type=str, default=None, help="Optional checkpoint path to resume from")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    run_training_experiment(
        config=config,
        run_dir=run_dir,
        manifest_path=args.manifest,
        schedule_path=args.schedule,
        resume_path=args.resume,
    )


if __name__ == "__main__":
    main()


