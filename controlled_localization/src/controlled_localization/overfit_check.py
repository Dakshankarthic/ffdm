"""Stage 4 Overfit verification check on 8 verified groups for Controlled Localization v1.1."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import torch
from torch.cuda.amp import GradScaler
from PIL import Image

from .models.detector import ControlledLocalizationDetector
from .losses.segmentation import compute_supervised_segmentation_loss
from .losses.consistency import compute_directed_consistency_loss, compute_outside_consistency_loss
from .train import build_optimizer, CosineWarmupScheduler, get_loss_ramp, save_checkpoint
from .data.canonicalize import apply_recipe
from .data.pairs import compute_dilated_mask
from .policy import InferencePolicy
from .evaluate import evaluate_dataset
from .data.schema import ManifestRecord, save_manifest
from .utils import seed_all, resolve_device, compute_sha256


def create_overfit_dataset(root_dir: Path, num_groups: int = 8) -> Tuple[List[Dict[str, torch.Tensor]], List[ManifestRecord]]:
    root_dir.mkdir(parents=True, exist_ok=True)
    images_dir = root_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.RandomState(42)
    groups = []
    manifest_records = []

    recipes = ["jpeg_q70", "jpeg_q85", "jpeg_q95", "blur_s05"]

    for i in range(num_groups):
        gid = f"grp_{i:02d}"
        recipe_id = recipes[i % len(recipes)]

        # Create clean background photograph texture [384, 384, 3]
        base_arr = (rng.rand(384, 384, 3) * 0.7 + 0.15).astype(np.float32)
        img_a_pil = Image.fromarray((base_arr * 255.0).astype(np.uint8))
        path_a = images_dir / f"{gid}_A.png"
        img_a_pil.save(path_a)

        # Create edited photograph C with a 50x50 object inserted at random location
        edit_arr = base_arr.copy()
        mask_arr = np.zeros((384, 384), dtype=np.float32)
        x0 = rng.randint(40, 290)
        y0 = rng.randint(40, 290)
        # Edit region with high-contrast distinct pattern
        edit_patch = (rng.rand(50, 50, 3) * 0.9 + 0.05).astype(np.float32)
        edit_arr[y0 : y0 + 50, x0 : x0 + 50] = edit_patch
        mask_arr[y0 : y0 + 50, x0 : x0 + 50] = 1.0

        img_c_pil = Image.fromarray((edit_arr * 255.0).astype(np.uint8))
        path_c = images_dir / f"{gid}_C.png"
        img_c_pil.save(path_c)

        mask_pil = Image.fromarray((mask_arr * 255.0).astype(np.uint8))
        path_mask = images_dir / f"{gid}_mask.png"
        mask_pil.save(path_mask)

        # Build degraded controls B and D
        img_b_pil, _ = apply_recipe(img_a_pil, recipe_id)
        path_b = images_dir / f"{gid}_B.png"
        img_b_pil.save(path_b)

        img_d_pil, _ = apply_recipe(img_c_pil, recipe_id)
        path_d = images_dir / f"{gid}_D.png"
        img_d_pil.save(path_d)

        # Compute dilated mask (16px)
        dilated = compute_dilated_mask(mask_arr, radius_px=16)

        # Manifest records for evaluation
        rec_clean = ManifestRecord(
            schema_version="1.1",
            sample_id=f"{gid}_clean",
            source_id=f"src_{i}",
            source_group_id=gid,
            pair_group_id=gid,
            split="train",
            view_role="A",
            image_path=str(path_a.resolve()),
            mask_path=None,
            ignore_mask_path=None,
            label=0,
            manipulation_family="pristine",
            editor_name="none",
            editor_version="none",
            edit_type="none",
            compositing_mode="native_pixel_paste",
            processing_recipe_id="identity",
            recipe_parameters={},
            width=384,
            height=384,
            mask_area_fraction=0.0,
            image_sha256=compute_sha256(path_a),
            mask_sha256=None,
            source_dataset="overfit_fixture",
            license_or_usage_notes="MIT",
            mask_verified=True,
            status="verified",
            geometry_verified=True,
            outside_pair_eligible=True,
            provenance_notes="Verified overfit group pristine sample",
        )
        rec_edit = ManifestRecord(
            schema_version="1.1",
            sample_id=f"{gid}_edit",
            source_id=f"src_{i}",
            source_group_id=gid,
            pair_group_id=gid,
            split="train",
            view_role="C",
            image_path=str(path_c.resolve()),
            mask_path=str(path_mask.resolve()),
            ignore_mask_path=None,
            label=1,
            manipulation_family="ai_local",
            editor_name="overfit_inpainting_sim",
            editor_version="1.0",
            edit_type="inpainting",
            compositing_mode="native_pixel_paste",
            processing_recipe_id="identity",
            recipe_parameters={},
            width=384,
            height=384,
            mask_area_fraction=float((50 * 50) / (384 * 384)),
            image_sha256=compute_sha256(path_c),
            mask_sha256=compute_sha256(path_mask),
            source_dataset="overfit_fixture",
            license_or_usage_notes="MIT",
            mask_verified=True,
            status="verified",
            geometry_verified=True,
            outside_pair_eligible=True,
            provenance_notes="Verified overfit group edited sample",
        )
        manifest_records.extend([rec_clean, rec_edit])

        group_tensors = {
            "img_a": torch.from_numpy(base_arr).permute(2, 0, 1).unsqueeze(0),
            "img_b": torch.from_numpy(np.array(img_b_pil, dtype=np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0),
            "img_c": torch.from_numpy(edit_arr).permute(2, 0, 1).unsqueeze(0),
            "img_d": torch.from_numpy(np.array(img_d_pil, dtype=np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0),
            "mask_c": torch.from_numpy(mask_arr).unsqueeze(0).unsqueeze(0),
            "dilated_mask_c": torch.from_numpy(dilated).unsqueeze(0).unsqueeze(0),
            "outside_eligible": True,
        }
        groups.append(group_tensors)

    manifest_p = root_dir / "overfit_manifest.jsonl"
    save_manifest(manifest_records, manifest_p)
    return groups, manifest_records


def run_overfit_experiment(
    out_dir: Path,
    num_steps: int = 60,
    variant: str = "T2",
) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    seed_all(42)
    device = resolve_device()

    # 1. Create 8 verified groups
    groups, manifest_records = create_overfit_dataset(out_dir / "dataset", num_groups=8)

    # 2. Build model, optimizer, scheduler, scaler
    detector = ControlledLocalizationDetector(pretrained_rgb=False).to(device)
    optimizer = build_optimizer(detector, rgb_lr=1e-3, new_lr=3e-3, weight_decay=1e-4)
    scheduler = CosineWarmupScheduler(optimizer, warmup_steps=10, max_steps=num_steps, min_lr_ratio=0.1)
    scaler = GradScaler(enabled=torch.cuda.is_available())

    loss_history = []
    start_time = time.perf_counter()

    detector.train()

    # 3. Training loop with 2 groups per step (8 forwards per step)
    for step in range(num_steps):
        scheduler.step(step)
        ramp = get_loss_ramp(step, ramp_steps=20)
        optimizer.zero_grad()

        # Step accumulates over 2 groups
        g_idx1 = (step * 2) % len(groups)
        g_idx2 = (step * 2 + 1) % len(groups)

        total_step_loss = 0.0

        for g_idx in (g_idx1, g_idx2):
            g = groups[g_idx]
            # Move group tensors to device
            g_dev = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in g.items()}

            # Forward and compute loss for group
            img_a = g_dev["img_a"]
            img_b = g_dev["img_b"]
            img_c = g_dev["img_c"]
            img_d = g_dev["img_d"]
            mask_c = g_dev["mask_c"]
            dil_mask = g_dev["dilated_mask_c"]
            empty_mask = torch.zeros_like(mask_c)

            out_a = detector(img_a)
            out_b = detector(img_b)
            out_c = detector(img_c)
            out_d = detector(img_d)

            l_seg_a, _, _ = compute_supervised_segmentation_loss(out_a["mask_logits"], empty_mask)
            l_seg_b, _, _ = compute_supervised_segmentation_loss(out_b["mask_logits"], empty_mask)
            l_seg_c, _, _ = compute_supervised_segmentation_loss(out_c["mask_logits"], mask_c)
            l_seg_d, _, _ = compute_supervised_segmentation_loss(out_d["mask_logits"], mask_c)

            l_seg = 0.25 * (l_seg_a + l_seg_b + l_seg_c + l_seg_d)

            if variant in ("T2", "T2O"):
                l_view_ab = compute_directed_consistency_loss(out_b["mask_logits"], out_a["mask_logits"], empty_mask)
                l_view_cd = compute_directed_consistency_loss(out_d["mask_logits"], out_c["mask_logits"], mask_c)
                l_view = 0.5 * (l_view_ab + l_view_cd)
            else:
                l_view = torch.tensor(0.0, device=device)

            if variant == "T2O":
                l_out_ac = compute_outside_consistency_loss(out_c["mask_logits"], out_a["mask_logits"], dil_mask)
                l_out_bd = compute_outside_consistency_loss(out_d["mask_logits"], out_b["mask_logits"], dil_mask)
                l_out = 0.5 * (l_out_ac + l_out_bd)
            else:
                l_out = torch.tensor(0.0, device=device)

            group_loss = l_seg + ramp * 0.1 * l_view + ramp * 0.1 * l_out
            # Divide by 2 for 2 accumulated groups per step
            acc_loss = group_loss / 2.0

            scaler.scale(acc_loss).backward()
            total_step_loss += group_loss.item() / 2.0

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(detector.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        loss_history.append(total_step_loss)

    train_runtime = time.perf_counter() - start_time

    # 4. Save final checkpoint
    ckpt_path = out_dir / "overfit_checkpoint.pt"
    save_checkpoint(
        ckpt_path,
        detector,
        optimizer,
        scheduler,
        scaler,
        step=num_steps,
        config={"variant": variant, "steps": num_steps},
    )

    # 5. Evaluate on the overfit dataset
    policy = InferencePolicy(
        policy_name="overfit_eval_policy",
        t_mask=0.5,
        t_image=0.5,
        score_direction="higher_is_fake",
    )
    eval_metrics = evaluate_dataset(
        manifest_records,
        detector,
        policy,
        device,
        out_dir / "eval_results",
    )

    initial_loss = loss_history[0]
    final_loss = loss_history[-1]
    achieved_iou = eval_metrics["localization"]["macro_iou"]
    det_auc = eval_metrics["detection"]["auroc"]

    report = {
        "status": "PASS" if final_loss < initial_loss else "FAIL",
        "variant": variant,
        "device": str(device),
        "steps": num_steps,
        "groups_count": len(groups),
        "total_train_runtime_seconds": float(train_runtime),
        "initial_loss": float(initial_loss),
        "final_loss": float(final_loss),
        "loss_reduction_pct": float((initial_loss - final_loss) / initial_loss * 100.0),
        "achieved_macro_iou": float(achieved_iou),
        "detection_auroc": float(det_auc),
        "checkpoint_path": str(ckpt_path.resolve()),
    }

    with open(out_dir / "overfit_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Overfit check complete on {device}: Initial Loss={initial_loss:.4f} -> Final Loss={final_loss:.4f} ({report['loss_reduction_pct']:.1f}% reduction). Macro IoU={achieved_iou:.4f}, Det AUC={det_auc:.4f}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 4 Overfit verification CLI")
    parser.add_argument("--out", type=str, default="I:/icvgip workshop/controlled_localization/runs/overfit_check", help="Output directory")
    parser.add_argument("--steps", type=int, default=60, help="Number of optimizer steps")
    parser.add_argument("--variant", type=str, default="T2", help="Model variant (T0, T1, T2, T2O)")
    args = parser.parse_args()

    run_overfit_experiment(Path(args.out), num_steps=args.steps, variant=args.variant)


if __name__ == "__main__":
    main()

