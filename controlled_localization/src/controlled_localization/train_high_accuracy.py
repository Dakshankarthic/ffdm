"""High-Accuracy Fine-Tuning Pipeline for Controlled Localization v1.1.

Optimizes for HIGH localization IoU while strictly suppressing clean-image false alarms.
Uses:
  1. Balanced segmentation loss with false-positive penalty on genuine images.
  2. Directed Bernoulli KL view consistency (T2).
  3. Fitness-driven checkpointing: fitness = mean_iou - 0.5 * genuine_positive_area.
  4. Initialization from T1_extended best checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from controlled_localization.models.detector import ControlledLocalizationDetector
from controlled_localization.losses.segmentation import compute_supervised_segmentation_loss
from controlled_localization.losses.consistency import compute_directed_consistency_loss
from controlled_localization.data.canonicalize import apply_recipe
from controlled_localization.infer import run_single_image_inference
from controlled_localization.policy import InferencePolicy
from controlled_localization.utils import seed_all


def load_sample(r: dict) -> tuple[Image.Image, np.ndarray]:
    im = Image.open(r["image"]).convert("RGB")
    if not r.get("mask"):
        m = np.zeros((im.height, im.width), dtype=np.float32)
    else:
        m = np.asarray(Image.open(r["mask"]), dtype=np.float32) / 255.0
    return im, m


def evaluate_val_set(
    model: ControlledLocalizationDetector,
    records: list[dict],
    device: torch.device,
    t_mask: float = 0.50,
) -> dict[str, float]:
    model.eval()
    policy = InferencePolicy(policy_name="val_eval", t_mask=t_mask, t_image=0.50)
    
    ious = []
    clean_fp_areas = []
    y_true = []
    y_scores = []

    for r in records:
        res = run_single_image_inference(r["image"], model, policy, device)
        p = res["merged_prob"]
        _, m = load_sample(r)
        
        y = m > 0.5
        pred = p > t_mask
        
        y_true.append(r["label"])
        y_scores.append(float(res["raw_score"]))

        if r["label"] == 1:
            inter = (pred & y).sum()
            union = (pred | y).sum()
            iou = float(inter / union) if union > 0 else 0.0
            ious.append(iou)
        else:
            clean_fp_areas.append(float(pred.mean()))

    mean_iou = float(np.mean(ious)) if ious else 0.0
    clean_fp = float(np.mean(clean_fp_areas)) if clean_fp_areas else 0.0
    auroc = float(roc_auc_score(y_true, y_scores)) if len(np.unique(y_true)) > 1 else 0.5

    # Fitness: high IoU penalized by clean false positive area
    fitness = mean_iou - 0.5 * clean_fp

    model.train()
    return {
        "mean_iou": round(mean_iou, 4),
        "clean_fp": round(clean_fp, 4),
        "auroc": round(auroc, 4),
        "fitness": round(fitness, 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train high-accuracy detector on real data")
    parser.add_argument("--steps", type=int, default=600, help="Number of optimizer steps")
    parser.add_argument("--init-ckpt", type=str, default="runs/real_development_20261114/T1_extended/best.pt", help="Warmstart checkpoint")
    parser.add_argument("--out-dir", type=str, default="runs/high_accuracy_real_t2", help="Output directory")
    parser.add_argument("--fp-penalty-weight", type=float, default=1.5, help="Weight on false positive suppression")
    parser.add_argument("--consistency-weight", type=float, default=0.15, help="Weight on directed KL consistency")
    args = parser.parse_args()

    seed_all(42)
    rng = np.random.default_rng(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base_dir = Path("I:/icvgip workshop/controlled_localization")
    manifest_p = base_dir / "runs/real_development_20261114/manifest.json"
    if not manifest_p.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_p}")

    records = json.loads(manifest_p.read_text(encoding="utf-8"))
    train_rows = [r for r in records if r["split"] == "train"]
    val_rows = [r for r in records if r["split"] == "val"]

    print(f"Loaded dataset: {len(train_rows)} training, {len(val_rows)} validation samples.")

    out_dir = base_dir / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Initialize model
    model = ControlledLocalizationDetector(pretrained_rgb=False)
    if args.init_ckpt and Path(args.init_ckpt).exists():
        print(f"Initializing weights from: {args.init_ckpt}")
        ckpt = torch.load(args.init_ckpt, map_location="cpu", weights_only=False)
        st = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        model.load_state_dict(st)
    model.to(device)
    model.train()

    # Preload training images into memory
    print("Pre-caching training images into RAM...")
    cache = {r["id"]: load_sample(r) for r in train_rows}
    classes = {
        0: [r for r in train_rows if r["label"] == 0],
        1: [r for r in train_rows if r["label"] == 1],
    }

    # Fine-tuning learning rates
    optimizer = torch.optim.AdamW(
        [
            {"params": [p for p in model.rgb_encoder.parameters() if p.requires_grad], "lr": 5e-5},
            {"params": list(model.res_encoder.parameters()) + list(model.decoder.parameters()), "lr": 1.5e-4},
        ],
        weight_decay=1e-4,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())

    recipes = ["identity", "jpeg_q70", "jpeg_q85", "jpeg_q95", "blur_s05"]

    # Initial validation score
    print("\nRunning initial baseline validation before fine-tuning...")
    init_metrics = evaluate_val_set(model, val_rows, device)
    print(f"Initial metrics: IoU={init_metrics['mean_iou']}, Clean FP={init_metrics['clean_fp']}, AUROC={init_metrics['auroc']}, Fitness={init_metrics['fitness']}")

    best_fitness = init_metrics["fitness"]
    best_iou = init_metrics["mean_iou"]

    log_path = out_dir / "train_log.jsonl"
    log_file = open(log_path, "w", encoding="utf-8")

    start_train_time = time.perf_counter()
    print(f"\nStarting high-accuracy fine-tuning for {args.steps} steps...")

    for step in range(1, args.steps + 1):
        xs, ys, is_clean = [], [], []

        # Balanced batch: 1 genuine image, 1 manipulated image
        for label in [0, 1]:
            r = classes[label][int(rng.integers(len(classes[label])))]
            im, m = cache[r["id"]]

            recipe = recipes[int(rng.integers(len(recipes)))]
            processed, _ = apply_recipe(im, recipe)

            a = np.asarray(im, dtype=np.float32) / 255.0
            b = np.asarray(processed, dtype=np.float32) / 255.0

            h, w = m.shape
            ph, pw = max(0, 384 - h), max(0, 384 - w)
            if ph > 0 or pw > 0:
                a = np.pad(a, ((0, ph), (0, pw), (0, 0)), mode="edge")
                b = np.pad(b, ((0, ph), (0, pw), (0, 0)), mode="edge")
                m = np.pad(m, ((0, ph), (0, pw)))
            h, w = m.shape

            # Crop selection: focus on edit mask 70% of the time for positive samples
            if label == 1 and rng.random() < 0.70 and (m > 0.5).any():
                pts = np.argwhere(m > 0.5)
                cy, cx = pts[int(rng.integers(len(pts)))]
                x = int(np.clip(cx - 192, 0, w - 384))
                y = int(np.clip(cy - 192, 0, h - 384))
            else:
                x = int(rng.integers(max(1, w - 384 + 1)))
                y = int(rng.integers(max(1, h - 384 + 1)))

            crop_a = a[y : y + 384, x : x + 384].transpose(2, 0, 1)
            crop_b = b[y : y + 384, x : x + 384].transpose(2, 0, 1)
            crop_m = m[y : y + 384, x : x + 384][None]

            xs.extend([crop_a, crop_b])
            ys.extend([crop_m, crop_m])
            is_clean.extend([label == 0, label == 0])

        x_t = torch.tensor(np.stack(xs), device=device)
        y_t = torch.tensor(np.stack(ys), device=device)
        clean_mask = torch.tensor(is_clean, device=device)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast("cuda"):
            out = model(x_t)
            z = out["mask_logits"]

            # Standard supervised segmentation loss
            l_seg, l_bce, l_dice = compute_supervised_segmentation_loss(z, y_t)

            # False-positive penalty strictly on genuine crops
            if clean_mask.any():
                z_clean = z[clean_mask]
                prob_clean = torch.sigmoid(z_clean)
                # Penalize any activation above 0.05 on genuine pixels
                l_clean_fp = F.softplus(z_clean).mean() + 2.0 * torch.clamp(prob_clean - 0.05, min=0.0).mean()
            else:
                l_clean_fp = torch.tensor(0.0, device=device)

            # Directed consistency loss: clean teacher (z[0::2]) guides degraded student (z[1::2])
            l_cons = compute_directed_consistency_loss(
                student_logits=z[1::2],
                teacher_logits=z[0::2],
                gt_mask=y_t[0::2],
            )

            # Total loss
            total_loss = l_seg + args.fp_penalty_weight * l_clean_fp + args.consistency_weight * l_cons

        if not torch.isfinite(total_loss):
            print(f"Warning: non-finite loss at step {step}, skipping update.")
            continue

        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        log_entry = {
            "step": step,
            "total_loss": round(float(total_loss), 4),
            "seg_loss": round(float(l_seg), 4),
            "clean_fp_loss": round(float(l_clean_fp), 4),
            "cons_loss": round(float(l_cons), 4),
            "lr": round(scheduler.get_last_lr()[0], 7),
        }
        log_file.write(json.dumps(log_entry) + "\n")
        log_file.flush()

        if step % 25 == 0:
            elapsed = int(time.perf_counter() - start_train_time)
            print(f"Step {step:3d}/{args.steps} | Loss: {total_loss:.4f} (seg: {l_seg:.4f}, fp: {l_clean_fp:.4f}, cons: {l_cons:.4f}) | {elapsed}s")

        # Validation check every 50 steps
        if step % 50 == 0 or step == args.steps:
            val_metrics = evaluate_val_set(model, val_rows, device)
            print(f"--> [Val Step {step}] IoU: {val_metrics['mean_iou']:.4f} | Clean FP: {val_metrics['clean_fp']:.4f} | AUROC: {val_metrics['auroc']:.4f} | Fitness: {val_metrics['fitness']:.4f}")

            # Save latest
            torch.save(
                {
                    "step": step,
                    "model_state_dict": model.state_dict(),
                    "val_metrics": val_metrics,
                },
                out_dir / "latest.pt",
            )

            # Save best by fitness
            if val_metrics["fitness"] > best_fitness or val_metrics["mean_iou"] > best_iou:
                if val_metrics["fitness"] > best_fitness:
                    best_fitness = val_metrics["fitness"]
                if val_metrics["mean_iou"] > best_iou:
                    best_iou = val_metrics["mean_iou"]
                print(f"*** New BEST checkpoint at step {step}! (IoU={val_metrics['mean_iou']}, Clean FP={val_metrics['clean_fp']}) ***")
                torch.save(
                    {
                        "step": step,
                        "model_state_dict": model.state_dict(),
                        "val_metrics": val_metrics,
                    },
                    out_dir / "best.pt",
                )

    log_file.close()
    total_time = time.perf_counter() - start_train_time
    print(f"\nTraining completed in {total_time:.1f}s. Best fitness: {best_fitness:.4f}")


if __name__ == "__main__":
    main()
