"""Phase 1: Matched Baseline Pilot Comparison.
Compares:
  Run A: Legacy noisy-crop supervision (crop_label = img_label regardless of edit presence)
  Run B: Corrected supervision (crop_label decoupled, smart edit sampling, verified masks)
Both evaluated under identical data, backbone, seed, optimizer, and compute budget.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import random
import time
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from controlled_localization.models.super_frame_v2_detector import SuperFRAMEV2Detector
from controlled_localization.data.corrected_dataset import CorrectedOpenSDIDataset
from controlled_localization.data.tiles import get_tile_grid_coords, blend_tiles_probability
from controlled_localization.losses.segmentation import compute_supervised_segmentation_loss
from controlled_localization.utils import seed_all


ROOT = Path("I:/icvgip workshop")
RUNS_DIR = ROOT / "controlled_localization/runs"
DEFAULT_MANIFEST = RUNS_DIR / "real_development_20261114/manifest_multigen.json"
INIT_CKPT = RUNS_DIR / "super_frame_v2_run/best.pt"


class LegacyOpenSDIDataset(Dataset):
    """Legacy dataset that passes whole-image label as crop label."""

    def __init__(self, records: List[Dict], crop_size: int = 384, is_train: bool = True) -> None:
        self.records = records
        self.crop_size = crop_size
        self.is_train = is_train

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, float, float, str]:
        rec = self.records[idx]
        img_p = Path(rec["image"])
        mask_val = rec.get("mask")
        img_label = float(rec["label"])
        sample_id = str(rec.get("id", idx))

        with Image.open(img_p) as img_pil:
            img_rgb = img_pil.convert("RGB")
            w, h = img_rgb.size
            img_arr = np.array(img_rgb, dtype=np.float32) / 255.0

        if img_label == 1.0 and mask_val is not None and Path(mask_val).exists():
            with Image.open(mask_val) as m_pil:
                m_arr = np.array(m_pil.convert("L"), dtype=np.float32)
                mask_binary = (m_arr > 127.5).astype(np.float32)
        else:
            mask_binary = np.zeros((h, w), dtype=np.float32)

        th, tw = self.crop_size, self.crop_size
        if self.is_train:
            if h > th and w > tw:
                y = random.randint(0, h - th)
                x = random.randint(0, w - tw)
                crop_rgb = img_arr[y : y + th, x : x + tw]
                crop_mask = mask_binary[y : y + th, x : x + tw]
            else:
                res_img = Image.fromarray((img_arr * 255).astype(np.uint8)).resize((tw, th), Image.BILINEAR)
                res_mask = Image.fromarray((mask_binary * 255).astype(np.uint8)).resize((tw, th), Image.NEAREST)
                crop_rgb = np.array(res_img, dtype=np.float32) / 255.0
                crop_mask = (np.array(res_mask, dtype=np.float32) > 127.5).astype(np.float32)

            if random.random() > 0.5:
                crop_rgb = np.fliplr(crop_rgb).copy()
                crop_mask = np.fliplr(crop_mask).copy()
        else:
            if h >= th and w >= tw:
                y = (h - th) // 2
                x = (w - tw) // 2
                crop_rgb = img_arr[y : y + th, x : x + tw]
                crop_mask = mask_binary[y : y + th, x : x + tw]
            else:
                res_img = Image.fromarray((img_arr * 255).astype(np.uint8)).resize((tw, th), Image.BILINEAR)
                res_mask = Image.fromarray((mask_binary * 255).astype(np.uint8)).resize((tw, th), Image.NEAREST)
                crop_rgb = np.array(res_img, dtype=np.float32) / 255.0
                crop_mask = (np.array(res_mask, dtype=np.float32) > 127.5).astype(np.float32)

        # In legacy mode, crop_label is forced equal to img_label!
        crop_label = img_label

        rgb_t = torch.from_numpy(crop_rgb).permute(2, 0, 1).float()
        mask_t = torch.from_numpy(crop_mask).unsqueeze(0).float()
        return rgb_t, mask_t, crop_label, img_label, sample_id


def evaluate_model_multievidence(
    model: nn.Module,
    records: List[Dict],
    device: torch.device,
    tile_size: int = 384,
    stride: int = 256,
    max_eval_samples: int = 150,
) -> Dict[str, float]:
    """Evaluates full image + native tile multi-evidence aggregation."""
    model.eval()
    all_ious = []
    clean_fps = []
    labels_all = []
    hybrid_scores = []
    whole_scores = []

    eval_recs = records[:max_eval_samples] if max_eval_samples > 0 else records

    with torch.no_grad():
        for rec in eval_recs:
            img_p = Path(rec["image"])
            mask_p = rec.get("mask")
            lbl = float(rec["label"])
            labels_all.append(lbl)

            with Image.open(img_p) as pil_img:
                rgb_full = pil_img.convert("RGB")
                orig_w, orig_h = rgb_full.size
                arr_full = np.array(rgb_full, dtype=np.float32) / 255.0

            # Ground truth mask
            if lbl == 1.0 and mask_p is not None and Path(mask_p).exists():
                with Image.open(mask_p) as m_img:
                    gt_mask = (np.array(m_img.convert("L"), dtype=np.float32) > 127.5).astype(np.float32)
            else:
                gt_mask = np.zeros((orig_h, orig_w), dtype=np.float32)

            # 1. Whole-image view (resized to 384x384)
            whole_resized = rgb_full.resize((tile_size, tile_size), Image.BILINEAR)
            whole_arr = np.array(whole_resized, dtype=np.float32) / 255.0
            whole_t = torch.from_numpy(whole_arr).permute(2, 0, 1).unsqueeze(0).float().to(device)

            with torch.amp.autocast("cuda"):
                whole_out = model(whole_t)
            whole_det_logit = whole_out["det_logits"][0, 0].item()
            s_whole = float(torch.sigmoid(torch.tensor(whole_det_logit)).item())
            whole_scores.append(s_whole)

            # 2. Native Tiles (Hann-window blended)
            coords = get_tile_grid_coords(orig_w, orig_h, tile_size=tile_size, stride=stride)
            # Sample at most 6 tiles for speed during eval
            if len(coords) > 6:
                coords = [coords[0], coords[len(coords) // 4], coords[len(coords) // 2], coords[3 * len(coords) // 4], coords[-1]]

            tile_logits = []
            for x, y in coords:
                # Extract tile (pad if needed)
                tx_end = min(orig_w, x + tile_size)
                ty_end = min(orig_h, y + tile_size)
                crop_patch = np.zeros((tile_size, tile_size, 3), dtype=np.float32)
                sub_arr = arr_full[y:ty_end, x:tx_end]
                crop_patch[: sub_arr.shape[0], : sub_arr.shape[1]] = sub_arr

                tile_t = torch.from_numpy(crop_patch).permute(2, 0, 1).unsqueeze(0).float().to(device)
                with torch.amp.autocast("cuda"):
                    t_out = model(tile_t)
                t_mask_l = t_out["mask_logits"][0, 0].cpu().numpy()
                tile_logits.append(t_mask_l)

            blended_map = blend_tiles_probability(
                tile_logits, coords, orig_w, orig_h, tile_size=tile_size
            )

            # Mask pooling score
            top_k_pixels = np.sort(blended_map.ravel())[-100:]
            s_pool = float(np.mean(top_k_pixels))

            # Calibrated Hybrid score
            s_hybrid = 0.15 * s_whole + 0.85 * s_pool
            hybrid_scores.append(s_hybrid)

            # Localization evaluation (at threshold 0.35)
            pred_bin = (blended_map > 0.35).astype(np.float32)

            if lbl == 1.0:
                inter = np.sum(pred_bin * gt_mask)
                union = np.sum((pred_bin + gt_mask) > 0)
                iou = float(inter / union) if union > 0 else 0.0
                all_ious.append(iou)
            else:
                fp_rate = float(np.mean(pred_bin > 0))
                clean_fps.append(fp_rate)

    mean_iou = float(np.mean(all_ious)) if all_ious else 0.0
    mean_fp = float(np.mean(clean_fps)) if clean_fps else 0.0

    try:
        auroc_whole = float(roc_auc_score(labels_all, whole_scores))
        auroc_hybrid = float(roc_auc_score(labels_all, hybrid_scores))
    except Exception:
        auroc_whole = 0.5
        auroc_hybrid = 0.5

    # Compute classification accuracy at optimal threshold on validation
    best_acc = 0.0
    best_f1 = 0.0
    for thresh in np.linspace(0.1, 0.9, 33):
        preds = [1 if s >= thresh else 0 for s in hybrid_scores]
        acc = accuracy_score(labels_all, preds)
        f1 = f1_score(labels_all, preds, zero_division=0)
        if acc > best_acc:
            best_acc = acc
            best_f1 = f1

    return {
        "mean_iou": round(mean_iou * 100, 2),
        "mean_fp_rate": round(mean_fp * 100, 3),
        "auroc_whole": round(auroc_whole * 100, 2),
        "auroc_hybrid": round(auroc_hybrid * 100, 2),
        "accuracy": round(best_acc * 100, 2),
        "f1": round(best_f1 * 100, 2),
    }


def train_single_regime(
    mode: str,
    train_recs: List[Dict],
    val_recs: List[Dict],
    steps: int = 300,
    batch_size: int = 4,
    grad_accum: int = 2,
    seed: int = 42,
    device: torch.device = torch.device("cuda"),
    save_dir: Path = RUNS_DIR / "phase1_pilot",
) -> Tuple[nn.Module, Dict]:
    seed_all(seed)
    print(f"\n=======================================================")
    print(f"STARTING PILOT REGIME: {mode.upper()} ({steps} steps)")
    print(f"=======================================================")

    save_dir.mkdir(parents=True, exist_ok=True)

    if mode == "legacy_noisy":
        train_ds = LegacyOpenSDIDataset(train_recs, crop_size=384, is_train=True)
    else:
        train_ds = CorrectedOpenSDIDataset(
            train_recs, crop_size=384, is_train=True, edit_crop_prob=0.65, strict_mask_validation=True
        )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0
    )

    # Instantiate model and initialize identical weights
    model = SuperFRAMEV2Detector(pretrained=True).to(device)
    if INIT_CKPT.exists():
        ckpt = torch.load(INIT_CKPT, map_location=device)
        sd = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        model.load_state_dict(sd, strict=False)
        print(f"Loaded starting weights from: {INIT_CKPT}")

    # Differential parameter groups
    backbone_params = []
    new_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "semantic_encoder" in name or "c_stage" in name:
            backbone_params.append(param)
        else:
            new_params.append(param)

    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": 3e-5},
            {"params": new_params, "lr": 1.5e-4},
        ],
        weight_decay=1e-4,
    )
    scaler = torch.amp.GradScaler("cuda")
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps, eta_min=1e-6)

    train_iter = iter(train_loader)
    optimizer.zero_grad()

    t0 = time.time()
    history = []

    for step in range(1, steps + 1):
        model.train()
        try:
            rgbs, masks, crop_labels, img_labels, _ = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            rgbs, masks, crop_labels, img_labels, _ = next(train_iter)

        rgbs = rgbs.to(device)
        masks = masks.to(device)
        crop_labels = crop_labels.to(device).unsqueeze(1).float()
        img_labels = img_labels.to(device).unsqueeze(1).float()

        with torch.amp.autocast("cuda"):
            out = model(rgbs)
            mask_logits = out["mask_logits"]
            det_logits = out["det_logits"]

            if mode == "legacy_noisy":
                # Old noisy regime: target is img_label
                targets = img_labels
                manip_mask = (targets > 0.5).squeeze(1)
                clean_mask = (targets < 0.5).squeeze(1)
            else:
                # Corrected regime: target is crop_label
                targets = crop_labels
                manip_mask = (targets > 0.5).squeeze(1)
                clean_mask = (targets < 0.5).squeeze(1)

            # Segmentation loss for crops containing edits
            if manip_mask.any():
                seg_loss, _, _ = compute_supervised_segmentation_loss(
                    mask_logits[manip_mask], masks[manip_mask]
                )
            else:
                seg_loss = torch.tensor(0.0, device=device)

            # False alarm penalty on pristine background crops
            if clean_mask.any():
                fp_probs = torch.sigmoid(mask_logits[clean_mask])
                fp_loss = fp_probs.mean()
            else:
                fp_loss = torch.tensor(0.0, device=device)

            # Detection classification loss
            det_loss = F.binary_cross_entropy_with_logits(det_logits, targets)

            # Total loss (scaled by grad_accum)
            total_loss = (seg_loss + 0.5 * det_loss + 0.3 * fp_loss) / grad_accum

        scaler.scale(total_loss).backward()

        if step % grad_accum == 0 or step == steps:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()

        if step % 50 == 0 or step == steps:
            curr_loss = total_loss.item() * grad_accum
            s_l = seg_loss.item()
            d_l = det_loss.item()
            f_l = fp_loss.item()
            print(
                f"[{mode.upper()}] Step {step:3d}/{steps} | Loss: {curr_loss:.4f} "
                f"(Seg: {s_l:.4f}, Det: {d_l:.4f}, FP: {f_l:.4f}) | Elapsed: {(time.time()-t0):.1f}s"
            )
            history.append({"step": step, "loss": curr_loss, "seg": s_l, "det": d_l, "fp": f_l})

    # Save checkpoint
    ckpt_path = save_dir / f"{mode}_pilot_checkpoint.pt"
    torch.save({"step": steps, "model_state_dict": model.state_dict(), "mode": mode}, ckpt_path)
    print(f"Saved {mode} checkpoint to: {ckpt_path}")

    # Validation evaluation
    print(f"Evaluating {mode} on validation set (280 images)...")
    val_results = evaluate_model_multievidence(model, val_recs, device, max_eval_samples=150)
    print(f"[{mode.upper()} Results] IoU: {val_results['mean_iou']}%, AUROC (Hybrid): {val_results['auroc_hybrid']}%, Acc: {val_results['accuracy']}%, F1: {val_results['f1']}%, FP Rate: {val_results['mean_fp_rate']}%")

    val_results["history"] = history
    val_results["checkpoint"] = str(ckpt_path)
    return model, val_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=200, help="Steps per regime (default: 200)")
    parser.add_argument("--manifest", type=str, default=str(DEFAULT_MANIFEST))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using compute device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    with open(args.manifest, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    train_recs = [r for r in manifest if r["split"] == "train"]
    val_recs = [r for r in manifest if r["split"] == "val"]
    print(f"Loaded {len(train_recs)} training records and {len(val_recs)} validation records.")

    # 1. Run A: Legacy Noisy Supervision
    _, results_a = train_single_regime(
        mode="legacy_noisy",
        train_recs=train_recs,
        val_recs=val_recs,
        steps=args.steps,
        seed=args.seed,
        device=device,
    )

    # 2. Run B: Corrected Supervision
    _, results_b = train_single_regime(
        mode="corrected",
        train_recs=train_recs,
        val_recs=val_recs,
        steps=args.steps,
        seed=args.seed,
        device=device,
    )

    # 3. Compile comparative report
    delta_iou = results_b["mean_iou"] - results_a["mean_iou"]
    delta_auroc = results_b["auroc_hybrid"] - results_a["auroc_hybrid"]
    delta_acc = results_b["accuracy"] - results_a["accuracy"]
    delta_f1 = results_b["f1"] - results_a["f1"]
    delta_fp = results_b["mean_fp_rate"] - results_a["mean_fp_rate"]

    comparison = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S IST"),
        "training_budget": {
            "steps": args.steps,
            "seed": args.seed,
            "batch_size": 4,
            "grad_accum": 2,
            "effective_batch_size": 8,
            "initialization": str(INIT_CKPT),
            "manifest": args.manifest,
            "train_samples": len(train_recs),
            "val_samples": len(val_recs),
        },
        "regimes": {
            "run_a_legacy_noisy": results_a,
            "run_b_corrected": results_b,
        },
        "delta_b_minus_a": {
            "delta_mean_iou": round(delta_iou, 2),
            "delta_auroc_hybrid": round(delta_auroc, 2),
            "delta_accuracy": round(delta_acc, 2),
            "delta_f1": round(delta_f1, 2),
            "delta_fp_rate": round(delta_fp, 3),
        },
    }

    out_json = ROOT / "controlled_localization/reports/phase1_matched_pilot_results.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)

    print("\n=======================================================")
    print("PHASE 1 MATCHED PILOT SUMMARY:")
    print("=======================================================")
    print(f"Metric                | Legacy Noisy | Corrected | Delta")
    print(f"----------------------|--------------|-----------|-------")
    print(f"Mean IoU (%)          | {results_a['mean_iou']:>12.2f} | {results_b['mean_iou']:>9.2f} | {delta_iou:>+6.2f}%")
    print(f"AUROC Hybrid (%)      | {results_a['auroc_hybrid']:>12.2f} | {results_b['auroc_hybrid']:>9.2f} | {delta_auroc:>+6.2f}%")
    print(f"Accuracy (%)          | {results_a['accuracy']:>12.2f} | {results_b['accuracy']:>9.2f} | {delta_acc:>+6.2f}%")
    print(f"F1 Score (%)          | {results_a['f1']:>12.2f} | {results_b['f1']:>9.2f} | {delta_f1:>+6.2f}%")
    print(f"Clean False Alarm (%) | {results_a['mean_fp_rate']:>12.3f} | {results_b['mean_fp_rate']:>9.3f} | {delta_fp:>+6.3f}%")
    print(f"Saved complete audit report to: {out_json}")


if __name__ == "__main__":
    main()
