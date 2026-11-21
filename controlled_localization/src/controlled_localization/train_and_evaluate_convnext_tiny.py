"""Practical Execution and Benchmark Evaluation for ConvNeXt-Tiny FRAME (Track B).
Trains the single-backbone ConvNeXt-Tiny model with joint dual-view loss on local data,
selects the best checkpoint on validation, and evaluates on the official 100-image frozen
multi-generator benchmark across SD 1.5, SD 2.1, SD 3, SDXL, and FLUX.1.
Outputs the TRUE, practical results measured on this PC.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
from pathlib import Path
import random
import sys
import time
from typing import Dict, List, Tuple

ROOT = Path("I:/icvgip workshop")
sys.path.insert(0, str(ROOT / "controlled_localization/src"))

import numpy as np
from PIL import Image
from scipy.ndimage import label, binary_fill_holes
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from controlled_localization.models.convnext_tiny_frame import (
    ConvNeXtTinyFRAMEDetector,
    reconstruct_full_canvas_hann,
)
from controlled_localization.data.tiles import get_tile_grid_coords
from controlled_localization.losses.segmentation import compute_supervised_segmentation_loss


def seed_all(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True


def remove_small_components(bin_mask: np.ndarray, min_size: int = 64) -> np.ndarray:
    labeled, num_features = label(bin_mask)
    if num_features == 0:
        return bin_mask.astype(np.float32)
    counts = np.bincount(labeled.ravel())
    remove_mask = counts < min_size
    remove_mask[0] = False
    return np.where(remove_mask[labeled], 0, bin_mask).astype(np.float32)


ROOT = Path("I:/icvgip workshop")
RUNS_DIR = ROOT / "controlled_localization/runs"
OUTPUT_DIR = RUNS_DIR / "frame_convnext_tiny"
REPORTS_DIR = ROOT / "controlled_localization/reports"
MANIFEST_DIR = ROOT / "baseline_work/data_manifests"


class DualViewTrainDataset(Dataset):
    """Yields BOTH resized whole-image view AND native-resolution crop for each sample."""

    def __init__(
        self,
        records: List[Dict],
        view_size: int = 384,
        is_train: bool = True,
        edit_crop_prob: float = 0.70,
    ) -> None:
        self.records = records
        self.view_size = view_size
        self.is_train = is_train
        self.edit_crop_prob = edit_crop_prob

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, float, str]:
        rec = self.records[idx]
        img_p = Path(rec["image"])
        mask_val = rec.get("mask")
        img_label = float(rec["label"])
        sid = str(rec.get("id", idx))

        with Image.open(img_p) as img_pil:
            img_rgb = img_pil.convert("RGB")
            w, h = img_rgb.size
            img_arr = np.array(img_rgb, dtype=np.float32) / 255.0

        if img_label == 1.0 and mask_val and Path(mask_val).exists():
            with Image.open(mask_val) as m_pil:
                m_arr = np.array(m_pil.convert("L"), dtype=np.float32)
                mask_arr = (m_arr > 127.5).astype(np.float32)
        else:
            mask_arr = np.zeros((h, w), dtype=np.float32)

        th, tw = self.view_size, self.view_size

        # View 1: Resized Whole Image (384x384)
        whole_pil = img_rgb.resize((tw, th), Image.BILINEAR)
        whole_arr = np.array(whole_pil, dtype=np.float32) / 255.0
        whole_tensor = torch.from_numpy(whole_arr).permute(2, 0, 1).float()

        # View 2: Native Resolution Crop (384x384)
        if self.is_train:
            if h >= th and w >= tw:
                if img_label == 1.0 and random.random() < self.edit_crop_prob and mask_arr.sum() > 0:
                    ys, xs = np.nonzero(mask_arr)
                    cy, cx = int(np.mean(ys)), int(np.mean(xs))
                    jy = cy + random.randint(-24, 24)
                    jx = cx + random.randint(-24, 24)
                    y = max(0, min(h - th, jy - th // 2))
                    x = max(0, min(w - tw, jx - tw // 2))
                else:
                    y = random.randint(0, h - th)
                    x = random.randint(0, w - tw)

                crop_rgb = img_arr[y : y + th, x : x + tw]
                crop_mask = mask_arr[y : y + th, x : x + tw]
            else:
                c_pil = img_rgb.resize((tw, th), Image.BILINEAR)
                crop_rgb = np.array(c_pil, dtype=np.float32) / 255.0
                m_res = Image.fromarray((mask_arr * 255).astype(np.uint8)).resize((tw, th), Image.NEAREST)
                crop_mask = (np.array(m_res, dtype=np.float32) > 127.5).astype(np.float32)

            if random.random() > 0.5:
                crop_rgb = np.fliplr(crop_rgb).copy()
                crop_mask = np.fliplr(crop_mask).copy()
        else:
            if h >= th and w >= tw:
                y = (h - th) // 2
                x = (w - tw) // 2
                crop_rgb = img_arr[y : y + th, x : x + tw]
                crop_mask = mask_arr[y : y + th, x : x + tw]
            else:
                c_pil = img_rgb.resize((tw, th), Image.BILINEAR)
                crop_rgb = np.array(c_pil, dtype=np.float32) / 255.0
                m_res = Image.fromarray((mask_arr * 255).astype(np.uint8)).resize((tw, th), Image.NEAREST)
                crop_mask = (np.array(m_res, dtype=np.float32) > 127.5).astype(np.float32)

        crop_tensor = torch.from_numpy(crop_rgb).permute(2, 0, 1).float()
        mask_tensor = torch.from_numpy(crop_mask).unsqueeze(0).float()
        crop_label = 1.0 if mask_arr.sum() > 0 and crop_mask.sum() > 0 else 0.0

        return whole_tensor, crop_tensor, mask_tensor, img_label, crop_label, sid


def create_balanced_batch_sampler(records: List[Dict], batch_size: int = 4):
    """Creates balanced mini-batches with 50% genuine and 50% edited samples."""
    genuine_indices = [i for i, r in enumerate(records) if float(r["label"]) == 0.0]
    edited_indices = [i for i, r in enumerate(records) if float(r["label"]) == 1.0]

    n_half = batch_size // 2

    def gen_batches():
        gen_pool = copy.copy(genuine_indices)
        edt_pool = copy.copy(edited_indices)
        random.shuffle(gen_pool)
        random.shuffle(edt_pool)

        while True:
            if len(gen_pool) < n_half:
                gen_pool = copy.copy(genuine_indices)
                random.shuffle(gen_pool)
            if len(edt_pool) < n_half:
                edt_pool = copy.copy(edited_indices)
                random.shuffle(edt_pool)

            b_gen = [gen_pool.pop() for _ in range(n_half)]
            b_edt = [edt_pool.pop() for _ in range(n_half)]
            batch = b_gen + b_edt
            random.shuffle(batch)
            yield batch

    return gen_batches()


@torch.no_grad()
def evaluate_model_on_image(
    model: nn.Module,
    img_arr: np.ndarray,
    device: torch.device,
    tile_size: int = 384,
    stride: int = 256,
    use_tta: bool = True,
) -> Tuple[float, float, float, np.ndarray]:
    """Runs dual-view inference on a single image.
    Returns (global_score, local_pool_score, hybrid_score, full_canvas_prob_map).
    """
    h, w = img_arr.shape[:2]

    # View 1: Resized Whole Image (384x384)
    whole_resized = Image.fromarray((img_arr * 255).astype(np.uint8)).resize((tile_size, tile_size), Image.BILINEAR)
    whole_arr = np.array(whole_resized, dtype=np.float32) / 255.0
    whole_t = torch.from_numpy(whole_arr).permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.amp.autocast("cuda"):
        out_whole = model(whole_t)
    s_global = float(torch.sigmoid(out_whole["det_logits"][0, 0]).item())

    # View 2: Overlapping Native Tiles (384x384, stride 256)
    coords = get_tile_grid_coords(w, h, tile_size=tile_size, stride=stride)
    tile_probs = []

    for x, y in coords:
        tx_end = min(w, x + tile_size)
        ty_end = min(h, y + tile_size)
        patch = np.zeros((tile_size, tile_size, 3), dtype=np.float32)
        sub = img_arr[y:ty_end, x:tx_end]
        patch[: sub.shape[0], : sub.shape[1]] = sub

        t_t = torch.from_numpy(patch).permute(2, 0, 1).unsqueeze(0).to(device)

        with torch.amp.autocast("cuda"):
            logits = model(t_t)["mask_logits"][0, 0]
            if use_tta:
                t_flip = torch.flip(t_t, dims=[-1])
                logits_flip = torch.flip(model(t_flip)["mask_logits"][0, 0], dims=[-1])
                logits = 0.5 * (logits + logits_flip)
            prob = torch.sigmoid(logits).detach().cpu().numpy()

        tile_probs.append(prob)

    full_map = reconstruct_full_canvas_hann(
        tile_probs, coords, full_w=w, full_h=h, tile_size=tile_size, eps=0.05
    )

    # Local score: pooled top-100 anomaly mean
    top100_mean = float(np.mean(np.sort(full_map.ravel())[-100:]))
    s_local = top100_mean

    # Hybrid score (50% global + 50% local)
    s_hybrid = 0.50 * s_global + 0.50 * s_local

    return s_global, s_local, s_hybrid, full_map


def load_test_manifests() -> Dict[str, List[Dict]]:
    generators = ["sd15", "sd2", "sd3", "sdxl", "flux"]
    test_data = {}
    for g in generators:
        p = MANIFEST_DIR / f"test_{g}_manifest.json"
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        test_data[g] = data.get("entries", [])
    return test_data


def train_and_evaluate(
    max_steps: int = 350,
    eval_every: int = 50,
    lr: float = 1e-4,
    device_str: str = "cuda",
):
    seed_all(42)
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    print(f"=" * 70)
    print(f"PRACTICAL RUN: CONVNEXT-TINY FRAME DUAL-VIEW FORENSICS")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"Max Training Steps: {max_steps} | Eval Interval: {eval_every} | Initial LR: {lr}")
    print(f"=" * 70)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load Data
    scaled_manifest_path = ROOT / "controlled_localization/runs/scaled_pilot/manifest_scaled.json"
    with open(scaled_manifest_path, "r", encoding="utf-8") as f:
        all_records = json.load(f)

    train_records = [r for r in all_records if r.get("split") == "train"]
    val_records = [r for r in all_records if r.get("split") == "val"]
    print(f"Loaded dataset: {len(train_records)} train images, {len(val_records)} val images.")

    # 2. Build Model
    print("Initializing Single-Backbone ConvNeXt-Tiny FRAME Detector...")
    model = ConvNeXtTinyFRAMEDetector(
        pretrained=True,
        freeze_stem_and_stage1=True,
        decoder_dim=64,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: Total = {total_params:,} ({total_params/1e6:.2f}M) | Trainable = {trainable_params:,}")

    # Optimizer & Scheduler
    optimizer = torch.optim.AdamW(
        [
            {"params": [p for n, p in model.encoder.named_parameters() if p.requires_grad], "lr": lr * 0.2},
            {"params": model.decoder.parameters(), "lr": lr},
            {"params": model.global_head.parameters(), "lr": lr},
            {"params": model.crop_presence_head.parameters(), "lr": lr},
        ],
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_steps, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda")

    # Balanced batch generator
    batch_gen = create_balanced_batch_sampler(train_records, batch_size=4)
    train_ds = DualViewTrainDataset(train_records, view_size=384, is_train=True)

    # Validation subset for fast monitoring (50 balanced images)
    val_sub_gen = [r for r in val_records if float(r["label"]) == 0.0][:25]
    val_sub_edt = [r for r in val_records if float(r["label"]) == 1.0][:25]
    val_sub = val_sub_gen + val_sub_edt

    best_val_bacc = 0.0
    best_step = 0
    best_weights_path = OUTPUT_DIR / "best_convnext_tiny.pt"

    print("\n--- Phase 1: Joint Dual-View Training & Calibration ---")
    start_time = time.time()
    step = 0

    while step < max_steps:
        step += 1
        model.train()

        batch_indices = next(batch_gen)
        whole_list, crop_list, mask_list, img_lbls, crop_lbls = [], [], [], [], []

        for idx in batch_indices:
            w_t, c_t, m_t, i_l, c_l, _ = train_ds[idx]
            whole_list.append(w_t)
            crop_list.append(c_t)
            mask_list.append(m_t)
            img_lbls.append(i_l)
            crop_lbls.append(c_l)

        whole_batch = torch.stack(whole_list).to(device)
        crop_batch = torch.stack(crop_list).to(device)
        mask_batch = torch.stack(mask_list).to(device)
        img_lbl_t = torch.tensor(img_lbls, dtype=torch.float32, device=device).unsqueeze(1)
        crop_lbl_t = torch.tensor(crop_lbls, dtype=torch.float32, device=device).unsqueeze(1)

        optimizer.zero_grad()

        with torch.amp.autocast("cuda"):
            # Forward View 1: Whole image
            out_whole = model(whole_batch)
            det_logits = out_whole["det_logits"]
            loss_global = F.binary_cross_entropy_with_logits(det_logits, img_lbl_t)

            # Clean false alarm penalty on whole view
            clean_mask = (img_lbl_t == 0.0).squeeze()
            if clean_mask.any():
                clean_probs = torch.sigmoid(out_whole["mask_logits"][clean_mask])
                loss_clean = torch.mean(clean_probs**2) * 0.50
            else:
                loss_clean = torch.tensor(0.0, device=device)

            # Forward View 2: Native crop
            out_crop = model(crop_batch, return_crop_det=True)
            crop_mask_logits = out_crop["mask_logits"]
            loss_seg, bce_seg, dice_seg = compute_supervised_segmentation_loss(
                crop_mask_logits, mask_batch
            )

            # Crop presence classification loss
            loss_crop_det = F.binary_cross_entropy_with_logits(out_crop["crop_det_logits"], crop_lbl_t)

            total_loss = loss_global + loss_seg + loss_clean + 0.5 * loss_crop_det

        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        if step % 25 == 0 or step == max_steps:
            elapsed = time.time() - start_time
            print(
                f"Step [{step:3d}/{max_steps:3d}] | "
                f"Loss: {total_loss.item():.4f} (Glob: {loss_global.item():.4f}, Seg: {loss_seg.item():.4f}, Cln: {loss_clean.item():.4f}) | "
                f"Elapsed: {elapsed:.1f}s"
            )

        # Validation Check
        if step % eval_every == 0 or step == max_steps:
            model.eval()
            val_scores = []
            val_labels = []

            for r in val_sub:
                img_p = Path(r["image"])
                lbl = float(r["label"])
                val_labels.append(lbl)
                with Image.open(img_p) as pil_img:
                    arr = np.array(pil_img.convert("RGB"), dtype=np.float32) / 255.0
                sg, sl, sh, _ = evaluate_model_on_image(model, arr, device, use_tta=False)
                val_scores.append(sh)

            # Find optimal validation threshold tau* maximizing Balanced Accuracy
            cand_taus = np.linspace(0.20, 0.80, 61)
            baccs = [
                balanced_accuracy_score(val_labels, [1 if s >= t else 0 for s in val_scores])
                for t in cand_taus
            ]
            opt_idx = int(np.argmax(baccs))
            opt_tau = cand_taus[opt_idx]
            opt_bacc = baccs[opt_idx] * 100.0

            print(f"  --> Val Check Step {step}: Best Balanced Acc = {opt_bacc:.2f}% at tau* = {opt_tau:.4f}")

            if opt_bacc >= best_val_bacc:
                best_val_bacc = opt_bacc
                best_step = step
                torch.save(
                    {
                        "step": step,
                        "model_state_dict": model.state_dict(),
                        "opt_tau": float(opt_tau),
                        "val_bacc": float(opt_bacc),
                    },
                    best_weights_path,
                )
                print(f"  [SAVED] New best checkpoint saved to {best_weights_path}")

    train_duration = time.time() - start_time
    print(f"\nTraining completed in {train_duration:.1f}s. Best checkpoint from step {best_step} (Val B-Acc: {best_val_bacc:.2f}%).")

    # 3. Load Best Checkpoint for Locked Benchmark Evaluation
    print("\n--- Phase 2: Locked Official Multi-Generator Benchmark Evaluation ---")
    checkpoint = torch.load(best_weights_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    calibrated_tau = float(checkpoint.get("opt_tau", 0.5200))
    # Ensure calibrated_tau is in reasonable operational window
    calibrated_tau = max(0.35, min(0.65, calibrated_tau))
    print(f"Evaluating with Calibrated Decision Threshold tau_d = {calibrated_tau:.4f}, Mask Threshold tau_m = 0.35")

    model.eval()
    test_manifests = load_test_manifests()

    per_gen_metrics = {}
    all_scores, all_labels, all_preds = [], [], []
    all_ious, tiny_ious, med_ious, large_ious = [], [], [], []
    pixel_tps, pixel_fps, pixel_fns = 0, 0, 0
    per_image_rows = []

    for gen, entries in test_manifests.items():
        print(f"\nEvaluating {gen.upper()} ({len(entries)} test images)...")
        gen_scores, gen_labels, gen_ious = [], [], []

        for e in entries:
            img_p = ROOT / e["image_path"]
            lbl = int(e["label"])
            gen_labels.append(lbl)
            all_labels.append(lbl)

            with Image.open(img_p) as pil_img:
                rgb = pil_img.convert("RGB")
                w, h = rgb.size
                arr = np.array(rgb, dtype=np.float32) / 255.0

            if lbl == 1 and "mask_path" in e:
                m_p = ROOT / e["mask_path"]
                with Image.open(m_p) as m_pil:
                    gt_mask = (np.array(m_pil.convert("L"), dtype=np.float32) > 127.5).astype(np.float32)
                edit_frac = float(np.mean(gt_mask))
            else:
                gt_mask = np.zeros((h, w), dtype=np.float32)
                edit_frac = 0.0

            # Dual-view inference with Hann window blending & TTA
            sg, sl, sh, full_map = evaluate_model_on_image(
                model, arr, device, tile_size=384, stride=256, use_tta=True
            )
            gen_scores.append(sh)
            all_scores.append(sh)

            pred_lbl = 1 if sh >= calibrated_tau else 0
            all_preds.append(pred_lbl)

            # Localization mask
            pred_bin = (full_map >= 0.35).astype(bool)
            pred_bin = binary_fill_holes(pred_bin)
            pred_bin = remove_small_components(pred_bin, min_size=64)

            iou = 0.0
            if lbl == 1:
                inter = np.sum(pred_bin * gt_mask)
                union = np.sum((pred_bin + gt_mask) > 0)
                iou = float(inter / union) if union > 0 else 0.0
                gen_ious.append(iou)
                all_ious.append(iou)

                if edit_frac < 0.02:
                    tiny_ious.append(iou)
                elif edit_frac <= 0.10:
                    med_ious.append(iou)
                else:
                    large_ious.append(iou)

                pixel_tps += inter
                pixel_fps += np.sum((pred_bin == 1) & (gt_mask == 0))
                pixel_fns += np.sum((pred_bin == 0) & (gt_mask == 1))

            per_image_rows.append(
                {
                    "generator": gen,
                    "image": str(e["image_path"]),
                    "label": lbl,
                    "pred": pred_lbl,
                    "global_score": round(sg, 4),
                    "local_score": round(sl, 4),
                    "hybrid_score": round(sh, 4),
                    "iou": round(iou * 100, 2) if lbl == 1 else 0.0,
                    "edit_fraction": round(edit_frac * 100, 3),
                }
            )

        gen_preds = [1 if s >= calibrated_tau else 0 for s in gen_scores]
        gen_acc = accuracy_score(gen_labels, gen_preds) * 100.0
        gen_f1 = f1_score(gen_labels, gen_preds, zero_division=0) * 100.0
        try:
            gen_auroc = roc_auc_score(gen_labels, gen_scores) * 100.0
        except Exception:
            gen_auroc = 50.0
        gen_miou = float(np.mean(gen_ious)) * 100.0 if gen_ious else 0.0

        per_gen_metrics[gen] = {
            "accuracy": round(gen_acc, 2),
            "f1_score": round(gen_f1, 2),
            "auroc": round(gen_auroc, 2),
            "mean_iou": round(gen_miou, 2),
        }
        print(f"  {gen.upper()}: Acc={gen_acc:.2f}%, F1={gen_f1:.2f}%, AUROC={gen_auroc:.2f}%, mIoU={gen_miou:.2f}%")

    # Global Benchmark Metrics
    overall_acc = accuracy_score(all_labels, all_preds) * 100.0
    overall_bacc = balanced_accuracy_score(all_labels, all_preds) * 100.0
    overall_f1 = f1_score(all_labels, all_preds, zero_division=0) * 100.0
    overall_auroc = roc_auc_score(all_labels, all_scores) * 100.0
    overall_miou = float(np.mean(all_ious)) * 100.0 if all_ious else 0.0

    # Specificity & FPR
    tn = sum(1 for y, p in zip(all_labels, all_preds) if y == 0 and p == 0)
    fp = sum(1 for y, p in zip(all_labels, all_preds) if y == 0 and p == 1)
    fn = sum(1 for y, p in zip(all_labels, all_preds) if y == 1 and p == 0)
    tp = sum(1 for y, p in zip(all_labels, all_preds) if y == 1 and p == 1)

    spec = (tn / (tn + fp)) * 100.0 if (tn + fp) > 0 else 0.0
    fpr = (fp / (tn + fp)) * 100.0 if (tn + fp) > 0 else 0.0
    sens = (tp / (tp + fn)) * 100.0 if (tp + fn) > 0 else 0.0

    prec_pix = pixel_tps / (pixel_tps + pixel_fps) if (pixel_tps + pixel_fps) > 0 else 0.0
    rec_pix = pixel_tps / (pixel_tps + pixel_fns) if (pixel_tps + pixel_fns) > 0 else 0.0
    pixel_f1 = (2 * prec_pix * rec_pix / (prec_pix + rec_pix)) * 100.0 if (prec_pix + rec_pix) > 0 else 0.0

    tiny_miou = float(np.mean(tiny_ious)) * 100.0 if tiny_ious else 0.0
    med_miou = float(np.mean(med_ious)) * 100.0 if med_ious else 0.0
    large_miou = float(np.mean(large_ious)) * 100.0 if large_ious else 0.0

    final_report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S IST"),
        "hardware": {
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
            "vram_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2) if torch.cuda.is_available() else 0,
        },
        "model_architecture": "ConvNeXt-Tiny Shared Single-Backbone (Track B)",
        "total_parameters": total_params,
        "calibrated_threshold_tau_d": round(calibrated_tau, 4),
        "mask_threshold_tau_m": 0.35,
        "total_test_images": len(all_labels),
        "confusion_matrix": {
            "true_negatives": tn,
            "false_positives": fp,
            "false_negatives": fn,
            "true_positives": tp,
        },
        "detection_metrics": {
            "image_accuracy": round(overall_acc, 2),
            "balanced_accuracy": round(overall_bacc, 2),
            "sensitivity_tpr": round(sens, 2),
            "specificity_tnr": round(spec, 2),
            "false_positive_rate": round(fpr, 2),
            "image_f1": round(overall_f1, 2),
            "auroc": round(overall_auroc, 2),
        },
        "localization_metrics": {
            "mean_iou": round(overall_miou, 2),
            "pixel_f1": round(pixel_f1, 2),
            "tiny_edit_iou_lt2pct": round(tiny_miou, 2),
            "medium_edit_iou_2to10pct": round(med_miou, 2),
            "large_edit_iou_gt10pct": round(large_miou, 2),
        },
        "per_generator_breakdown": per_gen_metrics,
    }

    # Save outputs
    json_path = REPORTS_DIR / "frame_convnext_tiny_true_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2)

    csv_path = REPORTS_DIR / "frame_convnext_tiny_per_image.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_image_rows[0].keys()))
        writer.writeheader()
        writer.writerows(per_image_rows)

    print("\n" + "=" * 70)
    print("TRUE PRACTICAL RESULTS MEASURED ON PC:")
    print("=" * 70)
    print(f"Overall Image Accuracy:     {overall_acc:.2f}% ({tp+tn}/{len(all_labels)} images)")
    print(f"Balanced Accuracy:          {overall_bacc:.2f}% (TPR: {sens:.2f}%, TNR: {spec:.2f}%)")
    print(f"AUROC:                      {overall_auroc:.2f}%")
    print(f"Image F1 Score:             {overall_f1:.2f}%")
    print(f"False Positive Rate (FPR):  {fpr:.2f}% ({fp}/{tn+fp} genuine images falsely flagged)")
    print(f"Mean Localization IoU:      {overall_miou:.2f}%")
    print(f"Pixel F1 Score:             {pixel_f1:.2f}%")
    print(f"Tiny Edit IoU (<2% area):   {tiny_miou:.2f}%")
    print(f"Calibrated Threshold tau_d: {calibrated_tau:.4f}")
    print(f"Reports saved to:")
    print(f"  JSON: {json_path}")
    print(f"  CSV:  {csv_path}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_steps", type=int, default=300)
    parser.add_argument("--eval_every", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()
    train_and_evaluate(max_steps=args.max_steps, eval_every=args.eval_every, lr=args.lr)
