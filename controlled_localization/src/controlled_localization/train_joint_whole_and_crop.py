"""Experiment 2: Joint Whole-Image and Crop Decision Training.
Solves the train/test domain mismatch and score compression:
1. Resized Whole Image (384x384): Trains whole-image classification head with img_label (0.0 genuine, 1.0 edited).
2. Native-Resolution Crop (384x384): Trains local segmentation & edit presence with crop_label.
3. Balanced Batch Sampling: 50% genuine, 50% edited images per batch to suppress false alarms.
4. Periodic Validation & Checkpoint Selection: Selects best checkpoint based on validation Balanced Accuracy.
5. Saves score distributions to measure true score separation.
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
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from controlled_localization.models.super_frame_v2_detector import SuperFRAMEV2Detector
from controlled_localization.data.tiles import get_tile_grid_coords, blend_tiles_probability
from controlled_localization.losses.segmentation import compute_supervised_segmentation_loss
from controlled_localization.utils import seed_all


ROOT = Path("I:/icvgip workshop")
RUNS_DIR = ROOT / "controlled_localization/runs"
DEV_MANIFEST = RUNS_DIR / "real_development_20261114/manifest_multigen.json"
INIT_CKPT = RUNS_DIR / "super_frame_v2_run/best.pt"


class JointDualViewDataset(Dataset):
    """Yields BOTH resized whole-image view AND native-resolution crop for each sample."""

    def __init__(
        self,
        records: List[Dict],
        view_size: int = 384,
        is_train: bool = True,
        edit_crop_prob: float = 0.65,
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
                    jy = cy + random.randint(-32, 32)
                    jx = cx + random.randint(-32, 32)
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

        crop_label = 1.0 if np.sum(crop_mask) > 0 else 0.0

        crop_rgb_tensor = torch.from_numpy(crop_rgb).permute(2, 0, 1).float()
        crop_mask_tensor = torch.from_numpy(crop_mask).unsqueeze(0).float()

        return whole_tensor, crop_rgb_tensor, crop_mask_tensor, img_label, crop_label, sid


class BalancedBatchSampler:
    """Ensures each batch contains an exact 50/50 balance of genuine and edited images."""

    def __init__(self, records: List[Dict], batch_size: int = 4) -> None:
        self.batch_size = batch_size
        self.half_batch = batch_size // 2
        self.genuine_indices = [i for i, r in enumerate(records) if float(r["label"]) == 0.0]
        self.edited_indices = [i for i, r in enumerate(records) if float(r["label"]) == 1.0]

    def __iter__(self):
        gen_copy = list(self.genuine_indices)
        edt_copy = list(self.edited_indices)
        random.shuffle(gen_copy)
        random.shuffle(edt_copy)

        idx_g, idx_e = 0, 0
        num_batches = min(len(gen_copy) // self.half_batch, len(edt_copy) // self.half_batch)

        for _ in range(num_batches):
            batch = gen_copy[idx_g : idx_g + self.half_batch] + edt_copy[idx_e : idx_e + self.half_batch]
            random.shuffle(batch)
            yield batch
            idx_g += self.half_batch
            idx_e += self.half_batch

    def __len__(self) -> int:
        return min(len(self.genuine_indices) // self.half_batch, len(self.edited_indices) // self.half_batch)


@torch.no_grad()
def evaluate_validation_separation(
    model: nn.Module,
    val_records: List[Dict],
    device: torch.device,
    tile_size: int = 384,
    max_samples: int = 140,
) -> Dict:
    model.eval()
    val_subset = val_records[:max_samples] if max_samples > 0 else val_records

    labels = []
    whole_scores = []
    pool_scores = []
    hybrid_scores = []
    ious = []

    for rec in val_subset:
        lbl = float(rec["label"])
        labels.append(lbl)

        with Image.open(rec["image"]) as img:
            arr_full = np.array(img.convert("RGB"), dtype=np.float32) / 255.0
            w, h = img.size

        if lbl == 1.0 and rec.get("mask") and Path(rec["mask"]).exists():
            with Image.open(rec["mask"]) as m_img:
                gt_mask = (np.array(m_img.convert("L"), dtype=np.float32) > 127.5).astype(np.float32)
        else:
            gt_mask = np.zeros((h, w), dtype=np.float32)

        # Whole view
        whole_pil = Image.fromarray((arr_full * 255).astype(np.uint8)).resize((tile_size, tile_size), Image.BILINEAR)
        whole_arr = np.array(whole_pil, dtype=np.float32) / 255.0
        whole_t = torch.from_numpy(whole_arr).permute(2, 0, 1).unsqueeze(0).float().to(device)

        with torch.amp.autocast("cuda"):
            w_out = model(whole_t)
        s_w = float(torch.sigmoid(w_out["det_logits"][0, 0]).item())
        whole_scores.append(s_w)

        # 4 Native tiles
        coords = get_tile_grid_coords(w, h, tile_size=tile_size, stride=256)
        if len(coords) > 4:
            coords = [coords[0], coords[len(coords) // 3], coords[2 * len(coords) // 3], coords[-1]]

        tile_logits = []
        for x, y in coords:
            tx_end = min(w, x + tile_size)
            ty_end = min(h, y + tile_size)
            patch = np.zeros((tile_size, tile_size, 3), dtype=np.float32)
            sub = arr_full[y:ty_end, x:tx_end]
            patch[: sub.shape[0], : sub.shape[1]] = sub

            t_t = torch.from_numpy(patch).permute(2, 0, 1).unsqueeze(0).float().to(device)
            with torch.amp.autocast("cuda"):
                t_out = model(t_t)
            tile_logits.append(t_out["mask_logits"][0, 0].detach().cpu().numpy())

        blended = blend_tiles_probability(tile_logits, coords, w, h, tile_size=tile_size)
        s_p = float(np.mean(np.sort(blended.ravel())[-100:]))
        pool_scores.append(s_p)

        s_h = 0.5 * s_w + 0.5 * s_p
        hybrid_scores.append(s_h)

        if lbl == 1.0:
            pred_bin = (blended > 0.35).astype(np.float32)
            inter = np.sum(pred_bin * gt_mask)
            union = np.sum((pred_bin + gt_mask) > 0)
            ious.append(float(inter / union) if union > 0 else 0.0)

    labels = np.array(labels, dtype=np.float32)
    gen_idx = (labels == 0)
    edt_idx = (labels == 1)

    # Score Separation Metrics
    gen_w_mean = float(np.mean([whole_scores[i] for i in range(len(labels)) if gen_idx[i]]))
    edt_w_mean = float(np.mean([whole_scores[i] for i in range(len(labels)) if edt_idx[i]]))
    separation_gap = edt_w_mean - gen_w_mean

    # Sweep threshold for Balanced Accuracy
    best_bacc = 0.0
    best_tau = 0.5
    best_acc = 0.0
    for tau in np.linspace(0.05, 0.95, 91):
        preds = [1 if s >= tau else 0 for s in whole_scores]
        bacc = balanced_accuracy_score(labels, preds)
        if bacc > best_bacc:
            best_bacc = bacc
            best_tau = tau
            best_acc = accuracy_score(labels, preds)

    try:
        auroc = float(roc_auc_score(labels, whole_scores))
    except Exception:
        auroc = 0.5

    mean_iou = float(np.mean(ious)) if ious else 0.0

    return {
        "val_balanced_acc": round(best_bacc * 100, 2),
        "val_acc": round(best_acc * 100, 2),
        "val_auroc": round(auroc * 100, 2),
        "val_iou": round(mean_iou * 100, 2),
        "val_tau": round(best_tau, 4),
        "gen_whole_mean": round(gen_w_mean, 4),
        "edt_whole_mean": round(edt_w_mean, 4),
        "separation_gap": round(separation_gap, 4),
    }


def train_joint(steps: int = 400, seed: int = 42, val_interval: int = 50) -> None:
    seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n=======================================================")
    print(f"EXPERIMENT 2: JOINT WHOLE-IMAGE & CROP TRAINING ({device})")
    print(f"=======================================================")

    with open(DEV_MANIFEST, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    train_recs = [r for r in manifest if r["split"] == "train"]
    val_recs = [r for r in manifest if r["split"] == "val"]
    print(f"Loaded {len(train_recs)} training records ({sum(1 for r in train_recs if r['label']==0)} genuine, {sum(1 for r in train_recs if r['label']==1)} edited).")

    train_ds = JointDualViewDataset(train_recs, view_size=384, is_train=True)
    batch_sampler = BalancedBatchSampler(train_recs, batch_size=4)
    train_loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=0)

    model = SuperFRAMEV2Detector(pretrained=True).to(device)
    if INIT_CKPT.exists():
        ckpt = torch.load(INIT_CKPT, map_location=device)
        sd = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(sd, strict=False)
        print(f"Initialized weights from: {INIT_CKPT}")

    # Differential learning rates
    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "semantic_encoder" in name or "c_stage" in name:
            backbone_params.append(param)
        else:
            head_params.append(param)

    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": 2e-5},
            {"params": head_params, "lr": 1.5e-4},
        ],
        weight_decay=1e-4,
    )
    scaler = torch.amp.GradScaler("cuda")
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps, eta_min=1e-6)

    grad_accum = 2
    train_iter = iter(train_loader)
    optimizer.zero_grad()

    best_val_bacc = -1.0
    best_state = None
    save_dir = RUNS_DIR / "experiment2_joint_training"
    save_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = save_dir / "best_joint_dualview.pt"

    t0 = time.time()
    for step in range(1, steps + 1):
        model.train()
        try:
            whole_t, crop_t, crop_mask_t, img_labels, crop_labels, _ = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            whole_t, crop_t, crop_mask_t, img_labels, crop_labels, _ = next(train_iter)

        whole_t = whole_t.to(device)
        crop_t = crop_t.to(device)
        crop_mask_t = crop_mask_t.to(device)
        img_labels = img_labels.to(device).unsqueeze(1).float()
        crop_labels = crop_labels.to(device).unsqueeze(1).float()

        with torch.amp.autocast("cuda"):
            # 1. Forward pass on RESIZED WHOLE IMAGES
            out_whole = model(whole_t)
            w_det_logits = out_whole["det_logits"]
            w_mask_logits = out_whole["mask_logits"]

            # Whole-image classification loss (matches how model is evaluated on whole images!)
            loss_w_det = F.binary_cross_entropy_with_logits(w_det_logits, img_labels)

            # Whole-image false alarm penalty on genuine images
            w_clean = (img_labels < 0.5).squeeze(1)
            if w_clean.any():
                loss_w_fp = torch.sigmoid(w_mask_logits[w_clean]).mean()
            else:
                loss_w_fp = torch.tensor(0.0, device=device)

            # 2. Forward pass on NATIVE CROPS
            out_crop = model(crop_t)
            c_det_logits = out_crop["det_logits"]
            c_mask_logits = out_crop["mask_logits"]

            # Crop edit presence loss
            loss_c_det = F.binary_cross_entropy_with_logits(c_det_logits, crop_labels)

            # Crop segmentation loss (only on crops containing actual edit pixels)
            c_manip = (crop_labels > 0.5).squeeze(1)
            if c_manip.any():
                loss_seg, _, _ = compute_supervised_segmentation_loss(
                    c_mask_logits[c_manip], crop_mask_t[c_manip]
                )
            else:
                loss_seg = torch.tensor(0.0, device=device)

            # Crop false alarm penalty on pristine crops
            c_clean = (crop_labels < 0.5).squeeze(1)
            if c_clean.any():
                loss_c_fp = torch.sigmoid(c_mask_logits[c_clean]).mean()
            else:
                loss_c_fp = torch.tensor(0.0, device=device)

            # Total joint loss
            total_loss = (
                loss_w_det + loss_seg + 0.5 * loss_c_det + 0.3 * (loss_w_fp + loss_c_fp)
            ) / grad_accum

        scaler.scale(total_loss).backward()

        if step % grad_accum == 0 or step == steps:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()

        if step % 25 == 0 or step == steps:
            print(
                f"[Joint Train] Step {step:3d}/{steps} | Loss: {(total_loss.item()*grad_accum):.4f} "
                f"(W-Det: {loss_w_det.item():.4f}, Seg: {loss_seg.item():.4f}, C-Det: {loss_c_det.item():.4f}) "
                f"| Elapsed: {(time.time()-t0):.1f}s"
            )

        # Validation check and checkpoint selection
        if step % val_interval == 0 or step == steps:
            val_metrics = evaluate_validation_separation(model, val_recs, device, max_samples=100)
            print(
                f"  --> [Validation @ Step {step}] Balanced Acc: {val_metrics['val_balanced_acc']}% | "
                f"AUROC: {val_metrics['val_auroc']}% | IoU: {val_metrics['val_iou']}% | "
                f"Score Gap (Edt - Gen): {val_metrics['separation_gap']} (Gen: {val_metrics['gen_whole_mean']}, Edt: {val_metrics['edt_whole_mean']})"
            )

            if val_metrics["val_balanced_acc"] > best_val_bacc:
                best_val_bacc = val_metrics["val_balanced_acc"]
                best_state = copy.deepcopy(model.state_dict())
                torch.save(
                    {
                        "step": step,
                        "model_state_dict": best_state,
                        "metrics": val_metrics,
                    },
                    best_ckpt_path,
                )
                print(f"  *** New Best Checkpoint Saved! (Val B-Acc: {best_val_bacc}%) ***")

    print(f"\nTraining Complete. Best Checkpoint: {best_ckpt_path} (Val B-Acc: {best_val_bacc}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=300)
    args = parser.parse_args()
    train_joint(steps=args.steps)
