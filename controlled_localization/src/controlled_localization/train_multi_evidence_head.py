"""Train Multi-Evidence Classification Head on Whole Images + Verified Native Tiles.
Keeps SuperFRAME v2 backbones completely FROZEN.
Supervises:
  1. Resized Whole Image (global context & lighting)
  2. Verified Edit Tile (native resolution; guaranteed to contain the inpainting)
  3. Hard Genuine Background Tile (textured/compressed non-manipulated region)
Supervises combined prediction with image-level label, and individual tiles with local ground-truth.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image
from sklearn.metrics import roc_auc_score, accuracy_score, balanced_accuracy_score, f1_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from controlled_localization.models.super_frame_v2_detector import SuperFRAMEV2Detector
from controlled_localization.models.multi_evidence_head import MultiEvidenceDetectionHead
from controlled_localization.utils import seed_all, resolve_device


ROOT = Path("I:/icvgip workshop")
RUNS_DIR = ROOT / "controlled_localization/runs"
DEFAULT_MANIFEST = RUNS_DIR / "real_development_20261114/manifest_multigen.json"
CHECKPOINT_V2 = RUNS_DIR / "super_frame_v2_run/best.pt"


def sample_verified_tiles(
    img_arr: np.ndarray,
    mask_arr: np.ndarray,
    label: float,
    tile_size: int = 384,
) -> Tuple[np.ndarray, float, np.ndarray, float]:
    """
    Samples:
      Tile 1: Verified edit tile (if label==1.0) or random genuine tile (if label==0.0)
      Tile 2: Hard negative genuine tile (textured / high gradient)
    """
    h, w, _ = img_arr.shape

    # Helper: compute gradient energy of a gray patch
    gray = np.dot(img_arr[..., :3], [0.299, 0.587, 0.114])
    gy, gx = np.gradient(gray)
    grad_mag = np.sqrt(gx**2 + gy**2)

    if h < tile_size or w < tile_size:
        # Resize if image is smaller than tile
        res_img = Image.fromarray((img_arr * 255).astype(np.uint8)).resize((tile_size, tile_size), Image.BILINEAR)
        res_arr = np.array(res_img, dtype=np.float32) / 255.0
        return res_arr, label, res_arr, 0.0

    # 1. Tile 1: Verified Edit Tile (if manipulated)
    tile1_arr = None
    tile1_label = 0.0

    if label == 1.0 and np.sum(mask_arr) > 0:
        ys, xs = np.nonzero(mask_arr)
        cy = int(np.mean(ys))
        cx = int(np.mean(xs))
        # Add small random jitter (+-32px)
        jy = cy + random.randint(-32, 32)
        jx = cx + random.randint(-32, 32)
        # Clamp to bounds
        y1 = max(0, min(h - tile_size, jy - tile_size // 2))
        x1 = max(0, min(w - tile_size, jx - tile_size // 2))
        tile1_arr = img_arr[y1 : y1 + tile_size, x1 : x1 + tile_size]
        tile1_label = 1.0
    else:
        # Random tile for genuine image
        y1 = random.randint(0, h - tile_size)
        x1 = random.randint(0, w - tile_size)
        tile1_arr = img_arr[y1 : y1 + tile_size, x1 : x1 + tile_size]
        tile1_label = 0.0

    # 2. Tile 2: Hard Genuine Tile (textured/compressed non-manipulated region)
    # Candidate positions: evaluate a small grid to find highest texture where mask is 0
    best_cand_y, best_cand_x = 0, 0
    max_texture = -1.0
    cand_ys = np.linspace(0, h - tile_size, num=min(4, h - tile_size + 1), dtype=int)
    cand_xs = np.linspace(0, w - tile_size, num=min(4, w - tile_size + 1), dtype=int)

    for cy_i in cand_ys:
        for cx_i in cand_xs:
            m_patch = mask_arr[cy_i : cy_i + tile_size, cx_i : cx_i + tile_size]
            # Genuine tile must have minimal or zero edit pixels
            if label == 1.0 and np.mean(m_patch) > 0.01:
                continue
            texture_score = float(np.mean(grad_mag[cy_i : cy_i + tile_size, cx_i : cx_i + tile_size]))
            if texture_score > max_texture:
                max_texture = texture_score
                best_cand_y = cy_i
                best_cand_x = cx_i

    tile2_arr = img_arr[best_cand_y : best_cand_y + tile_size, best_cand_x : best_cand_x + tile_size]
    tile2_label = 0.0

    return tile1_arr, tile1_label, tile2_arr, tile2_label


class MultiEvidenceDataset(Dataset):
    """Loads whole image + verified edit tile + hard genuine tile."""

    def __init__(self, records: List[Dict], tile_size: int = 384, is_train: bool = True) -> None:
        self.records = records
        self.tile_size = tile_size
        self.is_train = is_train

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, torch.Tensor]:
        rec = self.records[idx]
        img_p = Path(rec["image"])
        mask_val = rec.get("mask")
        label = float(rec["label"])

        img = Image.open(img_p).convert("RGB")
        w, h = img.size

        # Whole image resized to tile_size
        img_whole = img.resize((self.tile_size, self.tile_size), Image.BILINEAR)
        arr_whole = np.array(img_whole, dtype=np.float32) / 255.0

        # Load mask
        if label == 1.0 and mask_val is not None and Path(mask_val).exists():
            mask_img = Image.open(mask_val).convert("L")
            mask_arr = (np.array(mask_img, dtype=np.float32) > 127.5).astype(np.float32)
        else:
            mask_arr = np.zeros((h, w), dtype=np.float32)

        arr_img = np.array(img, dtype=np.float32) / 255.0

        # Sample tiles
        t1, l1, t2, l2 = sample_verified_tiles(arr_img, mask_arr, label, self.tile_size)

        # Augmentation on train
        if self.is_train and random.random() > 0.5:
            arr_whole = np.fliplr(arr_whole).copy()
            t1 = np.fliplr(t1).copy()
            t2 = np.fliplr(t2).copy()

        # To tensor
        ten_whole = torch.from_numpy(arr_whole).permute(2, 0, 1).float()
        ten_t1 = torch.from_numpy(t1).permute(2, 0, 1).float()
        ten_t2 = torch.from_numpy(t2).permute(2, 0, 1).float()

        tile_labels = torch.tensor([l1, l2], dtype=torch.float32)

        return ten_whole, ten_t1, ten_t2, label, tile_labels


def train_head() -> None:
    seed_all(42)
    device = resolve_device()
    save_dir = ROOT / "controlled_localization/runs/multi_evidence_head_v2"
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading frozen SuperFRAME v2 from: {CHECKPOINT_V2}")
    ckpt = torch.load(CHECKPOINT_V2, map_location=device)
    sd = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt

    detector = SuperFRAMEV2Detector(pretrained=False).to(device)
    detector.load_state_dict(sd)
    detector.eval()

    # Freeze all detector parameters
    for p in detector.parameters():
        p.requires_grad = False

    # Instantiate trainable head
    head = MultiEvidenceDetectionHead(in_dim=257, hidden_dim=64).to(device)
    head.train()

    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion_bce = nn.BCEWithLogitsLoss()

    # Load dataset
    with open(DEFAULT_MANIFEST, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    train_recs = [r for r in manifest if r["split"] == "train"]
    val_recs = [r for r in manifest if r["split"] == "val"]
    print(f"Dataset: {len(train_recs)} train records, {len(val_recs)} val records.")

    train_ds = MultiEvidenceDataset(train_recs, tile_size=384, is_train=True)
    val_ds = MultiEvidenceDataset(val_recs, tile_size=384, is_train=False)

    batch_size = 8
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    num_epochs = 12
    best_val_auc = -1.0
    best_head_path = save_dir / "best_head.pt"

    print("\nStarting Multi-Evidence Head Training...")
    for epoch in range(1, num_epochs + 1):
        head.train()
        total_loss = 0.0
        num_batches = 0
        t0 = time.time()

        for whole, t1, t2, labels, tile_labels in train_loader:
            whole = whole.to(device)
            t1 = t1.to(device)
            t2 = t2.to(device)
            labels = labels.to(device).unsqueeze(1)
            tile_labels = tile_labels.to(device).unsqueeze(2)  # [B, 2, 1]

            # Extract features through frozen detector
            with torch.no_grad():
                out_w = detector(whole, return_features=True)
                out_t1 = detector(t1, return_features=True)
                out_t2 = detector(t2, return_features=True)

                e_whole = out_w["det_feat"]
                e_t1 = out_t1["det_feat"]
                e_t2 = out_t2["det_feat"]

            e_tiles = torch.stack([e_t1, e_t2], dim=1)  # [B, 2, 257]

            optimizer.zero_grad()
            comb_logit, whole_logit, tile_logits = head(e_whole, e_tiles)

            loss_comb = criterion_bce(comb_logit, labels)
            loss_whole = criterion_bce(whole_logit, labels)
            loss_tiles = criterion_bce(tile_logits, tile_labels)

            loss = loss_comb + 0.5 * loss_whole + 0.5 * loss_tiles
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / max(1, num_batches)

        # Validation
        head.eval()
        val_y_true = []
        val_comb_preds = []
        val_whole_preds = []

        with torch.no_grad():
            for whole, t1, t2, labels, _ in val_loader:
                whole = whole.to(device)
                t1 = t1.to(device)
                t2 = t2.to(device)
                labels = labels.to(device)

                out_w = detector(whole, return_features=True)
                out_t1 = detector(t1, return_features=True)
                out_t2 = detector(t2, return_features=True)

                e_tiles = torch.stack([out_t1["det_feat"], out_t2["det_feat"]], dim=1)
                comb_logit, whole_logit, _ = head(out_w["det_feat"], e_tiles)

                val_y_true.extend(labels.cpu().numpy().tolist())
                val_comb_preds.extend(torch.sigmoid(comb_logit).squeeze(1).cpu().numpy().tolist())
                val_whole_preds.extend(torch.sigmoid(whole_logit).squeeze(1).cpu().numpy().tolist())

        val_y = np.array(val_y_true)
        val_scores = np.array(val_comb_preds)
        val_auc = float(roc_auc_score(val_y, val_scores))
        val_acc = float(accuracy_score(val_y, (val_scores > 0.5).astype(int)))
        val_f1 = float(f1_score(val_y, (val_scores > 0.5).astype(int), zero_division=0))

        elapsed = time.time() - t0
        print(f"Epoch {epoch:2d}/{num_epochs:2d} ({elapsed:.1f}s) - Train Loss: {avg_loss:.4f} | Val AUROC: {val_auc:.4f} | Val Acc: {val_acc*100:.2f}% | Val F1: {val_f1*100:.2f}%")

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save({
                "head_state_dict": head.state_dict(),
                "val_auc": val_auc,
                "val_acc": val_acc,
                "epoch": epoch,
            }, best_head_path)
            print(f"  --> Saved new best head: {best_head_path} (Val AUROC: {val_auc:.4f})")

    print(f"\nTraining Complete. Best Validation AUROC: {best_val_auc:.4f}")


if __name__ == "__main__":
    train_head()
