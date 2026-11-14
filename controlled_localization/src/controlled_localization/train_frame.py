"""Dedicated high-accuracy training pipeline for the FRAME Architecture.
Trains on 1,120 real OpenSDI images with multi-scale segmentation + image-level detection loss.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import time
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from controlled_localization.models.frame_detector import FRAMEDetector
from controlled_localization.models.super_frame_detector import SuperFRAMEDetector
from controlled_localization.models.super_frame_v2_detector import SuperFRAMEV2Detector
from controlled_localization.losses.segmentation import compute_supervised_segmentation_loss
from controlled_localization.utils import seed_all


ROOT = Path("I:/icvgip workshop")
RUNS_DIR = ROOT / "controlled_localization/runs"
DEFAULT_MANIFEST = RUNS_DIR / "real_development_20261114/manifest.json"


class OpenSDIDataset(Dataset):
    """Dataset loading real OpenSDI images and binary manipulation masks."""

    def __init__(self, records: List[Dict], crop_size: int = 384, is_train: bool = True) -> None:
        self.records = records
        self.crop_size = crop_size
        self.is_train = is_train

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, float, str]:
        rec = self.records[idx]
        img_p = Path(rec["image"])
        mask_val = rec.get("mask")
        label = float(rec["label"])

        # Load RGB image
        img = Image.open(img_p).convert("RGB")
        w, h = img.size

        # Load or generate mask
        if label == 1.0 and mask_val is not None and Path(mask_val).exists():
            mask_img = Image.open(mask_val).convert("L")
            mask_arr = np.array(mask_img, dtype=np.float32)
            # Polarity: values > 127.5 denote manipulated pixels
            mask_binary = (mask_arr > 127.5).astype(np.float32)
        else:
            mask_binary = np.zeros((h, w), dtype=np.float32)

        img_arr = np.array(img, dtype=np.float32) / 255.0  # [H, W, 3]

        # Training random crop / Eval center crop or resize
        th, tw = self.crop_size, self.crop_size
        if self.is_train:
            # Data augmentation: random crop + random horizontal flip
            if h > th and w > tw:
                y = random.randint(0, h - th)
                x = random.randint(0, w - tw)
                img_arr = img_arr[y : y + th, x : x + tw]
                mask_binary = mask_binary[y : y + th, x : x + tw]
            elif h != th or w != tw:
                # Resize if smaller than crop size
                img_resized = Image.fromarray((img_arr * 255).astype(np.uint8)).resize((tw, th), Image.BILINEAR)
                mask_resized = Image.fromarray((mask_binary * 255).astype(np.uint8)).resize((tw, th), Image.NEAREST)
                img_arr = np.array(img_resized, dtype=np.float32) / 255.0
                mask_binary = (np.array(mask_resized, dtype=np.float32) > 127.5).astype(np.float32)

            if random.random() > 0.5:
                img_arr = np.fliplr(img_arr).copy()
                mask_binary = np.fliplr(mask_binary).copy()
        else:
            # Validation: center crop
            if h >= th and w >= tw:
                y = (h - th) // 2
                x = (w - tw) // 2
                img_arr = img_arr[y : y + th, x : x + tw]
                mask_binary = mask_binary[y : y + th, x : x + tw]
            else:
                img_resized = Image.fromarray((img_arr * 255).astype(np.uint8)).resize((tw, th), Image.BILINEAR)
                mask_resized = Image.fromarray((mask_binary * 255).astype(np.uint8)).resize((tw, th), Image.NEAREST)
                img_arr = np.array(img_resized, dtype=np.float32) / 255.0
                mask_binary = (np.array(mask_resized, dtype=np.float32) > 127.5).astype(np.float32)

        # Convert to torch tensors
        rgb_tensor = torch.from_numpy(img_arr).permute(2, 0, 1).float()  # [3, H, W]
        mask_tensor = torch.from_numpy(mask_binary).unsqueeze(0).float()  # [1, H, W]

        return rgb_tensor, mask_tensor, label, rec.get("id", str(idx))


def evaluate_validation(
    model: nn.Module, val_loader: DataLoader, device: torch.device
) -> Dict[str, float]:
    model.eval()
    all_ious = []
    clean_fps = []
    labels_all = []
    scores_all = []

    with torch.no_grad():
        for rgbs, masks, labels, _ in val_loader:
            rgbs = rgbs.to(device)
            masks = masks.to(device)
            labels = labels.to(device)

            out = model(rgbs)
            mask_logits = out["mask_logits"]
            det_logits = out["det_logits"]

            probs = torch.sigmoid(mask_logits)
            preds = (probs > 0.5).float()
            det_scores = torch.sigmoid(det_logits).squeeze(1).cpu().numpy()

            for i in range(rgbs.size(0)):
                lbl = labels[i].item()
                labels_all.append(lbl)
                scores_all.append(float(det_scores[i]))

                pred_i = preds[i, 0]
                gt_i = masks[i, 0]

                if lbl == 1.0:
                    inter = (pred_i * gt_i).sum().item()
                    union = ((pred_i + gt_i) > 0).sum().item()
                    iou = inter / union if union > 0 else 0.0
                    all_ious.append(iou)
                else:
                    fp = (pred_i > 0).float().mean().item()
                    clean_fps.append(fp)

    mean_iou = float(np.mean(all_ious)) if all_ious else 0.0
    mean_fp = float(np.mean(clean_fps)) if clean_fps else 0.0

    try:
        if len(set(labels_all)) > 1:
            auroc = float(roc_auc_score(labels_all, scores_all))
        else:
            auroc = 0.5
    except Exception:
        auroc = 0.5

    return {
        "val_iou": mean_iou,
        "val_fp": mean_fp,
        "val_auroc": auroc,
    }


def train_frame_model(
    steps: int = 1200,
    batch_size: int = 4,
    val_interval: int = 100,
    save_dir: str = "runs/frame_run",
    model_type: str = "frame",
    manifest_path: str = "",
    initialize: str = "",
) -> None:
    seed_all(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting {model_type.upper()} high-accuracy training on {device}...")

    out_p = ROOT / "controlled_localization" / save_dir
    out_p.mkdir(parents=True, exist_ok=True)

    # 1. Load manifest
    m_p = Path(manifest_path) if manifest_path else DEFAULT_MANIFEST
    print(f"Loading dataset from: {m_p}")
    with open(m_p, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    train_recs = [r for r in manifest if r["split"] == "train"]
    val_recs = [r for r in manifest if r["split"] == "val"]
    print(f"Loaded {len(train_recs)} train records and {len(val_recs)} val records.")

    train_ds = OpenSDIDataset(train_recs, crop_size=384, is_train=True)
    val_ds = OpenSDIDataset(val_recs, crop_size=384, is_train=False)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    # 2. Instantiate Model
    if model_type == "super_v2":
        print("Instantiating SuperFRAMEV2Detector (ConvNeXt-Base + DINOv2-Base + Cross-Attention FPN)...")
        model = SuperFRAMEV2Detector(pretrained=True).to(device)
    elif model_type == "super":
        print("Instantiating SuperFRAMEDetector (ConvNeXt-Small + DINOv2-Small + 60ch Forensic)...")
        model = SuperFRAMEDetector(pretrained=True).to(device)
    else:
        print("Instantiating FRAMEDetector (ConvNeXt-Tiny + 60ch Forensic)...")
        model = FRAMEDetector(pretrained_semantic=True).to(device)

    # Optional checkpoint initialization
    if initialize and Path(initialize).exists():
        print(f"Initializing model weights from checkpoint: {initialize}")
        ckpt = torch.load(initialize, map_location=device)
        sd = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        model.load_state_dict(sd, strict=False)
        print("Successfully loaded initialized weights!")

    # 3. Differential parameter groups
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
            {"params": backbone_params, "lr": 5e-5},
            {"params": new_params, "lr": 2e-4},
        ],
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps, eta_min=1e-6)

    best_fitness = -1e9
    train_iter = iter(train_loader)
    history = []

    print(f"Training for {steps} steps with checkpointing every {val_interval} steps...")

    t0 = time.time()
    for step in range(1, steps + 1):
        model.train()
        try:
            rgbs, masks, labels, _ = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            rgbs, masks, labels, _ = next(train_iter)

        rgbs = rgbs.to(device)
        masks = masks.to(device)
        labels = labels.to(device).unsqueeze(1)  # [B, 1]

        optimizer.zero_grad()
        out = model(rgbs)
        mask_logits = out["mask_logits"]
        det_logits = out["det_logits"]

        # 1. Supervised segmentation loss (for manipulated images)
        manip_mask = (labels > 0.5).squeeze(1)
        if manip_mask.any():
            seg_loss, _, _ = compute_supervised_segmentation_loss(
                mask_logits[manip_mask], masks[manip_mask]
            )
        else:
            seg_loss = torch.tensor(0.0, device=device)

        # 2. Clean-image false positive penalty
        clean_mask = (labels < 0.5).squeeze(1)
        if clean_mask.any():
            clean_probs = torch.sigmoid(mask_logits[clean_mask])
            fp_loss = clean_probs.mean()
        else:
            fp_loss = torch.tensor(0.0, device=device)

        # 3. Image-level detection loss (BCE)
        det_loss = F.binary_cross_entropy_with_logits(det_logits, labels)

        # Total combined objective
        loss = seg_loss + 0.5 * det_loss + 0.3 * fp_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        if step % 20 == 0 or step == steps:
            print(
                f"Step {step:4d}/{steps} | Loss: {loss.item():.4f} "
                f"(Seg: {seg_loss.item():.4f}, Det: {det_loss.item():.4f}, FP: {fp_loss.item():.4f}) "
                f"| Elapsed: {(time.time()-t0):.1f}s"
            )

        # Validation check
        if step % val_interval == 0 or step == steps:
            val_metrics = evaluate_validation(model, val_loader, device)
            val_iou = val_metrics["val_iou"]
            val_fp = val_metrics["val_fp"]
            val_auroc = val_metrics["val_auroc"]

            # Fitness: prioritize high IoU and high AUROC while suppressing false alarms
            fitness = val_iou * 0.7 + (val_auroc - 0.5) * 0.3 - val_fp * 0.2

            is_best = fitness > best_fitness
            if is_best:
                best_fitness = fitness
                torch.save(
                    {
                        "step": step,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "fitness": fitness,
                        "metrics": val_metrics,
                    },
                    out_p / "best.pt",
                )

            torch.save(
                {
                    "step": step,
                    "model_state_dict": model.state_dict(),
                    "fitness": fitness,
                    "metrics": val_metrics,
                },
                out_p / "latest.pt",
            )

            record = {
                "step": step,
                "loss": loss.item(),
                "val_iou": val_iou,
                "val_fp": val_fp,
                "val_auroc": val_auroc,
                "fitness": fitness,
                "is_best": is_best,
            }
            history.append(record)

            print(
                f"--> [Val Step {step}] IoU: {val_iou*100:.2f}% | "
                f"AUROC: {val_auroc:.4f} | Clean FP: {val_fp*100:.2f}% | "
                f"Fitness: {fitness:.4f} {'*** BEST ***' if is_best else ''}"
            )

    # Save training log
    with open(out_p / "train_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nTraining completed! Best checkpoint saved to {out_p / 'best.pt'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--val-interval", type=int, default=100)
    parser.add_argument("--save-dir", type=str, default="runs/frame_run")
    parser.add_argument("--model-type", type=str, default="frame", choices=["frame", "super", "super_v2"])
    parser.add_argument("--manifest", type=str, default="", help="Path to manifest JSON")
    parser.add_argument("--initialize", type=str, default="", help="Path to checkpoint .pt to initialize")
    args = parser.parse_args()

    train_frame_model(
        steps=args.steps,
        batch_size=args.batch_size,
        val_interval=args.val_interval,
        save_dir=args.save_dir,
        model_type=args.model_type,
        manifest_path=args.manifest,
        initialize=args.initialize,
    )
