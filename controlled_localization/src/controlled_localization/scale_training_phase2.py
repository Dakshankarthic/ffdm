"""Phase 2: Scaled Supervised Training Pipeline.
Builds the largest defensible scaled dataset from verified OpenSDIDplus & OpenSDI parquet shards,
strictly enforces zero test leakage (SHA-256 and ID exclusion), and trains SuperFRAME v2
with decoupled supervision, AMP, and gradient accumulation.
"""

from __future__ import annotations

import glob
import hashlib
import io
import json
import os
from pathlib import Path
import re
import time
from typing import Dict, List, Set, Tuple

import numpy as np
from PIL import Image
import pyarrow.parquet as pq
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from controlled_localization.models.super_frame_v2_detector import SuperFRAMEV2Detector
from controlled_localization.data.corrected_dataset import CorrectedOpenSDIDataset
from controlled_localization.data.tiles import get_tile_grid_coords, blend_tiles_probability
from controlled_localization.losses.segmentation import compute_supervised_segmentation_loss
from controlled_localization.utils import seed_all


ROOT = Path("I:/icvgip workshop")
RUNS_DIR = ROOT / "controlled_localization/runs"
SCALED_DATA_DIR = RUNS_DIR / "scaled_pilot_data"
BASE_DEV_MANIFEST = RUNS_DIR / "real_development_20261114/manifest_multigen.json"
INIT_CKPT = RUNS_DIR / "super_frame_v2_run/best.pt"


def get_excluded_test_hashes() -> Tuple[Set[str], Set[str]]:
    """Collect SHA-256 hashes and numeric tokens from test files to prevent test leakage."""
    manifest_dir = ROOT / "baseline_work/data_manifests"
    test_files = glob.glob(str(manifest_dir / "test_*_manifest.json"))
    hashes = set()
    num_tokens = set()

    for tf in test_files:
        with open(tf, "r", encoding="utf-8") as f:
            data = json.load(f)
        for e in data.get("entries", []):
            if "image_sha256" in e:
                hashes.add(e["image_sha256"])
            m = re.findall(r"\d+", e.get("image_path", ""))
            for n in m:
                if len(n) > 2:
                    num_tokens.add(n.lstrip("0"))

    # Also check actual test image files on disk
    test_img_files = glob.glob(str(ROOT / "baseline_work/data/test_*/*.*"))
    for f in test_img_files:
        m = re.findall(r"\d+", os.path.basename(f))
        for n in m:
            if len(n) > 2:
                num_tokens.add(n.lstrip("0"))

    return hashes, num_tokens


def extract_parquet_samples(
    parquet_path: Path,
    generator_name: str,
    target_count: int,
    excluded_hashes: Set[str],
    excluded_tokens: Set[str],
) -> List[Dict]:
    """Extracts valid, non-leaking samples from a parquet shard."""
    print(f"Reading {parquet_path.name} for {generator_name} (target: {target_count})...")
    SCALED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    table = pq.read_table(parquet_path, columns=["key", "image", "mask", "label"])

    records = []
    num_rows = table.num_rows

    for i in range(num_rows):
        if len(records) >= target_count:
            break

        key = table["key"][i].as_py()
        label = int(table["label"][i].as_py())

        # Check numeric ID exclusion
        tokens = re.findall(r"\d+", key)
        if any(tok.lstrip("0") in excluded_tokens for tok in tokens):
            continue

        img_struct = table["image"][i].as_py()
        mask_struct = table["mask"][i].as_py()

        img_bytes = img_struct.get("bytes") if isinstance(img_struct, dict) else None
        if not img_bytes:
            continue

        # Check SHA-256 hash exclusion
        img_sha = hashlib.sha256(img_bytes).hexdigest()
        if img_sha in excluded_hashes:
            continue

        sid = f"{generator_name}_{tokens[-1] if tokens else i}"
        img_out = SCALED_DATA_DIR / f"{sid}.jpg"
        mask_out = SCALED_DATA_DIR / f"{sid}_mask.png"

        # Save to disk if not exists
        if not img_out.exists():
            with open(img_out, "wb") as f:
                f.write(img_bytes)

        if label == 1 and mask_struct:
            mask_bytes = mask_struct.get("bytes") if isinstance(mask_struct, dict) else None
            if mask_bytes and not mask_out.exists():
                with open(mask_out, "wb") as f:
                    f.write(mask_bytes)

        records.append({
            "id": sid,
            "source_id": sid,
            "label": float(label),
            "image": str(img_out),
            "mask": str(mask_out) if label == 1 and mask_out.exists() else None,
            "split": "train",
            "sha256": img_sha,
            "generator": generator_name,
            "key": key,
        })

    print(f"Extracted {len(records)} verified samples for {generator_name}.")
    return records


def build_scaled_manifest() -> Tuple[List[Dict], List[Dict]]:
    """Builds the comprehensive training and validation manifest."""
    manifest_out = RUNS_DIR / "scaled_pilot/manifest_scaled.json"
    manifest_out.parent.mkdir(parents=True, exist_ok=True)

    if manifest_out.exists():
        with open(manifest_out, "r", encoding="utf-8") as f:
            all_records = json.load(f)
        train_recs = [r for r in all_records if r["split"] == "train"]
        val_recs = [r for r in all_records if r["split"] == "val"]
        print(f"Loaded existing scaled manifest: {len(train_recs)} train, {len(val_recs)} val.")
        return train_recs, val_recs

    excluded_hashes, excluded_tokens = get_excluded_test_hashes()
    print(f"Found {len(excluded_hashes)} excluded test hashes and {len(excluded_tokens)} excluded tokens.")

    # 1. Start with existing development manifest (1140 train, 280 val)
    with open(BASE_DEV_MANIFEST, "r", encoding="utf-8") as f:
        dev_records = json.load(f)

    # 2. Extract FLUX samples from cached parquet
    flux_parquets = glob.glob("C:/Users/DK11/.cache/huggingface/hub/datasets--nebula--OpenSDIDplus/**/data/flux-*.parquet", recursive=True)
    flux_recs = []
    if flux_parquets:
        flux_recs = extract_parquet_samples(
            Path(flux_parquets[0]), "flux", target_count=500,
            excluded_hashes=excluded_hashes, excluded_tokens=excluded_tokens
        )

    # 3. Extract SD1.5 samples from cached parquet
    sd15_parquets = glob.glob("C:/Users/DK11/.cache/huggingface/hub/datasets--nebula--OpenSDI_train/**/data/sd15-*.parquet", recursive=True)
    sd15_recs = []
    if sd15_parquets:
        sd15_recs = extract_parquet_samples(
            Path(sd15_parquets[0]), "sd15", target_count=500,
            excluded_hashes=excluded_hashes, excluded_tokens=excluded_tokens
        )

    # Combine
    combined = dev_records + flux_recs + sd15_recs

    # Deduplicate by SHA-256
    seen_hashes = set()
    deduped = []
    for r in combined:
        h = r.get("sha256")
        if h and h in seen_hashes:
            continue
        if h:
            seen_hashes.add(h)
        deduped.append(r)

    train_recs = [r for r in deduped if r["split"] == "train"]
    val_recs = [r for r in deduped if r["split"] == "val"]

    with open(manifest_out, "w", encoding="utf-8") as f:
        json.dump(deduped, f, indent=2)

    print(f"\nSaved scaled manifest with {len(deduped)} total samples ({len(train_recs)} train, {len(val_recs)} val) to: {manifest_out}")
    return train_recs, val_recs


def evaluate_val_quick(
    model: nn.Module,
    records: List[Dict],
    device: torch.device,
    tile_size: int = 384,
    max_eval_samples: int = 140,
) -> Dict[str, float]:
    """Validation evaluation with multi-evidence whole + native tile pooling."""
    model.eval()
    all_ious = []
    clean_fps = []
    labels_all = []
    hybrid_scores = []

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

            if lbl == 1.0 and mask_p is not None and Path(mask_p).exists():
                with Image.open(mask_p) as m_img:
                    gt_mask = (np.array(m_img.convert("L"), dtype=np.float32) > 127.5).astype(np.float32)
            else:
                gt_mask = np.zeros((orig_h, orig_w), dtype=np.float32)

            # Whole image view (384x384)
            whole_resized = rgb_full.resize((tile_size, tile_size), Image.BILINEAR)
            whole_arr = np.array(whole_resized, dtype=np.float32) / 255.0
            whole_t = torch.from_numpy(whole_arr).permute(2, 0, 1).unsqueeze(0).float().to(device)

            with torch.amp.autocast("cuda"):
                whole_out = model(whole_t)
            s_whole = float(torch.sigmoid(whole_out["det_logits"][0, 0]).item())

            # 4 Native Tiles
            coords = get_tile_grid_coords(orig_w, orig_h, tile_size=tile_size, stride=256)
            if len(coords) > 4:
                coords = [coords[0], coords[len(coords) // 3], coords[2 * len(coords) // 3], coords[-1]]

            tile_logits = []
            for x, y in coords:
                tx_end = min(orig_w, x + tile_size)
                ty_end = min(orig_h, y + tile_size)
                crop_patch = np.zeros((tile_size, tile_size, 3), dtype=np.float32)
                sub_arr = arr_full[y:ty_end, x:tx_end]
                crop_patch[: sub_arr.shape[0], : sub_arr.shape[1]] = sub_arr

                tile_t = torch.from_numpy(crop_patch).permute(2, 0, 1).unsqueeze(0).float().to(device)
                with torch.amp.autocast("cuda"):
                    t_out = model(tile_t)
                tile_logits.append(t_out["mask_logits"][0, 0].cpu().numpy())

            blended_map = blend_tiles_probability(tile_logits, coords, orig_w, orig_h, tile_size=tile_size)
            s_pool = float(np.mean(np.sort(blended_map.ravel())[-100:]))
            s_hybrid = 0.15 * s_whole + 0.85 * s_pool
            hybrid_scores.append(s_hybrid)

            pred_bin = (blended_map > 0.35).astype(np.float32)
            if lbl == 1.0:
                inter = np.sum(pred_bin * gt_mask)
                union = np.sum((pred_bin + gt_mask) > 0)
                iou = float(inter / union) if union > 0 else 0.0
                all_ious.append(iou)
            else:
                clean_fps.append(float(np.mean(pred_bin > 0)))

    mean_iou = float(np.mean(all_ious)) if all_ious else 0.0
    mean_fp = float(np.mean(clean_fps)) if clean_fps else 0.0

    try:
        auroc = float(roc_auc_score(labels_all, hybrid_scores))
    except Exception:
        auroc = 0.5

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
        "auroc_hybrid": round(auroc * 100, 2),
        "accuracy": round(best_acc * 100, 2),
        "f1": round(best_f1 * 100, 2),
    }


def train_scaled_model(steps: int = 300, seed: int = 42) -> None:
    seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n=======================================================")
    print(f"PHASE 2: SCALED TRAINING ON {device} ({steps} STEPS)")
    print(f"=======================================================")

    train_recs, val_recs = build_scaled_manifest()

    train_ds = CorrectedOpenSDIDataset(
        train_recs, crop_size=384, is_train=True, edit_crop_prob=0.70, strict_mask_validation=False
    )
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, drop_last=True, num_workers=0)

    model = SuperFRAMEV2Detector(pretrained=True).to(device)
    if INIT_CKPT.exists():
        ckpt = torch.load(INIT_CKPT, map_location=device)
        sd = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        model.load_state_dict(sd, strict=False)
        print(f"Initialized from existing checkpoint: {INIT_CKPT}")

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

    grad_accum = 2
    train_iter = iter(train_loader)
    optimizer.zero_grad()

    t0 = time.time()
    for step in range(1, steps + 1):
        model.train()
        try:
            rgbs, masks, crop_labels, _, _ = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            rgbs, masks, crop_labels, _, _ = next(train_iter)

        rgbs = rgbs.to(device)
        masks = masks.to(device)
        crop_labels = crop_labels.to(device).unsqueeze(1).float()

        with torch.amp.autocast("cuda"):
            out = model(rgbs)
            mask_logits = out["mask_logits"]
            det_logits = out["det_logits"]

            manip_mask = (crop_labels > 0.5).squeeze(1)
            clean_mask = (crop_labels < 0.5).squeeze(1)

            if manip_mask.any():
                seg_loss, _, _ = compute_supervised_segmentation_loss(
                    mask_logits[manip_mask], masks[manip_mask]
                )
            else:
                seg_loss = torch.tensor(0.0, device=device)

            if clean_mask.any():
                fp_loss = torch.sigmoid(mask_logits[clean_mask]).mean()
            else:
                fp_loss = torch.tensor(0.0, device=device)

            det_loss = F.binary_cross_entropy_with_logits(det_logits, crop_labels)
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
            print(
                f"[Scaled Training] Step {step:3d}/{steps} | Loss: {(total_loss.item()*grad_accum):.4f} "
                f"(Seg: {seg_loss.item():.4f}, Det: {det_loss.item():.4f}, FP: {fp_loss.item():.4f}) "
                f"| Elapsed: {(time.time()-t0):.1f}s"
            )

    # Save scaled checkpoint
    out_dir = RUNS_DIR / "scaled_pilot"
    out_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt = out_dir / "best_superframe_v2_scaled.pt"
    torch.save({"step": steps, "model_state_dict": model.state_dict(), "num_train_samples": len(train_recs)}, best_ckpt)
    print(f"\nSaved scaled checkpoint to: {best_ckpt}")

    # Evaluate validation
    print("Evaluating scaled checkpoint on validation set...")
    val_res = evaluate_val_quick(model, val_recs, device)
    print(f"Scaled Validation Metrics: IoU={val_res['mean_iou']}%, Acc={val_res['accuracy']}%, F1={val_res['f1']}%, AUROC={val_res['auroc_hybrid']}%, FP Rate={val_res['mean_fp_rate']}%")

    report_p = ROOT / "controlled_localization/reports/phase2_scaled_training_results.json"
    with open(report_p, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S IST"),
            "steps": steps,
            "train_samples": len(train_recs),
            "val_samples": len(val_recs),
            "checkpoint": str(best_ckpt),
            "metrics": val_res,
        }, f, indent=2)
    print(f"Saved Phase 2 report to: {report_p}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=250)
    args = parser.parse_args()
    train_scaled_model(steps=args.steps)
