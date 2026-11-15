"""Experiment 1: Re-evaluate Checkpoints with Separated Image and Mask Thresholds.
- Evaluates genuine vs edited score distributions on validation (280 images).
- Calibrates image-level thresholds on validation using Balanced Accuracy.
- Freezes thresholds and tests on the 100-image benchmark split.
- Separates image-level classification threshold from pixel mask threshold (t_mask=0.35).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
import torch
import torch.nn as nn

from controlled_localization.models.super_frame_detector import SuperFRAMEDetector
from controlled_localization.models.super_frame_v2_detector import SuperFRAMEV2Detector
from controlled_localization.data.tiles import get_tile_grid_coords, blend_tiles_probability
from controlled_localization.utils import seed_all


ROOT = Path("I:/icvgip workshop")
RUNS_DIR = ROOT / "controlled_localization/runs"
VAL_MANIFEST = RUNS_DIR / "real_development_20261114/manifest_multigen.json"
TEST_MANIFEST_DIR = ROOT / "baseline_work/data_manifests"


def compute_distribution_stats(scores: List[float]) -> Dict[str, float]:
    arr = np.array(scores, dtype=np.float32)
    if len(arr) == 0:
        return {}
    return {
        "count": int(len(arr)),
        "mean": round(float(np.mean(arr)), 4),
        "std": round(float(np.std(arr)), 4),
        "median": round(float(np.median(arr)), 4),
        "p5": round(float(np.percentile(arr, 5)), 4),
        "p25": round(float(np.percentile(arr, 25)), 4),
        "p75": round(float(np.percentile(arr, 75)), 4),
        "p95": round(float(np.percentile(arr, 95)), 4),
        "min": round(float(np.min(arr)), 4),
        "max": round(float(np.max(arr)), 4),
    }


def predict_scores(
    model: nn.Module,
    img_arr: np.ndarray,
    device: torch.device,
    tile_size: int = 384,
    stride: int = 256,
) -> Tuple[float, float, float]:
    """Returns (s_whole, s_pool, s_hybrid)."""
    h, w = img_arr.shape[:2]

    # Whole-image view
    whole_resized = Image.fromarray((img_arr * 255).astype(np.uint8)).resize((tile_size, tile_size), Image.BILINEAR)
    whole_arr = np.array(whole_resized, dtype=np.float32) / 255.0
    whole_t = torch.from_numpy(whole_arr).permute(2, 0, 1).unsqueeze(0).float().to(device)

    with torch.amp.autocast("cuda"):
        out_w = model(whole_t)
    s_whole = float(torch.sigmoid(out_w["det_logits"][0, 0]).item())

    # Native tiles
    coords = get_tile_grid_coords(w, h, tile_size=tile_size, stride=stride)
    tile_logits = []

    for x, y in coords:
        tx_end = min(w, x + tile_size)
        ty_end = min(h, y + tile_size)
        patch = np.zeros((tile_size, tile_size, 3), dtype=np.float32)
        sub = img_arr[y:ty_end, x:tx_end]
        patch[: sub.shape[0], : sub.shape[1]] = sub

        t_t = torch.from_numpy(patch).permute(2, 0, 1).unsqueeze(0).float().to(device)
        with torch.amp.autocast("cuda"):
            t_out = model(t_t)
        tile_logits.append(t_out["mask_logits"][0, 0].detach().cpu().numpy())

    blended_map = blend_tiles_probability(tile_logits, coords, w, h, tile_size=tile_size)
    s_pool = float(np.mean(np.sort(blended_map.ravel())[-100:]))
    s_hybrid = 0.15 * s_whole + 0.85 * s_pool

    return s_whole, s_pool, s_hybrid


@torch.no_grad()
def evaluate_checkpoint(
    ckpt_path: Path,
    model_type: str,
    val_records: List[Dict],
    test_records: List[Dict],
    device: torch.device,
) -> Dict:
    print(f"\nEvaluating Checkpoint: {ckpt_path.name} ({model_type})...")

    if model_type == "super_v2":
        model = SuperFRAMEV2Detector(pretrained=False).to(device)
    else:
        model = SuperFRAMEDetector(pretrained=False).to(device)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(sd, strict=False)
    model.eval()

    # 1. Validation Run: collect scores & calibrate thresholds
    val_labels = []
    val_whole = []
    val_pool = []
    val_hybrid = []

    for rec in val_records:
        lbl = float(rec["label"])
        with Image.open(rec["image"]) as img:
            arr = np.array(img.convert("RGB"), dtype=np.float32) / 255.0
        sw, sp, sh = predict_scores(model, arr, device)
        val_labels.append(lbl)
        val_whole.append(sw)
        val_pool.append(sp)
        val_hybrid.append(sh)

    val_labels = np.array(val_labels, dtype=np.float32)
    gen_mask = (val_labels == 0)
    edt_mask = (val_labels == 1)

    dist_val = {
        "score_whole": {
            "genuine": compute_distribution_stats([val_whole[i] for i in range(len(val_labels)) if gen_mask[i]]),
            "edited": compute_distribution_stats([val_whole[i] for i in range(len(val_labels)) if edt_mask[i]]),
        },
        "score_pool": {
            "genuine": compute_distribution_stats([val_pool[i] for i in range(len(val_labels)) if gen_mask[i]]),
            "edited": compute_distribution_stats([val_pool[i] for i in range(len(val_labels)) if edt_mask[i]]),
        },
        "score_hybrid": {
            "genuine": compute_distribution_stats([val_hybrid[i] for i in range(len(val_labels)) if gen_mask[i]]),
            "edited": compute_distribution_stats([val_hybrid[i] for i in range(len(val_labels)) if edt_mask[i]]),
        },
    }

    # Calibrate thresholds on validation maximizing Balanced Accuracy
    calibrated_thresholds = {}
    for name, s_list in [("whole", val_whole), ("pool", val_pool), ("hybrid", val_hybrid)]:
        best_bacc = 0.0
        best_tau = 0.5
        best_acc = 0.0
        best_f1 = 0.0
        for tau in np.linspace(0.01, 0.99, 197):
            preds = [1 if s >= tau else 0 for s in s_list]
            bacc = balanced_accuracy_score(val_labels, preds)
            if bacc > best_bacc:
                best_bacc = bacc
                best_tau = tau
                best_acc = accuracy_score(val_labels, preds)
                best_f1 = f1_score(val_labels, preds, zero_division=0)
        calibrated_thresholds[name] = {
            "tau": round(float(best_tau), 4),
            "val_balanced_acc": round(float(best_bacc) * 100, 2),
            "val_acc": round(float(best_acc) * 100, 2),
            "val_f1": round(float(best_f1) * 100, 2),
            "val_auroc": round(float(roc_auc_score(val_labels, s_list)) * 100, 2),
        }

    # 2. Test Run: evaluate with FROZEN thresholds
    test_labels = []
    test_whole = []
    test_pool = []
    test_hybrid = []
    test_gens = []

    for rec in test_records:
        lbl = float(rec["label"])
        with Image.open(ROOT / rec["image_path"]) as img:
            arr = np.array(img.convert("RGB"), dtype=np.float32) / 255.0
        sw, sp, sh = predict_scores(model, arr, device)
        test_labels.append(lbl)
        test_whole.append(sw)
        test_pool.append(sp)
        test_hybrid.append(sh)
        test_gens.append(rec.get("generator", "unknown"))

    test_labels = np.array(test_labels, dtype=np.float32)

    frozen_test_results = {}
    for name, s_list in [("whole", test_whole), ("pool", test_pool), ("hybrid", test_hybrid)]:
        tau = calibrated_thresholds[name]["tau"]
        preds = [1 if s >= tau else 0 for s in s_list]
        acc = accuracy_score(test_labels, preds)
        bacc = balanced_accuracy_score(test_labels, preds)
        f1 = f1_score(test_labels, preds, zero_division=0)
        try:
            auroc = roc_auc_score(test_labels, s_list)
        except Exception:
            auroc = 0.5

        # Confusion
        tp = sum(1 for i in range(len(test_labels)) if test_labels[i] == 1 and preds[i] == 1)
        tn = sum(1 for i in range(len(test_labels)) if test_labels[i] == 0 and preds[i] == 0)
        fp = sum(1 for i in range(len(test_labels)) if test_labels[i] == 0 and preds[i] == 1)
        fn = sum(1 for i in range(len(test_labels)) if test_labels[i] == 1 and preds[i] == 0)

        # Per generator
        per_g = {}
        for g in set(test_gens):
            g_idx = [i for i, x in enumerate(test_gens) if x == g]
            g_lbl = test_labels[g_idx]
            g_pred = [preds[i] for i in g_idx]
            g_scores = [s_list[i] for i in g_idx]
            per_g[g] = {
                "acc": round(accuracy_score(g_lbl, g_pred) * 100, 2),
                "f1": round(f1_score(g_lbl, g_pred, zero_division=0) * 100, 2),
                "auroc": round(roc_auc_score(g_lbl, g_scores) * 100, 2) if len(set(g_lbl)) > 1 else 50.0,
            }

        frozen_test_results[name] = {
            "frozen_tau": tau,
            "accuracy": round(float(acc) * 100, 2),
            "balanced_accuracy": round(float(bacc) * 100, 2),
            "f1": round(float(f1) * 100, 2),
            "auroc": round(float(auroc) * 100, 2),
            "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
            "per_generator": per_g,
        }

    return {
        "checkpoint": ckpt_path.name,
        "model_type": model_type,
        "validation_distributions": dist_val,
        "calibrated_thresholds": calibrated_thresholds,
        "frozen_test_results": frozen_test_results,
    }


def main():
    seed_all(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Experiment 1 on {device}...")

    # Load validation manifest
    with open(VAL_MANIFEST, "r", encoding="utf-8") as f:
        all_dev = json.load(f)
    val_records = [r for r in all_dev if r["split"] == "val"]
    print(f"Loaded {len(val_records)} validation records.")

    # Load test manifests
    test_records = []
    for g in ["sd15", "sd2", "sd3", "sdxl", "flux"]:
        p = TEST_MANIFEST_DIR / f"test_{g}_manifest.json"
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        for e in data.get("entries", []):
            e_copy = dict(e)
            e_copy["generator"] = g
            test_records.append(e_copy)
    print(f"Loaded {len(test_records)} benchmark test records.")

    checkpoints = [
        ("super_frame_v2", RUNS_DIR / "super_frame_v2_run/best.pt", "super_v2"),
        ("super_frame_multigen", RUNS_DIR / "super_frame_multigen/best.pt", "super"),
        ("scaled_pilot_v2", RUNS_DIR / "scaled_pilot/best_superframe_v2_scaled.pt", "super_v2"),
        ("joint_dualview_v2", RUNS_DIR / "experiment2_joint_training/best_joint_dualview.pt", "super_v2"),
    ]

    out_p = ROOT / "controlled_localization/reports/experiment1_score_calibration_audit.json"
    all_results = {}
    if out_p.exists():
        try:
            with open(out_p, "r", encoding="utf-8") as f:
                all_results = json.load(f)
            print(f"Loaded existing results for: {list(all_results.keys())}")
        except Exception:
            all_results = {}

    for name, path, mtype in checkpoints:
        if path.exists() and name not in all_results:
            res = evaluate_checkpoint(path, mtype, val_records, test_records, device)
            all_results[name] = res
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    print("\n================================================================================")
    print("EXPERIMENT 1 SUMMARY: SCORE CALIBRATION & BENCHMARK ACCURACY (FROZEN TAU)")
    print("================================================================================")
    for name, r in all_results.items():
        print(f"\n--- Model: {name} ---")
        print("Score Type | Frozen Tau | Val B-Acc | Test Acc | Test B-Acc | Test F1 | Test AUROC | Confusion (TP/TN/FP/FN)")
        print("-----------|------------|-----------|----------|------------|---------|------------|------------------------")
        for stype in ["whole", "pool", "hybrid"]:
            v = r["calibrated_thresholds"][stype]
            t = r["frozen_test_results"][stype]
            c = t["confusion"]
            print(f"{stype:<10} | {t['frozen_tau']:>10.4f} | {v['val_balanced_acc']:>8.2f}% | {t['accuracy']:>7.2f}% | {t['balanced_accuracy']:>9.2f}% | {t['f1']:>6.2f}% | {t['auroc']:>9.2f}% | {c['tp']:>2}/{c['tn']:>2}/{c['fp']:>2}/{c['fn']:>2}")
    print(f"\nSaved complete distribution report to: {out_p}")


if __name__ == "__main__":
    main()
