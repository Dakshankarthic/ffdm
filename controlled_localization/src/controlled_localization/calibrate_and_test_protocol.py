"""Calibrated Three-Score Evaluation Protocol.
Calibrates combination alpha and decision thresholds on the VALIDATION split,
freezes them, and applies them to the official 100-sample multi-generator test set.
Direct comparison with FUSED protocol (SD1.5 trained vs 5 generator test split).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score, accuracy_score, balanced_accuracy_score, f1_score

from controlled_localization.models.super_frame_detector import SuperFRAMEDetector
from controlled_localization.models.super_frame_v2_detector import SuperFRAMEV2Detector
from controlled_localization.policy import InferencePolicy, compute_image_score
from controlled_localization.data.canonicalize import load_canonical_rgb
from controlled_localization.data.tiles import get_tile_grid_coords, blend_tiles_probability, pad_if_needed
from controlled_localization.utils import resolve_device, seed_all


def run_scores_for_image(
    img_path: Path,
    detector: torch.nn.Module,
    device: torch.device,
    tile_size: int = 384,
    stride: int = 256,
    pool_k: int = 2048,
) -> Tuple[float, float]:
    img, _ = load_canonical_rgb(img_path)
    orig_w, orig_h = img.size
    arr_img = np.array(img, dtype=np.float32) / 255.0

    # 1. Native-resolution tiled inference (Score 1: Pool)
    padded_img, _ = pad_if_needed(arr_img, tile_size=tile_size)
    padded_h, padded_w = padded_img.shape[:2]
    coords = get_tile_grid_coords(padded_w, padded_h, tile_size=tile_size, stride=stride)
    tile_logits = []

    with torch.no_grad():
        for x, y in coords:
            crop = padded_img[y : y + tile_size, x : x + tile_size]
            tensor_crop = torch.from_numpy(crop).permute(2, 0, 1).unsqueeze(0).to(device)
            out = detector(tensor_crop)
            tile_logits.append(out["mask_logits"].squeeze().cpu().numpy())

    merged_prob = blend_tiles_probability(
        tile_logits=tile_logits,
        coords=coords,
        image_width=orig_w,
        image_height=orig_h,
        tile_size=tile_size,
        use_logit_blending=False,
    )
    s_pool = float(compute_image_score(
        prob_map=merged_prob,
        pooling_method="top_k",
        k=pool_k,
    ))

    # 2. Resized whole image inference (Score 2: Whole)
    img_resized = img.resize((tile_size, tile_size), Image.BILINEAR)
    arr_resized = np.array(img_resized, dtype=np.float32) / 255.0
    tensor_whole = torch.from_numpy(arr_resized).permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        out_whole = detector(tensor_whole)
        det_logit = out_whole["det_logits"].item()
        s_whole = float(torch.sigmoid(torch.tensor(det_logit)).item())

    return s_pool, s_whole


def optimize_threshold(y_true: np.ndarray, scores: np.ndarray) -> Tuple[float, float]:
    """Finds threshold maximizing balanced accuracy on validation set."""
    best_tau = 0.5
    best_bal_acc = -1.0
    # Evaluate across fine percentiles of candidate scores
    taus = np.linspace(np.min(scores), np.max(scores), 101)
    for tau in taus:
        preds = (scores >= tau).astype(int)
        bal_acc = balanced_accuracy_score(y_true, preds)
        if bal_acc > best_bal_acc:
            best_bal_acc = bal_acc
            best_tau = float(tau)
    return best_tau, best_bal_acc


def evaluate_split(
    records: List[Dict],
    detector: torch.nn.Module,
    device: torch.device,
    workspace_root: Path,
    desc: str = "",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    y_trues, s_pools, s_wholes = [], [], []
    t0 = time.time()
    for i, r in enumerate(records):
        img_p = Path(r["image"]) if "image" in r else workspace_root / r["image_path"]
        if not img_p.exists():
            continue
        label = int(r["label"])
        sp, sw = run_scores_for_image(img_p, detector, device)
        y_trues.append(label)
        s_pools.append(sp)
        s_wholes.append(sw)
        if (i + 1) % 50 == 0 or (i + 1) == len(records):
            print(f"  [{desc}] Processed {i+1}/{len(records)} in {time.time()-t0:.1f}s")

    return np.array(y_trues), np.array(s_pools), np.array(s_wholes)


def compute_metrics_at_tau(y_true: np.ndarray, scores: np.ndarray, tau: float) -> Dict[str, float]:
    auc = float(roc_auc_score(y_true, scores)) if len(np.unique(y_true)) > 1 else 0.5
    preds = (scores >= tau).astype(int)
    acc = float(accuracy_score(y_true, preds))
    bal_acc = float(balanced_accuracy_score(y_true, preds)) if len(np.unique(y_true)) > 1 else 0.5
    f1 = float(f1_score(y_true, preds, zero_division=0))
    return {"roc_auc": round(auc, 4), "acc": round(acc, 4), "bal_acc": round(bal_acc, 4), "f1": round(f1, 4), "tau": round(tau, 4)}


def run_pipeline() -> None:
    seed_all(42)
    device = resolve_device()
    root = Path("I:/icvgip workshop")

    checkpoints = {
        "v1_sd15_only": {
            "path": root / "controlled_localization/runs/super_frame_run/best.pt",
            "val_manifest": root / "controlled_localization/runs/real_development_20261114/manifest.json",
        },
        "v1_multigen": {
            "path": root / "controlled_localization/runs/super_frame_multigen/best.pt",
            "val_manifest": root / "controlled_localization/runs/real_development_20261114/manifest_multigen.json",
        },
        "v2_cross_attn": {
            "path": root / "controlled_localization/runs/super_frame_v2_run/best.pt",
            "val_manifest": root / "controlled_localization/runs/real_development_20261114/manifest_multigen.json",
        },
    }

    test_manifests = {
        "sd15": root / "baseline_work/data_manifests/test_sd15_manifest.json",
        "sd2": root / "baseline_work/data_manifests/test_sd2_manifest.json",
        "sd3": root / "baseline_work/data_manifests/test_sd3_manifest.json",
        "sdxl": root / "baseline_work/data_manifests/test_sdxl_manifest.json",
        "flux": root / "baseline_work/data_manifests/test_flux_manifest.json",
    }

    final_report = {}

    for ckpt_key, cfg in checkpoints.items():
        ckpt_path = cfg["path"]
        print(f"\n=======================================================")
        print(f"PIPELINE: Calibrating & Evaluating {ckpt_key}")
        print(f"Path: {ckpt_path}")
        print(f"=======================================================")

        ckpt = torch.load(ckpt_path, map_location=device)
        sd = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt

        is_v2 = any("cross_attn16" in k for k in sd.keys())
        if is_v2:
            detector = SuperFRAMEV2Detector(pretrained=False).to(device)
        else:
            detector = SuperFRAMEDetector(pretrained=False).to(device)

        detector.load_state_dict(sd)
        detector.eval()

        # Step 1: Calibration on Validation Set (Subsample 100 balanced validation samples for quick calibration)
        with open(cfg["val_manifest"], "r", encoding="utf-8") as f:
            val_full = [r for r in json.load(f) if r["split"] == "val"]

        # Take 50 pos and 50 neg from val
        val_pos = [r for r in val_full if r["label"] == 1.0][:50]
        val_neg = [r for r in val_full if r["label"] == 0.0][:50]
        val_calib = val_pos + val_neg

        print(f"Running score calibration on {len(val_calib)} validation samples...")
        val_y, val_pool, val_whole = evaluate_split(val_calib, detector, device, root, desc="VAL")

        # Find best alpha on validation set using AUROC
        best_alpha = 0.5
        best_val_auc = -1.0
        for alpha in np.linspace(0.0, 1.0, 21):
            comb = alpha * val_whole + (1.0 - alpha) * val_pool
            auc = roc_auc_score(val_y, comb)
            if auc > best_val_auc:
                best_val_auc = auc
                best_alpha = float(alpha)

        val_hybrid = best_alpha * val_whole + (1.0 - best_alpha) * val_pool

        # Optimize frozen thresholds on validation set
        tau_pool, _ = optimize_threshold(val_y, val_pool)
        tau_whole, _ = optimize_threshold(val_y, val_whole)
        tau_hybrid, _ = optimize_threshold(val_y, val_hybrid)

        print(f"Calibration Results on Validation:")
        print(f"  Best Alpha: {best_alpha:.2f} (Val AUROC: {best_val_auc:.4f})")
        print(f"  Calibrated Tau Pool:   {tau_pool:.4f}")
        print(f"  Calibrated Tau Whole:  {tau_whole:.4f}")
        print(f"  Calibrated Tau Hybrid: {tau_hybrid:.4f}")

        # Step 2: Test on 100-sample test benchmark using FROZEN thresholds
        print(f"\nEvaluating on official 100-sample test set with FROZEN parameters...")
        test_y_all, test_pool_all, test_whole_all = [], [], []
        per_gen_test = {}

        for gen_name, m_path in test_manifests.items():
            with open(m_path, "r", encoding="utf-8") as f:
                t_entries = json.load(f).get("entries", [])
            gy, gp, gw = evaluate_split(t_entries, detector, device, root, desc=f"TEST-{gen_name.upper()}")
            gh = best_alpha * gw + (1.0 - best_alpha) * gp

            per_gen_test[gen_name] = {
                "pool": compute_metrics_at_tau(gy, gp, tau_pool),
                "whole": compute_metrics_at_tau(gy, gw, tau_whole),
                "hybrid": compute_metrics_at_tau(gy, gh, tau_hybrid),
            }

            test_y_all.extend(gy)
            test_pool_all.extend(gp)
            test_whole_all.extend(gw)

        test_y = np.array(test_y_all)
        test_pool = np.array(test_pool_all)
        test_whole = np.array(test_whole_all)
        test_hybrid = best_alpha * test_whole + (1.0 - best_alpha) * test_pool

        overall_metrics = {
            "score_pool": compute_metrics_at_tau(test_y, test_pool, tau_pool),
            "score_whole": compute_metrics_at_tau(test_y, test_whole, tau_whole),
            "score_hybrid": compute_metrics_at_tau(test_y, test_hybrid, tau_hybrid),
        }

        final_report[ckpt_key] = {
            "calibration": {
                "best_alpha": round(best_alpha, 2),
                "tau_pool": round(tau_pool, 4),
                "tau_whole": round(tau_whole, 4),
                "tau_hybrid": round(tau_hybrid, 4),
            },
            "overall_test": overall_metrics,
            "per_generator": per_gen_test,
        }

    # Save to JSON
    out_file = root / "controlled_localization/reports/calibrated_fused_protocol_results.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2)

    print("\n" + "=" * 90)
    print("CALIBRATED TEST BENCHMARK COMPARISON TABLE (FUSED PROTOCOL)")
    print("=" * 90)
    print(f"{'Checkpoint':<16} | {'Score Method':<14} | {'Mean Acc (%)':<14} | {'Mean BalAcc (%)':<16} | {'Mean F1 (%)':<14} | {'ROC-AUC'}")
    print("-" * 90)
    for name, d in final_report.items():
        for s_name, label in [("score_pool", "Score 1: Pool"), ("score_whole", "Score 2: Whole"), ("score_hybrid", "Score 3: Hybrid")]:
            m = d["overall_test"][s_name]
            print(f"{name:<16} | {label:<14} | {m['acc']*100:<14.2f} | {m['bal_acc']*100:<16.2f} | {m['f1']*100:<14.2f} | {m['roc_auc']:.4f}")
        print("-" * 90)
    print(f"Saved complete results to: {out_file}")


if __name__ == "__main__":
    run_pipeline()
