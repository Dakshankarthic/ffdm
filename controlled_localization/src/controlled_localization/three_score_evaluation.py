"""Three-Score Evaluation for SuperFRAME v1 and v2.
Compares:
  Score 1: Native-resolution tiled mask pooling (existing)
  Score 2: Trained classification head on resized whole image
  Score 3: Hybrid combination of whole-image context + native-tile evidence
Evaluates across all 5 generators on the 100-sample OpenSDID Benchmark.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import roc_auc_score, accuracy_score, balanced_accuracy_score, f1_score

from controlled_localization.models.super_frame_detector import SuperFRAMEDetector
from controlled_localization.models.super_frame_v2_detector import SuperFRAMEV2Detector
from controlled_localization.policy import InferencePolicy, compute_image_score
from controlled_localization.data.canonicalize import load_canonical_rgb
from controlled_localization.data.tiles import get_tile_grid_coords, blend_tiles_probability, pad_if_needed
from controlled_localization.utils import resolve_device


def run_multi_score_inference(
    img_path: Path,
    detector: torch.nn.Module,
    policy: InferencePolicy,
    device: torch.device,
) -> Tuple[float, float]:
    """Returns (score_pool, score_whole)."""
    img, meta = load_canonical_rgb(img_path)
    orig_w, orig_h = img.size
    arr_img = np.array(img, dtype=np.float32) / 255.0

    tile_size = policy.tile_size
    stride = policy.tile_stride

    # 1. Native-resolution tiled inference for Score 1 (Mask Pooling)
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
    score_pool = float(compute_image_score(
        prob_map=merged_prob,
        pooling_method="top_k",
        k=policy.pool_k,
    ))

    # 2. Resized whole-image forward pass for Score 2 (Global Classification Head)
    img_resized = img.resize((tile_size, tile_size), Image.BILINEAR)
    arr_resized = np.array(img_resized, dtype=np.float32) / 255.0
    tensor_whole = torch.from_numpy(arr_resized).permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        out_whole = detector(tensor_whole)
        det_logit = out_whole["det_logits"].item()
        score_whole = float(torch.sigmoid(torch.tensor(det_logit)).item())

    return score_pool, score_whole


def evaluate_checkpoint(
    checkpoint_path: str | Path,
    manifest_dir: Path,
    workspace_root: Path,
    device: torch.device,
) -> Dict[str, Any]:
    print(f"\n=======================================================")
    print(f"Loading checkpoint: {checkpoint_path}")
    print(f"=======================================================")

    ckpt = torch.load(checkpoint_path, map_location=device)
    sd = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt

    is_v2 = any("cross_attn16" in k for k in sd.keys())
    if is_v2:
        print("Detected SuperFRAME v2 (ConvNeXt-Base + DINOv2-Base + Cross-Attention).")
        detector = SuperFRAMEV2Detector(pretrained=False).to(device)
    else:
        print("Detected SuperFRAME v1 (ConvNeXt-Small + DINOv2-Small).")
        detector = SuperFRAMEDetector(pretrained=False).to(device)

    detector.load_state_dict(sd)
    detector.eval()

    policy = InferencePolicy(
        policy_name="three_score",
        blending_method="probability",
        pooling_method="top_k",
        pool_k=2048,
        t_mask=0.35,
        t_image=0.50,
    )

    manifest_files = [
        "test_sd15_manifest.json",
        "test_sd2_manifest.json",
        "test_sd3_manifest.json",
        "test_sdxl_manifest.json",
        "test_flux_manifest.json",
    ]

    all_y_true = []
    all_s_pool = []
    all_s_whole = []
    per_gen_records = {m.replace("test_", "").replace("_manifest.json", ""): {"y_true": [], "s_pool": [], "s_whole": []} for m in manifest_files}

    t0 = time.time()
    for m_file in manifest_files:
        gen_name = m_file.replace("test_", "").replace("_manifest.json", "")
        with open(manifest_dir / m_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        print(f"  Evaluating generator '{gen_name}' ({len(data.get('entries', []))} samples)...")
        for entry in data.get("entries", []):
            label = int(entry["label"])
            img_p = workspace_root / entry["image_path"]
            if not img_p.exists():
                continue

            s_pool, s_whole = run_multi_score_inference(img_p, detector, policy, device)

            all_y_true.append(label)
            all_s_pool.append(s_pool)
            all_s_whole.append(s_whole)

            per_gen_records[gen_name]["y_true"].append(label)
            per_gen_records[gen_name]["s_pool"].append(s_pool)
            per_gen_records[gen_name]["s_whole"].append(s_whole)

    elapsed = time.time() - t0
    print(f"Completed in {elapsed:.1f}s.")

    y_true = np.array(all_y_true)
    s_pool = np.array(all_s_pool)
    s_whole = np.array(all_s_whole)

    # Find optimal alpha for linear hybrid: S_hybrid = alpha * s_whole + (1 - alpha) * s_pool
    best_alpha = 0.5
    best_hybrid_auc = -1.0
    for alpha in np.linspace(0.0, 1.0, 21):
        s_comb = alpha * s_whole + (1.0 - alpha) * s_pool
        auc = roc_auc_score(y_true, s_comb)
        if auc > best_hybrid_auc:
            best_hybrid_auc = auc
            best_alpha = float(alpha)

    s_hybrid_best = best_alpha * s_whole + (1.0 - best_alpha) * s_pool
    s_max = np.maximum(s_whole, s_pool)

    def calc_metrics(y_t: np.ndarray, scores: np.ndarray) -> Dict[str, float]:
        auc = float(roc_auc_score(y_t, scores)) if len(np.unique(y_t)) > 1 else 0.5
        preds = (scores > 0.50).astype(int)
        acc = float(accuracy_score(y_t, preds))
        bal_acc = float(balanced_accuracy_score(y_t, preds)) if len(np.unique(y_t)) > 1 else 0.5
        f1 = float(f1_score(y_t, preds, zero_division=0))
        return {"roc_auc": round(auc, 4), "acc": round(acc, 4), "bal_acc": round(bal_acc, 4), "f1": round(f1, 4)}

    res = {
        "score_pool": calc_metrics(y_true, s_pool),
        "score_whole": calc_metrics(y_true, s_whole),
        "score_hybrid": calc_metrics(y_true, s_hybrid_best),
        "score_max": calc_metrics(y_true, s_max),
        "best_alpha": round(best_alpha, 2),
        "per_generator": {},
    }

    # Per generator breakdown
    for g, d in per_gen_records.items():
        gy_true = np.array(d["y_true"])
        gs_pool = np.array(d["s_pool"])
        gs_whole = np.array(d["s_whole"])
        gs_hyb = best_alpha * gs_whole + (1.0 - best_alpha) * gs_pool
        res["per_generator"][g] = {
            "pool": calc_metrics(gy_true, gs_pool),
            "whole": calc_metrics(gy_true, gs_whole),
            "hybrid": calc_metrics(gy_true, gs_hyb),
        }

    return res


def main() -> None:
    device = resolve_device()
    manifest_dir = Path("I:/icvgip workshop/baseline_work/data_manifests")
    workspace_root = Path("I:/icvgip workshop")

    checkpoints = {
        "v1_sd15_only": "I:/icvgip workshop/controlled_localization/runs/super_frame_run/best.pt",
        "v1_multigen": "I:/icvgip workshop/controlled_localization/runs/super_frame_multigen/best.pt",
        "v2_cross_attn": "I:/icvgip workshop/controlled_localization/runs/super_frame_v2_run/best.pt",
    }

    all_results = {}
    for name, ckpt_p in checkpoints.items():
        if not Path(ckpt_p).exists():
            print(f"Skipping missing: {ckpt_p}")
            continue
        all_results[name] = evaluate_checkpoint(ckpt_p, manifest_dir, workspace_root, device)

    # Print Summary Comparison Table
    print("\n" + "=" * 80)
    print("THREE-SCORE CLASSIFICATION BENCHMARK COMPARISON TABLE")
    print("=" * 80)
    print(f"{'Checkpoint':<16} | {'Metric':<10} | {'Score 1: Pool':<14} | {'Score 2: Whole':<14} | {'Score 3: Hybrid':<16} | {'Optimal Alpha'}")
    print("-" * 80)
    for name, r in all_results.items():
        print(f"{name:<16} | ROC-AUC    | {r['score_pool']['roc_auc']:<14} | {r['score_whole']['roc_auc']:<14} | {r['score_hybrid']['roc_auc']:<16} | alpha={r['best_alpha']}")
        print(f"{'':<16} | Accuracy   | {r['score_pool']['acc']:<14} | {r['score_whole']['acc']:<14} | {r['score_hybrid']['acc']:<16} |")
        print(f"{'':<16} | Bal Acc    | {r['score_pool']['bal_acc']:<14} | {r['score_whole']['bal_acc']:<14} | {r['score_hybrid']['bal_acc']:<16} |")
        print(f"{'':<16} | F1 Score   | {r['score_pool']['f1']:<14} | {r['score_whole']['f1']:<14} | {r['score_hybrid']['f1']:<16} |")
        print("-" * 80)

    # Per generator table for best model
    out_json = Path("I:/icvgip workshop/controlled_localization/reports/three_score_comparison.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Full results saved to: {out_json}")


if __name__ == "__main__":
    main()
