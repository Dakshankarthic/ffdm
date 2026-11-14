"""Evaluates the trained Multi-Evidence Head on the 100-sample OpenSDID Benchmark across all 5 generators.
Extracts whole-image context + native-resolution tiled forensic evidence.
Compares directly to FUSED and DeCLIP baselines.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import roc_auc_score, accuracy_score, balanced_accuracy_score, f1_score

from controlled_localization.models.super_frame_v2_detector import SuperFRAMEV2Detector
from controlled_localization.models.multi_evidence_head import MultiEvidenceDetectionHead
from controlled_localization.data.canonicalize import load_canonical_rgb
from controlled_localization.data.tiles import get_tile_grid_coords, pad_if_needed
from controlled_localization.utils import resolve_device, seed_all


ROOT = Path("I:/icvgip workshop")
RUNS_DIR = ROOT / "controlled_localization/runs"
CKPT_DETECTOR = RUNS_DIR / "super_frame_v2_run/best.pt"
CKPT_HEAD = RUNS_DIR / "multi_evidence_head_v2/best_head.pt"


def evaluate_sample(
    img_path: Path,
    detector: SuperFRAMEV2Detector,
    head: MultiEvidenceDetectionHead,
    device: torch.device,
    tile_size: int = 384,
    stride: int = 256,
) -> Tuple[float, float, float]:
    """
    Returns:
      (score_combined, score_whole, max_score_tile)
    """
    img, _ = load_canonical_rgb(img_path)
    orig_w, orig_h = img.size
    arr_img = np.array(img, dtype=np.float32) / 255.0

    # 1. Resized Whole Image
    img_resized = img.resize((tile_size, tile_size), Image.BILINEAR)
    arr_resized = np.array(img_resized, dtype=np.float32) / 255.0
    ten_whole = torch.from_numpy(arr_resized).permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        out_whole = detector(ten_whole, return_features=True)
        e_whole = out_whole["det_feat"]  # [1, 257]

    # 2. Native-Resolution Tiling
    padded_img, _ = pad_if_needed(arr_img, tile_size=tile_size)
    padded_h, padded_w = padded_img.shape[:2]
    coords = get_tile_grid_coords(padded_w, padded_h, tile_size=tile_size, stride=stride)

    tile_feats = []
    tile_scores = []

    with torch.no_grad():
        for x, y in coords:
            crop = padded_img[y : y + tile_size, x : x + tile_size]
            ten_crop = torch.from_numpy(crop).permute(2, 0, 1).unsqueeze(0).to(device)
            out_tile = detector(ten_crop, return_features=True)
            e_tile = out_tile["det_feat"]  # [1, 257]
            tile_feats.append(e_tile)

            # Individual tile classification
            t_logit = head.tile_mlp(e_tile)
            t_prob = torch.sigmoid(t_logit).item()
            tile_scores.append(t_prob)

    # Stack tile features: [1, N, 257]
    e_tiles = torch.cat(tile_feats, dim=0).unsqueeze(0)  # [1, N, 257]

    with torch.no_grad():
        comb_logit, whole_logit, t_logits = head(e_whole, e_tiles)
        score_comb = float(torch.sigmoid(comb_logit).item())
        score_whole = float(torch.sigmoid(whole_logit).item())
        max_score_tile = float(np.max(tile_scores))

    return score_comb, score_whole, max_score_tile


def run_benchmark() -> None:
    seed_all(42)
    device = resolve_device()

    print("================================================================================")
    print("EVALUATING MULTI-EVIDENCE HEAD ON 100-SAMPLE OPENSDID BENCHMARK")
    print("================================================================================")

    # 1. Load models
    print(f"Loading SuperFRAME v2 from: {CKPT_DETECTOR}")
    ckpt_det = torch.load(CKPT_DETECTOR, map_location=device)
    sd_det = ckpt_det["model_state_dict"] if "model_state_dict" in ckpt_det else ckpt_det
    detector = SuperFRAMEV2Detector(pretrained=False).to(device)
    detector.load_state_dict(sd_det)
    detector.eval()

    print(f"Loading trained MultiEvidenceHead from: {CKPT_HEAD}")
    ckpt_head = torch.load(CKPT_HEAD, map_location=device)
    sd_head = ckpt_head["head_state_dict"] if "head_state_dict" in ckpt_head else ckpt_head
    head = MultiEvidenceDetectionHead(in_dim=257, hidden_dim=64).to(device)
    head.load_state_dict(sd_head)
    head.eval()

    manifest_dir = ROOT / "baseline_work/data_manifests"
    manifest_files = [
        "test_sd15_manifest.json",
        "test_sd2_manifest.json",
        "test_sd3_manifest.json",
        "test_sdxl_manifest.json",
        "test_flux_manifest.json",
    ]

    all_y_true = []
    all_s_comb = []
    all_s_whole = []
    all_s_tile = []

    per_generator = {}

    t0 = time.time()
    for m_file in manifest_files:
        gen_name = m_file.replace("test_", "").replace("_manifest.json", "")
        with open(manifest_dir / m_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        print(f"Evaluating generator: {gen_name.upper()} ({len(data.get('entries', []))} samples)...")
        gen_y, gen_comb, gen_whole, gen_tile = [], [], [], []

        for entry in data.get("entries", []):
            label = int(entry["label"])
            img_p = ROOT / entry["image_path"]
            if not img_p.exists():
                continue

            sc, sw, st = evaluate_sample(img_p, detector, head, device)

            all_y_true.append(label)
            all_s_comb.append(sc)
            all_s_whole.append(sw)
            all_s_tile.append(st)

            gen_y.append(label)
            gen_comb.append(sc)
            gen_whole.append(sw)
            gen_tile.append(st)

        def calc(y: List[int], scores: List[float], tau: float = 0.5) -> Dict[str, float]:
            y_arr = np.array(y)
            s_arr = np.array(scores)
            auc = float(roc_auc_score(y_arr, s_arr)) if len(np.unique(y_arr)) > 1 else 0.5
            preds = (s_arr >= tau).astype(int)
            acc = float(accuracy_score(y_arr, preds))
            bal_acc = float(balanced_accuracy_score(y_arr, preds)) if len(np.unique(y_arr)) > 1 else 0.5
            f1 = float(f1_score(y_arr, preds, zero_division=0))
            return {"roc_auc": round(auc, 4), "acc": round(acc, 4), "bal_acc": round(bal_acc, 4), "f1": round(f1, 4)}

        per_generator[gen_name] = {
            "combined": calc(gen_y, gen_comb),
            "whole": calc(gen_y, gen_whole),
            "tile": calc(gen_y, gen_tile),
        }

    total_time = time.time() - t0
    print(f"Benchmark completed in {total_time:.1f}s.")

    y_all = np.array(all_y_true)
    sc_all = np.array(all_s_comb)
    sw_all = np.array(all_s_whole)
    st_all = np.array(all_s_tile)

    def calc_overall(scores: np.ndarray, tau: float = 0.5) -> Dict[str, float]:
        auc = float(roc_auc_score(y_all, scores))
        preds = (scores >= tau).astype(int)
        acc = float(accuracy_score(y_all, preds))
        bal_acc = float(balanced_accuracy_score(y_all, preds))
        f1 = float(f1_score(y_all, preds, zero_division=0))
        return {"roc_auc": round(auc, 4), "acc": round(acc, 4), "bal_acc": round(bal_acc, 4), "f1": round(f1, 4)}

    results = {
        "overall": {
            "multi_evidence_combined": calc_overall(sc_all),
            "whole_image_only": calc_overall(sw_all),
            "max_tile_only": calc_overall(st_all),
        },
        "per_generator": per_generator,
    }

    out_json = ROOT / "controlled_localization/reports/multi_evidence_benchmark_results.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 90)
    print("MULTI-EVIDENCE CLASSIFIER BENCHMARK SUMMARY (SUPERFRAME v2)")
    print("=" * 90)
    print(f"{'Method':<26} | {'ROC-AUC':<10} | {'Accuracy (%)':<14} | {'Bal Acc (%)':<14} | {'F1 Score (%)'}")
    print("-" * 90)
    for m_key, label in [
        ("multi_evidence_combined", "Multi-Evidence (Combined)"),
        ("whole_image_only", "Whole-Image Head Only"),
        ("max_tile_only", "Max-Tile Evidence Only"),
    ]:
        m = results["overall"][m_key]
        print(f"{label:<26} | {m['roc_auc']:<10.4f} | {m['acc']*100:<14.2f} | {m['bal_acc']*100:<14.2f} | {m['f1']*100:.2f}%")
    print("-" * 90)

    print("\nPER-GENERATOR BREAKDOWN (COMBINED MULTI-EVIDENCE):")
    print(f"{'Generator':<12} | {'ROC-AUC':<10} | {'Accuracy (%)':<14} | {'Bal Acc (%)':<14} | {'F1 Score (%)'}")
    print("-" * 75)
    for g, data in per_generator.items():
        m = data["combined"]
        print(f"{g.upper():<12} | {m['roc_auc']:<10.4f} | {m['acc']*100:<14.2f} | {m['bal_acc']*100:<14.2f} | {m['f1']*100:.2f}%")
    print("-" * 75)
    print(f"Results saved to: {out_json}")


if __name__ == "__main__":
    run_benchmark()
