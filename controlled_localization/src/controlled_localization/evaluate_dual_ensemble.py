"""SuperFRAME Dual-Model Ensemble Benchmark: v2 (Cross-Attention) + v1 (MultiGen).
Features:
- Dual-Architecture Complementary Fusion:
  * SuperFRAME v2: Dual-Base (ConvNeXt-Base + DINOv2-Base) + Cross-Attention FPN (excels at large edits, SD3, FLUX).
  * SuperFRAME v1: ConvNeXt-Small + DINOv2-Small + MultiGen (excels at fine texture, SD1.5, SD2).
- Test-Time Augmentation (Horizontal Flip Ensembling).
- Morphological Postprocessing (hole filling + small island noise suppression).
- Comprehensive benchmark reporting across edit-size bins and all 5 generators.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
from scipy import ndimage
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score, accuracy_score, balanced_accuracy_score, f1_score

from controlled_localization.models.super_frame_detector import SuperFRAMEDetector
from controlled_localization.models.super_frame_v2_detector import SuperFRAMEV2Detector
from controlled_localization.policy import InferencePolicy, compute_image_score
from controlled_localization.data.canonicalize import load_canonical_rgb
from controlled_localization.data.tiles import get_tile_grid_coords, blend_tiles_probability, pad_if_needed
from controlled_localization.metrics import compute_binary_confusion
from controlled_localization.utils import resolve_device, seed_all


ROOT = Path("I:/icvgip workshop")
CKPT_V1 = ROOT / "controlled_localization/runs/super_frame_multigen/best.pt"
CKPT_V2 = ROOT / "controlled_localization/runs/super_frame_v2_run/best.pt"


def get_tiled_prob_map(
    detector: torch.nn.Module,
    arr_img: np.ndarray,
    orig_w: int,
    orig_h: int,
    device: torch.device,
    tile_size: int = 384,
    stride: int = 256,
    use_tta: bool = True,
) -> np.ndarray:
    def infer_single(arr: np.ndarray) -> np.ndarray:
        padded, _ = pad_if_needed(arr, tile_size=tile_size)
        ph, pw = padded.shape[:2]
        coords = get_tile_grid_coords(pw, ph, tile_size=tile_size, stride=stride)
        tile_logits = []
        with torch.no_grad():
            for x, y in coords:
                crop = padded[y : y + tile_size, x : x + tile_size]
                ten = torch.from_numpy(crop).permute(2, 0, 1).unsqueeze(0).to(device)
                out = detector(ten)
                tile_logits.append(out["mask_logits"].squeeze().cpu().numpy())
        return blend_tiles_probability(
            tile_logits=tile_logits,
            coords=coords,
            image_width=orig_w,
            image_height=orig_h,
            tile_size=tile_size,
            use_logit_blending=False,
        )

    prob = infer_single(arr_img)
    if use_tta:
        arr_flip = np.fliplr(arr_img).copy()
        prob_flip = infer_single(arr_flip)
        prob = 0.5 * (prob + np.fliplr(prob_flip))
    return prob


def postprocess_mask(
    prob: np.ndarray,
    t_mask: float = 0.35,
    min_size: int = 64,
    fill_holes: bool = True,
) -> np.ndarray:
    bin_mask = (prob > t_mask).astype(np.uint8)
    if bin_mask.sum() == 0:
        return bin_mask
    if fill_holes:
        bin_mask = ndimage.binary_fill_holes(bin_mask).astype(np.uint8)
    if min_size > 0:
        labeled, num_features = ndimage.label(bin_mask)
        if num_features > 0:
            sizes = ndimage.sum(bin_mask, labeled, range(1, num_features + 1))
            mask_sizes = np.array(sizes) >= min_size
            remove_labels = np.where(~mask_sizes)[0] + 1
            for lbl in remove_labels:
                bin_mask[labeled == lbl] = 0
    return bin_mask


def evaluate_dual_ensemble(
    w_v2: float = 0.65,
    t_mask: float = 0.35,
    min_size: int = 64,
    use_tta: bool = True,
    fill_holes: bool = True,
) -> Dict[str, Any]:
    seed_all(42)
    device = resolve_device()

    print("================================================================================")
    print(f"EVALUATING SUPERFRAME DUAL ENSEMBLE (v2 w={w_v2:.2f} + v1 w={1-w_v2:.2f})")
    print(f"Parameters: TTA={use_tta}, fill_holes={fill_holes}, min_size={min_size}, t_mask={t_mask}")
    print("================================================================================")

    # 1. Load Model 1 (v1 MultiGen)
    print(f"Loading SuperFRAME v1 MultiGen from: {CKPT_V1}")
    ckpt1 = torch.load(CKPT_V1, map_location=device)
    sd1 = ckpt1["model_state_dict"] if "model_state_dict" in ckpt1 else ckpt1
    m1 = SuperFRAMEDetector(pretrained=False).to(device)
    m1.load_state_dict(sd1)
    m1.eval()

    # 2. Load Model 2 (v2 Cross-Attention)
    print(f"Loading SuperFRAME v2 Cross-Attention from: {CKPT_V2}")
    ckpt2 = torch.load(CKPT_V2, map_location=device)
    sd2 = ckpt2["model_state_dict"] if "model_state_dict" in ckpt2 else ckpt2
    m2 = SuperFRAMEV2Detector(pretrained=False).to(device)
    m2.load_state_dict(sd2)
    m2.eval()

    manifest_dir = ROOT / "baseline_work/data_manifests"
    manifest_files = [
        "test_sd15_manifest.json",
        "test_sd2_manifest.json",
        "test_sd3_manifest.json",
        "test_sdxl_manifest.json",
        "test_flux_manifest.json",
    ]

    all_ious = []
    all_f1s = []
    all_clean_fps = []

    # Edit size bins for localization
    size_bins = {"tiny": [], "medium": [], "large": []}  # <2%, 2-10%, >10%
    per_gen = {}

    t0 = time.time()
    for m_file in manifest_files:
        gen_name = m_file.replace("test_", "").replace("_manifest.json", "")
        with open(manifest_dir / m_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        entries = data.get("entries", [])
        g_ious, g_f1s, g_clean_fps = [], [], []
        print(f"  Evaluating {gen_name.upper()} ({len(entries)} samples)...", flush=True)

        for entry in entries:
            label = int(entry["label"])
            img_p = ROOT / entry["image_path"]
            if not img_p.exists():
                continue

            img, _ = load_canonical_rgb(img_p)
            w, h = img.size
            arr_img = np.array(img, dtype=np.float32) / 255.0

            # 1. Forward passes for v1 and v2
            p1 = get_tiled_prob_map(m1, arr_img, w, h, device, use_tta=use_tta)
            p2 = get_tiled_prob_map(m2, arr_img, w, h, device, use_tta=use_tta)

            # 2. Ensemble probability map
            p_ens = w_v2 * p2 + (1.0 - w_v2) * p1

            # 3. Postprocess
            pred_bin = postprocess_mask(p_ens, t_mask=t_mask, min_size=min_size, fill_holes=fill_holes)

            # Ground truth
            mask_rel = entry.get("mask_path")
            if label == 1 and mask_rel:
                mask_p = ROOT / mask_rel
                with Image.open(mask_p) as m_img:
                    m_gray = m_img.convert("L").resize((w, h), resample=Image.NEAREST)
                    gt_mask = (np.array(m_gray, dtype=np.float32) > 127.0).astype(np.uint8)
            else:
                gt_mask = np.zeros((h, w), dtype=np.uint8)

            cm = compute_binary_confusion(pred_bin, gt_mask)

            if label == 1:
                edit_frac = float(np.mean(gt_mask > 0))
                iou = cm.iou
                f1 = cm.f1

                g_ious.append(iou)
                g_f1s.append(f1)
                all_ious.append(iou)
                all_f1s.append(f1)

                if edit_frac < 0.02:
                    size_bins["tiny"].append(iou)
                elif edit_frac <= 0.10:
                    size_bins["medium"].append(iou)
                else:
                    size_bins["large"].append(iou)
            else:
                fp_pct = float(np.mean(pred_bin > 0)) * 100.0
                g_clean_fps.append(fp_pct)
                all_clean_fps.append(fp_pct)

        per_gen[gen_name] = {
            "mean_iou": float(np.mean(g_ious)) if g_ious else 0.0,
            "mean_f1": float(np.mean(g_f1s)) if g_f1s else 0.0,
            "clean_fp": float(np.mean(g_clean_fps)) if g_clean_fps else 0.0,
        }

    total_time = time.time() - t0

    mean_iou = float(np.mean(all_ious)) if all_ious else 0.0
    mean_f1 = float(np.mean(all_f1s)) if all_f1s else 0.0
    mean_clean_fp = float(np.mean(all_clean_fps)) if all_clean_fps else 0.0

    tiny_iou = float(np.mean(size_bins["tiny"])) if size_bins["tiny"] else 0.0
    med_iou = float(np.mean(size_bins["medium"])) if size_bins["medium"] else 0.0
    large_iou = float(np.mean(size_bins["large"])) if size_bins["large"] else 0.0

    results = {
        "overall": {
            "mean_iou": round(mean_iou, 4),
            "pixel_f1": round(mean_f1, 4),
            "clean_fp": round(mean_clean_fp, 4),
            "tiny_iou": round(tiny_iou, 4),
            "med_iou": round(med_iou, 4),
            "large_iou": round(large_iou, 4),
            "runtime_seconds": round(total_time, 1),
        },
        "per_generator": per_gen,
        "config": {
            "w_v2": w_v2,
            "t_mask": t_mask,
            "min_size": min_size,
            "use_tta": use_tta,
            "fill_holes": fill_holes,
        },
    }

    out_json = ROOT / "controlled_localization/reports/dual_ensemble_benchmark_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 80)
    print("SUPERFRAME DUAL ENSEMBLE OFFICIAL BENCHMARK RESULTS")
    print("=" * 80)
    print(f"Overall Localization Mean IoU:   {mean_iou * 100:.2f}%")
    print(f"Large Edit (>10%) IoU:           {large_iou * 100:.2f}%")
    print(f"Medium Edit (2-10%) IoU:         {med_iou * 100:.2f}%")
    print(f"Tiny Edit (<2%) IoU:             {tiny_iou * 100:.2f}%")
    print(f"Localization Pixel F1 Score:     {mean_f1 * 100:.2f}%")
    print(f"Clean False Alarm Rate:          {mean_clean_fp:.2f}%")
    print(f"Evaluation Runtime:              {total_time:.1f}s")
    print("-" * 80)
    print("Per-Generator Inpainting Breakdown:")
    for g, s in per_gen.items():
        print(f"  {g.upper():<10} | IoU: {s['mean_iou']*100:6.2f}% | F1: {s['mean_f1']*100:6.2f}% | Clean FP: {s['clean_fp']:5.2f}%")
    print("=" * 80)
    print(f"Results saved to: {out_json}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--w-v2", type=float, default=0.65)
    parser.add_argument("--t-mask", type=float, default=0.35)
    parser.add_argument("--min-size", type=int, default=64)
    parser.add_argument("--use-tta", action="store_true")
    parser.add_argument("--fill-holes", action="store_true")
    args = parser.parse_args()

    evaluate_dual_ensemble(
        w_v2=args.w_v2,
        t_mask=args.t_mask,
        min_size=args.min_size,
        use_tta=args.use_tta,
        fill_holes=args.fill_holes,
    )
