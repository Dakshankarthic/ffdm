"""Phase 5: Locked Paper-Quality Final Benchmark Evaluation.
Runs strictly on the frozen 100-image multi-generator benchmark:
  - SD 1.5 (20 images: 10 edited, 10 genuine)
  - SD 2.1 (20 images: 10 edited, 10 genuine)
  - SD 3   (20 images: 10 edited, 10 genuine)
  - SDXL   (20 images: 10 edited, 10 genuine)
  - FLUX.1 (20 images: 10 edited, 10 genuine)
Evaluates:
  1. SuperFRAME v2 Single Model (Multi-Evidence Whole + Native Tiles)
  2. SuperFRAME Dual Ensemble (+Flip TTA)
  3. Comparison with FUSED, DeCLIP, and VAAS.
"""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Dict, List

import numpy as np
from PIL import Image
from scipy.ndimage import label, binary_fill_holes


def remove_small_components(bin_mask: np.ndarray, min_size: int = 64) -> np.ndarray:
    labeled, num_features = label(bin_mask)
    if num_features == 0:
        return bin_mask.astype(np.float32)
    counts = np.bincount(labeled.ravel())
    remove_mask = counts < min_size
    remove_mask[0] = False
    return np.where(remove_mask[labeled], 0, bin_mask).astype(np.float32)
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import torch
import torch.nn as nn

from controlled_localization.models.super_frame_detector import SuperFRAMEDetector
from controlled_localization.models.super_frame_v2_detector import SuperFRAMEV2Detector
from controlled_localization.data.tiles import get_tile_grid_coords, blend_tiles_probability
from controlled_localization.utils import seed_all


ROOT = Path("I:/icvgip workshop")
RUNS_DIR = ROOT / "controlled_localization/runs"
V2_CKPT = RUNS_DIR / "super_frame_v2_run/best.pt"
V1_CKPT = RUNS_DIR / "super_frame_multigen/best.pt"
MANIFEST_DIR = ROOT / "baseline_work/data_manifests"


def load_test_manifests() -> Dict[str, List[Dict]]:
    generators = ["sd15", "sd2", "sd3", "sdxl", "flux"]
    test_data = {}
    for g in generators:
        p = MANIFEST_DIR / f"test_{g}_manifest.json"
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        test_data[g] = data.get("entries", [])
    return test_data


@torch.no_grad()
def predict_single_image(
    v2_model: nn.Module,
    v1_model: nn.Module | None,
    img_arr: np.ndarray,
    device: torch.device,
    use_ensemble: bool = True,
    tile_size: int = 384,
) -> Tuple[float, np.ndarray]:
    h, w = img_arr.shape[:2]

    # Whole image view (384x384)
    whole_resized = Image.fromarray((img_arr * 255).astype(np.uint8)).resize((tile_size, tile_size), Image.BILINEAR)
    whole_arr = np.array(whole_resized, dtype=np.float32) / 255.0
    whole_t = torch.from_numpy(whole_arr).permute(2, 0, 1).unsqueeze(0).float().to(device)

    with torch.amp.autocast("cuda"):
        out_v2_w = v2_model(whole_t)
    s_whole_v2 = float(torch.sigmoid(out_v2_w["det_logits"][0, 0]).item())

    if use_ensemble and v1_model is not None:
        with torch.amp.autocast("cuda"):
            out_v1_w = v1_model(whole_t)
        s_whole_v1 = float(torch.sigmoid(out_v1_w["det_logits"][0, 0]).item())
        s_whole = 0.65 * s_whole_v2 + 0.35 * s_whole_v1
    else:
        s_whole = s_whole_v2

    # Native tiles (Hann-blended)
    coords = get_tile_grid_coords(w, h, tile_size=tile_size, stride=256)
    tile_logits = []

    for x, y in coords:
        tx_end = min(w, x + tile_size)
        ty_end = min(h, y + tile_size)
        patch = np.zeros((tile_size, tile_size, 3), dtype=np.float32)
        sub = img_arr[y:ty_end, x:tx_end]
        patch[: sub.shape[0], : sub.shape[1]] = sub

        t_t = torch.from_numpy(patch).permute(2, 0, 1).unsqueeze(0).float().to(device)

        with torch.amp.autocast("cuda"):
            t_v2 = v2_model(t_t)["mask_logits"][0, 0].detach().cpu().numpy()
            if use_ensemble and v1_model is not None:
                t_v1 = v1_model(t_t)["mask_logits"][0, 0].detach().cpu().numpy()
                t_comb = 0.65 * t_v2 + 0.35 * t_v1
            else:
                t_comb = t_v2

            # Flip TTA
            if use_ensemble:
                t_flip = torch.flip(t_t, dims=[-1])
                t_v2_f = torch.flip(v2_model(t_flip)["mask_logits"][0, 0], dims=[-1]).detach().cpu().numpy()
                if v1_model is not None:
                    t_v1_f = torch.flip(v1_model(t_flip)["mask_logits"][0, 0], dims=[-1]).detach().cpu().numpy()
                    t_comb_f = 0.65 * t_v2_f + 0.35 * t_v1_f
                else:
                    t_comb_f = t_v2_f
                t_comb = 0.5 * (t_comb + t_comb_f)

        tile_logits.append(t_comb)

    blended_map = blend_tiles_probability(tile_logits, coords, w, h, tile_size=tile_size)

    # Mask pool score
    s_pool = float(np.mean(np.sort(blended_map.ravel())[-100:]))
    final_score = 0.15 * s_whole + 0.85 * s_pool

    # Postprocessing
    pred_bin = (blended_map > 0.35).astype(bool)
    pred_bin = binary_fill_holes(pred_bin)
    pred_bin = remove_small_components(pred_bin, min_size=64)

    return final_score, pred_bin


def run_benchmark_eval():
    seed_all(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running locked paper benchmark evaluation on {device}...")

    # Load models
    v2_model = SuperFRAMEV2Detector(pretrained=False).to(device)
    sd2 = torch.load(V2_CKPT, map_location=device)
    v2_model.load_state_dict(sd2.get("model_state_dict", sd2), strict=False)
    v2_model.eval()

    v1_model = SuperFRAMEDetector(pretrained=False).to(device)
    sd1 = torch.load(V1_CKPT, map_location=device)
    v1_model.load_state_dict(sd1.get("model_state_dict", sd1), strict=False)
    v1_model.eval()

    test_data = load_test_manifests()

    per_gen_results = {}
    all_ious = []
    tiny_ious = []
    med_ious = []
    large_ious = []
    pixel_tps = 0
    pixel_fps = 0
    pixel_fns = 0

    all_scores = []
    all_labels = []

    for gen, entries in test_data.items():
        print(f"Evaluating {gen.upper()} (20 images)...")
        gen_scores = []
        gen_labels = []
        gen_ious = []

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

            score, pred_bin = predict_single_image(
                v2_model, v1_model, arr, device, use_ensemble=True
            )
            gen_scores.append(score)
            all_scores.append(score)

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

        # Generator metrics at threshold 0.35
        preds = [1 if s >= 0.35 else 0 for s in gen_scores]
        acc = accuracy_score(gen_labels, preds)
        f1 = f1_score(gen_labels, preds, zero_division=0)
        try:
            auroc = roc_auc_score(gen_labels, gen_scores)
        except Exception:
            auroc = 0.5

        per_gen_results[gen] = {
            "accuracy": round(acc * 100, 2),
            "f1": round(f1 * 100, 2),
            "auroc": round(auroc * 100, 2),
            "mean_iou": round(float(np.mean(gen_ious)) * 100, 2) if gen_ious else 0.0,
        }

    # Global benchmark metrics
    preds_all = [1 if s >= 0.35 else 0 for s in all_scores]
    mean_acc = accuracy_score(all_labels, preds_all)
    mean_f1 = f1_score(all_labels, preds_all, zero_division=0)
    mean_auroc = roc_auc_score(all_labels, all_scores)
    mean_iou = float(np.mean(all_ious))

    precision = pixel_tps / (pixel_tps + pixel_fps) if (pixel_tps + pixel_fps) > 0 else 0.0
    recall = pixel_tps / (pixel_tps + pixel_fns) if (pixel_tps + pixel_fns) > 0 else 0.0
    pixel_f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S IST"),
        "total_test_images": len(all_labels),
        "protocol": "SuperFRAME Dual Ensemble (+TTA) on Official 100-Image Frozen Multi-Generator Benchmark",
        "global_detection_metrics": {
            "accuracy": round(mean_acc * 100, 2),
            "f1_score": round(mean_f1 * 100, 2),
            "auroc": round(mean_auroc * 100, 2),
        },
        "global_localization_metrics": {
            "mean_iou": round(mean_iou * 100, 2),
            "pixel_f1": round(pixel_f1 * 100, 2),
            "tiny_edit_iou": round(float(np.mean(tiny_ious)) * 100, 2) if tiny_ious else 0.0,
            "medium_edit_iou": round(float(np.mean(med_ious)) * 100, 2) if med_ious else 0.0,
            "large_edit_iou": round(float(np.mean(large_ious)) * 100, 2) if large_ious else 0.0,
        },
        "per_generator_metrics": per_gen_results,
        "baseline_comparison": {
            "SuperFRAME_Dual_Ensemble": {
                "mean_acc": round(mean_acc * 100, 2),
                "mean_iou": round(mean_iou * 100, 2),
                "pixel_f1": round(pixel_f1 * 100, 2),
                "flux_acc": per_gen_results["flux"]["accuracy"],
                "tiny_edit_iou": round(float(np.mean(tiny_ious)) * 100, 2),
            },
            "DeCLIP": {
                "mean_acc": 50.00,
                "mean_iou": 25.20,
                "pixel_f1": 30.30,
                "flux_acc": 50.00,
                "tiny_edit_iou": 5.10,
            },
            "VAAS": {
                "mean_acc": 50.00,
                "mean_iou": 16.33,
                "pixel_f1": 21.05,
                "flux_acc": 50.00,
                "tiny_edit_iou": 0.95,
            },
            "FUSED_Published": {
                "mean_acc": 89.50,
                "mean_iou": "N/A (Detection only)",
                "flux_acc": 77.20,
            },
        },
    }

    out_p = ROOT / "controlled_localization/reports/paper_locked_final_eval.json"
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n=======================================================")
    print("FINAL PAPER LOCKED BENCHMARK RESULTS:")
    print("=======================================================")
    print(f"Overall Detection Accuracy : {mean_acc*100:.2f}% | F1: {mean_f1*100:.2f}% | AUROC: {mean_auroc*100:.2f}%")
    print(f"Overall Localization IoU   : {mean_iou*100:.2f}% | Pixel F1: {pixel_f1*100:.2f}%")
    print(f"  - Tiny Edits (<2%)       : {report['global_localization_metrics']['tiny_edit_iou']:.2f}% (VAAS: 0.95%, DeCLIP: 5.10%)")
    print(f"  - Medium Edits (2-10%)   : {report['global_localization_metrics']['medium_edit_iou']:.2f}%")
    print(f"  - Large Edits (>10%)     : {report['global_localization_metrics']['large_edit_iou']:.2f}% (DeCLIP: 28.00%)")
    print("\nPer-Generator Metrics:")
    for g, r in per_gen_results.items():
        print(f"  {g.upper():<6}: Acc = {r['accuracy']:>6.2f}% | F1 = {r['f1']:>6.2f}% | AUROC = {r['auroc']:>6.2f}% | IoU = {r['mean_iou']:>6.2f}%")
    print(f"\nSaved final evaluation report to: {out_p}")
    return report


if __name__ == "__main__":
    run_benchmark_eval()
