"""Phase 3: Error Diagnosis and Hard-Example Mining.
Audits validation set predictions (TP, TN, FP, FN) by generator, edit size, and resolution.
Identifies hard negative textures and tiny-edit misses for targeted training and RL formulation.
"""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image
import torch
import torch.nn as nn

from controlled_localization.models.super_frame_v2_detector import SuperFRAMEV2Detector
from controlled_localization.data.tiles import get_tile_grid_coords, blend_tiles_probability


ROOT = Path("I:/icvgip workshop")
RUNS_DIR = ROOT / "controlled_localization/runs"
SCALED_CKPT = RUNS_DIR / "scaled_pilot/best_superframe_v2_scaled.pt"
VAL_MANIFEST = RUNS_DIR / "scaled_pilot/manifest_scaled.json"


def run_error_audit(max_samples: int = 200) -> Dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading checkpoint {SCALED_CKPT} on {device}...")

    model = SuperFRAMEV2Detector(pretrained=False).to(device)
    ckpt = torch.load(SCALED_CKPT, map_location=device)
    sd = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(sd, strict=False)
    model.eval()

    with open(VAL_MANIFEST, "r", encoding="utf-8") as f:
        all_recs = json.load(f)

    val_recs = [r for r in all_recs if r["split"] == "val"][:max_samples]
    print(f"Auditing {len(val_recs)} validation samples...")

    fp_list = []
    fn_list = []
    tp_list = []
    tn_list = []

    per_generator = {}
    edit_size_breakdown = {
        "tiny (<2%)": {"total": 0, "detected": 0, "missed": 0, "ious": []},
        "medium (2-10%)": {"total": 0, "detected": 0, "missed": 0, "ious": []},
        "large (>10%)": {"total": 0, "detected": 0, "missed": 0, "ious": []},
    }

    tile_size = 384
    thresh = 0.35  # standard calibrated threshold

    with torch.no_grad():
        for rec in val_recs:
            img_p = Path(rec["image"])
            mask_p = rec.get("mask")
            lbl = int(rec["label"])
            gen = rec.get("generator", rec.get("scope", "unknown")).replace("multi_", "")

            if gen not in per_generator:
                per_generator[gen] = {"tp": 0, "tn": 0, "fp": 0, "fn": 0, "ious": []}

            with Image.open(img_p) as pil_img:
                rgb_full = pil_img.convert("RGB")
                w, h = rgb_full.size
                arr_full = np.array(rgb_full, dtype=np.float32) / 255.0

            if lbl == 1 and mask_p is not None and Path(mask_p).exists():
                with Image.open(mask_p) as m_img:
                    gt_mask = (np.array(m_img.convert("L"), dtype=np.float32) > 127.5).astype(np.float32)
                edit_frac = float(np.mean(gt_mask))
            else:
                gt_mask = np.zeros((h, w), dtype=np.float32)
                edit_frac = 0.0

            # Whole image view
            whole_resized = rgb_full.resize((tile_size, tile_size), Image.BILINEAR)
            whole_arr = np.array(whole_resized, dtype=np.float32) / 255.0
            whole_t = torch.from_numpy(whole_arr).permute(2, 0, 1).unsqueeze(0).float().to(device)

            with torch.amp.autocast("cuda"):
                whole_out = model(whole_t)
            s_whole = float(torch.sigmoid(whole_out["det_logits"][0, 0]).item())

            # 4 native tiles
            coords = get_tile_grid_coords(w, h, tile_size=tile_size, stride=256)
            if len(coords) > 4:
                coords = [coords[0], coords[len(coords) // 3], coords[2 * len(coords) // 3], coords[-1]]

            tile_logits = []
            for x, y in coords:
                tx_end = min(w, x + tile_size)
                ty_end = min(h, y + tile_size)
                crop_patch = np.zeros((tile_size, tile_size, 3), dtype=np.float32)
                sub_arr = arr_full[y:ty_end, x:tx_end]
                crop_patch[: sub_arr.shape[0], : sub_arr.shape[1]] = sub_arr

                tile_t = torch.from_numpy(crop_patch).permute(2, 0, 1).unsqueeze(0).float().to(device)
                with torch.amp.autocast("cuda"):
                    t_out = model(tile_t)
                tile_logits.append(t_out["mask_logits"][0, 0].cpu().numpy())

            blended_map = blend_tiles_probability(tile_logits, coords, w, h, tile_size=tile_size)
            s_pool = float(np.mean(np.sort(blended_map.ravel())[-100:]))
            s_hybrid = 0.15 * s_whole + 0.85 * s_pool

            pred_bin = (blended_map > thresh).astype(np.float32)
            is_pred_edited = s_hybrid >= 0.35

            if lbl == 1:
                inter = np.sum(pred_bin * gt_mask)
                union = np.sum((pred_bin + gt_mask) > 0)
                iou = float(inter / union) if union > 0 else 0.0
                per_generator[gen]["ious"].append(iou)

                # Edit size bin
                if edit_frac < 0.02:
                    tier = "tiny (<2%)"
                elif edit_frac <= 0.10:
                    tier = "medium (2-10%)"
                else:
                    tier = "large (>10%)"

                edit_size_breakdown[tier]["total"] += 1
                edit_size_breakdown[tier]["ious"].append(iou)

                if is_pred_edited:
                    tp_list.append({"id": rec["id"], "score": s_hybrid, "iou": iou, "gen": gen, "frac": edit_frac})
                    per_generator[gen]["tp"] += 1
                    edit_size_breakdown[tier]["detected"] += 1
                else:
                    fn_list.append({"id": rec["id"], "score": s_hybrid, "iou": iou, "gen": gen, "frac": edit_frac, "img": str(img_p)})
                    per_generator[gen]["fn"] += 1
                    edit_size_breakdown[tier]["missed"] += 1
            else:
                fp_rate = float(np.mean(pred_bin > 0))
                if is_pred_edited or fp_rate > 0.05:
                    fp_list.append({"id": rec["id"], "score": s_hybrid, "fp_rate": fp_rate, "gen": gen, "img": str(img_p)})
                    per_generator[gen]["fp"] += 1
                else:
                    tn_list.append({"id": rec["id"], "score": s_hybrid, "fp_rate": fp_rate, "gen": gen})
                    per_generator[gen]["tn"] += 1

    # Summarize metrics
    total_samples = len(val_recs)
    acc = (len(tp_list) + len(tn_list)) / total_samples if total_samples > 0 else 0.0

    diagnosis = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S IST"),
        "total_audited": total_samples,
        "confusion": {
            "true_positives": len(tp_list),
            "true_negatives": len(tn_list),
            "false_positives": len(fp_list),
            "false_negatives": len(fn_list),
            "accuracy": round(acc * 100, 2),
        },
        "per_generator": {
            g: {
                "tp": v["tp"], "tn": v["tn"], "fp": v["fp"], "fn": v["fn"],
                "acc": round((v["tp"] + v["tn"]) / (v["tp"] + v["tn"] + v["fp"] + v["fn"]) * 100, 2) if (v["tp"] + v["tn"] + v["fp"] + v["fn"]) > 0 else 0,
                "mean_iou": round(float(np.mean(v["ious"])) * 100, 2) if v["ious"] else 0.0,
            }
            for g, v in per_generator.items()
        },
        "edit_size_breakdown": {
            tier: {
                "total": data["total"],
                "detected": data["detected"],
                "missed": data["missed"],
                "detection_rate": round(data["detected"] / data["total"] * 100, 2) if data["total"] > 0 else 0.0,
                "mean_iou": round(float(np.mean(data["ious"])) * 100, 2) if data["ious"] else 0.0,
            }
            for tier, data in edit_size_breakdown.items()
        },
        "hard_examples_mined": {
            "num_hard_fps": len(fp_list),
            "sample_hard_fps": fp_list[:5],
            "num_hard_fns": len(fn_list),
            "sample_hard_fns": fn_list[:5],
        },
    }

    out_p = ROOT / "controlled_localization/reports/phase3_error_diagnosis.json"
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(diagnosis, f, indent=2)

    print(f"\n=======================================================")
    print(f"PHASE 3 ERROR AUDIT SUMMARY (Total {total_samples} samples):")
    print(f"=======================================================")
    print(f"TP: {len(tp_list)} | TN: {len(tn_list)} | FP: {len(fp_list)} | FN: {len(fn_list)} | Acc: {acc*100:.2f}%")
    print(f"\nEdit Size Breakdown:")
    for tier, stats in diagnosis["edit_size_breakdown"].items():
        print(f"  {tier:<16}: Det Rate = {stats['detection_rate']:>6.2f}% | Mean IoU = {stats['mean_iou']:>6.2f}% (Total: {stats['total']})")
    print(f"\nPer-Generator Accuracy & IoU:")
    for g, stats in diagnosis["per_generator"].items():
        print(f"  {g:<12}: Acc = {stats['acc']:>6.2f}% | Mean IoU = {stats['mean_iou']:>6.2f}%")
    print(f"\nSaved diagnosis to: {out_p}")
    return diagnosis


if __name__ == "__main__":
    run_error_audit(max_samples=200)
