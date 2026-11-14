"""Dataset evaluation CLI and metrics aggregation for Controlled Localization v1.1."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import torch
from PIL import Image

from .data.schema import load_manifest, ManifestRecord
from .infer import run_single_image_inference
from .models.detector import ControlledLocalizationDetector
from .policy import InferencePolicy
from .metrics import (
    compute_binary_confusion,
    assign_edit_tier,
    StreamingHistogramAP,
    ConfusionMatrix,
)
from .utils import resolve_device, create_heatmap_overlay


def evaluate_dataset(
    manifest_records: List[ManifestRecord],
    detector: ControlledLocalizationDetector,
    policy: InferencePolicy,
    device: torch.device,
    out_dir: Path,
    data_root: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Run evaluation across an entire manifest of images.
    Collects per-image predictions and macro-averaged metrics.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir = out_dir / "overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)

    stream_ap = StreamingHistogramAP(num_bins=4096)

    per_image_rows = []
    edit_tier_ious: Dict[str, List[float]] = {
        "<0.5%": [],
        "0.5%-2%": [],
        "2%-10%": [],
        ">10%": [],
        "tiny_(<2%)": [],
    }
    edit_tier_f1s: Dict[str, List[float]] = {
        "<0.5%": [],
        "0.5%-2%": [],
        "2%-10%": [],
        ">10%": [],
        "tiny_(<2%)": [],
    }
    clean_fp_pcts: List[float] = []

    y_true_img: List[int] = []
    y_scores_img: List[float] = []

    start_eval_time = time.perf_counter()

    for idx, rec in enumerate(manifest_records, 1):
        # Resolve image path
        img_p = Path(rec.image_path)
        if not img_p.is_absolute() and data_root is not None:
            img_p = data_root / img_p

        # Run inference (strictly without ground truth inputs!)
        inf_res = run_single_image_inference(img_p, detector, policy, device)

        pred_prob = inf_res["merged_prob"]
        pred_bin = (pred_prob > policy.t_mask).astype(np.uint8)

        # Ground truth
        label = rec.label
        y_true_img.append(label)
        y_scores_img.append(inf_res["raw_score"])

        # Load ground truth mask if available
        if rec.mask_path is not None:
            mask_p = Path(rec.mask_path)
            if not mask_p.is_absolute() and data_root is not None:
                mask_p = data_root / mask_p

            with Image.open(mask_p) as m_img:
                m_gray = m_img.convert("L").resize((rec.width, rec.height), resample=Image.NEAREST)
                gt_mask = (np.array(m_gray, dtype=np.float32) > 127.0).astype(np.uint8)
        else:
            gt_mask = np.zeros((rec.height, rec.width), dtype=np.uint8)

        # Compute confusion
        cm = compute_binary_confusion(pred_bin, gt_mask)

        # Update streaming pixel AP
        stream_ap.update(pred_prob, gt_mask)

        # Edit fraction and tier
        edit_frac = rec.mask_area_fraction
        tier = assign_edit_tier(edit_frac)

        if label == 1:
            tier_key = tier if tier in edit_tier_ious else ">10%"
            edit_tier_ious[tier_key].append(cm.iou)
            edit_tier_f1s[tier_key].append(cm.f1)
            if edit_frac < 0.02:
                edit_tier_ious["tiny_(<2%)"].append(cm.iou)
                edit_tier_f1s["tiny_(<2%)"].append(cm.f1)
        else:
            clean_fp = float(np.mean(pred_bin > 0)) * 100.0
            clean_fp_pcts.append(clean_fp)

        # Save visual overlay
        overlay = create_heatmap_overlay(inf_res["canonical_image"], pred_prob, threshold=policy.t_mask)
        overlay.save(overlay_dir / f"{rec.sample_id}_overlay.png")

        row = {
            "sample_id": rec.sample_id,
            "split": rec.split,
            "label": label,
            "edit_fraction": edit_frac,
            "edit_tier": tier,
            "raw_score": inf_res["raw_score"],
            "is_fake_pred": int(inf_res["is_fake"]),
            "loc_iou": cm.iou,
            "loc_f1": cm.f1,
            "loc_precision": cm.precision,
            "loc_recall": cm.recall,
            "tp_pixels": cm.tp,
            "fp_pixels": cm.fp,
            "fn_pixels": cm.fn,
            "tn_pixels": cm.tn,
            "runtime_sec": inf_res["runtime_seconds"],
        }
        per_image_rows.append(row)

    total_eval_time = time.perf_counter() - start_eval_time

    # Compute image-level detection metrics
    y_true_arr = np.array(y_true_img)
    y_scores_arr = np.array(y_scores_img)

    has_both_classes = len(np.unique(y_true_arr)) > 1
    if has_both_classes:
        from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score, f1_score
        det_auc = float(roc_auc_score(y_true_arr, y_scores_arr))
        det_ap = float(average_precision_score(y_true_arr, y_scores_arr))
        preds_arr = (y_scores_arr > policy.t_image).astype(int) if policy.score_direction == "higher_is_fake" else (y_scores_arr < policy.t_image).astype(int)
        det_acc = float(np.mean(preds_arr == y_true_arr))
        det_bal_acc = float(balanced_accuracy_score(y_true_arr, preds_arr))
        det_f1 = float(f1_score(y_true_arr, preds_arr, zero_division=0))
    else:
        det_auc = 0.0
        det_ap = 0.0
        det_acc = 0.0
        det_bal_acc = 0.0
        det_f1 = 0.0

    # Macro localization IoU across all edited images
    all_edit_ious = [r["loc_iou"] for r in per_image_rows if r["label"] == 1]
    all_edit_f1s = [r["loc_f1"] for r in per_image_rows if r["label"] == 1]

    macro_iou = float(np.mean(all_edit_ious)) if all_edit_ious else 0.0
    macro_f1 = float(np.mean(all_edit_f1s)) if all_edit_f1s else 0.0
    mean_clean_fp = float(np.mean(clean_fp_pcts)) if clean_fp_pcts else 0.0
    pixel_corpus_ap = float(stream_ap.compute_ap())

    summary_metrics = {
        "evaluation_count": len(manifest_records),
        "edited_count": len(all_edit_ious),
        "genuine_count": len(clean_fp_pcts),
        "total_runtime_seconds": float(total_eval_time),
        "avg_seconds_per_image": float(total_eval_time / max(1, len(manifest_records))),
        "detection": {
            "auroc": det_auc,
            "average_precision": det_ap,
            "accuracy": det_acc,
            "balanced_accuracy": det_bal_acc,
            "f1": det_f1,
            "t_image": policy.t_image,
        },
        "localization": {
            "macro_iou": macro_iou,
            "macro_f1": macro_f1,
            "corpus_pixel_ap": pixel_corpus_ap,
            "t_mask": policy.t_mask,
            "clean_false_positive_pixel_pct": mean_clean_fp,
            "by_edit_tier_iou": {k: float(np.mean(v)) if v else None for k, v in edit_tier_ious.items()},
            "by_edit_tier_f1": {k: float(np.mean(v)) if v else None for k, v in edit_tier_f1s.items()},
        },
    }

    # Save metrics.json
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary_metrics, f, indent=2)

    # Save per_image_predictions.csv
    with open(out_dir / "per_image_predictions.csv", "w", newline="", encoding="utf-8") as f:
        if per_image_rows:
            writer = csv.DictWriter(f, fieldnames=list(per_image_rows[0].keys()))
            writer.writeheader()
            writer.writerows(per_image_rows)

    return summary_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Controlled Localization v1.1 Evaluation CLI")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained checkpoint (.pt)")
    parser.add_argument("--manifest", type=str, required=True, help="Path to JSONL manifest")
    parser.add_argument("--policy", type=str, required=True, help="Path to frozen inference_policy.json")
    parser.add_argument("--out", type=str, required=True, help="Output directory")
    parser.add_argument("--data-root", type=str, default=None, help="Root directory for relative manifest paths")
    args = parser.parse_args()

    out_dir = Path(args.out)
    records = load_manifest(args.manifest, data_root=args.data_root)
    policy = InferencePolicy.load(args.policy)
    device = resolve_device()

    detector = ControlledLocalizationDetector(pretrained_rgb=False)
    ckpt = torch.load(args.checkpoint, map_location=device)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    detector.load_state_dict(state_dict)
    detector.to(device)
    detector.eval()

    root_p = Path(args.data_root) if args.data_root else None
    metrics = evaluate_dataset(records, detector, policy, device, out_dir, data_root=root_p)
    print(f"Evaluation completed. Macro IoU: {metrics['localization']['macro_iou']:.4f}, Det AUC: {metrics['detection']['auroc']:.4f}")


if __name__ == "__main__":
    main()

