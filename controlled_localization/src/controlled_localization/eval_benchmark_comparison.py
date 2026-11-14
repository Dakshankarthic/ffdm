"""Evaluate Controlled Localization on 100-sample OpenSDID Benchmark and compare with VAAS baseline."""

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
from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score, f1_score

from controlled_localization.models.detector import ControlledLocalizationDetector
from controlled_localization.models.frame_detector import FRAMEDetector
from controlled_localization.models.super_frame_detector import SuperFRAMEDetector
from controlled_localization.models.super_frame_v2_detector import SuperFRAMEV2Detector
from controlled_localization.policy import InferencePolicy
from controlled_localization.infer import run_single_image_inference
from controlled_localization.metrics import compute_binary_confusion, StreamingHistogramAP
from controlled_localization.utils import resolve_device


def bootstrap_ci(values: List[float], n_boot: int = 1000, ci: float = 0.95, seed: int = 42) -> List[float]:
    """Compute empirical bootstrap confidence interval."""
    if not values:
        return [0.0, 0.0]
    if len(values) == 1:
        return [float(values[0]), float(values[0])]
    rng = np.random.default_rng(seed)
    arr = np.array(values)
    boot_means = []
    for _ in range(n_boot):
        sample = rng.choice(arr, size=len(arr), replace=True)
        boot_means.append(np.mean(sample))
    alpha = (1.0 - ci) / 2.0
    low = float(np.percentile(boot_means, alpha * 100.0))
    high = float(np.percentile(boot_means, (1.0 - alpha) * 100.0))
    return [round(low, 4), round(high, 4)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate on OpenSDID benchmark test manifests")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint .pt")
    parser.add_argument("--manifest-dir", type=str, default="I:/icvgip workshop/baseline_work/data_manifests", help="Dir with test manifests")
    parser.add_argument("--workspace-root", type=str, default="I:/icvgip workshop", help="Root dir to resolve relative paths")
    parser.add_argument("--baseline-metrics", type=str, default="I:/icvgip workshop/baseline_work/reports/metrics.json", help="VAAS baseline metrics JSON")
    parser.add_argument("--out-dir", type=str, default="I:/icvgip workshop/controlled_localization/reports", help="Output directory")
    parser.add_argument("--t-mask", type=float, default=0.50, help="Localization threshold")
    parser.add_argument("--t-image", type=float, default=0.50, help="Image classification threshold")
    args = parser.parse_args()

    device = resolve_device()
    print(f"Loading checkpoint from: {args.checkpoint} on {device}...")

    ckpt = torch.load(args.checkpoint, map_location=device)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt

    is_super_v2 = any("cross_attn16" in k for k in state_dict.keys())
    is_super_frame = any("c_stage" in k for k in state_dict.keys())
    is_frame = any("semantic_encoder" in k for k in state_dict.keys())

    if is_super_v2:
        print("Detected SuperFRAME v2 architecture (ConvNeXt-Base + DINOv2-Base + Cross-Attention). Instantiating SuperFRAMEV2Detector...")
        detector = SuperFRAMEV2Detector(pretrained=False)
    elif is_super_frame:
        print("Detected SuperFRAME architecture (ConvNeXt-Small + DINOv2-Small). Instantiating SuperFRAMEDetector...")
        detector = SuperFRAMEDetector(pretrained=False)
    elif is_frame:
        print("Detected FRAME model architecture. Instantiating FRAMEDetector...")
        detector = FRAMEDetector(pretrained_semantic=False)
    else:
        print("Detected baseline architecture. Instantiating ControlledLocalizationDetector...")
        detector = ControlledLocalizationDetector(pretrained_rgb=False)

    detector.load_state_dict(state_dict)
    detector.to(device)
    detector.eval()

    policy = InferencePolicy(
        policy_name="benchmark_eval",
        blending_method="probability",
        pooling_method="top_k",
        pool_k=2048,
        t_mask=args.t_mask,
        t_image=args.t_image,
        score_direction="higher_is_fake",
    )

    manifest_files = [
        "test_sd15_manifest.json",
        "test_sd2_manifest.json",
        "test_sd3_manifest.json",
        "test_sdxl_manifest.json",
        "test_flux_manifest.json",
    ]

    workspace_root = Path(args.workspace_root)
    manifest_dir = Path(args.manifest_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stream_ap = StreamingHistogramAP(num_bins=4096)

    per_generator_results: Dict[str, Any] = {}
    all_per_image_rows: List[Dict[str, Any]] = []

    total_images_processed = 0
    start_all_time = time.perf_counter()

    for m_file in manifest_files:
        gen_name = m_file.replace("test_", "").replace("_manifest.json", "")
        manifest_p = manifest_dir / m_file
        if not manifest_p.exists():
            print(f"Warning: Manifest not found: {manifest_p}")
            continue

        with open(manifest_p, "r", encoding="utf-8") as f:
            data = json.load(f)

        entries = data.get("entries", [])
        print(f"\nProcessing generator '{gen_name}' ({len(entries)} samples)...")

        gen_ious: List[float] = []
        gen_f1s: List[float] = []
        gen_clean_fps: List[float] = []
        gen_y_true: List[int] = []
        gen_y_scores: List[float] = []

        gen_tier_ious = {"<2%": [], "2-10%": [], ">10%": []}
        gen_tier_f1s = {"<2%": [], "2-10%": [], ">10%": []}

        for entry in entries:
            sample_id = entry["sample_id"]
            img_rel = entry["image_path"]
            mask_rel = entry.get("mask_path")
            label = int(entry["label"])
            edit_pct = float(entry.get("edit_area_pct", 0.0))
            edit_tier = entry.get("edit_tier", "<2%" if edit_pct < 2.0 else ("2-10%" if edit_pct <= 10.0 else ">10%"))

            img_p = workspace_root / img_rel
            if not img_p.exists():
                print(f"Warning: image missing: {img_p}")
                continue

            # Run inference
            inf_res = run_single_image_inference(img_p, detector, policy, device)
            pred_prob = inf_res["merged_prob"]
            pred_bin = (pred_prob > policy.t_mask).astype(np.uint8)
            raw_score = inf_res["raw_score"]

            gen_y_true.append(label)
            gen_y_scores.append(raw_score)

            # Ground truth mask
            h, w = pred_prob.shape
            if label == 1 and mask_rel:
                mask_p = workspace_root / mask_rel
                with Image.open(mask_p) as m_img:
                    m_gray = m_img.convert("L").resize((w, h), resample=Image.NEAREST)
                    gt_mask = (np.array(m_gray, dtype=np.float32) > 127.0).astype(np.uint8)
            else:
                gt_mask = np.zeros((h, w), dtype=np.uint8)

            cm = compute_binary_confusion(pred_bin, gt_mask)
            stream_ap.update(pred_prob, gt_mask)

            if label == 1:
                gen_ious.append(cm.iou)
                gen_f1s.append(cm.f1)
                tier_k = edit_tier if edit_tier in gen_tier_ious else ">10%"
                gen_tier_ious[tier_k].append(cm.iou)
                gen_tier_f1s[tier_k].append(cm.f1)
                clean_fp = 0.0
            else:
                clean_fp = float(np.mean(pred_bin > 0)) * 100.0
                gen_clean_fps.append(clean_fp)

            row = {
                "sample_id": sample_id,
                "generator": gen_name,
                "label": label,
                "edit_pct": edit_pct,
                "edit_tier": edit_tier,
                "raw_score": raw_score,
                "is_fake_pred": int(raw_score > policy.t_image),
                "loc_iou": cm.iou,
                "loc_f1": cm.f1,
                "clean_fp_pct": clean_fp,
                "runtime_sec": inf_res["runtime_seconds"],
            }
            all_per_image_rows.append(row)
            total_images_processed += 1

        # Generator aggregate metrics
        y_true_np = np.array(gen_y_true)
        y_scores_np = np.array(gen_y_scores)
        preds_np = (y_scores_np > policy.t_image).astype(int)

        gen_auc = float(roc_auc_score(y_true_np, y_scores_np)) if len(np.unique(y_true_np)) > 1 else 0.5
        gen_acc = float(np.mean(preds_np == y_true_np))
        gen_bal_acc = float(balanced_accuracy_score(y_true_np, preds_np)) if len(np.unique(y_true_np)) > 1 else 0.5

        mean_iou = float(np.mean(gen_ious)) if gen_ious else 0.0
        mean_f1 = float(np.mean(gen_f1s)) if gen_f1s else 0.0
        mean_clean_fp = float(np.mean(gen_clean_fps)) if gen_clean_fps else 0.0

        per_generator_results[gen_name] = {
            "num_total": len(entries),
            "num_real": len(gen_clean_fps),
            "num_fake": len(gen_ious),
            "detection": {
                "roc_auc": round(gen_auc, 4),
                "accuracy": round(gen_acc, 4),
                "balanced_accuracy": round(gen_bal_acc, 4),
            },
            "localization": {
                "mean_iou": round(mean_iou, 4),
                "iou_95_ci": bootstrap_ci(gen_ious),
                "mean_f1": round(mean_f1, 4),
                "f1_95_ci": bootstrap_ci(gen_f1s),
                "clean_image_false_alarm_mean_pct": round(mean_clean_fp, 4),
                "tiers": {
                    k: {
                        "n_samples": len(gen_tier_ious[k]),
                        "mean_iou": round(float(np.mean(gen_tier_ious[k])), 4) if gen_tier_ious[k] else None,
                        "mean_f1": round(float(np.mean(gen_tier_f1s[k])), 4) if gen_tier_f1s[k] else None,
                    }
                    for k in ["<2%", "2-10%", ">10%"]
                },
            },
        }

    total_eval_time = time.perf_counter() - start_all_time

    # Macro summary across all generators
    all_fake_ious = [r["loc_iou"] for r in all_per_image_rows if r["label"] == 1]
    all_fake_f1s = [r["loc_f1"] for r in all_per_image_rows if r["label"] == 1]
    all_clean_fps = [r["clean_fp_pct"] for r in all_per_image_rows if r["label"] == 0]
    all_y_true = np.array([r["label"] for r in all_per_image_rows])
    all_y_scores = np.array([r["raw_score"] for r in all_per_image_rows])
    all_preds = (all_y_scores > policy.t_image).astype(int)

    macro_auc = float(roc_auc_score(all_y_true, all_y_scores))
    macro_acc = float(np.mean(all_preds == all_y_true))
    macro_bal_acc = float(balanced_accuracy_score(all_y_true, all_preds))
    macro_f1 = float(f1_score(all_y_true, all_preds))
    macro_iou = float(np.mean(all_fake_ious))
    macro_loc_f1 = float(np.mean(all_fake_f1s))
    macro_clean_fp = float(np.mean(all_clean_fps))
    corpus_pixel_ap = float(stream_ap.compute_ap())

    # Tier macro aggregation
    tier_agg: Dict[str, Any] = {}
    for t_name in ["<2%", "2-10%", ">10%"]:
        t_ious = [r["loc_iou"] for r in all_per_image_rows if r["label"] == 1 and r["edit_tier"] == t_name]
        t_f1s = [r["loc_f1"] for r in all_per_image_rows if r["label"] == 1 and r["edit_tier"] == t_name]
        tier_agg[t_name] = {
            "sample_count": len(t_ious),
            "mean_iou": round(float(np.mean(t_ious)), 4) if t_ious else None,
            "mean_f1": round(float(np.mean(t_f1s)), 4) if t_f1s else None,
        }

    # Load baseline metrics for direct comparison
    baseline_metrics = {}
    if Path(args.baseline_metrics).exists():
        with open(args.baseline_metrics, "r", encoding="utf-8") as f:
            baseline_metrics = json.load(f)

    comparison_results = {
        "model_name": "Controlled Localization v1.1 (T2O)",
        "checkpoint": args.checkpoint,
        "evaluation_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_test_images": total_images_processed,
        "total_runtime_seconds": round(total_eval_time, 2),
        "per_generator": per_generator_results,
        "macro_average": {
            "num_generators": len(per_generator_results),
            "total_test_images": total_images_processed,
            "total_real": len(all_clean_fps),
            "total_fake": len(all_fake_ious),
            "detection_accuracy": round(macro_acc, 4),
            "detection_balanced_accuracy": round(macro_bal_acc, 4),
            "detection_f1": round(macro_f1, 4),
            "detection_roc_auc": round(macro_auc, 4),
            "localization_mean_iou": round(macro_iou, 4),
            "localization_iou_95_ci": bootstrap_ci(all_fake_ious),
            "localization_mean_f1": round(macro_loc_f1, 4),
            "localization_f1_95_ci": bootstrap_ci(all_fake_f1s),
            "clean_image_false_alarm_pct": round(macro_clean_fp, 4),
            "corpus_pixel_ap": round(corpus_pixel_ap, 4),
            "aggregated_edit_tiers": tier_agg,
        },
        "baseline_comparison": {
            "baseline_model": "OBA-Research/VAAS (v2-base-df2023)",
            "vaas_macro_iou": baseline_metrics.get("macro_average", {}).get("localization_mean_iou"),
            "vaas_tiny_edit_iou": baseline_metrics.get("macro_average", {}).get("aggregated_edit_tiers", {}).get("<2%", {}).get("mean_iou"),
            "vaas_clean_fp_pct": baseline_metrics.get("macro_average", {}).get("clean_image_false_alarm_pct"),
            "vaas_roc_auc": baseline_metrics.get("macro_average", {}).get("detection_roc_auc"),
        },
    }

    # Save JSON metrics
    metrics_path = out_dir / "opensdid_benchmark_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(comparison_results, f, indent=2)

    # Save per-image CSV
    csv_path = out_dir / "opensdid_per_image.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        if all_per_image_rows:
            writer = csv.DictWriter(f, fieldnames=list(all_per_image_rows[0].keys()))
            writer.writeheader()
            writer.writerows(all_per_image_rows)

    # Save Markdown comparison table
    md_path = out_dir / "opensdid_benchmark_comparison.md"
    vaas_macro = baseline_metrics.get("macro_average", {})
    t2o_macro = comparison_results["macro_average"]

    md_content = f"""# OpenSDID Benchmark Evaluation: Controlled Localization v1.1 vs. VAAS Baseline

Comprehensive evaluation on the frozen 100-sample test benchmark across 5 diffusion architectures (SD 1.5, SD 2.1, SD 3, SDXL, and FLUX.1-schnell).

## Benchmark Comparison Milestone Table

| Metric | VAAS Baseline (v2-base-df2023) | Controlled Localization v1.1 (T2O) | Delta / Improvement |
| :--- | :---: | :---: | :---: |
| **Detection ROC-AUC** | {vaas_macro.get('detection_roc_auc', 0.442):.4f} | **{t2o_macro['detection_roc_auc']:.4f}** | **{t2o_macro['detection_roc_auc'] - vaas_macro.get('detection_roc_auc', 0.442):+.4f}** |
| **Detection Balanced Acc** | {vaas_macro.get('detection_balanced_accuracy', 0.460):.4f} | **{t2o_macro['detection_balanced_accuracy']:.4f}** | **{t2o_macro['detection_balanced_accuracy'] - vaas_macro.get('detection_balanced_accuracy', 0.460):+.4f}** |
| **Localization Mean IoU** | {vaas_macro.get('localization_mean_iou', 0.1633):.4f} ({vaas_macro.get('localization_mean_iou', 0.1633)*100:.2f}%) | **{t2o_macro['localization_mean_iou']:.4f} ({t2o_macro['localization_mean_iou']*100:.2f}%)** | **{(t2o_macro['localization_mean_iou'] - vaas_macro.get('localization_mean_iou', 0.1633))*100:+.2f}%** |
| **Localization Mean F1** | {vaas_macro.get('localization_mean_f1', 0.2313):.4f} | **{t2o_macro['localization_mean_f1']:.4f}** | **{t2o_macro['localization_mean_f1'] - vaas_macro.get('localization_mean_f1', 0.2313):+.4f}** |
| **Tiny Edit (<2%) IoU** | {vaas_macro.get('aggregated_edit_tiers', {}).get('<2%', {}).get('mean_iou', 0.0095):.4f} ({vaas_macro.get('aggregated_edit_tiers', {}).get('<2%', {}).get('mean_iou', 0.0095)*100:.2f}%) | **{tier_agg['<2%']['mean_iou']:.4f} ({tier_agg['<2%']['mean_iou']*100:.2f}%)** | **{(tier_agg['<2%']['mean_iou'] - vaas_macro.get('aggregated_edit_tiers', {}).get('<2%', {}).get('mean_iou', 0.0095))*100:+.2f}%** |
| **Medium Edit (2-10%) IoU** | {vaas_macro.get('aggregated_edit_tiers', {}).get('2-10%', {}).get('mean_iou', 0.1897):.4f} | **{tier_agg['2-10%']['mean_iou']:.4f}** | **{tier_agg['2-10%']['mean_iou'] - vaas_macro.get('aggregated_edit_tiers', {}).get('2-10%', {}).get('mean_iou', 0.1897):+.4f}** |
| **Large Edit (>10%) IoU** | {vaas_macro.get('aggregated_edit_tiers', {}).get('>10%', {}).get('mean_iou', 0.1722):.4f} | **{tier_agg['>10%']['mean_iou']:.4f}** | **{tier_agg['>10%']['mean_iou'] - vaas_macro.get('aggregated_edit_tiers', {}).get('>10%', {}).get('mean_iou', 0.1722):+.4f}** |
| **Clean False Alarm %** | {vaas_macro.get('clean_image_false_alarm_pct', 16.5139):.2f}% | **{t2o_macro['clean_image_false_alarm_pct']:.2f}%** | **{t2o_macro['clean_image_false_alarm_pct'] - vaas_macro.get('clean_image_false_alarm_pct', 16.5139):+.2f}%** |

---

## Per-Generator Breakdown

| Generator | Fake/Real | T2O Det AUC | T2O Mean IoU [95% CI] | VAAS Mean IoU |
| :--- | :---: | :---: | :---: | :---: |
"""
    for gen, g_res in per_generator_results.items():
        v_gen = baseline_metrics.get("per_generator", {}).get(gen, {}).get("localization", {})
        v_iou = v_gen.get("mean_iou", "N/A")
        ci_str = f"[{g_res['localization']['iou_95_ci'][0]:.4f}, {g_res['localization']['iou_95_ci'][1]:.4f}]"
        md_content += f"| **{gen.upper()}** | {g_res['num_fake']}/{g_res['num_real']} | {g_res['detection']['roc_auc']:.4f} | **{g_res['localization']['mean_iou']:.4f}** {ci_str} | {v_iou} |\n"

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    print(f"\nBenchmark evaluation finished successfully in {total_eval_time:.1f}s!")
    print(f"Results written to: {metrics_path} and {md_path}")


if __name__ == "__main__":
    main()

