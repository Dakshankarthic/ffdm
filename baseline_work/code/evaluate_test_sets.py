import os
import sys
import hashlib
import json
import csv
import time
from pathlib import Path

# Ensure code dir is in sys.path
CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score, accuracy_score, balanced_accuracy_score, precision_recall_fscore_support

from baseline_adapter import VAASAdapter
from test_metrics_and_checks import compute_per_image_iou_f1, classify_edit_tier, compute_classification_metrics

ROOT_DIR = Path("I:/icvgip workshop")
PRED_BASE = ROOT_DIR / "baseline_work" / "predictions"
PRED_BASE.mkdir(parents=True, exist_ok=True)
REPORTS_DIR = ROOT_DIR / "baseline_work" / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
FAILURE_DIR = REPORTS_DIR / "failure_cases"
FAILURE_DIR.mkdir(parents=True, exist_ok=True)
CONFIG_DIR = ROOT_DIR / "baseline_work" / "configs"

def bootstrap_ci(values, n_boot=1000, seed=42, ci=95):
    if len(values) == 0:
        return [float("nan"), float("nan")]
    rng = np.random.RandomState(seed)
    boot_means = []
    vals = np.array(values, dtype=float)
    for _ in range(n_boot):
        sample = rng.choice(vals, size=len(vals), replace=True)
        boot_means.append(float(np.mean(sample)))
    alpha = (100 - ci) / 2.0
    low = float(np.percentile(boot_means, alpha))
    high = float(np.percentile(boot_means, 100 - alpha))
    return [round(low, 4), round(high, 4)]

def run_evaluation():
    print("======================================================================", flush=True)
    print("STEP 5: RUN HELD-OUT INDEPENDENT TEST BENCHMARK ON 5 GENERATORS", flush=True)
    print("======================================================================", flush=True)
    
    # Load locked config
    locked_cfg_path = CONFIG_DIR / "locked_evaluation_config.json"
    assert locked_cfg_path.exists(), "Locked evaluation config not found!"
    with open(locked_cfg_path, "r", encoding="utf-8") as f:
        locked_cfg = json.load(f)
        
    img_thresh = locked_cfg["image_score_threshold"]
    mask_thresh = locked_cfg["mask_threshold"]
    score_dir = locked_cfg["score_direction"]
    score_metric = locked_cfg["image_score_metric"]
    
    print(f"Using locked configuration: {locked_cfg_path}", flush=True)
    print(f"  Score Metric: {score_metric} | Image Threshold: {img_thresh} | Direction: {score_dir}", flush=True)
    print(f"  Mask Threshold: {mask_thresh} | Interpolation: {locked_cfg['mask_resize_interpolation']}", flush=True)
    
    adapter_cfg = {
        "model_repo": "OBA-Research/vaas",
        "model_variant": "v2-base-df2023",
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }
    adapter = VAASAdapter(adapter_cfg)
    adapter.load_model()
    print("Baseline model loaded onto device:", adapter.device, flush=True)
    
    generators = ["sd15", "sd2", "sdxl", "sd3", "flux"]
    all_per_image_records = []
    metrics_summary = {
        "model": "OBA-Research/vaas (v2-base-df2023)",
        "locked_config": locked_cfg,
        "evaluation_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "per_generator": {},
        "macro_average": {}
    }
    
    for gen in generators:
        split_name = f"test_{gen}"
        split_pred_dir = PRED_BASE / split_name
        split_pred_dir.mkdir(parents=True, exist_ok=True)
        
        manifest_file = ROOT_DIR / "baseline_work" / "data_manifests" / f"{split_name}_manifest.json"
        with open(manifest_file, "r", encoding="utf-8") as f:
            manifest_data = json.load(f)
            
        entries = manifest_data["entries"]
        print(f"\n--- Evaluating Split: {split_name} ({len(entries)} images) ---", flush=True)
        
        y_true = []
        y_scores = []
        loc_ious = []
        loc_f1s = []
        
        # Binned localization
        loc_bins = {"<2%": {"ious": [], "f1s": []}, "2-10%": {"ious": [], "f1s": []}, ">10%": {"ious": [], "f1s": []}}
        clean_false_positive_areas = []
        
        for idx, e in enumerate(entries):
            img_path = ROOT_DIR / e["image_path"]
            label = e["label"]
            y_true.append(label)
            
            # PURE BLIND PREDICT
            pred = adapter.predict(str(img_path))
            raw_map = pred["raw_map"]
            
            # Save raw map
            map_name = f"{e['sample_id']}_map.npy"
            map_save_path = split_pred_dir / map_name
            np.save(map_save_path, raw_map)
            
            s_val = pred["raw_scores"][score_metric]
            y_scores.append(s_val)
            # Binary classification decision using locked threshold and direction
            # For lower_is_fake: pred = 1 if score <= threshold
            det_pred = 1 if s_val <= img_thresh else 0
            
            loc_iou = None
            loc_f1 = None
            clean_fp_pct = None
            
            # Mask resize to native image/mask size using nearest neighbor
            pred_pil = Image.fromarray(raw_map)
            pred_resized = np.array(pred_pil.resize((e["width"], e["height"]), resample=Image.NEAREST))
            bin_pred_mask = (pred_resized >= mask_thresh).astype(np.uint8)
            
            if label == 1:
                # Fake image: evaluate localization against GT mask
                gt_mask_path = ROOT_DIR / e["mask_path"]
                gt_mask = np.array(Image.open(gt_mask_path).convert("L"))
                gt_bin = (gt_mask > 127).astype(np.uint8)
                
                iou, f1 = compute_per_image_iou_f1(bin_pred_mask, gt_bin)
                loc_iou = round(iou, 4)
                loc_f1 = round(f1, 4)
                loc_ious.append(iou)
                loc_f1s.append(f1)
                
                tier = e["edit_tier"]
                if tier in loc_bins:
                    loc_bins[tier]["ious"].append(iou)
                    loc_bins[tier]["f1s"].append(f1)
            else:
                # Clean genuine image: measure false alarm area
                clean_fp_pixels = int(bin_pred_mask.sum())
                clean_fp_pct = round(clean_fp_pixels / bin_pred_mask.size * 100.0, 4)
                clean_false_positive_areas.append(clean_fp_pct)
                
            rec = {
                "sample_id": e["sample_id"],
                "generator": gen,
                "split": split_name,
                "image_path": e["image_path"],
                "mask_path": e["mask_path"],
                "label": label,
                "edit_tier": e["edit_tier"],
                "edit_area_pct": e["edit_area_pct"],
                "S_F": pred["raw_scores"]["S_F"],
                "S_P": pred["raw_scores"]["S_P"],
                "S_H": pred["raw_scores"]["S_H"],
                "det_score_used": s_val,
                "det_pred": det_pred,
                "det_correct": int(det_pred == label),
                "loc_iou": loc_iou,
                "loc_f1": loc_f1,
                "clean_fp_pct": clean_fp_pct,
                "runtime_sec": pred["runtime_sec"],
                "map_path": str(map_save_path.relative_to(ROOT_DIR)),
                "image_sha256": e["image_sha256"]
            }
            all_per_image_records.append(rec)
            print(f"  [{idx+1}/{len(entries)}] {e['sample_id']} | Label: {label} | Score: {s_val:.4f} | Det: {det_pred} | IoU: {loc_iou} | FP%: {clean_fp_pct}", flush=True)

        # Compute split metrics
        det_m = compute_classification_metrics(y_true, y_scores, threshold=img_thresh, score_direction=score_dir)
        mean_iou = float(np.mean(loc_ious)) if loc_ious else float("nan")
        mean_f1 = float(np.mean(loc_f1s)) if loc_f1s else float("nan")
        mean_clean_fp = float(np.mean(clean_false_positive_areas)) if clean_false_positive_areas else 0.0
        
        # Binned tiers breakdown
        tier_results = {}
        for tier_name, t_data in loc_bins.items():
            n_samples = len(t_data["ious"])
            tier_results[tier_name] = {
                "n_samples": n_samples,
                "mean_iou": round(float(np.mean(t_data["ious"])), 4) if n_samples > 0 else "N/A",
                "mean_f1": round(float(np.mean(t_data["f1s"])), 4) if n_samples > 0 else "N/A",
            }
            
        # Bootstrap CIs for main metrics
        iou_ci = bootstrap_ci(loc_ious)
        f1_ci = bootstrap_ci(loc_f1s)
        
        split_summary = {
            "generator": gen,
            "split": split_name,
            "num_total": len(entries),
            "num_real": sum(1 for y in y_true if y == 0),
            "num_fake": sum(1 for y in y_true if y == 1),
            "detection": {
                "TP": det_m["TP"], "FP": det_m["FP"], "TN": det_m["TN"], "FN": det_m["FN"],
                "accuracy": round(det_m["Accuracy"], 4),
                "balanced_accuracy": round(det_m["Balanced_Accuracy"], 4),
                "precision": round(det_m["Precision"], 4),
                "recall": round(det_m["Recall"], 4),
                "f1": round(det_m["F1"], 4),
                "fpr": round(det_m["FPR"], 4),
                "roc_auc": round(det_m["ROC_AUC"], 4) if not np.isnan(det_m["ROC_AUC"]) else None,
                "score_direction": score_dir,
                "threshold": img_thresh
            },
            "localization": {
                "evaluated_n_fake": len(loc_ious),
                "mean_iou": round(mean_iou, 4),
                "iou_95_ci": iou_ci,
                "mean_f1": round(mean_f1, 4),
                "f1_95_ci": f1_ci,
                "mask_threshold": mask_thresh,
                "tiers": tier_results,
                "clean_image_false_alarm_mean_pct": round(mean_clean_fp, 4)
            }
        }
        metrics_summary["per_generator"][gen] = split_summary
        print(f"Split {gen} Summary: Det_Acc={det_m['Accuracy']:.3f}, ROC_AUC={det_m['ROC_AUC']:.3f}, Loc_IoU={mean_iou:.4f}, Loc_F1={mean_f1:.4f}", flush=True)

    # Save per_image.csv
    per_image_csv = REPORTS_DIR / "per_image.csv"
    with open(per_image_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_per_image_records[0].keys()))
        writer.writeheader()
        writer.writerows(all_per_image_records)
    print(f"\nSaved all 100 test per-image records to {per_image_csv}", flush=True)
    
    # Compute Macro-Average across generators
    macro_acc = float(np.mean([metrics_summary["per_generator"][g]["detection"]["accuracy"] for g in generators]))
    macro_bal_acc = float(np.mean([metrics_summary["per_generator"][g]["detection"]["balanced_accuracy"] for g in generators]))
    macro_det_f1 = float(np.mean([metrics_summary["per_generator"][g]["detection"]["f1"] for g in generators]))
    macro_auc = float(np.mean([metrics_summary["per_generator"][g]["detection"]["roc_auc"] for g in generators if metrics_summary["per_generator"][g]["detection"]["roc_auc"] is not None]))
    macro_iou = float(np.mean([metrics_summary["per_generator"][g]["localization"]["mean_iou"] for g in generators]))
    macro_loc_f1 = float(np.mean([metrics_summary["per_generator"][g]["localization"]["mean_f1"] for g in generators]))
    macro_clean_fp = float(np.mean([metrics_summary["per_generator"][g]["localization"]["clean_image_false_alarm_mean_pct"] for g in generators]))
    
    # Aggregated tier localization across all test fake samples (50 fake samples total)
    all_tier_ious = {"<2%": [], "2-10%": [], ">10%": []}
    all_tier_f1s = {"<2%": [], "2-10%": [], ">10%": []}
    for r in all_per_image_records:
        if r["label"] == 1 and r["edit_tier"] in all_tier_ious:
            all_tier_ious[r["edit_tier"]].append(r["loc_iou"])
            all_tier_f1s[r["edit_tier"]].append(r["loc_f1"])
            
    aggregated_tiers = {}
    for t_name in ["<2%", "2-10%", ">10%"]:
        n_t = len(all_tier_ious[t_name])
        aggregated_tiers[t_name] = {
            "sample_count": n_t,
            "mean_iou": round(float(np.mean(all_tier_ious[t_name])), 4) if n_t > 0 else "N/A",
            "mean_f1": round(float(np.mean(all_tier_f1s[t_name])), 4) if n_t > 0 else "N/A"
        }
        
    metrics_summary["macro_average"] = {
        "num_generators": len(generators),
        "total_test_images": len(all_per_image_records),
        "total_real": sum(1 for r in all_per_image_records if r["label"] == 0),
        "total_fake": sum(1 for r in all_per_image_records if r["label"] == 1),
        "detection_accuracy": round(macro_acc, 4),
        "detection_balanced_accuracy": round(macro_bal_acc, 4),
        "detection_f1": round(macro_det_f1, 4),
        "detection_roc_auc": round(macro_auc, 4),
        "localization_mean_iou": round(macro_iou, 4),
        "localization_mean_f1": round(macro_loc_f1, 4),
        "clean_image_false_alarm_pct": round(macro_clean_fp, 4),
        "aggregated_edit_tiers": aggregated_tiers
    }
    
    # Save metrics.json
    metrics_json_path = REPORTS_DIR / "metrics.json"
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2)
    print(f"Saved complete metrics summary to {metrics_json_path}", flush=True)
    
    # Generate Failure Case Overlays for top failure modes
    print("\nGenerating failure case overlays...", flush=True)
    generate_failure_visualizations(all_per_image_records)

def generate_failure_visualizations(records):
    # 1. Tiny edit failure (<2% edit)
    tiny_fakes = [r for r in records if r["label"] == 1 and r["edit_tier"] == "<2%"]
    if tiny_fakes:
        sample = sorted(tiny_fakes, key=lambda x: x["loc_iou"])[0] # lowest IoU
        create_overlay(sample, "failure_tiny_edit_missed.png", "Tiny Edit (<2%) Missed Localization")
        
    # 2. Large edit miss (high GT area, lowest IoU)
    large_fakes = [r for r in records if r["label"] == 1 and r["edit_tier"] == ">10%"]
    if large_fakes:
        sample = sorted(large_fakes, key=lambda x: x["loc_iou"])[0]
        create_overlay(sample, "failure_large_edit_missed.png", "Large Edit Missed Localization")
        
    # 3. Clean image with false alarm (highest FP pct)
    clean_reals = [r for r in records if r["label"] == 0]
    if clean_reals:
        sample = sorted(clean_reals, key=lambda x: x["clean_fp_pct"], reverse=True)[0]
        create_clean_overlay(sample, "failure_clean_image_false_alarm.png", "Genuine Image False Alarm")

def create_overlay(sample, filename, title):
    try:
        img = Image.open(ROOT_DIR / sample["image_path"]).convert("RGB")
        W, H = img.size
        gt_mask = np.array(Image.open(ROOT_DIR / sample["mask_path"]).convert("L")) > 127
        
        map_path = ROOT_DIR / sample["map_path"]
        raw_map = np.load(map_path)
        pred_pil = Image.fromarray(raw_map)
        pred_resized = np.array(pred_pil.resize((W, H), resample=Image.NEAREST))
        pred_bin = pred_resized >= 0.5
        
        # Create RGB overlay: Green = GT, Red = Pred, Yellow = Overlap
        overlay = np.array(img, dtype=np.uint8)
        # Red tint on pred
        overlay[pred_bin, 0] = np.clip(overlay[pred_bin, 0].astype(int) + 120, 0, 255).astype(np.uint8)
        # Green tint on GT
        overlay[gt_mask, 1] = np.clip(overlay[gt_mask, 1].astype(int) + 120, 0, 255).astype(np.uint8)
        
        out_img = Image.fromarray(overlay)
        save_path = FAILURE_DIR / filename
        out_img.save(save_path)
        print(f"  Saved overlay: {save_path.name} (Sample: {sample['sample_id']}, IoU: {sample['loc_iou']})", flush=True)
    except Exception as e:
        print(f"  Failed to create overlay for {sample['sample_id']}: {e}", flush=True)

def create_clean_overlay(sample, filename, title):
    try:
        img = Image.open(ROOT_DIR / sample["image_path"]).convert("RGB")
        W, H = img.size
        map_path = ROOT_DIR / sample["map_path"]
        raw_map = np.load(map_path)
        pred_pil = Image.fromarray(raw_map)
        pred_resized = np.array(pred_pil.resize((W, H), resample=Image.NEAREST))
        pred_bin = pred_resized >= 0.5
        
        overlay = np.array(img, dtype=np.uint8)
        overlay[pred_bin, 0] = np.clip(overlay[pred_bin, 0].astype(int) + 140, 0, 255).astype(np.uint8)
        
        out_img = Image.fromarray(overlay)
        save_path = FAILURE_DIR / filename
        out_img.save(save_path)
        print(f"  Saved clean overlay: {save_path.name} (Sample: {sample['sample_id']}, FP%: {sample['clean_fp_pct']}%)", flush=True)
    except Exception as e:
        print(f"  Failed to create clean overlay for {sample['sample_id']}: {e}", flush=True)

if __name__ == "__main__":
    run_evaluation()

