print("=== ENTERING RUN_VALIDATION.PY ===", flush=True)
import os
import sys
import traceback
from pathlib import Path

# Ensure code dir is in sys.path
CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

try:
    import hashlib
    import json
    import csv
    import time
    import numpy as np
    import torch
    from PIL import Image
    from sklearn.metrics import roc_auc_score, accuracy_score, balanced_accuracy_score, precision_recall_fscore_support

    from baseline_adapter import VAASAdapter
    from test_metrics_and_checks import compute_per_image_iou_f1, classify_edit_tier, compute_classification_metrics
    print("All modules imported in run_validation.py", flush=True)
except Exception as e:
    print("EXCEPTION DURING IMPORT:", flush=True)
    traceback.print_exc()
    sys.exit(1)

ROOT_DIR = Path("I:/icvgip workshop")
PRED_DIR = ROOT_DIR / "baseline_work" / "predictions" / "val_sd15"
PRED_DIR.mkdir(parents=True, exist_ok=True)
CONFIG_DIR = ROOT_DIR / "baseline_work" / "configs"
CONFIG_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR = ROOT_DIR / "baseline_work" / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

def run_validation():
    try:
        print("======================================================================", flush=True)
        print("STEP 4.6 & 5: VALIDATION RUN ON val_sd15 (CALIBRATION & THRESHOLD LOCK)", flush=True)
        print("======================================================================", flush=True)
        
        manifest_path = ROOT_DIR / "baseline_work" / "data_manifests" / "val_sd15_manifest.json"
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
            
        entries = manifest["entries"]
        print(f"Loaded {len(entries)} validation samples (Real: {manifest['num_real']}, Fake: {manifest['num_fake']})", flush=True)
        
        cfg = {
            "model_repo": "OBA-Research/vaas",
            "model_variant": "v2-base-df2023",
            "device": "cuda" if torch.cuda.is_available() else "cpu"
        }
        adapter = VAASAdapter(cfg)
        adapter.load_model()
        print("Model initialized.", flush=True)
        
        records = []
        y_true = []
        s_h_scores = []
        s_f_scores = []
        s_p_scores = []
        
        ious_05 = []
        f1s_05 = []
        
        for idx, e in enumerate(entries):
            img_path = ROOT_DIR / e["image_path"]
            label = e["label"]
            y_true.append(label)
            
            # BLIND INFERENCE
            pred = adapter.predict(str(img_path))
            raw_map = pred["raw_map"]
            
            # Save raw map as float32 .npy
            map_filename = f"{e['sample_id']}_map.npy"
            map_save_path = PRED_DIR / map_filename
            np.save(map_save_path, raw_map)
            
            rec = {
                "sample_id": e["sample_id"],
                "image_path": e["image_path"],
                "label": label,
                "edit_tier": e["edit_tier"],
                "edit_area_pct": e["edit_area_pct"],
                "S_F": pred["raw_scores"]["S_F"],
                "S_P": pred["raw_scores"]["S_P"],
                "S_H": pred["raw_scores"]["S_H"],
                "runtime_sec": pred["runtime_sec"],
                "map_path": str(map_save_path.relative_to(ROOT_DIR))
            }
            records.append(rec)
            s_h_scores.append(pred["raw_scores"]["S_H"])
            s_f_scores.append(pred["raw_scores"]["S_F"])
            s_p_scores.append(pred["raw_scores"]["S_P"])
            
            # Localization evaluation if fake
            if label == 1 and e["mask_path"]:
                gt_mask_path = ROOT_DIR / e["mask_path"]
                gt_mask = np.array(Image.open(gt_mask_path).convert("L"))
                gt_bin = (gt_mask > 127).astype(np.uint8)
                
                # Resize pred map to GT resolution using NEAREST NEIGHBOR
                pred_pil = Image.fromarray(raw_map)
                pred_resized = np.array(pred_pil.resize((gt_bin.shape[1], gt_bin.shape[0]), resample=Image.NEAREST))
                pred_bin_05 = (pred_resized >= 0.5).astype(np.uint8)
                
                iou, f1 = compute_per_image_iou_f1(pred_bin_05, gt_bin)
                ious_05.append(iou)
                f1s_05.append(f1)
                rec["loc_iou_05"] = round(iou, 4)
                rec["loc_f1_05"] = round(f1, 4)
            else:
                rec["loc_iou_05"] = None
                rec["loc_f1_05"] = None
                
            print(f"  [{idx+1}/{len(entries)}] {e['sample_id']} | Label: {label} | S_H: {pred['raw_scores']['S_H']:.4f} | Runtime: {pred['runtime_sec']:.3f}s", flush=True)

        val_pred_csv = REPORTS_DIR / "val_predictions.csv"
        with open(val_pred_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
            writer.writeheader()
            writer.writerows(records)
        print(f"Saved validation predictions to {val_pred_csv}", flush=True)
        
        real_s_h = [s for s, y in zip(s_h_scores, y_true) if y == 0]
        fake_s_h = [s for s, y in zip(s_h_scores, y_true) if y == 1]
        
        mean_real_s_h = float(np.mean(real_s_h))
        mean_fake_s_h = float(np.mean(fake_s_h))
        
        print("\n--- SCORE POLARITY ANALYSIS (VALIDATION SET) ---", flush=True)
        print(f"Mean S_H (Real / 0): {mean_real_s_h:.4f} (median: {np.median(real_s_h):.4f})", flush=True)
        print(f"Mean S_H (Fake / 1): {mean_fake_s_h:.4f} (median: {np.median(fake_s_h):.4f})", flush=True)
        
        auc_lower_is_fake = float(roc_auc_score(y_true, [-s for s in s_h_scores]))
        auc_higher_is_fake = float(roc_auc_score(y_true, s_h_scores))
        print(f"ROC-AUC if 'lower_is_fake' (code definition): {auc_lower_is_fake:.4f}", flush=True)
        print(f"ROC-AUC if 'higher_is_fake' (paper description): {auc_higher_is_fake:.4f}", flush=True)
        
        best_thresh = 0.5
        best_bal_acc = -1.0
        for t in np.linspace(min(s_h_scores), max(s_h_scores), 100):
            preds = [int(s <= t) for s in s_h_scores]
            bal_acc = balanced_accuracy_score(y_true, preds)
            if bal_acc > best_bal_acc:
                best_bal_acc = bal_acc
                best_thresh = float(t)
                
        print(f"\nOptimal Image Score Threshold on Validation Set: {best_thresh:.4f} (Bal_Acc: {best_bal_acc:.4f})", flush=True)
        
        mean_iou_05 = float(np.mean(ious_05)) if ious_05 else 0.0
        mean_f1_05 = float(np.mean(f1s_05)) if f1s_05 else 0.0
        print(f"Validation Localization at fixed threshold 0.5: Mean IoU = {mean_iou_05:.4f}, Mean F1 = {mean_f1_05:.4f}", flush=True)
        
        locked_config = {
            "locked_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "model_id": "OBA-Research/vaas:v2-base-df2023",
            "validation_manifest_hash": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            "random_seed": 42,
            "score_direction": "lower_is_fake",
            "score_direction_rationale": "Empirically verified on val_sd15 and in vaas 0.2.0 hybrid_score.py where S_P = 1.0 - mean(pred_sig) and S_F = exp(-delta/sigma). Both components output fidelity (normality) scores where real images score higher and anomalous/fake images score lower.",
            "documented_polarity_conflict": "VAAS paper text (Section 4) describes S_F as an anomaly deviation (higher=anomalous), while vaas 0.2.0 code and model card calculate fidelity S_F=exp(-delta/sigma). We report raw scores and explicitly document this version divergence.",
            "image_score_metric": "S_H",
            "image_score_threshold": round(best_thresh, 4),
            "image_threshold_selection_rule": "Maximizes balanced accuracy on frozen validation set val_sd15 (grid search on validation only).",
            "mask_threshold": 0.5,
            "mask_threshold_selection_rule": "Standard sigmoid boundary 0.5 matching FUSED and standard segmentation literature.",
            "mask_resize_interpolation": "PIL.Image.NEAREST",
            "zero_union_policy": "If both GT and Pred empty -> IoU=1.0, F1=1.0. If one empty and one non-empty -> IoU=0.0, F1=0.0."
        }
        
        locked_cfg_path = CONFIG_DIR / "locked_evaluation_config.json"
        locked_cfg_path.write_text(json.dumps(locked_config, indent=2), encoding="utf-8")
        print(f"\n>>> LOCKED EVALUATION CONFIG SAVED TO: {locked_cfg_path} <<<", flush=True)
    except Exception as e:
        print("EXCEPTION IN RUN_VALIDATION:", flush=True)
        traceback.print_exc()

if __name__ == "__main__":
    run_validation()
