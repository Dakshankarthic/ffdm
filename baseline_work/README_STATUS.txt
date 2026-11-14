================================================================================
BASELINE WORKSTREAM STATUS & DELIVERABLES INVENTORY
================================================================================
Project: FDDM — AI-Edited Image Detection and Localization
Workspace: I:\icvgip workshop\baseline_work\
Last Updated: 13 November 2026 (Asia/Kolkata)
Audience: Gemini / Antigravity IDE Baseline Evaluation

1. STATUS OVERVIEW
------------------
The baseline evaluation workstream is 100% operational, fully reproducible, and completely isolated within I:\icvgip workshop\baseline_work\. No modifications were made to EchoSeam-Net, Pixel Lab, or existing root files.

- Environment Audit: COMPLETED (Windows 11, PyTorch 2.6.0+cu124, CUDA 12.4, RTX 2070 SUPER 8 GB, peak VRAM 454.72 MB). Log: logs\environment.txt.
- Unit Testing & Integrity Checks: COMPLETED (7/7 unit tests passed in 0.004s, covering hand-calculated IoU, confusion matrix, empty mask policies, and leakage audit). Log: logs\quality_checks_log.txt.
- Baseline Model Audit:
  * VAAS (v2-base-df2023): OPERATIONAL & EVALUATED (MIT License).
  * FUSED: Weights and architecture verified (MIT License). Single-image inference feasible; full training blocked on 8 GB GPU (authors used 2x H100s).
  * DeCLIP: Source present (CC BY-NC-SA 4.0); public GCS checkpoint URL verified (~2.23 GB).
  * TruFor: BLOCKED / NOT DOWNLOADED (requires acceptance of custom noncommercial terms).
  * IFA-Net: BLOCKED / NOT AVAILABLE (no code or model weights released).
  Table: reports\baseline_status_table.md.
- Dataset Preparation & Zero-Leakage Audit: COMPLETED.
  * 120 total samples from OpenSDID (CC BY-SA 4.0):
    - 20 validation samples (val_sd15) from nebula/OpenSDI_train
    - 100 test samples (20 per generator: SD1.5, SD2.1, SDXL, SD3, Flux.1) from nebula/OpenSDI_test
  * Balanced 1:1 real vs fake across all splits.
  * Stratified into <2%, 2–10%, >10% ground-truth edit area tiers.
  * Cross-split SHA256 audit: 0% overlap between train/val and test splits.
  Manifests: data_manifests\*.json and *.csv.
- Threshold Calibration: COMPLETED & FROZEN on val_sd15 only.
  * Decision threshold: 0.7075 (grid search maximizing balanced accuracy).
  * Mask threshold: 0.50 (standard sigmoid boundary).
  * Interpolation: Nearest-neighbor (PIL.Image.NEAREST).
  * Frozen config: configs\locked_evaluation_config.json.
- Independent Test Benchmark: COMPLETED across all 100 test images (50 real, 50 fake).
  * All raw maps saved as float32 .npy in predictions\test_{gen}\.
  * All predictions, scores, and metrics in reports\per_image.csv and reports\metrics.json.
  * Formal scientific evaluation report: reports\baseline_report.txt.
  * Failure case visual overlays: reports\failure_cases\*.png.

2. QUANTITATIVE BENCHMARK SUMMARY (VAAS v2-base-df2023)
-------------------------------------------------------
- Test Size: 100 images (50 genuine, 50 partial diffusion inpaintings) across 5 generators.
- Macro Detection Accuracy: 46.00% (Balanced Acc: 46.00%, Det F1: 46.79%, ROC-AUC: 44.20%).
- Macro Localization Mean IoU: 16.33% (95% CI: [8.9%, 24.6%]).
- Macro Localization Mean Pixel F1: 23.13% (95% CI: [13.1%, 33.9%]).
- Tiny Edit (<2% area, N=5) Mean IoU: 0.95% (F1: 1.83%, 80% complete misses).
- Small Edit (2–10% area, N=21) Mean IoU: 18.97% (F1: 26.00%).
- Large Edit (>10% area, N=24) Mean IoU: 17.22% (F1: 25.06%).
- Clean Image False Alarm Area: 16.51% of clean pixels flagged as anomalous.

3. REMAINING BLOCKERS AND ACTION ITEMS
--------------------------------------
1. FUSED Local Inference: Checkpoint fused_opensdi.pth (~306 MB) is available on Hugging Face; running full inference requires downloading timm's convnext_xxlarge weights (~3.3 GB). It fits in 8 GB VRAM in fp16 with batch size 1, but full retraining is impossible without multi-GPU cluster access (2x H100s).
2. TruFor License: Requires explicit user instruction before downloading code or weights due to custom clickwrap/browsewrap license terms.
3. EchoSeam-Net Comparison: The baseline protocol and frozen test manifests are ready. When EchoSeam-Net inference is ready, it must be evaluated on the identical manifests in data_manifests\ without modifying these baseline numbers.

4. INVENTORY OF ARTIFACTS
-------------------------
- baseline_work\README_STATUS.txt (this file)
- baseline_work\reports\baseline_report.txt
- baseline_work\reports\metrics.json
- baseline_work\reports\per_image.csv
- baseline_work\reports\baseline_status_table.md
- baseline_work\reports\val_predictions.csv
- baseline_work\reports\failure_cases\failure_tiny_edit_missed.png
- baseline_work\reports\failure_cases\failure_large_edit_missed.png
- baseline_work\reports\failure_cases\failure_clean_image_false_alarm.png
- baseline_work\configs\locked_evaluation_config.json
- baseline_work\data_manifests\val_sd15_manifest.json (.csv)
- baseline_work\data_manifests\test_sd15_manifest.json (.csv)
- baseline_work\data_manifests\test_sd2_manifest.json (.csv)
- baseline_work\data_manifests\test_sdxl_manifest.json (.csv)
- baseline_work\data_manifests\test_sd3_manifest.json (.csv)
- baseline_work\data_manifests\test_flux_manifest.json (.csv)
- baseline_work\logs\environment.txt
- baseline_work\logs\quality_checks_log.txt
- baseline_work\logs\smoke_check_log.json
- baseline_work\code\baseline_adapter.py
- baseline_work\code\test_metrics_and_checks.py
- baseline_work\code\prepare_frozen_datasets.py
- baseline_work\code\run_validation.py
- baseline_work\code\evaluate_test_sets.py
- baseline_work\predictions\val_sd15\*.npy (20 files)
- baseline_work\predictions\test_sd15\*.npy (20 files)
- baseline_work\predictions\test_sd2\*.npy (20 files)
- baseline_work\predictions\test_sdxl\*.npy (20 files)
- baseline_work\predictions\test_sd3\*.npy (20 files)
- baseline_work\predictions\test_flux\*.npy (20 files)
================================================================================

