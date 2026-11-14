# OpenSDID Benchmark Evaluation: Controlled Localization v1.1 vs. VAAS Baseline

Comprehensive evaluation on the frozen 100-sample test benchmark across 5 diffusion architectures (SD 1.5, SD 2.1, SD 3, SDXL, and FLUX.1-schnell).

## Benchmark Comparison Milestone Table

| Metric | VAAS Baseline (v2-base-df2023) | Controlled Localization v1.1 (T2O) | Delta / Improvement |
| :--- | :---: | :---: | :---: |
| **Detection ROC-AUC** | 0.4420 | **0.7272** | **+0.2852** |
| **Detection Balanced Acc** | 0.4600 | **0.5000** | **+0.0400** |
| **Localization Mean IoU** | 0.1633 (16.33%) | **0.3086 (30.86%)** | **+14.53%** |
| **Localization Mean F1** | 0.2313 | **0.4089** | **+0.1776** |
| **Tiny Edit (<2%) IoU** | 0.0095 (0.95%) | **0.0589 (5.89%)** | **+4.94%** |
| **Medium Edit (2-10%) IoU** | 0.1897 | **0.2606** | **+0.0709** |
| **Large Edit (>10%) IoU** | 0.1722 | **0.4026** | **+0.2304** |
| **Clean False Alarm %** | 16.51% | **7.05%** | **-9.46%** |

---

## Per-Generator Breakdown

| Generator | Fake/Real | T2O Det AUC | T2O Mean IoU [95% CI] | VAAS Mean IoU |
| :--- | :---: | :---: | :---: | :---: |
| **SD15** | 10/10 | 0.9200 | **0.5121** [0.3087, 0.6945] | 0.1992 |
| **SD2** | 10/10 | 0.7500 | **0.3394** [0.1673, 0.5280] | 0.1359 |
| **SD3** | 10/10 | 0.8000 | **0.2807** [0.1458, 0.4167] | 0.1395 |
| **SDXL** | 10/10 | 0.5100 | **0.1878** [0.0865, 0.2887] | 0.1163 |
| **FLUX** | 10/10 | 0.6200 | **0.2231** [0.1005, 0.3527] | 0.2256 |
