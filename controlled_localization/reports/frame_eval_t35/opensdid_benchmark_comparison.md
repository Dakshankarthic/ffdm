# OpenSDID Benchmark Evaluation: Controlled Localization v1.1 vs. VAAS Baseline

Comprehensive evaluation on the frozen 100-sample test benchmark across 5 diffusion architectures (SD 1.5, SD 2.1, SD 3, SDXL, and FLUX.1-schnell).

## Benchmark Comparison Milestone Table

| Metric | VAAS Baseline (v2-base-df2023) | Controlled Localization v1.1 (T2O) | Delta / Improvement |
| :--- | :---: | :---: | :---: |
| **Detection ROC-AUC** | 0.4420 | **0.5468** | **+0.1048** |
| **Detection Balanced Acc** | 0.4600 | **0.5000** | **+0.0400** |
| **Localization Mean IoU** | 0.1633 (16.33%) | **0.2460 (24.60%)** | **+8.27%** |
| **Localization Mean F1** | 0.2313 | **0.3514** | **+0.1201** |
| **Tiny Edit (<2%) IoU** | 0.0095 (0.95%) | **0.0181 (1.81%)** | **+0.86%** |
| **Medium Edit (2-10%) IoU** | 0.1897 | **0.2030** | **+0.0133** |
| **Large Edit (>10%) IoU** | 0.1722 | **0.3311** | **+0.1589** |
| **Clean False Alarm %** | 16.51% | **13.65%** | **-2.87%** |

---

## Per-Generator Breakdown

| Generator | Fake/Real | T2O Det AUC | T2O Mean IoU [95% CI] | VAAS Mean IoU |
| :--- | :---: | :---: | :---: | :---: |
| **SD15** | 10/10 | 0.7600 | **0.3890** [0.2455, 0.5522] | 0.1992 |
| **SD2** | 10/10 | 0.5600 | **0.2908** [0.1672, 0.4300] | 0.1359 |
| **SD3** | 10/10 | 0.6900 | **0.2433** [0.1253, 0.3604] | 0.1395 |
| **SDXL** | 10/10 | 0.3900 | **0.1426** [0.0617, 0.2271] | 0.1163 |
| **FLUX** | 10/10 | 0.4300 | **0.1643** [0.0684, 0.2732] | 0.2256 |
