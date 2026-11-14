# OpenSDID Benchmark Evaluation: Controlled Localization v1.1 vs. VAAS Baseline

Comprehensive evaluation on the frozen 100-sample test benchmark across 5 diffusion architectures (SD 1.5, SD 2.1, SD 3, SDXL, and FLUX.1-schnell).

## Benchmark Comparison Milestone Table

| Metric | VAAS Baseline (v2-base-df2023) | Controlled Localization v1.1 (T2O) | Delta / Improvement |
| :--- | :---: | :---: | :---: |
| **Detection ROC-AUC** | 0.4420 | **0.6196** | **+0.1776** |
| **Detection Balanced Acc** | 0.4600 | **0.5100** | **+0.0500** |
| **Localization Mean IoU** | 0.1633 (16.33%) | **0.1147 (11.47%)** | **-4.86%** |
| **Localization Mean F1** | 0.2313 | **0.1815** | **-0.0498** |
| **Tiny Edit (<2%) IoU** | 0.0095 (0.95%) | **0.0230 (2.30%)** | **+1.35%** |
| **Medium Edit (2-10%) IoU** | 0.1897 | **0.1005** | **-0.0892** |
| **Large Edit (>10%) IoU** | 0.1722 | **0.1462** | **-0.0260** |
| **Clean False Alarm %** | 16.51% | **5.14%** | **-11.37%** |

---

## Per-Generator Breakdown

| Generator | Fake/Real | T2O Det AUC | T2O Mean IoU [95% CI] | VAAS Mean IoU |
| :--- | :---: | :---: | :---: | :---: |
| **SD15** | 10/10 | 0.5500 | **0.1103** [0.0376, 0.2192] | 0.1992 |
| **SD2** | 10/10 | 0.6000 | **0.1496** [0.0794, 0.2253] | 0.1359 |
| **SD3** | 10/10 | 0.6600 | **0.1314** [0.0218, 0.2519] | 0.1395 |
| **SDXL** | 10/10 | 0.5900 | **0.0879** [0.0371, 0.1480] | 0.1163 |
| **FLUX** | 10/10 | 0.6800 | **0.0942** [0.0276, 0.1782] | 0.2256 |
