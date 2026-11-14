# OpenSDID Benchmark Evaluation: Controlled Localization v1.1 vs. VAAS Baseline

Comprehensive evaluation on the frozen 100-sample test benchmark across 5 diffusion architectures (SD 1.5, SD 2.1, SD 3, SDXL, and FLUX.1-schnell).

## Benchmark Comparison Milestone Table

| Metric | VAAS Baseline (v2-base-df2023) | Controlled Localization v1.1 (T2O) | Delta / Improvement |
| :--- | :---: | :---: | :---: |
| **Detection ROC-AUC** | 0.4420 | **0.6204** | **+0.1784** |
| **Detection Balanced Acc** | 0.4600 | **0.5000** | **+0.0400** |
| **Localization Mean IoU** | 0.1633 (16.33%) | **0.1700 (17.00%)** | **+0.67%** |
| **Localization Mean F1** | 0.2313 | **0.2580** | **+0.0267** |
| **Tiny Edit (<2%) IoU** | 0.0095 (0.95%) | **0.0157 (1.57%)** | **+0.62%** |
| **Medium Edit (2-10%) IoU** | 0.1897 | **0.0921** | **-0.0976** |
| **Large Edit (>10%) IoU** | 0.1722 | **0.2703** | **+0.0981** |
| **Clean False Alarm %** | 16.51% | **21.03%** | **+4.52%** |

---

## Per-Generator Breakdown

| Generator | Fake/Real | T2O Det AUC | T2O Mean IoU [95% CI] | VAAS Mean IoU |
| :--- | :---: | :---: | :---: | :---: |
| **SD15** | 10/10 | 0.4600 | **0.2109** [0.0940, 0.3623] | 0.1992 |
| **SD2** | 10/10 | 0.5200 | **0.2153** [0.1065, 0.3350] | 0.1359 |
| **SD3** | 10/10 | 0.7100 | **0.1490** [0.0710, 0.2403] | 0.1395 |
| **SDXL** | 10/10 | 0.6900 | **0.1350** [0.0563, 0.2138] | 0.1163 |
| **FLUX** | 10/10 | 0.7200 | **0.1398** [0.0495, 0.2471] | 0.2256 |
