# OpenSDID Benchmark Evaluation: Controlled Localization v1.1 vs. VAAS Baseline

Comprehensive evaluation on the frozen 100-sample test benchmark across 5 diffusion architectures (SD 1.5, SD 2.1, SD 3, SDXL, and FLUX.1-schnell).

## Benchmark Comparison Milestone Table

| Metric | VAAS Baseline (v2-base-df2023) | Controlled Localization v1.1 (T2O) | Delta / Improvement |
| :--- | :---: | :---: | :---: |
| **Detection ROC-AUC** | 0.4420 | **0.7272** | **+0.2852** |
| **Detection Balanced Acc** | 0.4600 | **0.5000** | **+0.0400** |
| **Localization Mean IoU** | 0.1633 (16.33%) | **0.3070 (30.70%)** | **+14.37%** |
| **Localization Mean F1** | 0.2313 | **0.4115** | **+0.1802** |
| **Tiny Edit (<2%) IoU** | 0.0095 (0.95%) | **0.0575 (5.75%)** | **+4.80%** |
| **Medium Edit (2-10%) IoU** | 0.1897 | **0.2400** | **+0.0503** |
| **Large Edit (>10%) IoU** | 0.1722 | **0.4176** | **+0.2454** |
| **Clean False Alarm %** | 16.51% | **9.08%** | **-7.43%** |

---

## Per-Generator Breakdown

| Generator | Fake/Real | T2O Det AUC | T2O Mean IoU [95% CI] | VAAS Mean IoU |
| :--- | :---: | :---: | :---: | :---: |
| **SD15** | 10/10 | 0.9200 | **0.4929** [0.2963, 0.6735] | 0.1992 |
| **SD2** | 10/10 | 0.7500 | **0.3457** [0.1764, 0.5269] | 0.1359 |
| **SD3** | 10/10 | 0.8000 | **0.2732** [0.1472, 0.4032] | 0.1395 |
| **SDXL** | 10/10 | 0.5100 | **0.1947** [0.0899, 0.2979] | 0.1163 |
| **FLUX** | 10/10 | 0.6200 | **0.2285** [0.1104, 0.3544] | 0.2256 |
