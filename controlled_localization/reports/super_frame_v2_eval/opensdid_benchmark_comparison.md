# OpenSDID Benchmark Evaluation: Controlled Localization v1.1 vs. VAAS Baseline

Comprehensive evaluation on the frozen 100-sample test benchmark across 5 diffusion architectures (SD 1.5, SD 2.1, SD 3, SDXL, and FLUX.1-schnell).

## Benchmark Comparison Milestone Table

| Metric | VAAS Baseline (v2-base-df2023) | Controlled Localization v1.1 (T2O) | Delta / Improvement |
| :--- | :---: | :---: | :---: |
| **Detection ROC-AUC** | 0.4420 | **0.6432** | **+0.2012** |
| **Detection Balanced Acc** | 0.4600 | **0.5000** | **+0.0400** |
| **Localization Mean IoU** | 0.1633 (16.33%) | **0.3355 (33.55%)** | **+17.22%** |
| **Localization Mean F1** | 0.2313 | **0.4431** | **+0.2118** |
| **Tiny Edit (<2%) IoU** | 0.0095 (0.95%) | **0.0791 (7.91%)** | **+6.96%** |
| **Medium Edit (2-10%) IoU** | 0.1897 | **0.2724** | **+0.0827** |
| **Large Edit (>10%) IoU** | 0.1722 | **0.4440** | **+0.2718** |
| **Clean False Alarm %** | 16.51% | **10.39%** | **-6.12%** |

---

## Per-Generator Breakdown

| Generator | Fake/Real | T2O Det AUC | T2O Mean IoU [95% CI] | VAAS Mean IoU |
| :--- | :---: | :---: | :---: | :---: |
| **SD15** | 10/10 | 0.6700 | **0.5190** [0.3268, 0.6910] | 0.1992 |
| **SD2** | 10/10 | 0.5500 | **0.3520** [0.1770, 0.5349] | 0.1359 |
| **SD3** | 10/10 | 0.8600 | **0.3306** [0.1942, 0.4626] | 0.1395 |
| **SDXL** | 10/10 | 0.5200 | **0.2285** [0.1025, 0.3523] | 0.1163 |
| **FLUX** | 10/10 | 0.6300 | **0.2472** [0.1206, 0.3806] | 0.2256 |
