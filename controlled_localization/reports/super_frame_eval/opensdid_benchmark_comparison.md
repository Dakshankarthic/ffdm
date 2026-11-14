# OpenSDID Benchmark Evaluation: Controlled Localization v1.1 vs. VAAS Baseline

Comprehensive evaluation on the frozen 100-sample test benchmark across 5 diffusion architectures (SD 1.5, SD 2.1, SD 3, SDXL, and FLUX.1-schnell).

## Benchmark Comparison Milestone Table

| Metric | VAAS Baseline (v2-base-df2023) | Controlled Localization v1.1 (T2O) | Delta / Improvement |
| :--- | :---: | :---: | :---: |
| **Detection ROC-AUC** | 0.4420 | **0.6388** | **+0.1968** |
| **Detection Balanced Acc** | 0.4600 | **0.5000** | **+0.0400** |
| **Localization Mean IoU** | 0.1633 (16.33%) | **0.3027 (30.27%)** | **+13.94%** |
| **Localization Mean F1** | 0.2313 | **0.4028** | **+0.1715** |
| **Tiny Edit (<2%) IoU** | 0.0095 (0.95%) | **0.0588 (5.88%)** | **+4.93%** |
| **Medium Edit (2-10%) IoU** | 0.1897 | **0.2428** | **+0.0531** |
| **Large Edit (>10%) IoU** | 0.1722 | **0.4058** | **+0.2336** |
| **Clean False Alarm %** | 16.51% | **9.31%** | **-7.21%** |

---

## Per-Generator Breakdown

| Generator | Fake/Real | T2O Det AUC | T2O Mean IoU [95% CI] | VAAS Mean IoU |
| :--- | :---: | :---: | :---: | :---: |
| **SD15** | 10/10 | 0.7400 | **0.4917** [0.2864, 0.6716] | 0.1992 |
| **SD2** | 10/10 | 0.5200 | **0.3546** [0.1764, 0.5449] | 0.1359 |
| **SD3** | 10/10 | 0.7400 | **0.2819** [0.1602, 0.3963] | 0.1395 |
| **SDXL** | 10/10 | 0.4200 | **0.1808** [0.0780, 0.2838] | 0.1163 |
| **FLUX** | 10/10 | 0.7100 | **0.2043** [0.0830, 0.3315] | 0.2256 |
