# OpenSDID Benchmark Evaluation: Controlled Localization v1.1 vs. VAAS Baseline

Comprehensive evaluation on the frozen 100-sample test benchmark across 5 diffusion architectures (SD 1.5, SD 2.1, SD 3, SDXL, and FLUX.1-schnell).

## Benchmark Comparison Milestone Table

| Metric | VAAS Baseline (v2-base-df2023) | Controlled Localization v1.1 (T2O) | Delta / Improvement |
| :--- | :---: | :---: | :---: |
| **Detection ROC-AUC** | 0.4420 | **0.5468** | **+0.1048** |
| **Detection Balanced Acc** | 0.4600 | **0.5000** | **+0.0400** |
| **Localization Mean IoU** | 0.1633 (16.33%) | **0.2310 (23.10%)** | **+6.77%** |
| **Localization Mean F1** | 0.2313 | **0.3307** | **+0.0994** |
| **Tiny Edit (<2%) IoU** | 0.0095 (0.95%) | **0.0108 (1.08%)** | **+0.13%** |
| **Medium Edit (2-10%) IoU** | 0.1897 | **0.2032** | **+0.0135** |
| **Large Edit (>10%) IoU** | 0.1722 | **0.3012** | **+0.1290** |
| **Clean False Alarm %** | 16.51% | **10.84%** | **-5.68%** |

---

## Per-Generator Breakdown

| Generator | Fake/Real | T2O Det AUC | T2O Mean IoU [95% CI] | VAAS Mean IoU |
| :--- | :---: | :---: | :---: | :---: |
| **SD15** | 10/10 | 0.7600 | **0.3826** [0.2317, 0.5513] | 0.1992 |
| **SD2** | 10/10 | 0.5600 | **0.2697** [0.1503, 0.4047] | 0.1359 |
| **SD3** | 10/10 | 0.6900 | **0.2451** [0.1242, 0.3642] | 0.1395 |
| **SDXL** | 10/10 | 0.3900 | **0.1250** [0.0517, 0.2009] | 0.1163 |
| **FLUX** | 10/10 | 0.4300 | **0.1326** [0.0498, 0.2322] | 0.2256 |
