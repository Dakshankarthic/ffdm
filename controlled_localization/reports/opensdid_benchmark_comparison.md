# OpenSDID Benchmark Evaluation: Controlled Localization v1.1 vs. VAAS Baseline

Comprehensive evaluation on the frozen 100-sample test benchmark across 5 diffusion architectures (SD 1.5, SD 2.1, SD 3, SDXL, and FLUX.1-schnell).

## Benchmark Comparison Milestone Table

| Metric | VAAS Baseline (v2-base-df2023) | Controlled Localization v1.1 (T2O) | Delta / Improvement |
| :--- | :---: | :---: | :---: |
| **Detection ROC-AUC** | 0.4420 | **0.5348** | **+0.0928** |
| **Detection Balanced Acc** | 0.4600 | **0.5000** | **+0.0400** |
| **Localization Mean IoU** | 0.1633 (16.33%) | **0.0444 (4.44%)** | **-11.89%** |
| **Localization Mean F1** | 0.2313 | **0.0809** | **-0.1504** |
| **Tiny Edit (<2%) IoU** | 0.0095 (0.95%) | **0.0573 (5.73%)** | **+4.78%** |
| **Medium Edit (2-10%) IoU** | 0.1897 | **0.0463** | **-0.1434** |
| **Large Edit (>10%) IoU** | 0.1722 | **0.0402** | **-0.1320** |
| **Clean False Alarm %** | 16.51% | **4.42%** | **-12.10%** |

---

## Per-Generator Breakdown

| Generator | Fake/Real | T2O Det AUC | T2O Mean IoU [95% CI] | VAAS Mean IoU |
| :--- | :---: | :---: | :---: | :---: |
| **SD15** | 10/10 | 0.5100 | **0.0601** [0.0291, 0.0980] | 0.1992 |
| **SD2** | 10/10 | 0.4800 | **0.0274** [0.0067, 0.0543] | 0.1359 |
| **SD3** | 10/10 | 0.3400 | **0.0293** [0.0065, 0.0565] | 0.1395 |
| **SDXL** | 10/10 | 0.6700 | **0.0312** [0.0160, 0.0458] | 0.1163 |
| **FLUX** | 10/10 | 0.6000 | **0.0743** [0.0368, 0.1179] | 0.2256 |
