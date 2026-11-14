# OpenSDID Benchmark Evaluation: Controlled Localization v1.1 vs. VAAS Baseline

Comprehensive evaluation on the frozen 100-sample test benchmark across 5 diffusion architectures (SD 1.5, SD 2.1, SD 3, SDXL, and FLUX.1-schnell).

## Benchmark Comparison Milestone Table

| Metric | VAAS Baseline (v2-base-df2023) | Controlled Localization v1.1 (T2O) | Delta / Improvement |
| :--- | :---: | :---: | :---: |
| **Detection ROC-AUC** | 0.4420 | **0.6404** | **+0.1984** |
| **Detection Balanced Acc** | 0.4600 | **0.5100** | **+0.0500** |
| **Localization Mean IoU** | 0.1633 (16.33%) | **0.1677 (16.77%)** | **+0.44%** |
| **Localization Mean F1** | 0.2313 | **0.2456** | **+0.0143** |
| **Tiny Edit (<2%) IoU** | 0.0095 (0.95%) | **0.0279 (2.79%)** | **+1.84%** |
| **Medium Edit (2-10%) IoU** | 0.1897 | **0.0804** | **-0.1093** |
| **Large Edit (>10%) IoU** | 0.1722 | **0.2731** | **+0.1009** |
| **Clean False Alarm %** | 16.51% | **14.69%** | **-1.82%** |

---

## Per-Generator Breakdown

| Generator | Fake/Real | T2O Det AUC | T2O Mean IoU [95% CI] | VAAS Mean IoU |
| :--- | :---: | :---: | :---: | :---: |
| **SD15** | 10/10 | 0.6400 | **0.1783** [0.0390, 0.3449] | 0.1992 |
| **SD2** | 10/10 | 0.7300 | **0.2689** [0.1229, 0.4343] | 0.1359 |
| **SD3** | 10/10 | 0.7000 | **0.1359** [0.0474, 0.2406] | 0.1395 |
| **SDXL** | 10/10 | 0.6400 | **0.1214** [0.0611, 0.2045] | 0.1163 |
| **FLUX** | 10/10 | 0.5300 | **0.1337** [0.0469, 0.2393] | 0.2256 |
