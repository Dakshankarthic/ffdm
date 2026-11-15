# Multi Paper Comparison of Image Edit Detection Methods

Updated 15 November 2026.

## Literature benchmark

This comparison includes 15 methods. Accuracy and image F1 are averages across five OpenSDID generators, reported at threshold 0.5 in [FUSED Table 2](https://arxiv.org/html/2608.28302v1). These are benchmark results attributed to FUSED, not necessarily results in the original papers. The reported training setting is SD1.5 only.

| Method | Original venue or publication status | Accuracy (%) | Image F1 (%) |
|---|---|---:|---:|
| FUSED | arXiv 2026; conference/journal venue unverified | 89.5 | 88.0 |
| MaskCLIP | CVPR 2025, OpenSDI | 82.0 | 77.8 |
| RINE | ECCV 2024 | 80.3 | 76.0 |
| PSCC-Net | IEEE TCSVT 2022 | 76.8 | 68.0 |
| NPR | CVPR 2024 | 76.4 | 74.9 |
| CAT-Net | IJCV 2022, cited journal version | 76.2 | 65.6 |
| UniFD | CVPR 2023 | 75.7 | 72.2 |
| MVSS-Net | IEEE TPAMI; online 2022 | 75.1 | 64.6 |
| DeCLIP | WACV 2025 | 73.1 | 71.4 |
| GramNet | CVPR 2020 | 72.3 | 67.2 |
| CNNDetection | CVPR 2020 | 70.9 | 61.6 |
| TruFor | CVPR 2023 | 69.8 | 58.6 |
| ObjectFormer | CVPR 2022 | 66.3 | 54.8 |
| FreqNet | AAAI 2024 | 66.3 | 56.4 |
| IML-ViT | arXiv 2023; venue unverified for this paper | 56.4 | 53.6 |

## Local experiments

These results use the repeatedly inspected 100-image development subset and differing training and scoring configurations. They must not be ranked directly against the literature table.

| Local configuration | Accuracy (%) | Image F1 (%) | AUROC | Source |
|---|---:|---:|---:|---|
| Two-model ensemble with flip augmentation | 70.0 | 62.50 | 0.7452 | paper_locked_final_eval.json |
| Earlier Base model, whole-image score | 64.0 | 60.00 | 0.6872 | experiment1_score_calibration_audit.json |
| Earlier Small multi-generator model, whole-image score | 64.0 | 63.27 | 0.6836 | experiment1_score_calibration_audit.json |
| Scaled pilot, combined score | 65.0 | 61.54 | 0.6700 | experiment1_score_calibration_audit.json |
| Joint whole-image/crop model, whole-image score | 59.0 | 45.33 | 0.6450 | experiment1_score_calibration_audit.json |

The ensemble reuses an image threshold of 0.9358. Calibration for the exact ensemble configuration still requires verification. The filenames do not establish independent test status. Local figures are recorded results, not fresh reproductions performed for this document.

VAAS is an additional local baseline, not a verified peer-reviewed paper entry here. Its publication identity, checkpoint provenance and matched evaluation must be documented before it is included as a literature comparator. A downloaded checkpoint or model card alone does not establish a publication venue.

## Interpretation for the paper

The literature table covers broader prior work than the three-method summary it replaces. The local table identifies the models actually evaluated in this project. Inclusion of a method in the literature table does not mean it has been reproduced locally.

Before making a comparative performance claim, match training generator exposure, source splits, test population, preprocessing, score thresholds and metric aggregation. Report image detection separately from pixel localization. Do not interpret AUROC as accuracy or genuine predicted mask area as image-level false-positive rate.

The previous comparison contained unsupported approximate metrics, hardware estimates and victory claims. Those are omitted. Original versions are retained separately for traceability.

## Publication references

- [FUSED record](https://arxiv.org/abs/2608.28302) and [benchmark and method bibliography](https://arxiv.org/html/2608.28302v1).
- [MaskCLIP in OpenSDI, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Wang_OpenSDI_Spotting_Diffusion-Generated_Images_in_the_Open_World_CVPR_2025_paper.html).
- [RINE, ECCV 2024](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/9160_ECCV_2024_paper.php).
- [DeCLIP, WACV 2025](https://openaccess.thecvf.com/content/WACV2025/html/Smeu_DeCLIP_Decoding_CLIP_Representations_for_Deepfake_Localization_WACV_2025_paper.html).
- [PSCC-Net, author-hosted TCSVT paper](https://jhc.sjtu.edu.cn/~xiaohongliu/papers/2022psccnet.pdf).
- [MVSS-Net, author-hosted TPAMI paper](https://www.lixirong.net/pub/tpami2022-mvssnet.pdf).
- [IML-ViT paper record](https://arxiv.org/abs/2307.14863). The separate IMDL-BenCo publication should not be mistaken for the publication venue of this paper.

Other original venue details follow the method bibliography linked above. Publication status and reported benchmark conditions must be rechecked when preparing the final submission.
