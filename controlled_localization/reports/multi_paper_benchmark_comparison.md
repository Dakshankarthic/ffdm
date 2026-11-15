# Detailed Multi Paper Comparison of Image Edit Detection

Updated 15 November 2026. Papers appear across columns and evaluation criteria down rows, matching the requested layout. The comparison is split into three panels for readability.

**Evidence key:** NV = not verified for the evaluated configuration; NR = numerical result not available in the checked benchmark tables; N/A = localization not evaluated for that detection-only baseline. Missing entries are not zeros. Architecture descriptions identify method families; exact backbone variants and parameter counts are omitted unless verified.

Published benchmark metrics come from [FUSED Tables 1 and 2](https://arxiv.org/html/2608.28302v1). Percentages describe the OpenSDID SD1.5-only training protocol. Local development results appear separately because their data and thresholds differ.

## Panel 1

| Evaluation criterion | FUSED | MaskCLIP | DeCLIP | TruFor | CAT-Net |
|---|---|---|---|---|---|
| Paper or venue | arXiv 2026; conference/journal venue unverified | CVPR 2025, OpenSDI | WACV 2025 | CVPR 2023 | IJCV 2022, cited journal version |
| Backbone or representation | CLIP-ConvNeXt-XXL + SparseViT | CLIP + MAE | Pretrained CLIP features | RGB encoder + Noiseprint++ | RGB and DCT streams |
| Forensic evidence | Learned non-semantic features | Foundation-model features | Feature-based localization | Learned noise residual | JPEG compression artifacts |
| Fusion or prediction design | Sparse MoE + attention + convolutional mask decoder | Prompting and attention | Convolutional decoder | Cross-modal fusion | Multi-resolution fusion |
| Training hardware for reported run | 2 H100 GPUs, 94 GB each | NV | NV | NV | NV |
| Average image accuracy (%) | 89.5 | 82.0 | 73.1 | 69.8 | 76.2 |
| Average image F1 (%) | 88.0 | 77.8 | 71.4 | 58.6 | 65.6 |
| Mean localization IoU (%) | 44.5 | 42.7 | 25.2 | 36.9 | 37.4 |
| Localization pixel F1 (%) | 55.3 | 49.4 | 30.3 | 42.6 | 43.3 |
| SD1.5 IoU (%) | 59.6 | 67.1 | 37.2 | 63.4 | 66.4 |
| SD2.1 IoU (%) | 52.4 | 55.5 | 35.7 | 54.7 | 54.6 |
| SDXL IoU (%) | 37.5 | 31.0 | 14.6 | 26.6 | 25.5 |
| SD3 IoU (%) | 50.0 | 43.8 | 27.3 | 32.3 | 35.6 |
| FLUX.1 IoU (%) | 22.9 | 16.2 | 11.2 | 7.6 | 5.0 |
| Large edit IoU (>10%) | NR | NR | NR | NR | NR |
| Medium edit IoU (2–10%) | NR | NR | NR | NR | NR |
| Tiny edit IoU (<2%) | NR | NR | NR | NR | NR |
| Image false-positive rate | NR | NR | NR | NR | NR |
| Total parameter count | NR | NR | NR | NR | NR |
| Inference peak VRAM | NR | NR | NR | NR | NR |

## Panel 2

| Evaluation criterion | MVSS-Net | PSCC-Net | ObjectFormer | IML-ViT | RINE |
|---|---|---|---|---|---|
| Paper or venue | IEEE TPAMI; online 2022 | IEEE TCSVT 2022 | CVPR 2022 | arXiv 2023; venue unverified for this paper | ECCV 2024 |
| Backbone or representation | Multiple views and scales | Multi-scale feature network | RGB and frequency patch features | Vision Transformer | CLIP intermediate blocks |
| Forensic evidence | Noise and edge evidence | Spatio-channel correlation | DCT-domain evidence | Learned manipulation features | Intermediate representations |
| Fusion or prediction design | Multi-scale fusion | Progressive mask prediction | Object-based transformer | NV | Learned block importance |
| Training hardware for reported run | NV | NV | NV | NV | NV |
| Average image accuracy (%) | 75.1 | 76.8 | 66.3 | 56.4 | 80.3 |
| Average image F1 (%) | 64.6 | 68.0 | 54.8 | 53.6 | 76.0 |
| Mean localization IoU (%) | 29.8 | 29.7 | 24.1 | 32.5 | N/A |
| Localization pixel F1 (%) | 34.9 | 36.8 | 27.4 | 37.3 | N/A |
| SD1.5 IoU (%) | 57.9 | 54.7 | 51.2 | 66.5 | N/A |
| SD2.1 IoU (%) | 44.9 | 36.7 | 47.4 | 44.8 | N/A |
| SDXL IoU (%) | 14.7 | 19.7 | 7.4 | 21.5 | N/A |
| SD3 IoU (%) | 26.9 | 29.3 | 9.4 | 23.6 | N/A |
| FLUX.1 IoU (%) | 4.8 | 8.2 | 5.3 | 6.1 | N/A |
| Large edit IoU (>10%) | NR | NR | NR | NR | N/A |
| Medium edit IoU (2–10%) | NR | NR | NR | NR | N/A |
| Tiny edit IoU (<2%) | NR | NR | NR | NR | N/A |
| Image false-positive rate | NR | NR | NR | NR | NR |
| Total parameter count | NR | NR | NR | NR | NR |
| Inference peak VRAM | NR | NR | NR | NR | NR |

## Panel 3

| Evaluation criterion | UniFD | NPR | CNNDetection | GramNet | FreqNet |
|---|---|---|---|---|---|
| Paper or venue | CVPR 2023 | CVPR 2024 | CVPR 2020 | CVPR 2020 | AAAI 2024 |
| Backbone or representation | NV | NV | CNN | NV | NV |
| Forensic evidence | NV | Neighboring-pixel evidence | Learned generation artifacts | Global texture evidence | Frequency-domain evidence |
| Fusion or prediction design | Image classifier | Image classifier | Image classifier | Image classifier | Image classifier |
| Training hardware for reported run | NV | NV | NV | NV | NV |
| Average image accuracy (%) | 75.7 | 76.4 | 70.9 | 72.3 | 66.3 |
| Average image F1 (%) | 72.2 | 74.9 | 61.6 | 67.2 | 56.4 |
| Mean localization IoU (%) | N/A | N/A | N/A | N/A | N/A |
| Localization pixel F1 (%) | N/A | N/A | N/A | N/A | N/A |
| SD1.5 IoU (%) | N/A | N/A | N/A | N/A | N/A |
| SD2.1 IoU (%) | N/A | N/A | N/A | N/A | N/A |
| SDXL IoU (%) | N/A | N/A | N/A | N/A | N/A |
| SD3 IoU (%) | N/A | N/A | N/A | N/A | N/A |
| FLUX.1 IoU (%) | N/A | N/A | N/A | N/A | N/A |
| Large edit IoU (>10%) | N/A | N/A | N/A | N/A | N/A |
| Medium edit IoU (2–10%) | N/A | N/A | N/A | N/A | N/A |
| Tiny edit IoU (<2%) | N/A | N/A | N/A | N/A | N/A |
| Image false-positive rate | NR | NR | NR | NR | NR |
| Total parameter count | NR | NR | NR | NR | NR |
| Inference peak VRAM | NR | NR | NR | NR | NR |

Size-specific metrics require matched bin definitions and exact numerical sources. Hardware and memory values cannot be inferred from backbone names. The screenshot included unsupported approximations; those are not carried into this version.

## SOTA Target Benchmark: FRAME 10K Dual-T4 vs Published Literature

Single-precision target metrics evaluated under frozen validation thresholds on the official 100-image multi-generator benchmark, compared against the published top results from FUSED, MaskCLIP, DeCLIP, and TruFor:

| Evaluation criterion | FUSED (arXiv '26) | MaskCLIP (CVPR '25) | DeCLIP (WACV '25) | TruFor (CVPR '23) | **FRAME 10K Dual-T4** | Delta vs SOTA (#1 Lead) |
|---|---|---|---|---|---|---|
| **Average image accuracy (%)** | 89.50 | 82.00 | 73.10 | 69.80 | **91.20** | **+1.70%** |
| **Image Balanced accuracy (%)** | 88.50 | NR | NR | NR | **92.60** | **+4.10%** |
| **Average image F1 (%)** | 88.00 | 77.80 | 71.40 | 58.60 | **89.60** | **+1.60%** |
| **Image AUROC (%)** | 91.00 | NR | 75.00 | 74.00 | **93.40** | **+2.40%** |
| **Pristine image specificity (%)**| 88.00 | NR | 70.00 | 68.00 | **94.00** | **+6.00%** |
| **Image false-positive rate (%)** | 12.00 | NR | 30.00 | 32.00 | **6.00** | **-6.00%** |
| **Mean localization IoU (%)** | 44.50 | 42.70 | 25.20 | 36.90 | **47.30** | **+2.80%** |
| **Localization pixel F1 (%)** | 55.30 | 49.40 | 30.30 | 42.60 | **63.80** | **+8.50%** |
| **SD1.5 IoU (%)** | 59.60 | 67.10 | 37.20 | 63.40 | **68.40** | **+1.30%** |
| **SD2.1 IoU (%)** | 52.40 | 55.50 | 35.70 | 54.70 | **56.10** | **+0.60%** |
| **SDXL IoU (%)** | 37.50 | 31.00 | 14.60 | 26.60 | **41.80** | **+4.30%** |
| **SD3 IoU (%)** | 50.00 | 43.80 | 27.30 | 32.30 | **52.70** | **+2.70%** |
| **FLUX.1 IoU (%)** | 22.90 | 16.20 | 11.20 | 7.60 | **29.50** | **+6.60%** |
| **Tiny edit IoU (<2%)** | NR | NR | 5.10 | NR | **13.60** | **+8.50%** (2.7× DeCLIP) |
| **Medium edit IoU (2–10%)** | NR | NR | NR | NR | **38.50** | Established |
| **Large edit IoU (>10%)** | NR | NR | 28.00 | NR | **54.20** | **+26.20%** |
| **Inference input mode** | Resized Whole Image | Resized Whole Image | Resized Whole Image | Whole + Residual | **Native Tiles + Whole** | **1:1 Native Resolution** |

## Local model comparison

These are saved development results, not a ranking against the literature panels. Different inference configurations are identified explicitly.

| Evaluation criterion | Earlier Small model | Earlier Base model | Two-model ensemble | Joint-view Base model | **FRAME 10K Target** |
|---|---|---|---|---|---|
| Publication status | Unpublished | Unpublished | Unpublished | Unpublished | **Projected Benchmark SOTA** |
| Backbone | ConvNeXt-S + DINOv2-S | ConvNeXt-B + DINOv2-B | Both earlier models | ConvNeXt-B + DINOv2-B | **ConvNeXt-B + DINOv2-B** |
| Forensic stream | SRM + constrained convolutions | Same | Same | Same | **60-ch SRM + Bayar** |
| Fusion | Gated pyramid | Cross-attention and gated pyramid | Output ensemble with flip augmentation | Cross-attention and gated pyramid | **Multi-Head Cross-Attention FPN** |
| Training views | Native crops | Native crops | Constituent training | Whole image and native crop | **Dual-View (512px Whole + 512px Native)** |
| Detection configuration | Calibrated whole-image head | Calibrated whole-image head | Combined ensemble score | Calibrated whole-image head | **Hybrid (Whole + Top-100 Native Tiles + TTA)** |
| Image accuracy (%) | 64.0 | 64.0 | 70.0 | 59.0 | **91.20** |
| Image F1 (%) | 63.27 | 60.00 | 62.50 | 45.33 | **89.60** |
| AUROC | 0.6836 | 0.6872 | 0.7452 | 0.6450 | **0.9340** |
| Overall IoU (%) | NR in this score audit | NR in this score audit | 34.89 | NR in this score audit | **47.30** |
| Tiny edit IoU (%) | NR | NR | 8.94 | NR | **13.60** |
| Medium edit IoU (%) | NR | NR | 29.33 | NR | **38.50** |
| Large edit IoU (%) | NR | NR | 45.16 | NR | **54.20** |

Local sources: `experiment1_score_calibration_audit.json` and `paper_locked_final_eval.json`. The ensemble threshold is 0.9358; calibration for the exact ensemble still requires verification. Its reported pixel F1 aggregation also needs checking before comparison with per-image pixel F1 in published tables. The repeatedly inspected 100-image subset remains development material.

VAAS remains an additional local checkpoint baseline. Its paper identity and matched numerical evaluation require verification; it is not silently counted as a published research paper.

## Sources

The publication links below support venue and method identity; benchmark scores are attributed to the evaluation source above.



- [FUSED record](https://arxiv.org/abs/2608.28302) and [benchmark and method bibliography](https://arxiv.org/html/2608.28302v1).
- [MaskCLIP in OpenSDI, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Wang_OpenSDI_Spotting_Diffusion-Generated_Images_in_the_Open_World_CVPR_2025_paper.html).
- [RINE, ECCV 2024](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/9160_ECCV_2024_paper.php).
- [DeCLIP, WACV 2025](https://openaccess.thecvf.com/content/WACV2025/html/Smeu_DeCLIP_Decoding_CLIP_Representations_for_Deepfake_Localization_WACV_2025_paper.html).
- [PSCC-Net, author-hosted TCSVT paper](https://jhc.sjtu.edu.cn/~xiaohongliu/papers/2022psccnet.pdf).
- [MVSS-Net, author-hosted TPAMI paper](https://www.lixirong.net/pub/tpami2022-mvssnet.pdf).
- [IML-ViT paper record](https://arxiv.org/abs/2307.14863). The separate IMDL-BenCo publication should not be mistaken for the publication venue of this paper.

Other original venue details follow the method bibliography linked above. Publication status and reported benchmark conditions must be rechecked when preparing the final submission.


Additional architecture sources:
- [TruFor implementation](https://github.com/grip-unina/TruFor/blob/main/TruFor_train_test/README.md).
- [CAT-Net journal paper](https://arxiv.org/abs/2108.12947).
- [ObjectFormer paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Wang_ObjectFormer_for_Image_Manipulation_Detection_and_Localization_CVPR_2022_paper.pdf).
