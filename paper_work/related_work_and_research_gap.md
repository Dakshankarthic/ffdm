# Related work and research gap — working notes

Updated 13 November 2026. Scope: passive detection and localization of AI edits from one image. This document contains paper-reported results and hypotheses, not our own experimental results.

## Closest methods

### FUSED (preprint, 2026)
Forensic and semantic streams with sparse mixture-of-experts fusion for AI-inpainting detection and localization. On its OpenSDID cross-generator protocol, authors report average image accuracy 89.5%, image F1 88.0%, per-image pixel IoU 44.5%, and pixel F1 55.3% across five generators. Training used SD1.5, and the average includes in-domain SD1.5. FUSED is not best for every generator. One fixed seed, two H100 GPUs reported for this experiment. Its paired INP/INP-X analysis already studies sensitivity to global reconstruction artifacts. Exact results, mask definitions, data and checkpoint settings must match for a fair comparison.
https://arxiv.org/html/2608.28302v1

### IFA-Net (CVPR 2026)
An initial localization prediction guides a second reconstruction-refinement step using an MAE; therefore neither two passes nor prediction-guided refinement alone is novel. The proposed local native-resolution crop verification is architecturally different, but its research value and novelty must be substantiated. Check the paper's training and test protocols carefully before using numerical results.
https://openaccess.thecvf.com/content/CVPR2026/papers/Zhang_Detecting_AI-Generated_Forgeries_via_Iterative_Manifold_Deviation_Amplification_CVPR_2026_paper.pdf

### TruFor (CVPR 2023)
General manipulation localization combines RGB and learned noise traces, outputs confidence estimates. It is an important baseline but its README says downloading/using the files implicitly signifies agreement with its custom license; source and weights were not downloaded in this work.
https://openaccess.thecvf.com/content/CVPR2023/papers/Guillaro_TruFor_Leveraging_All-Round_Clues_for_Trustworthy_Image_Forgery_Detection_and_CVPR_2023_paper.pdf
https://github.com/grip-unina/TruFor

### DiffusionPrint (CVPRW 2026)
Patch-level contrastive learning for generative forensic signals and whole-image reconstruction challenges. Relevant to any claim that local noise consistency or generative fingerprints are new.
https://openaccess.thecvf.com/content/CVPR2026W/AIMS/html/Giakoumoglou_DiffusionPrint_Learning_Generative_Fingerprints_for_Diffusion-Based_Inpainting_Localization_CVPRW_2026_paper.html

### ForenScope / GIFTBench (preprint, 2026)
Multi-depth and multi-scale features; multiple benchmark shifts. Related to high-resolution localization and cross-dataset evaluation. Training and test protocols differ from FUSED.
https://arxiv.org/abs/2610.01778

## Working hypothesis
When a local AI edit occupies a small part of an image and a previously unseen editor is used, a bounded high-resolution local inspection procedure may improve localization over a global-only baseline under an equal input and compute budget. Compare against simple overlapping tiles and prediction-guided refinement, including IFA-Net where feasible. This hypothesis may fail, especially if the proposal stage misses the edit or artifacts disappear after social-media processing.

## Evidence required before claiming contribution
1. Report <2%, 2–10%, >10% edit-area groups and the number of examples in each.
2. Measure crop coverage recall@K, pixel IoU/F1 and false positives on genuine images.
3. Compare global only, full/overlapping tiles and proposed crop selection at equal budgets.
4. Evaluate unseen editors, processing, and source-image grouping; document dataset provenance.
5. Ablate each feature and report uncertainty/multiple seeds when feasible. Do not claim a module is necessary if results do not support it.

No implementation, baseline predictions, trained model or empirical outcomes are reported here.

## Directly relevant prior art: VAAS neighborhood comparison
The VAAS paper describes cosine similarity between each patch and its spatial
neighbors as an anomaly cue (Section 4.2). Consequently, the broad concept of
EchoSeam's neighboring-feature cosine disagreement is already in the literature.
The exact fixed hollow 3x3 kernel and full proposed combination still require
a specific novelty comparison; neither identical implementation nor novelty
is established from this observation alone.
https://arxiv.org/html/2512.15512v1

## Development diagnostic — 13 November 2026
The installed vaas 0.2.0 v2-base-df2023 pipeline ran successfully on 16 viewer
assets from OpenSDI_train/sd15 (8 labeled real and 8 fake), selected using
systematic, nonrandom offsets; all are training split and source independence
is unchecked. Raw scores and 224x224 maps: vaas_development_preview/.
No independent benchmark accuracy or localization metric was calculated.
The paper describes a deviation S_F where greater is more anomalous, whereas
the installed code computes S_F=exp(-delta/sigma), which is higher when closer
to reference; S_P=1-mean(predicted map). Model card calls S_H anomaly intensity.
This discrepancy requires author/version clarification before deciding score
orientation, thresholding, AUC, or interpreting map polarity. Viewer JPEG masks
may be transcoded and ground-truth convention has not been verified.
