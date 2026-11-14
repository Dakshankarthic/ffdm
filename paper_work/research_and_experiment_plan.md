# Research plan: small AI-edited region localization

Status: WORKING RESEARCH PLAN, 13 November 2026. No detector has been trained or evaluated by this project. The user has a local RTX 2070 SUPER (8 GB) and access to Kaggle GPUs. Actual Kaggle allocation and usable time have not been confirmed.

## Working question
Can a high-resolution local inspection stage improve localization of very small AI-edited regions when the editor is unseen during training, while controlling false alarms on genuine photographs?

This is a hypothesis, not a demonstrated contribution or novelty claim. The model inspects one image at inference. Original images, masks and provenance support dataset preparation and evaluation only.

## Venue and timing
OpenReview FDDM submission invitation: https://openreview.net/group?id=ICVGIP.in/2026/Workshop/FDDM
The public invitation's duedate is 14 November 2026 18:30 UTC = 15 November 2026 00:00 IST (end of 14 November). OpenReview records an expiration 30 minutes later. Confirm requirements, extensions and cutoff with the venue. A research plan or one-image demo is not a completed experimental paper. Do not fabricate results. Main conference deadlines are not workshop deadlines.

## Closest prior work
- FUSED (preprint): https://arxiv.org/html/2608.28302v1 . Local inpainting detection/localization; cross-generator and global reconstruction artifact analysis. Compare matching protocols; authors used two H100s for OpenSDID.
- IFA-Net (CVPR 2026): https://openaccess.thecvf.com/content/CVPR2026/papers/Zhang_Detecting_AI-Generated_Forgeries_via_Iterative_Manifold_Deviation_Amplification_CVPR_2026_paper.pdf . Two-stage prediction-guided refinement already exists. Our high-resolution crop method would need a specific distinguishable contribution and controlled evaluation.
- TruFor (CVPR 2023): https://github.com/grip-unina/TruFor . General localization and reliability output; its README states that download/use signifies agreement to its custom license. This has not been accepted on the user's behalf.
- DeCLIP (WACV 2025): https://github.com/bit-ml/DeCLIP . Localizer with released code; official checkpoint links have presented a sign-in barrier. Local source under baselines/DeCLIP is present; no pretrained inference completed.
- DiffusionPrint (CVPRW 2026): https://openaccess.thecvf.com/content/CVPR2026W/AIMS/html/Giakoumoglou_DiffusionPrint_Learning_Generative_Fingerprints_for_Diffusion-Based_Inpainting_Localization_CVPRW_2026_paper.html . Patch-level forensic features and VAE reconstruction are close prior art.
- GIFTBench / ForenScope (preprint): https://arxiv.org/abs/2610.01778 . Multi-axis benchmark and multi-scale localization. Not directly comparable without matching training and test settings.

## Candidate datasets
1. OpenSDID: https://github.com/iamwangyabin/OpenSDI . Candidate primary public benchmark, multiple generators; CC BY-SA 4.0 and academic-use notice reported by authors. Verify dataset component licenses, access, exact masks, source identity and leakage. Generator-held-out does not automatically mean source-held-out. No data downloaded yet.
2. TGIF2: https://github.com/IDLabMedia/tgif-dataset . Candidate complementary evaluation with masks and original/edited data; verify access, reuse conditions, <2% mask counts and source split separation.
3. AutoSplice: https://github.com/shanface33/AutoSplice_Dataset . Candidate external and JPEG evaluation; authors restrict to academic research by educational institutions for noncommercial purposes; confirm applicability and terms.

For all data, partition by source image before deriving samples. Log identities, edit provenance, operation, editor, license, requested mask and reviewed ground-truth semantics. Check whether datasets already meet this requirement; do not silently alter published protocols.

## Minimum evaluation design
- Image detection: ROC-AUC, F1, precision, recall, false-positive rate on genuine images; accuracy with class balance stated.
- Localization: per-image pixel IoU and pixel F1 on edited images. State threshold, averaging, annotation conventions and handling of empty masks.
- Tiny-edit strata: <2%, 2–10%, >10% by ground-truth area, counts and uncertainty intervals; 2% is a proposed cutoff, not a proven threshold.
- Cross-generator and external dataset conditions, with held-out source scenes wherever identifiable; tune thresholds and checkpoints on validation only.
- Compression/resizing tests, and positives/negatives matched by processing to expose shortcuts.
- Record latency, peak GPU memory, image size, checkpoint, seed and conditions. Multiple seeds if feasible.
- For proposal-based zoom: Recall@K with a predefined criterion for crop coverage, compute and number of crops; compare global-only, equal-budget tiled, and selected-crop variants. A crop cannot refine an edit it never covers.

## Staged experimental variants
B0 Existing baseline (starting with available/checkpoint-compatible method). B1 simple global localizer. B2 B1 + fixed tiled high-resolution inference. B3 B1 + proposal-driven local refinement. After establishing these, individually ablate neighbor-ring, residual/wavelet features, reference selection, and boundary supervision. Use identical splits/resolution and specify equal-compute comparisons. No accuracy improvements are assumed.

## Review of EchoSeam proposal
See ../project_records/EchoSeam_Net_user_proposal.txt. It is a proposed method. Neighbor differences can be genuine texture; a low-scoring reference is not guaranteed clean; two stages are prior art; DINOv2 has 14-pixel patches and does not natively yield 64x64 patches from 512 pixels; a fixed weighted score is not calibrated; 2xT4 batch/memory estimates are unverified. Specify crop coordinate geometry, mixed GT/model crop training, no-edit cases, multiple edits and masks in native coordinates before implementation.

## Current next actions
1. Resolve venue timing and submission requirements. Request user confirmation only for missing facts such as an existing draft and compute access; no submission is authorized by this plan.
2. Get a license-compatible, accessible pretrained baseline running; preserve license terms and checkpoint provenance.
3. Prepare a small licensed validation dataset with reviewed masks and source/editor IDs, then expand. A single globe example cannot establish accuracy.
4. Run and record baseline, compare equal-budget high-resolution variants, and only then state measured results and a defensible contribution.

## Diagnostic data access check (13 November 2026)
The Hugging Face dataset viewer API responded for OpenSDI_train, sd15 split,
showing key, image, mask and label fields and 200,824 rows. Three consecutive
partial-edit training examples and viewer-provided image/mask JPEGs were saved
to ../research_data/opensdi_viewer_preview/ with source keys, hashes and sizes.
These three are convenience samples, all labeled fake, not an evaluation set;
the viewer may transcode assets. JPEG mask semantics, original dataset files,
source overlap and licenses require checking before numerical evaluation.
