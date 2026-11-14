# Working paper outline — not submission-ready

Working title: Evaluating High-Resolution Local Inspection for Small AI-Edited Image Localization

## Status
Research outline only. No trained EchoSeam-Net model, quantitative baseline, evaluation tables, or validated novelty claims currently exist. Do not submit as a completed experimental paper.

## Abstract structure (to complete after results)
Problem and why small edits matter; limitation supported by measured baseline; proposed high-resolution inspection and how it differs from existing two-stage work; datasets and held-out protocols; measured results and limitations. Leave numerical results blank until actually produced.

## 1. Introduction
Define the one-image input and image-level detection plus pixel-level localization outputs. Describe resolution loss as a hypothesis and support any claimed gap with literature and baseline evidence. State contributions only after studies are completed.

## 2. Related work
AI inpainting localization (FUSED, DiffusionPrint); prediction-guided refinement (IFA-Net); forensic and reliability baselines (TruFor); generalization and evaluation (OpenSDI, TGIF2, GIFTBench). Compare exact protocols and avoid universal SOTA claims.

## 3. Method
Define global localizer, proposal selection, native-resolution crop mapping, local refinement, optional confidence/reference handling and fusion. Specify training with both GT-guided and predicted crops, but inference from one image only. Distinguish implemented model from candidate components.

## 4. Experimental design
Datasets, licensing, source grouped splits, editor-held-out evaluation, mask semantics, 2% area strata and sample counts, implementation details, thresholds chosen on validation, baselines, equal-compute tiles and ablations.

## 5. Results
PENDING. Include detection, per-image pixel IoU/F1, tiny-region metrics, crop recall@K, false positive rates, multiple generator and compression conditions, compute, uncertainty intervals and representative failures.

## 6. Limitations and conclusion
Discuss proposal misses, domain shift, imperfect masks, small samples, data/compute limits, potential false accusations and results supported by the evidence.
