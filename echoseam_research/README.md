# EchoSeam research prototype

This is an untrained global plus tiled local research control. It takes a single RGB image and produces raw logits and a coverage map. It is not the complete EchoSeam-Net, has no pretrained weights, and its outputs do not detect AI editing reliably. It does not use the original image or ground truth during inference.

Goal: compare global-only, fixed overlapping tiles, and a later proposal-guided local inspection under matched data and compute budgets. Neighbor ring, seam head and wavelet residuals remain hypotheses. The model is deliberately small to enable iteration on the local 8 GB GPU; feasibility and real performance are not established.

The companion baseline agent writes to baseline_work; this research code uses echoseam_research. Full-resolution tiles can be slow or memory-intensive during training. Evaluation and dataset splits need to be established before training or claiming results.
