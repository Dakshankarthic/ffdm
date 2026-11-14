"""Unit tests for confusion matrices, streaming AP histogram, and pooling policies."""

import pytest
import numpy as np
from sklearn.metrics import average_precision_score

from controlled_localization.metrics import (
    compute_binary_confusion,
    assign_edit_tier,
    StreamingHistogramAP,
    ConfusionMatrix,
)
from controlled_localization.policy import compute_image_score


def test_hand_computed_confusion_matrix():
    """Verify hand-computed confusion cases, IoU, and F1."""
    # 4 pixels: TP, FP, FN, TN
    pred = np.array([1, 1, 0, 0])
    target = np.array([1, 0, 1, 0])

    cm = compute_binary_confusion(pred, target)
    assert cm.tp == 1
    assert cm.fp == 1
    assert cm.fn == 1
    assert cm.tn == 1

    # IoU = TP / (TP + FP + FN) = 1 / 3 = 0.333333
    assert abs(cm.iou - 1.0 / 3.0) < 1e-5

    # F1 = 2*TP / (2*TP + FP + FN) = 2 / 4 = 0.50
    assert abs(cm.f1 - 0.5) < 1e-5


def test_zero_union_empty_masks():
    """Empty GT and empty Pred yields IoU=1.0 and F1=1.0."""
    pred = np.array([0, 0, 0, 0])
    target = np.array([0, 0, 0, 0])

    cm = compute_binary_confusion(pred, target)
    assert cm.iou == 1.0
    assert cm.f1 == 1.0


def test_streaming_ap_vs_exact_sklearn():
    """
    Streaming histogram AP vs exact sklearn AP on a dataset with sparse positives and tails.
    Target absolute error <= 0.001 (Section 08 & Section 10).
    """
    rng = np.random.RandomState(42)
    # Generate 50,000 synthetic predictions with 1% positive fraction (sparse positives)
    num_samples = 50000
    y_true = (rng.rand(num_samples) < 0.01).astype(int)

    # Probabilities with signal: positives have higher mean
    y_scores = np.clip(
        rng.beta(0.5, 5.0, size=num_samples) + y_true * rng.beta(2.0, 1.0, size=num_samples) * 0.5,
        1e-6,
        1.0 - 1e-6,
    )

    # Exact sklearn AP
    exact_ap = float(average_precision_score(y_true, y_scores))

    # Streaming histogram AP
    stream_ap = StreamingHistogramAP(num_bins=4096, logit_min=-20.0, logit_max=20.0)
    # Stream in chunks
    chunk_size = 10000
    for i in range(0, num_samples, chunk_size):
        stream_ap.update(y_scores[i : i + chunk_size], y_true[i : i + chunk_size])

    approx_ap = stream_ap.compute_ap()
    error = abs(approx_ap - exact_ap)

    print(f"Exact AP: {exact_ap:.5f}, Streaming AP: {approx_ap:.5f}, Abs Error: {error:.5f}")
    assert error <= 0.001, f"Streaming AP error {error:.5f} exceeds tolerance 0.001!"


def test_pooling_images_smaller_than_k():
    """Fixed-K pooling works robustly for images smaller than K."""
    prob_map = np.array([[0.1, 0.9], [0.4, 0.6]], dtype=np.float32)  # 4 pixels total
    score = compute_image_score(prob_map, pooling_method="top_k", k=2048)
    # Should average all 4 pixels since N < K
    assert abs(score - 0.5) < 1e-5

