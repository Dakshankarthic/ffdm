"""Evaluation metrics, streaming histogram AP, and stratification for Controlled Localization v1.1."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score


@dataclass
class ConfusionMatrix:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0

    @property
    def iou(self) -> float:
        denom = self.tp + self.fp + self.fn
        if denom == 0:
            return 1.0  # Zero union convention: empty GT and empty Pred -> IoU=1.0
        return float(self.tp / denom)

    @property
    def f1(self) -> float:
        denom = 2 * self.tp + self.fp + self.fn
        if denom == 0:
            return 1.0
        return float(2 * self.tp / denom)

    @property
    def precision(self) -> float:
        denom = self.tp + self.fp
        if denom == 0:
            return 0.0
        return float(self.tp / denom)

    @property
    def recall(self) -> float:
        denom = self.tp + self.fn
        if denom == 0:
            return 0.0
        return float(self.tp / denom)


def compute_binary_confusion(
    pred_binary: np.ndarray,
    target_binary: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
) -> ConfusionMatrix:
    """Compute confusion matrix for binary masks."""
    p = pred_binary.astype(bool)
    t = target_binary.astype(bool)

    if valid_mask is not None:
        v = valid_mask.astype(bool)
        p = p & v
        t = t & v

    tp = int(np.logical_and(p, t).sum())
    fp = int(np.logical_and(p, ~t).sum())
    fn = int(np.logical_and(~p, t).sum())
    tn = int(np.logical_and(~p, ~t).sum())

    return ConfusionMatrix(tp=tp, fp=fp, fn=fn, tn=tn)


def assign_edit_tier(edit_fraction: float) -> str:
    """
    Stratification by full-image edit area fraction:
      - (0, 0.005)
      - [0.005, 0.02)
      - [0.02, 0.10)
      - [0.10, 1.0]
      - genuine (0.0)
    """
    if edit_fraction <= 0.0:
        return "genuine"
    elif edit_fraction < 0.005:
        return "<0.5%"
    elif edit_fraction < 0.02:
        return "0.5%-2%"
    elif edit_fraction < 0.10:
        return "2%-10%"
    else:
        return ">10%"


class StreamingHistogramAP:
    """
    Streaming int64 histogram on logit-scale bins for memory-bounded corpus pixel AP.
    Resolves audit finding F11:
      - 4096 edges spanning -20 to +20, plus underflow and overflow bins (4097 bins total).
      - Numerically stable logit transformation.
      - Computes step-wise sum(delta_recall * precision) without storing all corpus pixels in RAM.
    """

    def __init__(self, num_bins: int = 4096, logit_min: float = -20.0, logit_max: float = 20.0) -> None:
        self.num_bins = num_bins
        self.logit_min = logit_min
        self.logit_max = logit_max
        self.bin_edges = np.linspace(logit_min, logit_max, num_bins + 1, dtype=np.float32)

        # Histograms for positive (label=1) and negative (label=0) pixels: 4097 bins
        # Bin 0: < bin_edges[0]
        # Bin 1..num_bins: between bin_edges[i-1] and bin_edges[i]
        # Bin num_bins+1: >= bin_edges[-1]
        self.pos_counts = np.zeros(num_bins + 2, dtype=np.int64)
        self.neg_counts = np.zeros(num_bins + 2, dtype=np.int64)

    def _probs_to_logits(self, probs: np.ndarray) -> np.ndarray:
        p_clipped = np.clip(probs, 1e-9, 1.0 - 1e-9)
        return np.log(p_clipped / (1.0 - p_clipped))

    def update(self, probs: np.ndarray, targets: np.ndarray, valid_mask: Optional[np.ndarray] = None) -> None:
        """Accumulate counts from a single image or tile."""
        p = probs.ravel()
        y = (targets.ravel() > 0.5)

        if valid_mask is not None:
            v = (valid_mask.ravel() > 0.5)
            p = p[v]
            y = y[v]

        if len(p) == 0:
            return

        logits = self._probs_to_logits(p)

        # Bin index: np.digitize returns 1..num_bins for in-range values
        bin_indices = np.digitize(logits, self.bin_edges)

        pos_bins = bin_indices[y]
        neg_bins = bin_indices[~y]

        if len(pos_bins) > 0:
            bc = np.bincount(pos_bins, minlength=self.num_bins + 2)
            self.pos_counts += bc[: self.num_bins + 2]

        if len(neg_bins) > 0:
            bc = np.bincount(neg_bins, minlength=self.num_bins + 2)
            self.neg_counts += bc[: self.num_bins + 2]

    def compute_ap(self) -> float:
        """
        Compute Average Precision = sum(delta_recall * precision) from descending bins.
        """
        total_pos = int(np.sum(self.pos_counts))
        if total_pos == 0:
            return 0.0

        # Iterate from highest score bin (num_bins + 1) down to 0
        desc_pos = self.pos_counts[::-1]
        desc_neg = self.neg_counts[::-1]

        cum_tp = np.cumsum(desc_pos, dtype=np.float64)
        cum_fp = np.cumsum(desc_neg, dtype=np.float64)

        recalls = cum_tp / float(total_pos)
        denom = cum_tp + cum_fp
        precisions = np.where(denom > 0, cum_tp / np.maximum(denom, 1.0), 1.0)

        # Prepend 0 to recalls for delta calculation
        r_prev = np.concatenate(([0.0], recalls[:-1]))
        delta_recall = recalls - r_prev

        ap = float(np.sum(delta_recall * precisions))
        return min(max(ap, 0.0), 1.0)
