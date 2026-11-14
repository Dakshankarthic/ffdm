"""Mask, change detection, and background audit utilities for Controlled Localization v1.1."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple
import numpy as np
from PIL import Image


@dataclass
class AuditResult:
    sample_id: str
    width: int
    height: int
    mask_pixel_count: int
    mask_area_fraction: float
    inside_mask_l1_diff: Optional[float]
    outside_mask_l1_diff: Optional[float]
    frac_changed_inside: Optional[float]
    change_threshold_tau: float
    is_noop: bool
    status: str
    decision_reason: str


def audit_pair_and_mask(
    canonical_a: np.ndarray,
    canonical_c: Optional[np.ndarray],
    mask: Optional[np.ndarray],
    sample_id: str,
    tau: float = 0.02,  # default 2% pixel intensity difference threshold
    noop_fraction_threshold: float = 0.05,  # if <5% of mask pixels exceed tau, flag as no-op candidate
) -> AuditResult:
    """
    Audit pair A (unedited) and C (edited) with ground truth mask.
    Arrays are expected to be float32 in range [0, 1] with shape [H, W, 3] for images and [H, W] for mask.
    """
    h, w = canonical_a.shape[:2]

    if mask is None:
        return AuditResult(
            sample_id=sample_id,
            width=w,
            height=h,
            mask_pixel_count=0,
            mask_area_fraction=0.0,
            inside_mask_l1_diff=None,
            outside_mask_l1_diff=None,
            frac_changed_inside=None,
            change_threshold_tau=tau,
            is_noop=False,
            status="verified",
            decision_reason="Genuine unedited sample with no mask",
        )

    # Check mask dimensions match image
    if mask.shape[:2] != (h, w):
        raise ValueError(f"Mask shape {mask.shape[:2]} does not match image shape {(h, w)}")

    # Ensure mask is binary [0, 1]
    binary_mask = (mask > 0.5).astype(np.float32)
    mask_pixels = int(np.sum(binary_mask))
    total_pixels = h * w
    mask_frac = float(mask_pixels / total_pixels)

    if canonical_c is None:
        # Standalone mask audit
        return AuditResult(
            sample_id=sample_id,
            width=w,
            height=h,
            mask_pixel_count=mask_pixels,
            mask_area_fraction=mask_frac,
            inside_mask_l1_diff=None,
            outside_mask_l1_diff=None,
            frac_changed_inside=None,
            change_threshold_tau=tau,
            is_noop=False,
            status="verified" if mask_pixels > 0 else "quarantined",
            decision_reason="Standalone mask validated",
        )

    # Compute d(x) = mean_RGB(abs(C - A))
    diff = np.mean(np.abs(canonical_c - canonical_a), axis=-1)  # [H, W]

    if mask_pixels > 0:
        inside_diff = float(np.mean(diff[binary_mask > 0.5]))
        frac_changed = float(np.mean(diff[binary_mask > 0.5] > tau))
    else:
        inside_diff = 0.0
        frac_changed = 0.0

    outside_mask = binary_mask < 0.5
    outside_pixels = int(np.sum(outside_mask))
    if outside_pixels > 0:
        outside_diff = float(np.mean(diff[outside_mask]))
    else:
        outside_diff = 0.0

    is_noop = mask_pixels > 0 and frac_changed < noop_fraction_threshold
    status = "quarantined" if is_noop else "verified"
    reason = (
        f"Quarantined as no-op: only {frac_changed*100:.2f}% of mask pixels exceed tau={tau}"
        if is_noop
        else f"Verified: inside L1={inside_diff:.4f}, outside L1={outside_diff:.4f}, frac_changed={frac_changed:.4f}"
    )

    return AuditResult(
        sample_id=sample_id,
        width=w,
        height=h,
        mask_pixel_count=mask_pixels,
        mask_area_fraction=mask_frac,
        inside_mask_l1_diff=inside_diff,
        outside_mask_l1_diff=outside_diff,
        frac_changed_inside=frac_changed,
        change_threshold_tau=tau,
        is_noop=is_noop,
        status=status,
        decision_reason=reason,
    )

