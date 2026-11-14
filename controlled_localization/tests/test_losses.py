"""Unit tests for segmentation, directed consistency, and outside-mask losses."""

import pytest
import torch
import torch.nn.functional as F

from controlled_localization.losses.segmentation import compute_supervised_segmentation_loss
from controlled_localization.losses.consistency import (
    compute_bernoulli_kl,
    compute_directed_consistency_loss,
    compute_outside_consistency_loss,
)


def test_segmentation_loss_edge_cases():
    """All-negative, all-positive, tiny positive and all-ignore inputs are finite; no divide-by-zero."""
    # 1. All-negative
    z_neg = torch.randn(2, 1, 64, 64)
    y_neg = torch.zeros_like(z_neg)
    loss, bce, dice = compute_supervised_segmentation_loss(z_neg, y_neg)
    assert torch.isfinite(loss)
    assert dice.item() == 0.0  # Empty mask has Dice=0 per spec

    # 2. All-positive
    z_pos = torch.randn(2, 1, 64, 64)
    y_pos = torch.ones_like(z_pos)
    loss, bce, dice = compute_supervised_segmentation_loss(z_pos, y_pos)
    assert torch.isfinite(loss)
    assert dice.item() >= 0.0

    # 3. Tiny positive (1 pixel out of 4096)
    z_tiny = torch.randn(1, 1, 64, 64)
    y_tiny = torch.zeros_like(z_tiny)
    y_tiny[0, 0, 10, 10] = 1.0
    loss, bce, dice = compute_supervised_segmentation_loss(z_tiny, y_tiny)
    assert torch.isfinite(loss)
    assert dice.item() > 0.0

    # 4. All-ignore
    z_ign = torch.randn(1, 1, 64, 64)
    y_ign = torch.zeros_like(z_ign)
    valid_none = torch.zeros_like(z_ign)
    loss, bce, dice = compute_supervised_segmentation_loss(z_ign, y_ign, valid_mask=valid_none)
    assert torch.isfinite(loss)
    assert loss.item() == 0.0


def test_consistency_teacher_detached_gradients():
    """
    Detached consistency teacher has NO consistency gradient; student does.
    Clean view still receives gradient from its supervised segmentation loss (Section 10).
    """
    z_clean = torch.randn(1, 1, 32, 32, requires_grad=True)
    z_degraded = torch.randn(1, 1, 32, 32, requires_grad=True)
    gt_mask = torch.zeros(1, 1, 32, 32)

    # Consistency loss alone: A -> B
    l_view = compute_directed_consistency_loss(
        student_logits=z_degraded,
        teacher_logits=z_clean,
        gt_mask=gt_mask,
    )
    l_view.backward()

    # Student MUST receive gradient
    assert z_degraded.grad is not None
    assert torch.isfinite(z_degraded.grad).all()
    assert (z_degraded.grad.abs().sum() > 0.0)

    # Teacher MUST NOT receive gradient from consistency loss alone
    assert z_clean.grad is None, "Teacher logits received unwanted gradient from consistency loss!"

    # Now verify clean view STILL receives gradient when supervised loss is included
    z_clean.grad = None
    z_degraded.grad = None

    l_seg_clean, _, _ = compute_supervised_segmentation_loss(z_clean, gt_mask)
    l_view = compute_directed_consistency_loss(z_degraded, z_clean, gt_mask)
    total_loss = l_seg_clean + l_view
    total_loss.backward()

    assert z_clean.grad is not None, "Clean view failed to receive gradient from supervised loss!"
    assert z_degraded.grad is not None


def test_identical_predictions_zero_kl():
    """Identical predictions yield near-zero KL; extreme logits remain finite (Section 10)."""
    z = torch.randn(1, 1, 32, 32)
    kl = compute_bernoulli_kl(student_logits=z, teacher_logits=z)
    assert torch.isfinite(kl)
    assert kl.item() < 1e-5, f"KL between identical distributions should be ~0, got {kl.item()}"

    # Extreme logits (-50 and +50)
    z_extreme_stu = torch.tensor([[-50.0, 50.0], [50.0, -50.0]])
    z_extreme_tea = torch.tensor([[50.0, -50.0], [-50.0, 50.0]])
    kl_extreme = compute_bernoulli_kl(z_extreme_stu, z_extreme_tea)
    assert torch.isfinite(kl_extreme)

