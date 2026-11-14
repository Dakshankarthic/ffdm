"""Directed view consistency and outside-mask consistency losses for Controlled Localization v1.1."""

from __future__ import annotations

from typing import Optional, Tuple
import torch
import torch.nn.functional as F


def compute_bernoulli_kl(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Directed Bernoulli KL from teacher to student:
      KL(q || sigmoid(z)) = BCEWithLogits(z, q) - H(q)
    where q = sigmoid(teacher_logits).detach() and student z = student_logits.
    No gradient through teacher. All reductions in float32.
    """
    z = torch.clamp(student_logits.float(), -30.0, 30.0)
    q = torch.sigmoid(torch.clamp(teacher_logits.float(), -30.0, 30.0)).detach()

    # Stable BCEWithLogits: -[q*log(p) + (1-q)*log(1-p)]
    bce = F.binary_cross_entropy_with_logits(z, q, reduction="none")

    # Teacher binary entropy: H(q) = -[q*log(q) + (1-q)*log(1-q)]
    q_safe = torch.clamp(q, 1e-7, 1.0 - 1e-7)
    entropy = -(q_safe * torch.log(q_safe) + (1.0 - q_safe) * torch.log(1.0 - q_safe))

    kl = torch.clamp(bce - entropy, min=0.0)

    if mask is not None:
        m = (mask.float() > 0.5)
        if m.any():
            return kl[m].mean()
        else:
            return torch.tensor(0.0, device=z.device, dtype=torch.float32)
    else:
        return kl.mean()


def compute_directed_consistency_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    gt_mask: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Computes directed view consistency between clean teacher and degraded student.
    Reduces foreground and background separately, equally when both exist.
    """
    if valid_mask is not None:
        v = valid_mask.float() > 0.5
    else:
        v = torch.ones_like(student_logits, dtype=torch.bool)

    pos = (gt_mask.float() > 0.5) & v
    neg = (gt_mask.float() <= 0.5) & v

    has_pos = pos.any()
    has_neg = neg.any()

    if has_pos and has_neg:
        kl_pos = compute_bernoulli_kl(student_logits, teacher_logits, mask=pos)
        kl_neg = compute_bernoulli_kl(student_logits, teacher_logits, mask=neg)
        return 0.5 * (kl_pos + kl_neg)
    elif has_neg:
        return compute_bernoulli_kl(student_logits, teacher_logits, mask=neg)
    elif has_pos:
        return compute_bernoulli_kl(student_logits, teacher_logits, mask=pos)
    else:
        return torch.tensor(0.0, device=student_logits.device, dtype=torch.float32)


def compute_outside_consistency_loss(
    edited_logits: torch.Tensor,
    unedited_logits: torch.Tensor,
    dilated_mask: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Outside-mask consistency (T2O).
    Directed KL from unedited (teacher) to edited (student), evaluated strictly
    OUTSIDE the dilated edit mask and ignore regions.
    """
    # Outside mask: dilated_mask == 0
    outside = (dilated_mask.float() < 0.5)

    if valid_mask is not None:
        outside = outside & (valid_mask.float() > 0.5)

    if outside.any():
        return compute_bernoulli_kl(
            student_logits=edited_logits,
            teacher_logits=unedited_logits,
            mask=outside,
        )
    else:
        return torch.tensor(0.0, device=edited_logits.device, dtype=torch.float32)

