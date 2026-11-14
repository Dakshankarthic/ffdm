"""Supervised balanced BCE and Dice segmentation loss for Controlled Localization v1.1."""

from __future__ import annotations

from typing import Optional, Tuple
import torch
import torch.nn.functional as F


def compute_supervised_segmentation_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Balanced BCE + Dice loss per image/tile.
    All reductions performed in float32.

    Parameters:
      logits: [B, 1, H, W] raw model output logits.
      targets: [B, 1, H, W] ground truth binary labels {0, 1}.
      valid_mask: [B, 1, H, W] binary mask where 1=valid pixel, 0=ignore/padded.

    Returns:
      (total_loss, bce_loss, dice_loss)
    """
    # Ensure float32 calculations and clamp extreme logits for numerical stability
    z = torch.clamp(logits.float(), -30.0, 30.0)
    y = targets.float()

    if valid_mask is not None:
        v = valid_mask.float()
    else:
        v = torch.ones_like(z)

    b, c, h, w = z.shape

    batch_bce = []
    batch_dice = []

    for i in range(b):
        z_i = z[i, 0]
        y_i = y[i, 0]
        v_i = v[i, 0] > 0.5

        if not v_i.any():
            batch_bce.append(torch.tensor(0.0, device=z.device, dtype=torch.float32))
            batch_dice.append(torch.tensor(0.0, device=z.device, dtype=torch.float32))
            continue

        z_valid = z_i[v_i]
        y_valid = y_i[v_i]

        pos_mask = y_valid > 0.5
        neg_mask = ~pos_mask

        has_pos = pos_mask.any()
        has_neg = neg_mask.any()

        # Balanced BCE:
        # positive_loss = mean(softplus(-z)) over valid positive pixels
        # negative_loss = mean(softplus(z)) over valid negative pixels
        if has_pos and has_neg:
            pos_loss = F.softplus(-z_valid[pos_mask]).mean()
            neg_loss = F.softplus(z_valid[neg_mask]).mean()
            bce = 0.5 * (pos_loss + neg_loss)
        elif has_pos:
            bce = F.softplus(-z_valid[pos_mask]).mean()
        elif has_neg:
            bce = F.softplus(z_valid[neg_mask]).mean()
        else:
            bce = torch.tensor(0.0, device=z.device, dtype=torch.float32)

        # Dice loss: only computed when there are valid positive pixels
        if has_pos:
            p_valid = torch.sigmoid(z_valid)
            intersection = (p_valid * y_valid).sum()
            cardinality = p_valid.sum() + y_valid.sum()
            dice = 1.0 - (2.0 * intersection + 1e-6) / (cardinality + 1e-6)
        else:
            # For empty-mask tiles Dice=0; retain BCE, do not reward empty prediction as perfect in metrics
            dice = torch.tensor(0.0, device=z.device, dtype=torch.float32)

        batch_bce.append(bce)
        batch_dice.append(dice)

    mean_bce = torch.stack(batch_bce).mean()
    mean_dice = torch.stack(batch_dice).mean()
    total_lseg = mean_bce + mean_dice

    return total_lseg, mean_bce, mean_dice

