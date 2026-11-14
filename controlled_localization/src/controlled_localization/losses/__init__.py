"""Loss functions for Controlled Localization v1.1."""

from .segmentation import compute_supervised_segmentation_loss
from .consistency import compute_directed_consistency_loss, compute_outside_consistency_loss

__all__ = [
    "compute_supervised_segmentation_loss",
    "compute_directed_consistency_loss",
    "compute_outside_consistency_loss",
]

