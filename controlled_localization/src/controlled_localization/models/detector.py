"""Complete Controlled Localization Detector model for Controlled Localization v1.1."""

from __future__ import annotations

from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn

from .rgb_encoder import FrozenBNResNet18Encoder
from .residual_encoder import ResidualForensicEncoder
from .decoder import FPNDecoder


class ControlledLocalizationDetector(nn.Module):
    """
    Main model for Controlled Localization v1.1.
    Receives ONE image: forward(rgb) accepts float [B, 3, H, W] in [0, 1].
    Returns {'mask_logits': [B, 1, H, W], 'processing_logits': optional [B, R]}.
    Deployment receives no originals, masks, or recipes.
    """

    def __init__(
        self,
        pretrained_rgb: bool = True,
        weights_name: Optional[str] = "IMAGENET1K_V1",
        num_processing_classes: Optional[int] = None,
        rgb_only: bool = False,
    ) -> None:
        super().__init__()
        self.rgb_only = rgb_only
        self.rgb_encoder = FrozenBNResNet18Encoder(pretrained=pretrained_rgb, weights_name=weights_name)

        if not rgb_only:
            self.res_encoder = ResidualForensicEncoder()
        else:
            self.res_encoder = None

        self.decoder = FPNDecoder(num_processing_classes=num_processing_classes)

    def forward(self, rgb: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Input: rgb float32 tensor of shape [B, 3, H, W] with pixel values in [0, 1].
        Returns dictionary containing:
          - 'mask_logits': [B, 1, H, W]
          - 'processing_logits': [B, R] (optional, None for T0-T2O)
        """
        if rgb.ndim != 4 or rgb.shape[1] != 3:
            raise ValueError(f"Expected input shape [B, 3, H, W], got {rgb.shape}")

        b, c, h, w = rgb.shape
        rgb_feats = self.rgb_encoder(rgb)

        if not self.rgb_only:
            res_feats = self.res_encoder(rgb)
        else:
            # For RGB-only ablation, create zero dummy residual features
            res_feats = {
                "F2": torch.zeros((b, 32, h // 2, w // 2), device=rgb.device, dtype=rgb.dtype),
                "F4": torch.zeros((b, 64, h // 4, w // 4), device=rgb.device, dtype=rgb.dtype),
                "F8": torch.zeros((b, 96, h // 8, w // 8), device=rgb.device, dtype=rgb.dtype),
                "F16": torch.zeros((b, 128, h // 16, w // 16), device=rgb.device, dtype=rgb.dtype),
            }

        mask_logits, proc_logits = self.decoder(rgb_feats, res_feats, target_size=(h, w))

        out: Dict[str, torch.Tensor] = {"mask_logits": mask_logits}
        if proc_logits is not None:
            out["processing_logits"] = proc_logits

        return out

