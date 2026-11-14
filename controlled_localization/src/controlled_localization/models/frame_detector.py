"""FRAMEDetector: Forensic Reconstruction-Anomaly Mixture of Encoders.
Combines ConvNeXt visual semantics with a 60-channel SRM+Bayar forensic stream
and a dual-domain gated FPN decoder for state-of-the-art AI-inpainting localization.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn

from .convnext_encoder import ConvNeXtEncoder
from .forensic_encoder import EnhancedForensicEncoder
from .gated_decoder import DualStreamGatedDecoder


class FRAMEDetector(nn.Module):
    """
    Main model for the FRAME architecture.
    Receives ONE image: forward(rgb) accepts float [B, 3, H, W] in [0, 1].
    Returns:
      - 'mask_logits': [B, 1, H, W] (pixel-level inpainting localization)
      - 'det_logits': [B, 1] (image-level manipulation detection score)
      - 'processing_logits': optional [B, R]
    Deployment receives NO original photos, masks, or recipes.
    """

    def __init__(
        self,
        pretrained_semantic: bool = True,
        num_processing_classes: Optional[int] = None,
    ) -> None:
        super().__init__()
        # Semantic stream: ConvNeXt-Tiny (ImageNet pretrained, 7x7 depthwise convs, LayerNorm)
        self.semantic_encoder = ConvNeXtEncoder(
            pretrained=pretrained_semantic,
            freeze_stem_and_stage1=True,
        )

        # Forensic stream: 60-channel SRM (30 fixed) + Bayar (30 learnable) + GroupNorm ResBlocks
        self.forensic_encoder = EnhancedForensicEncoder()

        # Dual-stream gated decoder
        self.decoder = DualStreamGatedDecoder(
            semantic_dims=(768, 384, 192, 96),
            forensic_dims=(192, 144, 96, 48),
            decoder_dim=128,
            num_processing_classes=num_processing_classes,
        )

    def forward(self, rgb: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Input: rgb float32 tensor of shape [B, 3, H, W] with pixel values in [0, 1].
        Returns:
          - 'mask_logits': [B, 1, H, W]
          - 'det_logits': [B, 1]
          - 'processing_logits': [B, R] (optional)
        """
        if rgb.ndim != 4 or rgb.shape[1] != 3:
            raise ValueError(f"Expected input shape [B, 3, H, W], got {rgb.shape}")

        b, c, h, w = rgb.shape

        # Extract multi-scale semantic and forensic feature pyramids
        semantic_feats = self.semantic_encoder(rgb)
        forensic_feats = self.forensic_encoder(rgb)

        # Fuse via adaptive spatial gating
        mask_logits, det_logits, proc_logits = self.decoder(
            semantic_feats=semantic_feats,
            forensic_feats=forensic_feats,
            target_size=(h, w),
        )

        out: Dict[str, torch.Tensor] = {
            "mask_logits": mask_logits,
            "det_logits": det_logits,
        }
        if proc_logits is not None:
            out["processing_logits"] = proc_logits

        return out
