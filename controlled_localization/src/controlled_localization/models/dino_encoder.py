"""Frozen DINOv2 Vision Transformer encoder for Controlled Localization.
Extracts rich self-supervised multi-scale patch embeddings.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class FrozenDINOv2Encoder(nn.Module):
    """
    Frozen DINOv2 encoder.
    Zero optimizer parameters, all weights frozen.
    Extracts 4 intermediate representation stages from DINOv2 ViT (blocks 2, 5, 8, 11).
    Interpolates them to standard multi-scale feature pyramid representations:
      - S4: stride 4 [B, C, H // 4, W // 4]
      - S8: stride 8 [B, C, H // 8, W // 8]
      - S16: stride 16 [B, C, H // 16, W // 16]
      - S32: stride 32 [B, C, H // 32, W // 32]
    For ViT-Small/14, C = 384. For ViT-Base/14, C = 768.
    """

    def __init__(
        self,
        arch_name: str = "dinov2_vits14",
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.arch_name = arch_name

        if pretrained:
            self.model = torch.hub.load("facebookresearch/dinov2", arch_name)
        else:
            raise ValueError("DINOv2 must be initialized with pretrained weights")

        # Freeze completely
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()

        self.channels = 384 if "vits" in arch_name else 768
        self.patch_size = 14

        # ImageNet normalization parameters for DINOv2
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)
        )

    def train(self, mode: bool = True) -> "FrozenDINOv2Encoder":
        """Always keep underlying DINOv2 model in eval mode to prevent dropout/running stats drift."""
        super().train(mode)
        self.model.eval()
        return self

    def forward(self, rgb: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Input: rgb [B, 3, H, W] in [0, 1].
        Returns dict with:
          - 'S4': [B, C, H // 4, W // 4]
          - 'S8': [B, C, H // 8, W // 8]
          - 'S16': [B, C, H // 16, W // 16]
          - 'S32': [B, C, H // 32, W // 32]
        """
        b, c, h, w = rgb.shape

        # Normalize with ImageNet statistics
        rgb_norm = (rgb - self.mean) / self.std

        # Pad height and width to nearest multiple of patch_size (14)
        pad_h = (self.patch_size - h % self.patch_size) % self.patch_size
        pad_w = (self.patch_size - w % self.patch_size) % self.patch_size
        if pad_h > 0 or pad_w > 0:
            pad_mode = "reflect" if (h > pad_h and w > pad_w) else "replicate"
            x_in = F.pad(rgb_norm, (0, pad_w, 0, pad_h), mode=pad_mode)
        else:
            x_in = rgb_norm

        with torch.no_grad():
            # Extract intermediate layers [2, 5, 8, 11]
            raw_feats = self.model.get_intermediate_layers(x_in, n=[2, 5, 8, 11], reshape=True)

        f_shallow, f_mid_low, f_mid_high, f_deep = raw_feats

        # If we padded, crop out the corresponding patch ratio, or interpolate directly
        # Direct bilinear interpolation to target spatial resolutions gives smooth, exact alignment:
        h4, w4 = max(1, h // 4), max(1, w // 4)
        h8, w8 = max(1, h // 8), max(1, w // 8)
        h16, w16 = max(1, h // 16), max(1, w // 16)
        h32, w32 = max(1, h // 32), max(1, w // 32)

        s4 = F.interpolate(f_shallow, size=(h4, w4), mode="bilinear", align_corners=False)
        s8 = F.interpolate(f_mid_low, size=(h8, w8), mode="bilinear", align_corners=False)
        s16 = F.interpolate(f_mid_high, size=(h16, w16), mode="bilinear", align_corners=False)
        s32 = F.interpolate(f_deep, size=(h32, w32), mode="bilinear", align_corners=False)

        return {
            "S4": s4,
            "S8": s8,
            "S16": s16,
            "S32": s32,
        }
