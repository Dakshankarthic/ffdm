"""Single-Backbone ConvNeXt-Tiny FRAME Architecture (Track B).
Implements the unified single-backbone dual-view forensic architecture:
  - Single shared ImageNet-pretrained ConvNeXt-Tiny encoder (28M params)
  - 64-channel lightweight Feature Pyramid Network (FPN) decoder
  - Global scene classification head (from C32 GAP)
  - Pixel localization mask head (from P4)
  - Dual-view inference (Global resized 384x384 + Overlapping native 384x384 tiles)
  - 2D Hann window probability reconstruction with positive floor (eps=0.05)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from controlled_localization.models.convnext_encoder import ConvNeXtEncoder


def make_proj(in_channels: int, out_channels: int = 64) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
        nn.GroupNorm(num_groups=8, num_channels=out_channels),
        nn.GELU(),
    )


def make_smooth(channels: int = 64) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
        nn.GroupNorm(num_groups=8, num_channels=channels),
        nn.GELU(),
    )


class ConvNeXtTinyFPNDecoder(nn.Module):
    """Lightweight 64-channel Feature Pyramid Network decoder for ConvNeXt-Tiny."""

    def __init__(self, decoder_dim: int = 64) -> None:
        super().__init__()
        self.decoder_dim = decoder_dim

        # Lateral projections: C32 (768->64), C16 (384->64), C8 (192->64), C4 (96->64)
        self.proj32 = make_proj(768, decoder_dim)
        self.proj16 = make_proj(384, decoder_dim)
        self.proj8 = make_proj(192, decoder_dim)
        self.proj4 = make_proj(96, decoder_dim)

        # 3x3 smoothing layers
        self.smooth16 = make_smooth(decoder_dim)
        self.smooth8 = make_smooth(decoder_dim)
        self.smooth4 = make_smooth(decoder_dim)

        # 1x1 mask logits head
        self.mask_head = nn.Conv2d(decoder_dim, 1, kernel_size=1)

    def forward(
        self, feats: Dict[str, torch.Tensor], target_size: Tuple[int, int]
    ) -> torch.Tensor:
        c4, c8, c16, c32 = feats["C4"], feats["C8"], feats["C16"], feats["C32"]

        p32 = self.proj32(c32)

        p16_lat = self.proj16(c16)
        up32 = F.interpolate(p32, size=p16_lat.shape[-2:], mode="bilinear", align_corners=False)
        p16 = self.smooth16(p16_lat + up32)

        p8_lat = self.proj8(c8)
        up16 = F.interpolate(p16, size=p8_lat.shape[-2:], mode="bilinear", align_corners=False)
        p8 = self.smooth8(p8_lat + up16)

        p4_lat = self.proj4(c4)
        up8 = F.interpolate(p8, size=p4_lat.shape[-2:], mode="bilinear", align_corners=False)
        p4 = self.smooth4(p4_lat + up8)

        logits = self.mask_head(p4)
        logits = F.interpolate(logits, size=target_size, mode="bilinear", align_corners=False)
        return logits


class ConvNeXtTinyFRAMEDetector(nn.Module):
    """
    Track B: Single-Backbone FRAME Detector.
    Encodes both global resized view and local native crops using the SAME shared ConvNeXt-Tiny.
    Total parameters: ~28.5M.
    """

    def __init__(
        self,
        pretrained: bool = True,
        freeze_stem_and_stage1: bool = True,
        decoder_dim: int = 64,
    ) -> None:
        super().__init__()
        self.encoder = ConvNeXtEncoder(
            pretrained=pretrained,
            freeze_stem_and_stage1=freeze_stem_and_stage1,
        )
        self.decoder = ConvNeXtTinyFPNDecoder(decoder_dim=decoder_dim)

        # Global scene classification head (operates on C32 of resized whole image)
        self.global_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(768, 64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

        # Local crop presence head (operates on C32 of native crops)
        self.crop_presence_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(768, 64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def forward(
        self, rgb: torch.Tensor, return_crop_det: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Standard forward pass on a batch of images [B, 3, H, W].
        Returns:
          - 'mask_logits': [B, 1, H, W]
          - 'det_logits': [B, 1] (whole-image classification logits)
          - 'crop_det_logits': [B, 1] (optional local crop presence logits)
        """
        b, c, h, w = rgb.shape
        feats = self.encoder(rgb)
        mask_logits = self.decoder(feats, target_size=(h, w))
        det_logits = self.global_head(feats["C32"])

        out = {
            "mask_logits": mask_logits,
            "det_logits": det_logits,
        }
        if return_crop_det:
            out["crop_det_logits"] = self.crop_presence_head(feats["C32"])
        return out


def make_2d_hann_window(height: int, width: int, eps: float = 0.05) -> np.ndarray:
    """Creates a 2D separable Hann window with positive boundary floor."""
    wy = np.sin(np.pi * (np.arange(height) + 0.5) / height) ** 2
    wx = np.sin(np.pi * (np.arange(width) + 0.5) / width) ** 2
    w2d = np.outer(wy, wx) + eps
    return w2d.astype(np.float32)


def reconstruct_full_canvas_hann(
    tile_probs: List[np.ndarray],
    coords: List[Tuple[int, int]],
    full_w: int,
    full_h: int,
    tile_size: int = 384,
    eps: float = 0.05,
) -> np.ndarray:
    """Reconstructs full canvas probability map via 2D Hann window blending."""
    canvas = np.zeros((full_h, full_w), dtype=np.float32)
    weights = np.zeros((full_h, full_w), dtype=np.float32)
    hann = make_2d_hann_window(tile_size, tile_size, eps=eps)

    for prob, (x, y) in zip(tile_probs, coords):
        tx_end = min(full_w, x + tile_size)
        ty_end = min(full_h, y + tile_size)
        sub_h = ty_end - y
        sub_w = tx_end - x

        w_sub = hann[:sub_h, :sub_w]
        p_sub = prob[:sub_h, :sub_w]

        canvas[y:ty_end, x:tx_end] += p_sub * w_sub
        weights[y:ty_end, x:tx_end] += w_sub

    weights = np.maximum(weights, 1e-6)
    full_map = np.clip(canvas / weights, 0.0, 1.0)
    return full_map
