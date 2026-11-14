"""Dual-Domain Gated FPN Decoder for Controlled Localization.
Features adaptive spatial gating between Semantic (ConvNeXt / DINOv2) and Forensic (SRM+Bayar) representations.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


def make_proj(in_channels: int, out_channels: int = 128) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
        nn.GroupNorm(num_groups=8, num_channels=out_channels),
        nn.GELU(),
    )


def make_smooth(channels: int = 128) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
        nn.GroupNorm(num_groups=8, num_channels=channels),
        nn.GELU(),
    )


class SpatialGate(nn.Module):
    """
    Adaptive spatial gating between semantic and forensic features.
    Computes a per-channel attention gate alpha in [0, 1].
    alpha -> 1 favors semantic features; alpha -> 0 favors forensic features.
    """

    def __init__(self, channels: int = 128) -> None:
        super().__init__()
        self.gate_conv = nn.Sequential(
            nn.Conv2d(channels * 2, channels // 2, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=4, num_channels=channels // 2),
            nn.GELU(),
            nn.Conv2d(channels // 2, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, semantic: torch.Tensor, forensic: torch.Tensor) -> torch.Tensor:
        cat_feats = torch.cat([semantic, forensic], dim=1)
        gate = self.gate_conv(cat_feats)
        return gate * semantic + (1.0 - gate) * forensic


class DualStreamGatedDecoder(nn.Module):
    """
    128-channel Top-Down FPN Decoder with Gated Dual-Domain Fusion.
    Integrates:
      - Semantic stream: 4 stages at strides 32, 16, 8, 4
      - Forensic stream: 4 stages at strides 16, 8, 4, 2
    Outputs:
      - mask_logits: [B, 1, H, W]
      - det_logits: [B, 1]
    """

    def __init__(
        self,
        semantic_dims: Tuple[int, int, int, int] = (768, 384, 192, 96),  # (s32, s16, s8, s4)
        forensic_dims: Tuple[int, int, int, int] = (192, 144, 96, 48),    # (f16, f8, f4, f2)
        decoder_dim: int = 128,
        num_processing_classes: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.decoder_dim = decoder_dim
        s32_dim, s16_dim, s8_dim, s4_dim = semantic_dims
        f16_dim, f8_dim, f4_dim, f2_dim = forensic_dims

        # Projections for Semantic branch
        self.proj_s32 = make_proj(s32_dim, decoder_dim)
        self.proj_s16 = make_proj(s16_dim, decoder_dim)
        self.proj_s8 = make_proj(s8_dim, decoder_dim)
        self.proj_s4 = make_proj(s4_dim, decoder_dim)

        # Projections for Forensic branch
        self.proj_f16 = make_proj(f16_dim, decoder_dim)
        self.proj_f8 = make_proj(f8_dim, decoder_dim)
        self.proj_f4 = make_proj(f4_dim, decoder_dim)
        self.proj_f2 = make_proj(f2_dim, decoder_dim)

        # Adaptive Gating Modules
        self.gate16 = SpatialGate(decoder_dim)
        self.gate8 = SpatialGate(decoder_dim)
        self.gate4 = SpatialGate(decoder_dim)

        # Smoothing layers after fusion + upsampling addition
        self.smooth16 = make_smooth(decoder_dim)
        self.smooth8 = make_smooth(decoder_dim)
        self.smooth4 = make_smooth(decoder_dim)

        # Stride 2 fusion with high-res forensic features
        self.fuse2 = nn.Sequential(
            nn.Conv2d(decoder_dim * 2, decoder_dim, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=8, num_channels=decoder_dim),
            nn.GELU(),
            nn.Conv2d(decoder_dim, decoder_dim // 2, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=4, num_channels=decoder_dim // 2),
            nn.GELU(),
        )

        # Pixel mask head
        self.mask_head = nn.Conv2d(decoder_dim // 2, 1, kernel_size=1)

        # Image-level detection classification head (pooling P16, P32, and mask features)
        self.det_head = nn.Sequential(
            nn.Linear(decoder_dim * 2 + 1, 64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

        # Optional processing classifier head
        self.num_processing_classes = num_processing_classes
        if num_processing_classes is not None and num_processing_classes > 0:
            self.proc_classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(decoder_dim, 64),
                nn.GELU(),
                nn.Linear(64, num_processing_classes),
            )
        else:
            self.proc_classifier = None

    def forward(
        self,
        semantic_feats: Dict[str, torch.Tensor],
        forensic_feats: Dict[str, torch.Tensor],
        target_size: Tuple[int, int],
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Returns:
          - mask_logits: [B, 1, H, W]
          - det_logits: [B, 1]
          - proc_logits: [B, R] or None
        """
        # Support both 'C' keys (ConvNeXt) and 'S' keys (DINOv2)
        s32 = semantic_feats.get("C32", semantic_feats.get("S32"))
        s16 = semantic_feats.get("C16", semantic_feats.get("S16"))
        s8 = semantic_feats.get("C8", semantic_feats.get("S8"))
        s4 = semantic_feats.get("C4", semantic_feats.get("S4"))

        f16 = forensic_feats["F16"]
        f8 = forensic_feats["F8"]
        f4 = forensic_feats["F4"]
        f2 = forensic_feats["F2"]

        # Top-down FPN starting from stride 32
        p32 = self.proj_s32(s32)

        # Stride 16
        ps16 = self.proj_s16(s16)
        pf16 = self.proj_f16(f16)
        fused16 = self.gate16(ps16, pf16)
        up32 = F.interpolate(p32, size=fused16.shape[-2:], mode="bilinear", align_corners=False)
        p16 = self.smooth16(fused16 + up32)

        # Stride 8
        ps8 = self.proj_s8(s8)
        pf8 = self.proj_f8(f8)
        fused8 = self.gate8(ps8, pf8)
        up16 = F.interpolate(p16, size=fused8.shape[-2:], mode="bilinear", align_corners=False)
        p8 = self.smooth8(fused8 + up16)

        # Stride 4
        ps4 = self.proj_s4(s4)
        pf4 = self.proj_f4(f4)
        fused4 = self.gate4(ps4, pf4)
        up8 = F.interpolate(p8, size=fused4.shape[-2:], mode="bilinear", align_corners=False)
        p4 = self.smooth4(fused4 + up8)

        # Stride 2
        up4 = F.interpolate(p4, size=f2.shape[-2:], mode="bilinear", align_corners=False)
        pf2 = self.proj_f2(f2)
        p2 = self.fuse2(torch.cat([up4, pf2], dim=1))

        # Pixel mask logits at stride 1
        mask_stride2 = self.mask_head(p2)
        mask_logits = F.interpolate(mask_stride2, size=target_size, mode="bilinear", align_corners=False)

        # Image-level detection score
        pool16 = F.adaptive_avg_pool2d(p16, 1).flatten(1)
        pool32 = F.adaptive_avg_pool2d(p32, 1).flatten(1)
        mask_sig_max = torch.sigmoid(mask_logits).amax(dim=(-2, -1))  # [B, 1]
        det_input = torch.cat([pool16, pool32, mask_sig_max], dim=1)
        det_logits = self.det_head(det_input)

        proc_logits = None
        if self.proc_classifier is not None:
            proc_logits = self.proc_classifier(p16)

        return mask_logits, det_logits, proc_logits
