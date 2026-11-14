"""Multi-Head Cross-Attention Decoder (MH-CAD) for SuperFRAME v2.
Fuses high-frequency forensic residuals (Queries) with rich semantic representations (Keys & Values).
Enables long-range global reasoning for large inpainted regions and subtle flow-matching artifacts.
"""

from __future__ import annotations

import math
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


class MultiHeadCrossAttentionBlock(nn.Module):
    """
    Multi-Head Cross-Attention where Forensic tokens are Queries and Semantic tokens are Keys & Values.
    Includes residual connections, LayerNorm, and an MLP feedforward block.
    """

    def __init__(self, dim: int = 128, num_heads: int = 8, dropout: float = 0.1) -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)

        self.mha = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.Dropout(dropout),
        )

    def forward(self, forensic: torch.Tensor, semantic: torch.Tensor) -> torch.Tensor:
        """
        forensic: [B, C, H, W]
        semantic: [B, C, H, W]
        returns: [B, C, H, W]
        """
        b, c, h, w = forensic.shape
        n = h * w

        # [B, N, C]
        q = forensic.flatten(2).permute(0, 2, 1)
        kv = semantic.flatten(2).permute(0, 2, 1)

        # Cross attention: forensic queries semantic context
        q_norm = self.norm_q(q)
        kv_norm = self.norm_kv(kv)
        attn_out, _ = self.mha(query=q_norm, key=kv_norm, value=kv_norm)
        x = q + attn_out

        # FFN
        x = x + self.mlp(self.norm2(x))

        # Reshape back to [B, C, H, W]
        return x.permute(0, 2, 1).view(b, c, h, w)


class SpatialGate(nn.Module):
    """Adaptive spatial gating between semantic and forensic features."""

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


class CrossAttentionFPN(nn.Module):
    """
    128-channel Multi-Head Cross-Attention Decoder for SuperFRAME v2.
    Fuses ConvNeXt-Base + DINOv2-Base (Semantic) with 60-ch SRM+Bayar (Forensic).
    """

    def __init__(
        self,
        c_dims: Tuple[int, int, int, int] = (1024, 512, 256, 128),  # ConvNeXt-Base (C32, C16, C8, C4)
        d_dim: int = 768,                                          # DINOv2-Base (768)
        f_dims: Tuple[int, int, int, int] = (192, 144, 96, 48),    # Forensic (F16, F8, F4, F2)
        decoder_dim: int = 128,
    ) -> None:
        super().__init__()
        self.decoder_dim = decoder_dim
        c32, c16, c8, c4 = c_dims
        f16, f8, f4, f2 = f_dims

        # Semantic fusion projections [ConvNeXt + DINOv2 -> decoder_dim]
        self.proj_s32 = make_proj(c32 + d_dim, decoder_dim)
        self.proj_s16 = make_proj(c16 + d_dim, decoder_dim)
        self.proj_s8 = make_proj(c8 + d_dim, decoder_dim)
        self.proj_s4 = make_proj(c4 + d_dim, decoder_dim)

        # Forensic projections
        self.proj_f16 = make_proj(f16, decoder_dim)
        self.proj_f8 = make_proj(f8, decoder_dim)
        self.proj_f4 = make_proj(f4, decoder_dim)
        self.proj_f2 = make_proj(f2, decoder_dim)

        # Multi-Head Cross Attention at Stride 16 (where semantic context and forensic anomalies meet)
        self.cross_attn16 = MultiHeadCrossAttentionBlock(dim=decoder_dim, num_heads=8)

        # Spatial gates for multi-scale fusion
        self.gate16 = SpatialGate(decoder_dim)
        self.gate8 = SpatialGate(decoder_dim)
        self.gate4 = SpatialGate(decoder_dim)

        # FPN Smoothing layers
        self.smooth16 = make_smooth(decoder_dim)
        self.smooth8 = make_smooth(decoder_dim)
        self.smooth4 = make_smooth(decoder_dim)

        # High-resolution Stride-2 refinement with high-pass forensic stem
        self.fuse2 = nn.Sequential(
            nn.Conv2d(decoder_dim * 2, decoder_dim, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=8, num_channels=decoder_dim),
            nn.GELU(),
            nn.Conv2d(decoder_dim, decoder_dim // 2, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=4, num_channels=decoder_dim // 2),
            nn.GELU(),
        )

        # Pixel mask logit head
        self.mask_head = nn.Conv2d(decoder_dim // 2, 1, kernel_size=1)

        # Dedicated image-level binary classification head
        self.det_head = nn.Sequential(
            nn.Linear(decoder_dim * 2 + 1, 64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        c_feats: Dict[str, torch.Tensor],
        d_feats: Dict[str, torch.Tensor],
        f_feats: Dict[str, torch.Tensor],
        target_size: Tuple[int, int],
        return_features: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor] | Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # 1. Project concatenated Semantic streams (ConvNeXt + DINOv2)
        sem32 = self.proj_s32(torch.cat([c_feats["C32"], d_feats["S32"]], dim=1))
        sem16 = self.proj_s16(torch.cat([c_feats["C16"], d_feats["S16"]], dim=1))
        sem8 = self.proj_s8(torch.cat([c_feats["C8"], d_feats["S8"]], dim=1))
        sem4 = self.proj_s4(torch.cat([c_feats["C4"], d_feats["S4"]], dim=1))

        # 2. Top-down FPN with Cross-Attention
        p32 = sem32

        # Stride 16: Cross-Attention (Forensic queries Semantic) + Gating
        pf16 = self.proj_f16(f_feats["F16"])
        attn_f16 = self.cross_attn16(pf16, sem16)
        fused16 = self.gate16(sem16, attn_f16)
        up32 = F.interpolate(p32, size=fused16.shape[-2:], mode="bilinear", align_corners=False)
        p16 = self.smooth16(fused16 + up32)

        # Stride 8
        pf8 = self.proj_f8(f_feats["F8"])
        fused8 = self.gate8(sem8, pf8)
        up16 = F.interpolate(p16, size=fused8.shape[-2:], mode="bilinear", align_corners=False)
        p8 = self.smooth8(fused8 + up16)

        # Stride 4
        pf4 = self.proj_f4(f_feats["F4"])
        fused4 = self.gate4(sem4, pf4)
        up8 = F.interpolate(p8, size=fused4.shape[-2:], mode="bilinear", align_corners=False)
        p4 = self.smooth4(fused4 + up8)

        # Stride 2 refinement
        pf2 = self.proj_f2(f_feats["F2"])
        up4 = F.interpolate(p4, size=pf2.shape[-2:], mode="bilinear", align_corners=False)
        fused2 = self.fuse2(torch.cat([up4, pf2], dim=1))

        # Final pixel mask logits
        mask_logits_s2 = self.mask_head(fused2)
        mask_logits = F.interpolate(mask_logits_s2, size=target_size, mode="bilinear", align_corners=False)

        # Image-level detection logits
        gap32 = F.adaptive_avg_pool2d(p32, (1, 1)).flatten(1)
        gmp32 = F.adaptive_max_pool2d(p32, (1, 1)).flatten(1)
        k_pixels = min(2048, mask_logits.shape[-2] * mask_logits.shape[-1])
        flat_logits = mask_logits.flatten(1)
        topk_mean = torch.topk(flat_logits, k_pixels, dim=1)[0].mean(dim=1, keepdim=True)
        det_input = torch.cat([gap32, gmp32, topk_mean], dim=1)
        det_logits = self.det_head(det_input)

        if return_features:
            return mask_logits, det_logits, det_input
        return mask_logits, det_logits
