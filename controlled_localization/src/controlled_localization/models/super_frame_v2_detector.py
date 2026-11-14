"""SuperFRAME v2: Dual Foundation Scale (DINOv2-Base 86M + ConvNeXt-Base 88M) + 60-ch Forensic + Multi-Head Cross-Attention Decoder.
Designed to push localization IoU toward and beyond 50%+ on the OpenSDID Benchmark.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from .dino_encoder import FrozenDINOv2Encoder
from .forensic_encoder import EnhancedForensicEncoder
from .cross_attention_decoder import CrossAttentionFPN


class SuperFRAMEV2Detector(nn.Module):
    """
    SuperFRAME v2 Architecture.
    - Semantic Stream A: ConvNeXt-Base (88M params) - 4-stage hierarchical depthwise conv
    - Semantic Stream B: Frozen DINOv2-Base (86M params) - 768-dim self-supervised patch tokens
    - Forensic Stream: 60-channel SRM (30 fixed) + Bayar (30 learnable) residual encoder
    - Decoder: 128-channel Multi-Head Cross-Attention FPN with Stride-2 refinement
    """

    def __init__(
        self,
        pretrained: bool = True,
        freeze_dino: bool = True,
        freeze_convnext_early: bool = True,
    ) -> None:
        super().__init__()

        # 1. ConvNeXt-Base RGB Encoder
        weights = models.ConvNeXt_Base_Weights.IMAGENET1K_V1 if pretrained else None
        c_base = models.convnext_base(weights=weights)

        # c_base.features:
        # [0, 1] -> Stage 4 (128ch)
        # [2, 3] -> Stage 8 (256ch)
        # [4, 5] -> Stage 16 (512ch)
        # [6, 7] -> Stage 32 (1024ch)
        self.c_stage4 = nn.Sequential(c_base.features[0], c_base.features[1])
        self.c_stage8 = nn.Sequential(c_base.features[2], c_base.features[3])
        self.c_stage16 = nn.Sequential(c_base.features[4], c_base.features[5])
        self.c_stage32 = nn.Sequential(c_base.features[6], c_base.features[7])

        if freeze_convnext_early:
            for p in self.c_stage4.parameters():
                p.requires_grad = False
            for p in self.c_stage8.parameters():
                p.requires_grad = False

        # ImageNet normalization for ConvNeXt
        self.register_buffer(
            "rgb_mean", torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "rgb_std", torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)
        )

        # 2. Frozen DINOv2-Base Encoder (86M params, 768-dim)
        self.dino = FrozenDINOv2Encoder(arch_name="dinov2_vitb14", pretrained=True)

        # 3. Enhanced Forensic Encoder (60-ch SRM + Bayar)
        self.forensic = EnhancedForensicEncoder()

        # 4. Multi-Head Cross-Attention Decoder
        self.decoder = CrossAttentionFPN(
            c_dims=(1024, 512, 256, 128),
            d_dim=768,
            f_dims=(192, 144, 96, 48),
            decoder_dim=128,
        )

    def forward(self, rgb: torch.Tensor, return_features: bool = False) -> Dict[str, torch.Tensor]:
        """
        Input: rgb [B, 3, H, W] in [0, 1].
        Returns:
          - 'mask_logits': [B, 1, H, W]
          - 'det_logits': [B, 1]
          - 'det_feat': [B, 257] (if return_features=True)
        """
        b, c, h, w = rgb.shape

        # 1. ConvNeXt-Base hierarchical forward
        x_norm = (rgb - self.rgb_mean) / self.rgb_std
        c4 = self.c_stage4(x_norm)
        c8 = self.c_stage8(c4)
        c16 = self.c_stage16(c8)
        c32 = self.c_stage32(c16)

        c_feats = {"C4": c4, "C8": c8, "C16": c16, "C32": c32}

        # 2. DINOv2-Base forward (completely frozen)
        d_feats = self.dino(rgb)

        # 3. Forensic Encoder forward (SRM + Bayar 60ch)
        f_feats = self.forensic(rgb)

        # 4. Cross-Attention Decoder
        if return_features:
            mask_logits, det_logits, det_feat = self.decoder(
                c_feats=c_feats,
                d_feats=d_feats,
                f_feats=f_feats,
                target_size=(h, w),
                return_features=True,
            )
            return {
                "mask_logits": mask_logits,
                "det_logits": det_logits,
                "det_feat": det_feat,
            }
        else:
            mask_logits, det_logits = self.decoder(
                c_feats=c_feats,
                d_feats=d_feats,
                f_feats=f_feats,
                target_size=(h, w),
                return_features=False,
            )
            return {
                "mask_logits": mask_logits,
                "det_logits": det_logits,
            }
