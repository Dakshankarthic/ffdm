"""SuperFRAME: Dual-Semantic Foundation (ConvNeXt-Small + DINOv2-Small) + 60-ch Forensic Encoder.
Designed to surpass FUSED by combining hierarchical convolutions with self-supervised patch attention.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from .dino_encoder import FrozenDINOv2Encoder
from .forensic_encoder import EnhancedForensicEncoder
from .gated_decoder import SpatialGate, make_proj, make_smooth


class SuperFRAMEDecoder(nn.Module):
    """
    Dual-Semantic Gated Decoder.
    Fuses ConvNeXt-Small (local/hierarchical) + DINOv2-Small (global/patch attention) + SRM/Bayar (forensics).
    """

    def __init__(self, decoder_dim: int = 128) -> None:
        super().__init__()
        self.decoder_dim = decoder_dim

        # Semantic fusion: projects concatenated [ConvNeXt, DINOv2] to decoder_dim
        # Stride 32: ConvNeXt 768 + DINO 384 = 1152 -> 128
        self.proj_s32 = make_proj(768 + 384, decoder_dim)
        # Stride 16: ConvNeXt 384 + DINO 384 = 768 -> 128
        self.proj_s16 = make_proj(384 + 384, decoder_dim)
        # Stride 8: ConvNeXt 192 + DINO 384 = 576 -> 128
        self.proj_s8 = make_proj(192 + 384, decoder_dim)
        # Stride 4: ConvNeXt 96 + DINO 384 = 480 -> 128
        self.proj_s4 = make_proj(96 + 384, decoder_dim)

        # Forensic projections:
        self.proj_f16 = make_proj(192, decoder_dim)
        self.proj_f8 = make_proj(144, decoder_dim)
        self.proj_f4 = make_proj(96, decoder_dim)
        self.proj_f2 = make_proj(48, decoder_dim)

        # Spatial Gates
        self.gate16 = SpatialGate(decoder_dim)
        self.gate8 = SpatialGate(decoder_dim)
        self.gate4 = SpatialGate(decoder_dim)

        # Smooth layers
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

        # Final pixel mask head
        self.mask_head = nn.Conv2d(decoder_dim // 2, 1, kernel_size=1)

        # Image-level detection classification head
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
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Fuse Semantic Streams (ConvNeXt + DINOv2)
        sem32 = self.proj_s32(torch.cat([c_feats["C32"], d_feats["S32"]], dim=1))
        sem16 = self.proj_s16(torch.cat([c_feats["C16"], d_feats["S16"]], dim=1))
        sem8 = self.proj_s8(torch.cat([c_feats["C8"], d_feats["S8"]], dim=1))
        sem4 = self.proj_s4(torch.cat([c_feats["C4"], d_feats["S4"]], dim=1))

        # Top-down FPN
        p32 = sem32

        # Stride 16
        pf16 = self.proj_f16(f_feats["F16"])
        fused16 = self.gate16(sem16, pf16)
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

        # Stride 2
        up4 = F.interpolate(p4, size=f_feats["F2"].shape[-2:], mode="bilinear", align_corners=False)
        pf2 = self.proj_f2(f_feats["F2"])
        p2 = self.fuse2(torch.cat([up4, pf2], dim=1))

        # Pixel mask logits at stride 1
        mask_stride2 = self.mask_head(p2)
        mask_logits = F.interpolate(mask_stride2, size=target_size, mode="bilinear", align_corners=False)

        # Image-level detection score
        pool16 = F.adaptive_avg_pool2d(p16, 1).flatten(1)
        pool32 = F.adaptive_avg_pool2d(p32, 1).flatten(1)
        mask_sig_max = torch.sigmoid(mask_logits).amax(dim=(-2, -1))
        det_input = torch.cat([pool16, pool32, mask_sig_max], dim=1)
        det_logits = self.det_head(det_input)

        return mask_logits, det_logits


class SuperFRAMEDetector(nn.Module):
    """
    SuperFRAME Detector.
    Integrates:
      1. ConvNeXt-Small hierarchical convolutions
      2. DINOv2-Small self-supervised patch tokens
      3. 60-channel SRM + Bayar forensic noise stream
      4. Dual-Semantic Gated FPN Decoder
    """

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        # ConvNeXt-Small
        weights = models.ConvNeXt_Small_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.convnext_small(weights=weights)

        self.c_stage4 = nn.Sequential(backbone.features[0], backbone.features[1])
        self.c_stage8 = nn.Sequential(backbone.features[2], backbone.features[3])
        self.c_stage16 = nn.Sequential(backbone.features[4], backbone.features[5])
        self.c_stage32 = nn.Sequential(backbone.features[6], backbone.features[7])

        # Freeze stem and stage 1
        for p in self.c_stage4.parameters():
            p.requires_grad = False
        for p in self.c_stage8.parameters():
            p.requires_grad = False

        # DINOv2-Small (frozen)
        self.dino = FrozenDINOv2Encoder("dinov2_vits14")

        # Forensic Encoder
        self.forensic = EnhancedForensicEncoder()

        # Dual-Semantic Gated Decoder
        self.decoder = SuperFRAMEDecoder(decoder_dim=128)

        # ImageNet normalization
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)
        )

    def forward(self, rgb: torch.Tensor) -> Dict[str, torch.Tensor]:
        b, c, h, w = rgb.shape

        # Normalize
        x_norm = (rgb - self.mean) / self.std

        # ConvNeXt pyramid
        c4 = self.c_stage4(x_norm)
        c8 = self.c_stage8(c4)
        c16 = self.c_stage16(c8)
        c32 = self.c_stage32(c16)
        c_feats = {"C4": c4, "C8": c8, "C16": c16, "C32": c32}

        # DINOv2 self-attention tokens
        d_feats = self.dino(rgb)

        # Forensic frequency residuals
        f_feats = self.forensic(rgb)

        # Gated fusion
        mask_logits, det_logits = self.decoder(
            c_feats=c_feats,
            d_feats=d_feats,
            f_feats=f_feats,
            target_size=(h, w),
        )

        return {
            "mask_logits": mask_logits,
            "det_logits": det_logits,
        }
