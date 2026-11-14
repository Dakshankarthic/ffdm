"""64-channel FPN decoder with exact spatial size interpolation for Controlled Localization v1.1."""

from __future__ import annotations

from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


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


class FPNDecoder(nn.Module):
    """
    FPN decoder fusing RGB and Residual features.
    Resolves audit finding F1:
      - R32 (stride 32) -> P32
      - At strides 16, 8, 4: fuse proj(R_s) and proj(F_s) with 1x1 conv to 64, add upsampled P_2s, smooth.
      - At stride 2: fuse upsampled P4 and proj(F2) with 3x3 conv to 64.
      - Predict mask logits at stride 1 via 1x1 conv and exact interpolation to target H, W.
      - Optional T3 processing classification head on GAP(P16).
    """

    def __init__(self, num_processing_classes: Optional[int] = None) -> None:
        super().__init__()
        # Projections for RGB branch
        self.proj_r32 = make_proj(512, 64)
        self.proj_r16 = make_proj(256, 64)
        self.proj_r8 = make_proj(128, 64)
        self.proj_r4 = make_proj(64, 64)

        # Projections for Residual branch
        self.proj_f16 = make_proj(128, 64)
        self.proj_f8 = make_proj(96, 64)
        self.proj_f4 = make_proj(64, 64)
        self.proj_f2 = make_proj(32, 64)

        # Lateral fusion layers (128 -> 64)
        self.fuse16 = make_proj(128, 64)
        self.fuse8 = make_proj(128, 64)
        self.fuse4 = make_proj(128, 64)

        # Smoothing layers
        self.smooth16 = make_smooth(64)
        self.smooth8 = make_smooth(64)
        self.smooth4 = make_smooth(64)

        # Stride 2 fusion (128 -> 64) using Conv3x3 + GN + GELU
        self.fuse2 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=8, num_channels=64),
            nn.GELU(),
        )

        # Final prediction head
        self.head = nn.Conv2d(64, 1, kernel_size=1)

        # Optional T3 processing classifier
        self.num_processing_classes = num_processing_classes
        if num_processing_classes is not None and num_processing_classes > 0:
            self.proc_classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(64, 32),
                nn.GELU(),
                nn.Linear(32, num_processing_classes),
            )
        else:
            self.proc_classifier = None

    def forward(
        self,
        rgb_feats: Dict[str, torch.Tensor],
        res_feats: Dict[str, torch.Tensor],
        target_size: Tuple[int, int],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        target_size: (H, W) of input image.
        Returns:
          - mask_logits: [B, 1, H, W]
          - proc_logits: [B, R] or None
        """
        r32, r16, r8, r4 = rgb_feats["R32"], rgb_feats["R16"], rgb_feats["R8"], rgb_feats["R4"]
        f16, f8, f4, f2 = res_feats["F16"], res_feats["F8"], res_feats["F4"], res_feats["F2"]

        # Top-down FPN
        p32 = self.proj_r32(r32)

        # Stride 16
        lat16 = self.fuse16(torch.cat([self.proj_r16(r16), self.proj_f16(f16)], dim=1))
        up32 = F.interpolate(p32, size=lat16.shape[-2:], mode="bilinear", align_corners=False)
        p16 = self.smooth16(lat16 + up32)

        # Stride 8
        lat8 = self.fuse8(torch.cat([self.proj_r8(r8), self.proj_f8(f8)], dim=1))
        up16 = F.interpolate(p16, size=lat8.shape[-2:], mode="bilinear", align_corners=False)
        p8 = self.smooth8(lat8 + up16)

        # Stride 4
        lat4 = self.fuse4(torch.cat([self.proj_r4(r4), self.proj_f4(f4)], dim=1))
        up8 = F.interpolate(p8, size=lat4.shape[-2:], mode="bilinear", align_corners=False)
        p4 = self.smooth4(lat4 + up8)

        # Stride 2
        up4 = F.interpolate(p4, size=f2.shape[-2:], mode="bilinear", align_corners=False)
        p2 = self.fuse2(torch.cat([up4, self.proj_f2(f2)], dim=1))

        # Final prediction
        pred_p2 = self.head(p2)
        mask_logits = F.interpolate(pred_p2, size=target_size, mode="bilinear", align_corners=False)

        proc_logits = None
        if self.proc_classifier is not None:
            proc_logits = self.proc_classifier(p16)

        return mask_logits, proc_logits

