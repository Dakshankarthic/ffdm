"""Multi-scale Forensic Residual Encoder for Controlled Localization.
Processes 60-channel SRM+Bayar frequency residuals into hierarchical forensic features.
"""

from __future__ import annotations

from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

from .bayar_srm_stem import ForensicFrontEnd


class ForensicBlock(nn.Module):
    """
    Forensic residual block:
    Conv3x3 (stride=2 for downsampling) + GroupNorm + GELU + Conv3x3 + GroupNorm + GELU.
    Uses residual skip connection.
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 2) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(num_groups=8, num_channels=out_channels)
        self.act1 = nn.GELU()

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(num_groups=8, num_channels=out_channels)
        self.act2 = nn.GELU()

        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(num_groups=8, num_channels=out_channels),
            )
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.skip(x)
        out = self.act1(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        return self.act2(out + res)


class EnhancedForensicEncoder(nn.Module):
    """
    Forensic Encoder operating on 60-channel SRM+Bayar noise residuals.
    Produces multi-scale forensic features:
      - F2:  stride 2,  48 channels  [B, 48,  H // 2,  W // 2]
      - F4:  stride 4,  96 channels  [B, 96,  H // 4,  W // 4]
      - F8:  stride 8,  144 channels [B, 144, H // 8,  W // 8]
      - F16: stride 16, 192 channels [B, 192, H // 16, W // 16]
    """

    def __init__(self) -> None:
        super().__init__()
        self.stem = ForensicFrontEnd()

        # Initial projection from 60 -> 32 channels
        self.init_conv = nn.Sequential(
            nn.Conv2d(60, 32, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=8, num_channels=32),
            nn.GELU(),
        )

        # Stage 2: stride 2 -> 48 channels (F2)
        self.block2 = ForensicBlock(32, 48, stride=2)

        # Stage 4: stride 2 -> 96 channels (F4)
        self.block4 = ForensicBlock(48, 96, stride=2)

        # Stage 8: stride 2 -> 144 channels (F8)
        self.block8 = ForensicBlock(96, 144, stride=2)

        # Stage 16: stride 2 -> 192 channels (F16)
        self.block16 = ForensicBlock(144, 192, stride=2)

    def forward(self, rgb: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Input: rgb [B, 3, H, W] in [0, 1].
        Returns dict with keys: 'F2', 'F4', 'F8', 'F16'.
        """
        residuals = self.stem(rgb)
        x = self.init_conv(residuals)

        f2 = self.block2(x)
        f4 = self.block4(f2)
        f8 = self.block8(f4)
        f16 = self.block16(f8)

        return {
            "F2": f2,
            "F4": f4,
            "F8": f8,
            "F16": f16,
        }
