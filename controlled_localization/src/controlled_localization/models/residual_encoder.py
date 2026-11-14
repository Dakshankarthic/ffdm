"""Fixed high-pass stem and residual CNN encoder for Controlled Localization v1.1."""

from __future__ import annotations

from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F


class FixedHighPassStem(nn.Module):
    """
    Fixed high-pass filter stem. Zero optimizer parameters.
    Applies Dx, Dy, and Laplacian kernels grouped across RGB channels.
    """

    def __init__(self) -> None:
        super().__init__()
        # 3 exact 3x3 kernels
        # Dx = [[0, 0, 0], [-0.5, 0, 0.5], [0, 0, 0]]
        dx = torch.tensor([[0.0, 0.0, 0.0], [-0.5, 0.0, 0.5], [0.0, 0.0, 0.0]], dtype=torch.float32)

        # Dy = [[0, -0.5, 0], [0, 0, 0], [0, 0.5, 0]]
        dy = torch.tensor([[0.0, -0.5, 0.0], [0.0, 0.0, 0.0], [0.0, 0.5, 0.0]], dtype=torch.float32)

        # Lap = [[0, 0.25, 0], [0.25, -1, 0.25], [0, 0.25, 0]]
        lap = torch.tensor([[0.0, 0.25, 0.0], [0.25, -1.0, 0.25], [0.0, 0.25, 0.0]], dtype=torch.float32)

        # Output order: R_Dx, R_Dy, R_Lap, G_Dx, G_Dy, G_Lap, B_Dx, B_Dy, B_Lap
        # Grouped conv with groups=3: weight shape is [9, 1, 3, 3]
        kernels = torch.stack([dx, dy, lap, dx, dy, lap, dx, dy, lap], dim=0).unsqueeze(1)
        self.register_buffer("kernels", kernels)

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        """
        Input: rgb [B, 3, H, W] in [0, 1].
        Output: high-pass residuals [B, 9, H, W].
        Uses reflect padding for tiles (replicate fallback for tiny dimensions).
        """
        b, c, h, w = rgb.shape
        pad_mode = "reflect" if (h > 2 and w > 2) else "replicate"
        padded = F.pad(rgb, (1, 1, 1, 1), mode=pad_mode)
        # Grouped conv with groups=3
        out = F.conv2d(padded, self.kernels, groups=3)
        return out


class ResidualBlock(nn.Module):
    """
    Two-conv block: Conv3x3 stride=2 + GN(8, C) + GELU followed by Conv3x3 stride=1 + GN(8, C) + GELU.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(num_groups=8, num_channels=out_channels)
        self.act1 = nn.GELU()

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(num_groups=8, num_channels=out_channels)
        self.act2 = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act1(self.gn1(self.conv1(x)))
        x = self.act2(self.gn2(self.conv2(x)))
        return x


class ResidualForensicEncoder(nn.Module):
    """
    Four-stage residual CNN encoder operating on fixed high-pass residuals.
    Channels: 32 (stride 2), 64 (stride 4), 96 (stride 8), 128 (stride 16).
    """

    def __init__(self) -> None:
        super().__init__()
        self.stem = FixedHighPassStem()
        self.block1 = ResidualBlock(in_channels=9, out_channels=32)    # F2: stride 2
        self.block2 = ResidualBlock(in_channels=32, out_channels=64)   # F4: stride 4
        self.block3 = ResidualBlock(in_channels=64, out_channels=96)   # F8: stride 8
        self.block4 = ResidualBlock(in_channels=96, out_channels=128)  # F16: stride 16

    def forward(self, rgb: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Input: rgb [B, 3, H, W] in [0, 1].
        Returns:
          - 'F2':  [B, 32, H/2, W/2]
          - 'F4':  [B, 64, H/4, W/4]
          - 'F8':  [B, 96, H/8, W/8]
          - 'F16': [B, 128, H/16, W/16]
        """
        hp = self.stem(rgb)
        f2 = self.block1(hp)
        f4 = self.block2(f2)
        f8 = self.block3(f4)
        f16 = self.block4(f8)

        return {"F2": f2, "F4": f4, "F8": f8, "F16": f16}

