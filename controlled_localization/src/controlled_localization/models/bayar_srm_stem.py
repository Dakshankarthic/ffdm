"""Multi-kernel SRM (Steganalysis Rich Models) and learnable Bayar constrained stem.
Provides robust high-pass and forensic noise residual extraction for AI-inpainting detection.
"""

from __future__ import annotations

from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F


def build_srm_kernels() -> torch.Tensor:
    """
    Build 10 distinct SRM filter kernels.
    When applied across 3 RGB channels with grouped conv (10 kernels * 3 channels = 30 channels).
    Includes 1st order, 2nd order, 3x3 square, edge, and 5x5 filters.
    All filters strictly sum to 0 to cancel smooth/flat image content.
    """
    # 1. First order horizontal residual [1, 2]
    k1 = torch.tensor([
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0]
    ], dtype=torch.float32)

    # 2. First order vertical residual
    k2 = torch.tensor([
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0]
    ], dtype=torch.float32)

    # 3. First order diagonal residual
    k3 = torch.tensor([
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0]
    ], dtype=torch.float32)

    # 4. Second order horizontal residual [-1, 2, -1]
    k4 = torch.tensor([
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, -1.0, 2.0, -1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0]
    ], dtype=torch.float32) * 0.5

    # 5. Second order vertical residual
    k5 = torch.tensor([
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, 2.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0]
    ], dtype=torch.float32) * 0.5

    # 6. 3x3 Laplacian / High-pass
    k6 = torch.tensor([
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0],
        [0.0, 1.0, -4.0, 1.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0]
    ], dtype=torch.float32) * 0.25

    # 7. 3x3 Square residual filter
    k7 = torch.tensor([
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, -1.0, 2.0, -1.0, 0.0],
        [0.0, 2.0, -4.0, 2.0, 0.0],
        [0.0, -1.0, 2.0, -1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0]
    ], dtype=torch.float32) * 0.25

    # 8. 3x3 Edge residual filter
    k8 = torch.tensor([
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, -1.0, 2.0, -1.0, 0.0],
        [0.0, 2.0, -4.0, 2.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0]
    ], dtype=torch.float32) * 0.5

    # 9. 5x5 Square high-pass filter
    k9 = torch.tensor([
        [-1.0, 2.0, -2.0, 2.0, -1.0],
        [2.0, -6.0, 8.0, -6.0, 2.0],
        [-2.0, 8.0, -12.0, 8.0, -2.0],
        [2.0, -6.0, 8.0, -6.0, 2.0],
        [-1.0, 2.0, -2.0, 2.0, -1.0]
    ], dtype=torch.float32) * (1.0 / 12.0)

    # 10. 5x5 Edge high-pass filter
    k10 = torch.tensor([
        [-1.0, 2.0, -2.0, 2.0, -1.0],
        [2.0, -6.0, 8.0, -6.0, 2.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0]
    ], dtype=torch.float32) * (1.0 / 6.0)

    # Stack 10 kernels -> [10, 1, 5, 5]
    kernels = torch.stack([k1, k2, k3, k4, k5, k6, k7, k8, k9, k10], dim=0).unsqueeze(1)
    # Replicate for 3 RGB channels (groups=3): [30, 1, 5, 5]
    grouped_kernels = kernels.repeat(3, 1, 1, 1)
    return grouped_kernels


class SRMFiltersStem(nn.Module):
    """Fixed 30-channel SRM filter stem with zero trainable parameters."""

    def __init__(self) -> None:
        super().__init__()
        kernels = build_srm_kernels()
        self.register_buffer("kernels", kernels)

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        """rgb: [B, 3, H, W] in [0, 1] -> [B, 30, H, W] SRM high-pass residuals."""
        b, c, h, w = rgb.shape
        pad_mode = "reflect" if (h > 4 and w > 4) else "replicate"
        padded = F.pad(rgb, (2, 2, 2, 2), mode=pad_mode)
        return F.conv2d(padded, self.kernels, groups=3)


class BayarConv2d(nn.Module):
    """
    Learnable Bayar Constrained Convolution.
    Enforces the center kernel weight to -1 and surrounding weights to sum to 1,
    forcing the network to suppress image content and learn pure residual forensics.
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 30, kernel_size: int = 5) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

        # Initialize unconstrained weights
        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels, kernel_size, kernel_size) * 0.02
        )

        # Create binary mask where center is 0 and surroundings are 1
        mask = torch.ones(kernel_size, kernel_size, dtype=torch.float32)
        center = kernel_size // 2
        mask[center, center] = 0.0
        self.register_buffer("mask", mask.view(1, 1, kernel_size, kernel_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        # Surrounding weights only
        w_surr = self.weight * self.mask
        # Sum of surroundings along spatial dimensions
        sums = w_surr.sum(dim=(-2, -1), keepdim=True)
        # Avoid division by zero
        sums = torch.where(sums.abs() < 1e-6, torch.full_like(sums, 1e-6), sums)
        # Normalize surroundings so they sum to 1.0 per filter per channel
        w_norm = w_surr / sums
        # Center is strictly -1.0
        w_center = torch.zeros_like(self.weight)
        center = self.kernel_size // 2
        w_center[:, :, center, center] = -1.0
        effective_weight = w_norm + w_center

        pad_mode = "reflect" if (h > self.padding * 2 and w > self.padding * 2) else "replicate"
        padded = F.pad(x, (self.padding, self.padding, self.padding, self.padding), mode=pad_mode)
        return F.conv2d(padded, effective_weight, bias=None)


class ForensicFrontEnd(nn.Module):
    """
    Dual forensic front-end combining 30-channel fixed SRM and 30-channel learnable Bayar conv.
    Outputs 60-channel high-frequency forensic noise representation.
    """

    def __init__(self) -> None:
        super().__init__()
        self.srm = SRMFiltersStem()
        self.bayar = BayarConv2d(in_channels=3, out_channels=30, kernel_size=5)

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        """Input: [B, 3, H, W] -> Output: [B, 60, H, W]."""
        srm_out = self.srm(rgb)
        bayar_out = self.bayar(rgb)
        return torch.cat([srm_out, bayar_out], dim=1)
