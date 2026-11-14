"""ConvNeXt RGB Encoder for Controlled Localization.
Replaces ResNet-18 with modern 7x7 depthwise convolutions and LayerNorm.
Produces native 4-scale feature pyramids at strides 4, 8, 16, 32.
"""

from __future__ import annotations

from typing import Dict, Optional
import torch
import torch.nn as nn
import torchvision.models as models


class ConvNeXtEncoder(nn.Module):
    """
    ConvNeXt-Tiny hierarchical visual feature extractor.
    Native pyramid outputs:
      - C4:  stride 4,  96 channels  [B, 96, H // 4, W // 4]
      - C8:  stride 8,  192 channels [B, 192, H // 8, W // 8]
      - C16: stride 16, 384 channels [B, 384, H // 16, W // 16]
      - C32: stride 32, 768 channels [B, 768, H // 32, W // 32]
    """

    def __init__(
        self,
        pretrained: bool = True,
        freeze_stem_and_stage1: bool = True,
    ) -> None:
        super().__init__()
        weights = models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.convnext_tiny(weights=weights)

        # features:
        # 0: Conv2dNormActivation (downsample to stride 4, 96ch)
        # 1: Sequential of 3 ConvNeXt blocks
        # 2: Sequential downsample to stride 8, 192ch
        # 3: Sequential of 3 ConvNeXt blocks
        # 4: Sequential downsample to stride 16, 384ch
        # 5: Sequential of 9 ConvNeXt blocks
        # 6: Sequential downsample to stride 32, 768ch
        # 7: Sequential of 3 ConvNeXt blocks
        self.stage4 = nn.Sequential(backbone.features[0], backbone.features[1])
        self.stage8 = nn.Sequential(backbone.features[2], backbone.features[3])
        self.stage16 = nn.Sequential(backbone.features[4], backbone.features[5])
        self.stage32 = nn.Sequential(backbone.features[6], backbone.features[7])

        if freeze_stem_and_stage1:
            for p in self.stage4.parameters():
                p.requires_grad = False
            for p in self.stage8.parameters():
                p.requires_grad = False

        # ImageNet normalization
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)
        )

    def forward(self, rgb: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Input: rgb [B, 3, H, W] in [0, 1].
        Returns dict with keys: 'C4', 'C8', 'C16', 'C32'.
        """
        x = (rgb - self.mean) / self.std

        c4 = self.stage4(x)
        c8 = self.stage8(c4)
        c16 = self.stage16(c8)
        c32 = self.stage32(c16)

        return {
            "C4": c4,
            "C8": c8,
            "C16": c16,
            "C32": c32,
        }
