"""RGB backbone with frozen BatchNorm for Controlled Localization v1.1."""

from __future__ import annotations

from typing import Dict, List, Optional
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class FrozenBNResNet18Encoder(nn.Module):
    """
    ResNet-18 RGB encoder with ImageNet normalization and permanently frozen BatchNorm.
    Resolves audit finding F3:
      - BatchNorm affine parameters (weight, bias) frozen: requires_grad=False.
      - Running statistics (running_mean, running_var) frozen.
      - train(mode) overridden to force all BatchNorm2d modules into eval() mode.
      - Convolutions remain trainable.
    """

    def __init__(self, pretrained: bool = True, weights_name: Optional[str] = "IMAGENET1K_V1") -> None:
        super().__init__()
        if pretrained:
            weights = ResNet18_Weights.IMAGENET1K_V1 if weights_name == "IMAGENET1K_V1" else ResNet18_Weights.DEFAULT
            base = resnet18(weights=weights)
        else:
            base = resnet18(weights=None)

        # ImageNet normalization parameters registered as buffers
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        # Stride 4 stem: conv1 (stride 2) + bn1 + relu + maxpool (stride 2)
        self.conv1 = base.conv1
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool

        # Residual stages
        self.layer1 = base.layer1  # stride 4, channels 64
        self.layer2 = base.layer2  # stride 8, channels 128
        self.layer3 = base.layer3  # stride 16, channels 256
        self.layer4 = base.layer4  # stride 32, channels 512

        # Freeze all BatchNorm layers permanently
        self._freeze_all_bn()

    def _freeze_all_bn(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                if m.weight is not None:
                    m.weight.requires_grad = False
                if m.bias is not None:
                    m.bias.requires_grad = False

    def train(self, mode: bool = True) -> FrozenBNResNet18Encoder:
        super().train(mode)
        # Force all BatchNorm2d layers back to eval() after any train() call
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
        return self

    def forward(self, rgb: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Input: rgb float tensor [B, 3, H, W] in [0, 1].
        Returns features at strides 4, 8, 16, 32:
          - 'R4':  [B, 64, H/4, W/4]
          - 'R8':  [B, 128, H/8, W/8]
          - 'R16': [B, 256, H/16, W/16]
          - 'R32': [B, 512, H/32, W/32]
        """
        # Apply normalization
        x = (rgb - self.mean) / self.std

        # Stem (stride 4)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        r4 = self.layer1(x)

        r8 = self.layer2(r4)
        r16 = self.layer3(r8)
        r32 = self.layer4(r16)

        return {"R4": r4, "R8": r8, "R16": r16, "R32": r32}

