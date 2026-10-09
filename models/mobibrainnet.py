"""3D-MobiBrainNet adapted for three-channel prostate MRI classification."""
from __future__ import annotations

import torch
from torch import nn


class ConvBNAct(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, groups=1):
        super().__init__(
            nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride,
                      padding=kernel_size // 2, groups=groups, bias=False),
            nn.BatchNorm3d(out_channels), nn.ReLU6(inplace=True),
        )


class Mobile3DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.depthwise = ConvBNAct(in_channels, in_channels, 3, stride, in_channels)
        self.pointwise = ConvBNAct(in_channels, out_channels, 1)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


class MobiBrainNet3D(nn.Module):
    """MobiBrainNet backbone with configurable input channels/classes."""
    def __init__(self, in_channels=3, num_classes=2, dropout=0.30):
        super().__init__()
        self.features = nn.Sequential(
            ConvBNAct(in_channels, 16, stride=2),
            Mobile3DBlock(16, 32),
            Mobile3DBlock(32, 64, stride=2),
            Mobile3DBlock(64, 96, stride=2),
            Mobile3DBlock(96, 160, stride=2),
            Mobile3DBlock(160, 256, stride=2),
        )
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(256, num_classes))

    def forward(self, x):
        return self.classifier(self.pool(self.features(x)))
