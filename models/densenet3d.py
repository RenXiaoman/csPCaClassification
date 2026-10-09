"""MONAI DenseNet121 wrapper for shallow 3D MRI volumes."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from monai.networks.nets import DenseNet121


class DenseNet1213DClassifier(nn.Module):
    """3D DenseNet121 with depth padding for the project's 16-slice inputs."""

    def __init__(self, num_classes: int = 2, min_depth: int = 32) -> None:
        super().__init__()
        self.min_depth = min_depth
        self.backbone = DenseNet121(
            spatial_dims=3,
            in_channels=3,
            out_channels=num_classes,
            pretrained=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(f"expected [B, C, D, H, W], got {tuple(x.shape)}")
        if x.shape[2] < self.min_depth:
            x = F.pad(x, (0, 0, 0, 0, 0, self.min_depth - x.shape[2]))
        return self.backbone(x)
