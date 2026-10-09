"""MONAI EfficientNet wrapper for the project's shallow 3D MRI volumes."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from monai.networks.nets import EfficientNetBN


class EfficientNet3DClassifier(nn.Module):
    """3D EfficientNet-B0 with depth padding for 16-slice inputs.

    MONAI's EfficientNet downsamples all three spatial axes several times and
    therefore needs more than 16 voxels in the depth dimension. Padding only
    the depth axis keeps the in-plane resolution and modality channels intact.
    """

    def __init__(self, num_classes: int = 2, min_depth: int = 32) -> None:
        super().__init__()
        self.min_depth = min_depth
        self.backbone = EfficientNetBN(
            model_name="efficientnet-b0",
            spatial_dims=3,
            in_channels=3,
            num_classes=num_classes,
            pretrained=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(f"expected [B, C, D, H, W], got {tuple(x.shape)}")
        if x.shape[2] < self.min_depth:
            x = F.pad(x, (0, 0, 0, 0, 0, self.min_depth - x.shape[2]))
        return self.backbone(x)
