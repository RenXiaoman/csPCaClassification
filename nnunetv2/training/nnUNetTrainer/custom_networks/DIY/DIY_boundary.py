from __future__ import annotations

import torch
import torch.nn as nn

from monai.networks.blocks import UnetOutBlock

try:
    from .DIYgate import DIYGate
except ImportError:
    from DIYgate import DIYGate


class DIYBoundary(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        spatial_dims: int = 3,
        feature_size: int = 16,
        norm_name: tuple | str = "instance",
        dropout_rate: float = 0.0,
        depth: int = 4,
        boundary_channels: int = 1,
    ) -> None:
        super().__init__()
        self.backbone = DIYGate(
            in_channels=in_channels,
            out_channels=out_channels,
            spatial_dims=spatial_dims,
            feature_size=feature_size,
            norm_name=norm_name,
            dropout_rate=dropout_rate,
            depth=depth,
        )
        self.seg_head = self.backbone.out
        self.backbone.out = nn.Identity()
        self.boundary_head = UnetOutBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size,
            out_channels=boundary_channels,
        )

    def forward(self, x: torch.Tensor, training: bool = False):
        features = self.backbone(x)
        seg_logits = self.seg_head(features)
        if not training:
            return seg_logits
        boundary_logits = self.boundary_head(features)
        return seg_logits, boundary_logits


DIY_boundary = DIYBoundary


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DIYBoundary(in_channels=1, out_channels=2).to(device)
    inputs = torch.randn(1, 1, 16, 64, 64).to(device)
    seg, boundary = model(inputs, training=True)
    print(seg.shape, boundary.shape)
