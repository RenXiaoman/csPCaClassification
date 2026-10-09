from __future__ import annotations

import torch
import torch.nn as nn

from monai.networks.blocks import UnetOutBlock

from nnunetv2.training.nnUNetTrainer.custom_networks.DIY.ablation import DIYAblationBackbone


class PairwiseAuxAblationNet(nn.Module):
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
        use_frequency: bool = False,
        use_msag: bool = False,
    ) -> None:
        super().__init__()
        self.backbone = DIYAblationBackbone(
            in_channels=in_channels,
            out_channels=out_channels,
            spatial_dims=spatial_dims,
            feature_size=feature_size,
            norm_name=norm_name,
            dropout_rate=dropout_rate,
            depth=depth,
            use_frequency=use_frequency,
            use_msag=use_msag,
        )
        self.boundary_head = UnetOutBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size,
            out_channels=boundary_channels,
        )

    def forward(self, x: torch.Tensor, training: bool = False):
        features = self.backbone.forward_features(x)
        seg_logits = self.backbone.out(features)
        if not training:
            return seg_logits

        boundary_logits = self.boundary_head(features)
        return seg_logits, boundary_logits
