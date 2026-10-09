from __future__ import annotations

import torch
from torch import nn
from monai.networks.blocks import UnetOutBlock

from nnunetv2.training.nnUNetTrainer.custom_networks.DIY.ablation import DIYAblationBackbone
from nnunetv2.training.nnUNetTrainer.custom_networks.DIY.components import (
    ComplementaryMultiScaleAttentionGate,
)


class FreGNetAblationBackbone(DIYAblationBackbone):
    """Old ablation backbone with the new CMSAG substituted at all four skips."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        spatial_dims: int = 3,
        feature_size: int = 16,
        norm_name: tuple | str = "instance",
        dropout_rate: float = 0.0,
        depth: int = 4,
        use_frequency: bool = False,
        use_cmsag: bool = False,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            spatial_dims=spatial_dims,
            feature_size=feature_size,
            norm_name=norm_name,
            dropout_rate=dropout_rate,
            depth=depth,
            use_frequency=use_frequency,
            use_msag=use_cmsag,
        )
        self.use_frequency = use_frequency
        self.use_cmsag = use_cmsag

        if use_cmsag:
            decoder_channels = (8, 4, 2, 1)
            for decoder, channel_multiplier in zip(
                (self.decoder5, self.decoder4, self.decoder3, self.decoder2),
                decoder_channels,
            ):
                channels = feature_size * channel_multiplier
                decoder.attention = ComplementaryMultiScaleAttentionGate(
                    spatial_dims=spatial_dims,
                    f_int=channels,
                    f_g=channels,
                    f_l=channels,
                    dropout=dropout_rate,
                )


class FreGNetAuxAblationNet(nn.Module):
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
        use_cmsag: bool = False,
    ) -> None:
        super().__init__()
        self.backbone = FreGNetAblationBackbone(
            in_channels=in_channels,
            out_channels=out_channels,
            spatial_dims=spatial_dims,
            feature_size=feature_size,
            norm_name=norm_name,
            dropout_rate=dropout_rate,
            depth=depth,
            use_frequency=use_frequency,
            use_cmsag=use_cmsag,
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
        return seg_logits, self.boundary_head(features)
