from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn

from monai.networks.blocks import UnetOutBlock, UnetrBasicBlock, UnetrUpBlock
from monai.networks.blocks.convolutions import ResidualUnit
from monai.networks.blocks.dynunet_block import UnetBasicBlock, UnetResBlock, get_conv_layer
from monai.networks.layers.factories import Norm

try:
    from .components import MultiScaleAttentionGate
except ImportError:
    from components import MultiScaleAttentionGate


class MSAGUnetrUpBlock(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Sequence[int] | int,
        upsample_kernel_size: Sequence[int] | int,
        norm_name: tuple | str,
        dropout_rate: float = 0.0,
        res_block: bool = False,
    ) -> None:
        super().__init__()
        self.transp_conv = get_conv_layer(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=upsample_kernel_size,
            stride=upsample_kernel_size,
            conv_only=True,
            is_transposed=True,
        )
        self.attention = MultiScaleAttentionGate(
            spatial_dims=spatial_dims,
            f_int=out_channels,
            f_g=out_channels,
            f_l=out_channels,
            dropout=dropout_rate,
        )

        block = UnetResBlock if res_block else UnetBasicBlock
        self.conv_block = block(
            spatial_dims,
            out_channels + out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            norm_name=norm_name,
        )

    def forward(self, inp: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        out = self.transp_conv(inp)
        skip = self.attention(g=out, x=skip)
        out = torch.cat((out, skip), dim=1)
        return self.conv_block(out)


class SimpleConvEnhance(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        channels: int,
    ) -> None:
        super().__init__()
        self.conv = get_conv_layer(
            spatial_dims,
            channels,
            channels,
            kernel_size=3,
            stride=1,
            conv_only=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class DIYAblationBackbone(nn.Module):
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
        use_msag: bool = False,
    ) -> None:
        super().__init__()
        if depth != 4:
            raise ValueError(f"DIYAblationBackbone currently expects depth=4, got {depth}.")

        self.stem = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )
        if use_frequency:
            try:
                from .diy import wav_Enhance
            except ImportError:
                from diy import wav_Enhance

            self.enhance = nn.ModuleList(
                [wav_Enhance(in_dim=feature_size * (2 ** i)) for i in range(depth)]
            )
        else:
            self.enhance = nn.ModuleList(
                [SimpleConvEnhance(spatial_dims, feature_size * (2 ** i)) for i in range(depth)]
            )

        self.enc1 = ResidualUnit(
            spatial_dims=spatial_dims,
            in_channels=feature_size,
            out_channels=2 * feature_size,
            strides=2,
            kernel_size=3,
            adn_ordering="NDA",
            act=("leakyrelu", {"negative_slope": 0.2}),
            norm=Norm.BATCH,
            dropout=dropout_rate,
            dropout_dim=1,
        )
        self.enc2 = ResidualUnit(
            spatial_dims=spatial_dims,
            in_channels=2 * feature_size,
            out_channels=4 * feature_size,
            strides=2,
            kernel_size=3,
            adn_ordering="NDA",
            act=("leakyrelu", {"negative_slope": 0.2}),
            norm=Norm.BATCH,
            dropout=dropout_rate,
            dropout_dim=1,
        )
        self.enc3 = ResidualUnit(
            spatial_dims=spatial_dims,
            in_channels=4 * feature_size,
            out_channels=8 * feature_size,
            strides=2,
            kernel_size=3,
            adn_ordering="NDA",
            act=("leakyrelu", {"negative_slope": 0.2}),
            norm=Norm.BATCH,
            dropout=dropout_rate,
            dropout_dim=1,
        )
        self.enc4 = ResidualUnit(
            spatial_dims=spatial_dims,
            in_channels=8 * feature_size,
            out_channels=16 * feature_size,
            strides=2,
            kernel_size=3,
            adn_ordering="NDA",
            act=("leakyrelu", {"negative_slope": 0.2}),
            norm=Norm.BATCH,
            dropout=dropout_rate,
            dropout_dim=1,
        )

        up_block = MSAGUnetrUpBlock if use_msag else UnetrUpBlock
        self.decoder5 = up_block(
            spatial_dims=spatial_dims,
            in_channels=16 * feature_size,
            out_channels=8 * feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            dropout_rate=dropout_rate,
            res_block=True,
        ) if use_msag else up_block(
            spatial_dims=spatial_dims,
            in_channels=16 * feature_size,
            out_channels=8 * feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )
        self.decoder4 = up_block(
            spatial_dims=spatial_dims,
            in_channels=8 * feature_size,
            out_channels=4 * feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            dropout_rate=dropout_rate,
            res_block=True,
        ) if use_msag else up_block(
            spatial_dims=spatial_dims,
            in_channels=8 * feature_size,
            out_channels=4 * feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )
        self.decoder3 = up_block(
            spatial_dims=spatial_dims,
            in_channels=4 * feature_size,
            out_channels=2 * feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            dropout_rate=dropout_rate,
            res_block=True,
        ) if use_msag else up_block(
            spatial_dims=spatial_dims,
            in_channels=4 * feature_size,
            out_channels=2 * feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )
        self.decoder2 = up_block(
            spatial_dims=spatial_dims,
            in_channels=2 * feature_size,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            dropout_rate=dropout_rate,
            res_block=True,
        ) if use_msag else up_block(
            spatial_dims=spatial_dims,
            in_channels=2 * feature_size,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )

        self.out = UnetOutBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size,
            out_channels=out_channels,
        )

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.stem(x)
        x2 = self.enc1(self.enhance[0](x1))
        x3 = self.enc2(self.enhance[1](x2))
        x4 = self.enc3(self.enhance[2](x3))
        x5 = self.enc4(self.enhance[3](x4))

        up4 = self.decoder5(x5, x4)
        up3 = self.decoder4(up4, x3)
        up2 = self.decoder3(up3, x2)
        return self.decoder2(up2, x1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(self.forward_features(x))


class BoundaryOnlyNet(nn.Module):
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
