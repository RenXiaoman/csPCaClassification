from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from monai.networks.blocks import UnetOutBlock, UnetrBasicBlock
from monai.networks.blocks.dynunet_block import UnetBasicBlock, UnetResBlock, get_conv_layer
from monai.networks.blocks.convolutions import ResidualUnit
from monai.networks.layers.factories import Norm

try:
    from .diy import wav_Enhance
except ImportError:
    from diy import wav_Enhance


class Gate(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        # norm_name: tuple | str,
    ) -> None:
        super().__init__()
        self.out_channels = out_channels
        self.convS = get_conv_layer(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            conv_only=False,
        )
        
        self.convT = get_conv_layer(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            conv_only=False,
        )
        
        self.convR = get_conv_layer(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            conv_only=False,
        )
        
        self.conv1x1_out = get_conv_layer(
            spatial_dims,
            out_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            conv_only=False,
        )
        self.residual_proj = (
            nn.Identity()
            if in_channels == out_channels
            else get_conv_layer(
                spatial_dims,
                in_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                conv_only=False,
            )
        )
        self.output = get_conv_layer(
            spatial_dims,
            out_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            conv_only=False,
        )
        
        self.attn_dropout = nn.Dropout(0.1)
        self.non_linearity = nn.LeakyReLU()

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:  # x is skip
        b, c, d, h, w = x.size()  # x = g = [B, 1, 16, 256, 256]
        convR = self.convR(x)
        convS = self.convS(x)
        convT = self.convT(g)  # convS, convR, convT is [B, 1, 16, 256, 256]
                
        # Reshape R, S, T
        S_reshape = convS.reshape(b, self.out_channels, d * h * w)   
        R_reshape = convR.view(b, self.out_channels, d * h * w)  
        T_reshape = convT.view(b, self.out_channels, d * h * w)  # [B, out_channels, d * h * w]
        
        U = R_reshape + S_reshape
        Uid = F.softmax(U, dim=-1)
        T = F.softmax(T_reshape, dim=-1).transpose(1, 2)  
        
        Pid = torch.matmul(Uid, T) 
        prostate_target = self.non_linearity(Pid)
        prostate_target = self.attn_dropout(prostate_target)  # [B, C, C]
        
        
        
        nmap_reshape = Uid.view(b, self.out_channels, d * h * w)  # [B, C, d * h * w]

        
        nmap_prostate = nmap_reshape.mean(dim=2, keepdim=True)  # [5, C, 1]

        augmented_prostate_target = torch.matmul(prostate_target, nmap_prostate)  # [5, C, 1]

        enhanced_prostate = prostate_target + augmented_prostate_target  # [5, 32, 32]
        
        augmentedR = torch.matmul(enhanced_prostate.transpose(2, 1), nmap_reshape)
        augmentedR = torch.sigmoid(augmentedR)  # [B, C, d * h * w]
        
        output = torch.matmul(enhanced_prostate, augmentedR).reshape(b, self.out_channels, d, h, w)
        
        conv1x1_out = self.conv1x1_out(output)
        output = self.output(conv1x1_out + self.residual_proj(x))
        return torch.sigmoid(output)




class GatedUnetrUpBlock(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Sequence[int] | int,
        upsample_kernel_size: Sequence[int] | int,
        norm_name: tuple | str,
        res_block: bool = False,
    ) -> None:
        super().__init__()
        upsample_stride = upsample_kernel_size
        self.transp_conv = get_conv_layer(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=upsample_kernel_size,
            stride=upsample_stride,
            conv_only=True,
            is_transposed=True,
        )
        self.attention = Gate(
            spatial_dims=spatial_dims,
            in_channels=out_channels,
            out_channels=out_channels,
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
        out = self.transp_conv(inp)  # out = inp = skip = [1, 128, 2, 32, 32]
        skip = skip * self.attention(g=out, x=skip)  # [1, 128, 2, 32, 32]
        
        out = torch.cat((out, skip), dim=1)  # [1, 256, 2, 32, 32]
        return self.conv_block(out)  # [1, 128, 2, 32, 32]


class DIYGate(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        spatial_dims: int = 3,
        feature_size: int = 16,
        norm_name: tuple | str = "instance",
        dropout_rate: float = 0.0,
        depth: int = 4,
    ) -> None:
        super().__init__()
        if depth != 4:
            raise ValueError(f"DIYGate currently expects depth=4, got {depth}.")
        self.depth = depth

        self.stem = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )
        self.enhance = nn.ModuleList(
            [wav_Enhance(in_dim=feature_size * (2 ** i)) for i in range(depth)]
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

        self.decoder5 = GatedUnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=16 * feature_size,
            out_channels=8 * feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )
        self.decoder4 = GatedUnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=8 * feature_size,
            out_channels=4 * feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )
        self.decoder3 = GatedUnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=4 * feature_size,
            out_channels=2 * feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )
        self.decoder2 = GatedUnetrUpBlock(
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.stem(x)
        x2 = self.enc1(self.enhance[0](x1))
        x3 = self.enc2(self.enhance[1](x2))
        x4 = self.enc3(self.enhance[2](x3))
        x5 = self.enc4(self.enhance[3](x4))

        up4 = self.decoder5(x5, x4)
        up3 = self.decoder4(up4, x3)
        up2 = self.decoder3(up3, x2)
        up1 = self.decoder2(up2, x1)
        return self.out(up1)


DIY_gate = DIYGate


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    C =32
    # model = DIYGate(in_channels=1, out_channels=2).to(device)
    model = Gate(spatial_dims=3,in_channels=C, out_channels=C).to(device)
    g = torch.randn(5, C, 16, 256, 256).to(device)
    x = torch.randn(5, C, 16, 256, 256).to(device)
    outputs = model(g=g, x=x)
    # print(outputs.shape)
