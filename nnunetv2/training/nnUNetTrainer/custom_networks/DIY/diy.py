from __future__ import annotations

import itertools
from collections.abc import Sequence

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from torch.nn import LayerNorm

from monai.networks.blocks import PatchEmbed, UnetOutBlock, UnetrBasicBlock, UnetrUpBlock
from monai.networks.blocks.convolutions import Convolution, ResidualUnit
from monai.networks.layers.factories import Norm

import ptwt

try:
    from .components import SpatialAttention, ChannelAttention, LearnableGaussianFilterBank3D
except ImportError:
    from components import SpatialAttention, ChannelAttention, LearnableGaussianFilterBank3D


class AttentionGate(nn.Module):
    def __init__(
        self,
        spatial_dims: int = 3,
        in_channels: int = 16,
        gating_channels: int = 16,
        inter_channels: int | None = None,
        norm_name: tuple | str = Norm.BATCH,
    ):
        super().__init__()
        if inter_channels is None:
            inter_channels = max(in_channels // 2, 1)

        self.W_g = nn.Sequential(
            Convolution(
                spatial_dims=spatial_dims,
                in_channels=gating_channels,
                out_channels=inter_channels,
                kernel_size=1,
                strides=1,
                padding=0,
                conv_only=True,
            ),
            Norm[norm_name, spatial_dims](inter_channels),
        )
        self.W_x = nn.Sequential(
            Convolution(
                spatial_dims=spatial_dims,
                in_channels=in_channels,
                out_channels=inter_channels,
                kernel_size=1,
                strides=1,
                padding=0,
                conv_only=True,
            ),
            Norm[norm_name, spatial_dims](inter_channels),
        )
        self.psi = nn.Sequential(
            Convolution(
                spatial_dims=spatial_dims,
                in_channels=inter_channels,
                out_channels=1,
                kernel_size=1,
                strides=1,
                padding=0,
                conv_only=True,
            ),
            Norm[norm_name, spatial_dims](1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:  # [B, 128, 2, 32, 32]
        att = self.relu(self.W_g(g) + self.W_x(x))  # [B, 64, 2, 32, 32] + [B, 64, 2, 32, 32]
        att = self.psi(att)  # [B, 1, 2, 32, 32]
        return x * att





class ConvDWT(torch.nn.Module):
    def __init__(self, wavelet='db1', level=1, mode='zero'):
        super(ConvDWT, self).__init__()
        self.wavelet = wavelet
        self.mode = mode
        self.level = level
        
        
    def forward(self, x):
        b, c, d, h, w = x.shape
        if d == 1:
            coeffs = ptwt.wavedec2(x.squeeze(2), wavelet=self.wavelet, level=self.level, mode=self.mode)
            Yl = coeffs[0].unsqueeze(2)
            Yh = torch.stack(coeffs[1], dim=1)
            Yh = Yh.reshape(b, -1, 1, Yh.shape[-2], Yh.shape[-1])
            return torch.cat((Yl, Yh), dim=1)  # [B, 4*C, 1, H/2, W/2]

        coeffs = ptwt.wavedec3(x, wavelet=self.wavelet, level=self.level, mode=self.mode)
        Yl  = coeffs[0]  # Yl (B, C, D/2, H/2, W/2) for low-frequency LL
        Yh = coeffs[1:]  # List Yh for high-frequency:aad, ada,add,daa,dad,dda,ddd
        
        band_order = ("aad", "ada", "add", "daa", "dad", "dda", "ddd")
        Yh = torch.stack([Yh[0][band] for band in band_order], dim=1)  # [B, 7, C, D/2, H/2, W/2]
        Yh = Yh.reshape(Yh.shape[0], -1, Yh.shape[3], Yh.shape[4], Yh.shape[5])  # [B, 7*C, D/2, H/2, W/2]

        return torch.cat((Yl, Yh), dim=1)  # [B, 8*C, D/2, H/2, W/2]    


class ConvIDWT(nn.Module):
    def __init__(self, wavelet='db1', mode='zero'):
        super().__init__()
        self.wavelet = wavelet
        self.mode = mode
        self.band_order_3d = ("aad", "ada", "add", "daa", "dad", "dda", "ddd")

    def forward(self, low_freqs, high_freqs=None):
        if high_freqs is None:
            b, channels, d, h, w = low_freqs.shape
            if d == 1 and channels % 4 == 0:
                c = channels // 4
            elif channels % 8 == 0:
                c = channels // 8
            else:
                raise ValueError(f"Expected channels to be divisible by 4 or 8, got {channels}.")
            high_freqs = low_freqs[:, c:, :, :, :]
            low_freqs = low_freqs[:, :c, :, :, :]

        b, c, d, h, w = low_freqs.shape
        if high_freqs.shape[1] == 3 * c:
            expected_hf_channels = 3 * c
            if high_freqs.shape != (b, expected_hf_channels, d, h, w):
                raise ValueError(
                    f"Expected high_freqs shape {(b, expected_hf_channels, d, h, w)}, got {tuple(high_freqs.shape)}."
                )
            high_freqs = high_freqs.reshape(b, 3, c, h, w)
            detail = tuple(high_freqs[:, idx] for idx in range(3))
            return ptwt.waverec2((low_freqs.squeeze(2), detail), wavelet=self.wavelet).unsqueeze(2)

        expected_hf_channels = 7 * c
        if high_freqs.shape != (b, expected_hf_channels, d, h, w):
            raise ValueError(
                f"Expected high_freqs shape {(b, expected_hf_channels, d, h, w)}, got {tuple(high_freqs.shape)}."
            )

        high_freqs = high_freqs.reshape(b, 7, c, d, h, w)
        detail = {band: high_freqs[:, idx] for idx, band in enumerate(self.band_order_3d)}
        return ptwt.waverec3((low_freqs, detail), wavelet=self.wavelet)

class wav_Enhance(nn.Module): # Low-frequency Guided Feature Purification
    def __init__(self, wavelet='db1', mode='zero', in_dim=16):
        super().__init__()
        self.in_dim = in_dim
        
        
        self.dwt = ConvDWT(wavelet=wavelet, mode=mode)
        self.idwt = ConvIDWT(wavelet=wavelet, mode=mode)
        
        self.ca = ChannelAttention(in_dim)
        self.sa = SpatialAttention()
        self.gaussian_filter_2d_depth = LearnableGaussianFilterBank3D(
            kernel_size=3,
            num_filters=1,
            num_channels=3 * in_dim,
        )
        self.gaussian_filter_3d = LearnableGaussianFilterBank3D(
            kernel_size=3,
            num_filters=1,
            num_channels=7 * in_dim,
        )
        self.alpha = nn.Parameter(torch.tensor(0.5))
        # Populated during forward for optional feature visualization only.
        self.last_attended_low: torch.Tensor | None = None
        self.last_raw_high: torch.Tensor | None = None
        
    def forward(self, x):
        orig_dtype = x.dtype
        with torch.autocast(device_type=x.device.type, enabled=False):
            x = x.float()
            B, C, D, H, W = x.shape
            dwt_out = self.dwt(x)  # [B, 8*C, D/2, H/2, W/2] or [B, 4*C, 1, H/2, W/2]
            
            LL = dwt_out[:, :C, :, :, :]
            Yh = dwt_out[:, C:, :, :, :]

            ca_mask = self.ca(LL)
            sa_mask = self.sa(ca_mask * LL)
            att = ca_mask * sa_mask
            self.last_attended_low = (LL * att).detach()
            
            
            ############ 高斯差分高通增强滤波 ############
            gaussian_filter = self.gaussian_filter_2d_depth if D == 1 else self.gaussian_filter_3d
            self.last_raw_high = Yh.detach()
            Yh_low = gaussian_filter(Yh)
            Yh_high = Yh - Yh_low
            Yh = Yh + self.alpha * Yh_high
            
            subbands = 3 if D == 1 else 7
            Yh = Yh * att.repeat_interleave(subbands, dim=1)

            
            x_rec = self.idwt(LL, Yh)
        return x_rec.to(orig_dtype)
        
class DIY(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        spatial_dims: int = 3,
        feature_size: int = 16,
        norm_name: tuple | str = "instance",
        dropout_rate: float = 0.0,
        depth: int = 4,
        wavelet: str = "db1",
        mode: str = "zero",
    ) -> None:
        super().__init__()
        if depth != 4:
            raise ValueError(f"DIY currently expects depth=4, got {depth}.")
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
            [wav_Enhance(wavelet=wavelet, mode=mode, in_dim=feature_size * (2 ** i)) for i in range(depth)]
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
        
        
        self.decoder5 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=16 * feature_size,
            out_channels=8 * feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )
        
        self.decoder4 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=8 * feature_size,
            out_channels=4 * feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )
        
        self.decoder3 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=4 * feature_size,
            out_channels=2 * feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )
        
        self.decoder2 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=2 * feature_size,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )
        
        self.out = UnetOutBlock(spatial_dims=spatial_dims, 
                                in_channels=feature_size, 
                                out_channels=out_channels)
        
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
        out = self.out(up1)
        return out

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    
    model = DIY(in_channels=1, out_channels=2).to(device)
    input = torch.randn(1, 1, 16, 64, 64).to(device)
    output = model(input)
    
    print(output.shape)
