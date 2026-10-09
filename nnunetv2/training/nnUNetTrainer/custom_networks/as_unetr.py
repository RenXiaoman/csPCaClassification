# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
from torch.nn import init

from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks.unetr_block import UnetrBasicBlock, UnetrPrUpBlock, UnetrUpBlock
from monai.networks.blocks.convolutions import Convolution
from monai.networks.blocks.convolutions import ResidualUnit
from monai.networks.blocks import TransformerBlock
from monai.networks.layers.factories import Norm
from monai.networks.nets.vit import ViT
from monai.utils import deprecated_arg, ensure_tuple_rep
from .multiscale import DCFB


def fix_adsa_net_checkpoint(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        if 'gre_dcgf_' in key:
            # Replace 'gre_dcgf_X' with 'BiCR_X'
            new_key = key.replace('gre_dcgf_', 'BiCR_')
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value
    return new_state_dict

class ConvBlock(nn.Module):

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Sequence[int] | int = 3,
        strides: int = 1,
        dropout=0.0,
        act="relu",
    ):
        super().__init__()
        layers = [
            Convolution(
                spatial_dims=spatial_dims,
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                strides=strides,
                padding=None,
                adn_ordering="NDA",
                act=act,
                norm=Norm.BATCH,
                dropout=dropout,
            ),
            Convolution(
                spatial_dims=spatial_dims,
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                strides=1,
                padding=None,
                adn_ordering="NDA",
                act="relu",
                norm=Norm.BATCH,
                dropout=dropout,
            ),
        ]
        self.conv = nn.Sequential(*layers)

        # 添加残差连接
        self.use_residual = (in_channels == out_channels) and (strides == 1)
        if not self.use_residual and strides == 1:
            # 当通道数不同但stride=1时，使用1x1卷积调整维度
            self.residual_conv = Convolution(
                spatial_dims=spatial_dims,
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                strides=1,
                padding=None,
                adn_ordering="NDA",
                act=None,  # 残差连接不需要激活
                norm=Norm.BATCH,
                dropout=0.0,
            )
        else:
            self.residual_conv = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x_c: torch.Tensor = self.conv(x)

        # 处理残差连接
        if self.use_residual:
            # 直接相加
            return x_c + residual
        elif self.residual_conv is not None:
            # 使用1x1卷积调整维度后相加
            return x_c + self.residual_conv(residual)
        else:
            # 下采样或通道数不同且没有残差连接
            return x_c
    
    

# -----------------------------
# 基础:Pre-Norm 3D (卷积块与很多3D分割工程风格一致:先Norm+Act,再Conv)
# -----------------------------
def normalization(planes, norm='in'):
    if norm == 'bn':
        return nn.BatchNorm3d(planes)
    elif norm == 'gn':
        return nn.GroupNorm(4, planes)  # 组数可按需调
    elif norm == 'in':
        return nn.InstanceNorm3d(planes)
    else:
        raise ValueError(f'Unsupported norm: {norm}')

class GeneralConv3dPreNorm(nn.Module):
    """
    Pre-Norm：Norm -> Act -> Conv3d
    """
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1,
                 pad_type='zeros', norm='in', act='lrelu',
                 relu_slope=0.2, bias=True):
        super().__init__()
        self.norm = normalization(in_ch, norm=norm)
        if act == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif act == 'lrelu':
            self.act = nn.LeakyReLU(negative_slope=relu_slope, inplace=True)
        else:
            raise ValueError(f'Unsupported act: {act}')
        self.conv = nn.Conv3d(in_ch, out_ch, kernel_size=k, stride=s,
                              padding=p, padding_mode=pad_type, bias=bias)

    def forward(self, x):
        x = self.norm(x)
        x = self.act(x)
        x = self.conv(x)
        return x
    
    
# -----------------------------------------
# Sample-wise Gating Module (SGM)
# DCGF 子模块①：样本级门控（AttDual）
# 输入：curr ∈ [B, C, D, H, W], all_mod ∈ [B, 2C, D, H, W]
# 输出：curr * w，w ∈ [B,1,1,1,1]
# -----------------------------------------
class AttDual(nn.Module):  # SGM
    def __init__(self, in_channel: int):
        super().__init__()
        c = in_channel
        # 拼 curr 的 GAP 和 all_mod 的 GAP： [B, C,1,1,1] + [B, 2C,1,1,1] -> [B, 3C,1,1,1]
        self.weight_layer = nn.Sequential(
            nn.Conv3d(3 * c, 128, kernel_size=1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(128, 1, kernel_size=1, bias=True)  # 样本级标量门控
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, curr, all_mod):
        B, C, D, H, W = curr.shape
        curr_gap = torch.mean(curr,    dim=(2, 3, 4), keepdim=True)  # [B, C, 1,1,1]
        all_gap  = torch.mean(all_mod, dim=(2, 3, 4), keepdim=True)  # [B, 2C,1,1,1]
        x = torch.cat([curr_gap, all_gap], dim=1)                    # [B, 3C,1,1,1]
        w = self.weight_layer(x)                                     # [B, 1, 1,1,1]
        w = self.sigmoid(w)
        return curr * w


# -----------------------------------------
# ottleneck Fusion Module (BFM)
# DCGF 子模块②：双模态卷积融合（2C -> C -> C -> C）
# 和常见 1x1→3x3→1x1 融合头一致
# -----------------------------------------
class FusionLayerDual(nn.Module):
    def __init__(self, in_channel: int, norm='in'):
        super().__init__()
        c = in_channel
        self.block = nn.Sequential(
            GeneralConv3dPreNorm(2 * c, c, k=1, s=1, p=0, norm=norm),
            GeneralConv3dPreNorm(c,     c, k=3, s=1, p=1, norm=norm),
            GeneralConv3dPreNorm(c,     c, k=1, s=1, p=0, norm=norm),
        )

    def forward(self, x):
        # x: [B, 2C, D, H, W]
        return self.block(x)
    


# -----------------------------------------
#  Voxel-wise Confidence-Guided Refinement 
# out = Conv3d( Fusion(Concat) + Att(T2) + Att(Aux) )
# -----------------------------------------
class ACF(nn.Module):
    def __init__(self, in_channel: int, norm='in'):
        super().__init__()
        c = in_channel
        self.att_t2  = AttDual(c)
        self.att_aux = AttDual(c)
        self.fusion  = FusionLayerDual(c, norm=norm)     # 输入 2C，输出 C
        self.refine  = GeneralConv3dPreNorm(c, c, k=3, s=1, p=1, norm=norm)

    def forward(self, T2, Aux):
        all_mod   = torch.cat([T2, Aux], dim=1)         # [B, 2C, ...]
        trust_t2  = self.att_t2(T2, all_mod)            # [B, C, ...]
        trust_aux = self.att_aux(Aux, all_mod)          # [B, C, ...]
        fused     = self.fusion(all_mod)                # [B, C, ...]
        out       = self.refine(fused + trust_t2 + trust_aux)
        return out


# -----------------------------------------
# Sample-wise Modality Gating (SMG) 
# 基于样本级全局统计，softmax 生成两路权重，保证和为1
# 返回：coarse 融合 + 权重（便于可解释性）
# -----------------------------------------
class MRE(nn.Module):
    def __init__(self, in_channel: int):
        super().__init__()
        c = in_channel
        mid = max(c // 2, 1)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Conv3d(2 * c, mid, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid, 2, kernel_size=1, bias=True)  # 输出两路logits
        )

    def forward(self, T2, Aux):
        x = torch.cat([T2, Aux], dim=1)     # [B, 2C, D,H,W]
        x = self.pool(x)                    # [B, 2C, 1,1,1]
        logits = self.fc(x)                 # [B, 2,  1,1,1]
        weights = torch.softmax(logits, dim=1)
        w_t2  = weights[:, 0:1]             # [B,1,1,1,1]
        w_aux = weights[:, 1:2]             # [B,1,1,1,1]
        coarse = T2 * w_t2 + Aux * w_aux    # [B, C, D,H,W]
        return coarse, w_t2, w_aux
    

# -----------------------------------------
# GRE_DCGF3D：串行版本 = GRE（粗） -> DCGF（细）
# 粗粒度融合结果作为细粒度融合的输入
# 输出：out；可选返回权重与中间特征
# -----------------------------------------
class GRE_DCGF3D(nn.Module):
    """
    输入：
        T2  : [B, C, D, H, W]  主模态
        Aux : [B, C, D, H, W]  辅助模态（如 DWI/ADC 融合后的特征）
    输出：
        out : [B, C, D, H, W]
        （可选）(w_t2, w_aux), coarse, fine 便于可解释性与可视化
    """
    def __init__(self, in_channel: int, norm='in'):
        super().__init__()
        self.gre  = MRE(in_channel)
        self.dcgf = ACF(in_channel, norm=norm)

    def forward(self, T2, Aux, return_extras: bool = False):
        # 串行融合：粗粒度 -> 细粒度
        coarse, w_t2, w_aux = self.gre(T2, Aux)  # 粗融合 + 权重
        # 使用粗粒度融合结果作为细粒度融合的输入
        t2_refined = T2 * w_t2                    # 主模态优化
        aux_refined = Aux * w_aux                 # 辅助模态优化
        fine = self.dcgf(t2_refined, aux_refined) # 细融合
        out = fine                                # 直接输出细粒度结果
        if return_extras:
            return out, (w_t2, w_aux), coarse, fine
        return out



class CDSA_Net(nn.Module):
    """
    UNETR based on: "Hatamizadeh et al.,
    UNETR: Transformers for 3D Medical Image Segmentation <https://arxiv.org/abs/2103.10504>"
    """

    @deprecated_arg(
        name="pos_embed", since="1.2", removed="1.4", new_name="proj_type", msg_suffix="please use `proj_type` instead."
    )
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        img_size: Sequence[int] | int,
        feature_size: int = 16,
        hidden_size: int = 768,
        mlp_dim: int = 3072,
        num_heads: int = 12,
        pos_embed: str = "conv",
        proj_type: str = "conv",
        norm_name: tuple | str = "instance",
        conv_block: bool = True,
        res_block: bool = True,
        dropout_rate: float = 0.0,
        spatial_dims: int = 3,
        qkv_bias: bool = False,
        save_attn: bool = False,
    ) -> None:
        """
        Args:
            in_channels: dimension of input channels.
            out_channels: dimension of output channels.
            img_size: dimension of input image.
            feature_size: dimension of network feature size. Defaults to 16.
            hidden_size: dimension of hidden layer. Defaults to 768.
            mlp_dim: dimension of feedforward layer. Defaults to 3072.
            num_heads: number of attention heads. Defaults to 12.
            proj_type: patch embedding layer type. Defaults to "conv".
            norm_name: feature normalization type and arguments. Defaults to "instance".
            conv_block: if convolutional block is used. Defaults to True.
            res_block: if residual block is used. Defaults to True.
            dropout_rate: fraction of the input units to drop. Defaults to 0.0.
            spatial_dims: number of spatial dims. Defaults to 3.
            qkv_bias: apply the bias term for the qkv linear layer in self attention block. Defaults to False.
            save_attn: to make accessible the attention in self attention block. Defaults to False.

        .. deprecated:: 1.4
            ``pos_embed`` is deprecated in favor of ``proj_type``.

        Examples::

            # for single channel input 4-channel output with image size of (96,96,96), feature size of 32 and batch norm
            >>> net = UNETR(in_channels=1, out_channels=4, img_size=(96,96,96), feature_size=32, norm_name='batch')

             # for single channel input 4-channel output with image size of (96,96), feature size of 32 and batch norm
            >>> net = UNETR(in_channels=1, out_channels=4, img_size=96, feature_size=32, norm_name='batch', spatial_dims=2)

            # for 4-channel input 3-channel output with image size of (128,128,128), conv position embedding and instance norm
            >>> net = UNETR(in_channels=4, out_channels=3, img_size=(128,128,128), proj_type='conv', norm_name='instance')

        """

        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size should be divisible by num_heads.")

        self.num_layers = 12
        img_size = ensure_tuple_rep(img_size, spatial_dims)
        self.patch_size = ensure_tuple_rep(16, spatial_dims)
        self.feat_size = tuple(img_d // p_d for img_d, p_d in zip(img_size, self.patch_size))
        self.hidden_size = hidden_size
        self.classification = False
        
        self.vit = ViT(
            in_channels=in_channels,
            img_size=img_size,
            patch_size=self.patch_size,
            hidden_size=hidden_size,
            mlp_dim=mlp_dim,
            num_layers=self.num_layers,
            num_heads=num_heads,
            proj_type=proj_type,
            classification=self.classification,
            dropout_rate=dropout_rate,
            spatial_dims=spatial_dims,
            qkv_bias=qkv_bias,
            save_attn=save_attn,
        )
        self.encoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder2 = UnetrPrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=hidden_size,
            out_channels=feature_size * 2,
            num_layer=2,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )
        self.encoder3 = UnetrPrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=hidden_size,
            out_channels=feature_size * 4,
            num_layer=1,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )
        self.encoder4 = UnetrPrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=hidden_size,
            out_channels=feature_size * 8,
            num_layer=0,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )
        
        self.decoder4 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=hidden_size,
            out_channels=feature_size * 8,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        
        self.decoder3 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 8,
            out_channels=feature_size * 4,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 4,
            out_channels=feature_size * 2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder1 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 2,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        
        self.out = UnetOutBlock(spatial_dims=spatial_dims, 
                                in_channels=feature_size, 
                                out_channels=out_channels)
        
        # 添加多尺度输出头 - nnUNet风格深度监督
        self.out_dec3 = UnetOutBlock(spatial_dims=spatial_dims,
                                   in_channels=feature_size * 8,
                                   out_channels=out_channels)
        self.out_dec2 = UnetOutBlock(spatial_dims=spatial_dims,
                                   in_channels=feature_size * 4,
                                   out_channels=out_channels)
        self.out_dec1 = UnetOutBlock(spatial_dims=spatial_dims,
                                   in_channels=feature_size * 2,
                                   out_channels=out_channels)
        
        self.proj_axes = (0, spatial_dims + 1) + tuple(d + 1 for d in range(spatial_dims))
        
        self.proj_view_shape = list(self.feat_size) + [self.hidden_size]  # [1, 16, 16, 768]
        
        # Attention Unet
        self.conv_head = ConvBlock(
            spatial_dims=spatial_dims,
            in_channels=1,
            out_channels=16,
            dropout=dropout_rate,
            kernel_size=3,
        )
        
        self.trans_1 = Convolution(
            spatial_dims=spatial_dims,
            in_channels=16,
            out_channels=16,
            strides=1,
            kernel_size=3,
        )
        
        self.s_encoder1 = ResidualUnit(
            spatial_dims=spatial_dims,
            in_channels=16,
            out_channels=32,
            strides=2,
            kernel_size=3,
            adn_ordering="NDA",
            act=("leakyrelu", {"negative_slope": 0.2}),
            norm=Norm.BATCH,
            dropout=dropout_rate,
            dropout_dim=1
        )
        # self.s_encoder1 = ConvBlock(
        #     spatial_dims=spatial_dims,
        #     in_channels=16,
        #     out_channels=32,
        #     dropout=dropout_rate,
        #     kernel_size=3,
        #     strides=2,
        #     act='leakyrelu'
        # )
        
        self.trans_2 = ConvBlock(
            spatial_dims=spatial_dims,
            in_channels=16 * 2,
            out_channels=32,
            dropout=dropout_rate,
            kernel_size=3,
            strides=1,
        )
        
        self.s_encoder2 = ResidualUnit(
            spatial_dims=spatial_dims,
            in_channels=32,
            out_channels=64,
            strides=2,
            kernel_size=3,
            adn_ordering="NDA",
            act=("leakyrelu", {"negative_slope": 0.2}),
            norm=Norm.BATCH,
            dropout=dropout_rate,
            dropout_dim=1
        )
        # self.s_encoder2 = ConvBlock(
        #     spatial_dims=spatial_dims,
        #     in_channels=32,
        #     out_channels=64,
        #     dropout=dropout_rate,
        #     kernel_size=3,
        #     strides=2,
        #     act='leakyrelu'
        # )
        
        self.trans_3 = ConvBlock(
            spatial_dims=spatial_dims,
            in_channels=16 * 4,
            out_channels=64,
            dropout=dropout_rate,
            kernel_size=3,
            strides=1,
        )
        
        self.s_encoder3 = ResidualUnit(
            spatial_dims=spatial_dims,
            in_channels=64,
            out_channels=128,
            strides=2,
            kernel_size=3,
            adn_ordering="NDA",
            act=("leakyrelu", {"negative_slope": 0.2}),
            norm=Norm.BATCH,
            dropout=dropout_rate,
            dropout_dim=1
        )
        # self.s_encoder3 = ConvBlock(
        #     spatial_dims=spatial_dims,
        #     in_channels=64,
        #     out_channels=128,
        #     dropout=dropout_rate,
        #     kernel_size=3,
        #     strides=2,
        #     act='leakyrelu'
        # )
        
        self.trans_4 = ConvBlock(
            spatial_dims=spatial_dims,
            in_channels=16 * 8,
            out_channels=128,
            dropout=dropout_rate,
            kernel_size=3,
            strides=1,
        )
        
        self.s_encoder4 = ResidualUnit(
            spatial_dims=spatial_dims,
            in_channels=128,
            out_channels=256,
            strides=2,
            kernel_size=3,
            adn_ordering="NDA",
            act=("leakyrelu", {"negative_slope": 0.2}),
            norm=Norm.BATCH,
            dropout=dropout_rate,
            dropout_dim=1
        )
        # self.s_encoder4 = ConvBlock(
        #     spatial_dims=spatial_dims,
        #     in_channels=128,
        #     out_channels=256,
        #     dropout=dropout_rate,
        #     kernel_size=3,
        #     strides=2,
        #     act='leakyrelu'
        # )
        #一个1x1的卷积，输入为 256维，输出为hidden_size
        self.conv1x1 = nn.Conv3d(
            in_channels=256,   # 输入通道数
            out_channels=768,  # 输出通道数
            kernel_size=1,     # 1x1x1 卷积核
            stride=1,
            padding=0,         # 不填充，保持 D,H,W 不变
            )
        
        self.trans_5 = ConvBlock(
            spatial_dims=spatial_dims,
            in_channels=hidden_size,
            out_channels=hidden_size,
            dropout=dropout_rate,
            kernel_size=3,
            strides=1,
        )
        
        self.BiCR_1 = GRE_DCGF3D(in_channel=16, norm='in')
        self.BiCR_2 = GRE_DCGF3D(in_channel=32, norm='in')
        self.BiCR_3 = GRE_DCGF3D(in_channel=64, norm='in')
        self.BiCR_4 = GRE_DCGF3D(in_channel=128, norm='in')
        self.BiCR_5 = GRE_DCGF3D(in_channel=hidden_size, norm='in')

        # 在CNN分支encoder后面添加DCFB模块，增强多尺度特征提取
        
        self.dcfb1 = DCFB(
            in_channels=32,      # 输入通道数
            out_channels=32,     # 输出通道数（保持不变）
            stride=1,            # 步长为1，保持分辨率
            kernel_sizes=[1, 3, 5],  # 多尺度卷积核
            expansion_factor=2,  # 扩展因子
            dw_parallel=True,    # 并行处理
            add=True,            # 使用加法融合
            activation='relu6'   # 激活函数
        )
        self.dcfb2 = DCFB(
            in_channels=64,
            out_channels=64,
            stride=1,
            kernel_sizes=[1, 3, 5],
            expansion_factor=2,
            dw_parallel=True,
            add=True,
            activation='relu6'
        )
        self.dcfb3 = DCFB(
            in_channels=128,
            out_channels=128,
            stride=1,
            kernel_sizes=[1, 3, 5],
            expansion_factor=2,
            dw_parallel=True,
            add=True,
            activation='relu6'
        )
        self.dcfb4 = DCFB(
            in_channels=256,
            out_channels=256,
            stride=1,
            kernel_sizes=[1, 3, 5],
            expansion_factor=2,
            dw_parallel=True,
            add=True,
            activation='relu6'
        )
        self.dcfb5 = DCFB(
            in_channels=768,
            out_channels=768,
            stride=1,
            kernel_sizes=[1, 3, 5],
            expansion_factor=2,
            dw_parallel=True,
            add=True,
            activation='relu6'
        )


    def proj_feat(self, x):  # [B, 256, 768]
        new_view = [x.size(0)] + self.proj_view_shape  

        x = x.view(new_view)  # [B, 1, 16, 16, 768]
        x = x.permute(self.proj_axes).contiguous()  # [B, C, D, H, W]
        return x

    def forward(self, x_input):  # x_in [B, 3, 16, 256, 256]
        x_in, x_single = x_input[:, 0:2, :, :, :], x_input[:, 2:3, :, :, :]  # [B, 2, 16, 256, 256]
        
        #############################  Vit Encoder  #############################
        # x [B, 256, 768], hidden_states_out [12, B, 256, 768]
        x, hidden_states_out = self.vit(x_in)  
        
        enc1 = self.encoder1(x_in)  # [B, 16, 16, 256, 256]
        
        x2 = hidden_states_out[3]  # [B, 256, 768]
        enc2 = self.encoder2(self.proj_feat(x2))  # [B, 32, 8, 128, 128]
        
        x3 = hidden_states_out[6]
        enc3 = self.encoder3(self.proj_feat(x3))  # [B, 64, 4, 64, 64]
        
        x4 = hidden_states_out[9]
        enc4 = self.encoder4(self.proj_feat(x4))  # [B, 128, 2, 32, 32]
 
        dec4 = self.proj_feat(x)  # 这里的 x 实际上是 hidden_states_out[12]   [B, 768, 1, 16, 16]

        #############################  Attention UNet  #############################
        # 'x_single.shape' [B, 1, 16, 256, 256]
        x_single_conv = self.conv_head(x_single)  # [5, C=16, 16, 256, 256]
    
        merge_x_1 = self.BiCR_1(x_single_conv, enc1)
        merge_x_1 = self.trans_1(merge_x_1)  # [5, 16, 16, 256, 256])
        
        x2_single_in = self.s_encoder1(merge_x_1)  # 和enc2 均为 [5, 32, 8, 128, 128]
        x2_single_in = self.dcfb1(x2_single_in)    # 添加DCFB模块，保持shape [5, 32, 8, 128, 128]
        merge_x_2 = self.BiCR_2(x2_single_in, enc2)
        merge_x_2 = self.trans_2(merge_x_2)

        x3_single_in = self.s_encoder2(merge_x_2)  # 和enc3 均为 [5, 64, 4, 64, 64]
        x3_single_in = self.dcfb2(x3_single_in)    # 添加DCFB模块，保持shape [5, 64, 4, 64, 64]

        merge_x_3 = self.BiCR_3(x3_single_in, enc3)
        merge_x_3 = self.trans_3(merge_x_3)

        x_4_single_in = self.s_encoder3(merge_x_3)  # [5, 128, 2, 32, 32]
        x_4_single_in = self.dcfb3(x_4_single_in)   # 添加DCFB模块，保持shape [5, 128, 2, 32, 32]

        merge_x_4 = self.BiCR_4(x_4_single_in, enc4)
        merge_x_4 = self.trans_4(merge_x_4)

        x5_single_in = self.s_encoder4(merge_x_4)   # [5, 256, 1, 16, 16]
        x5_single_in = self.dcfb4(x5_single_in)     # 添加DCFB模块，保持shape [5, 256, 1, 16, 16]
        
        x5_single_in = self.conv1x1(x5_single_in)   # [5, 768, 1, 16, 16]
        x5_single_in = self.dcfb5(x5_single_in)     # 添加DCFB模块，保持shape [5, 768, 1, 16, 16]
        
        merge_x_5 = self.BiCR_5(x5_single_in, dec4)
        merge_x_5 = self.trans_5(merge_x_5)

        #############################  Decoder  #############################
        dec3 = self.decoder4(merge_x_5, merge_x_4)
        dec2 = self.decoder3(dec3, merge_x_3)
        dec1 = self.decoder2(dec2, merge_x_2)
        out = self.decoder1(dec1, merge_x_1)
        
        # nnUNet风格多尺度输出
        # output_dec3 = self.out_dec3(dec3)  # 1/8分辨率输出
        # output_dec2 = self.out_dec2(dec2)  # 1/4分辨率输出  
        # output_dec1 = self.out_dec1(dec1)  # 1/2分辨率输出
        output_final = self.out(out)       # 全分辨率输出
        
        # 返回多尺度输出列表，从粗到细
        # return [output_dec3, output_dec2, output_dec1, output_final]
        return output_final
    


#################################################################################
# -------------------------------
# TransConv block  (Eq.4)
# TransConv(x) = C1( GN( GELU( C1( BN( DWC(x) ) ) ) ) ) + x
# DWC: depthwise conv (k=5,5,5)  维持通道数不变
# -------------------------------


class TransConv3D(nn.Module):
    def __init__(self, channels: int, dw_kernel=5, groups_gn=8):
        super().__init__()
        pad = dw_kernel // 2
        self.dwc = nn.Conv3d(channels, channels, kernel_size=dw_kernel,
                             padding=pad, groups=channels, bias=False)  # DWC
        self.bn  = nn.BatchNorm3d(channels)
        self.pw1 = nn.Conv3d(channels, channels, kernel_size=1, bias=False)   # C1
        self.act = nn.GELU()                                                  # GL
        self.gn  = nn.GroupNorm(num_groups=min(groups_gn, channels), num_channels=channels)  # GN
        self.pw2 = nn.Conv3d(channels, channels, kernel_size=1, bias=False)   # C1 (second)

    def forward(self, x):
        r = x
        z = self.dwc(x)
        z = self.bn(z)
        z = self.pw1(z)
        z = self.act(z)
        z = self.gn(z)
        z = self.pw2(z)
        return z + r  # residual

# -----------------------------------------
# MPFA module  (Eq.1–3 + Fig.2(a))
# 输入:  F_ap, F_vp  (B, C, D, H, W)
# 输出:  F_vap      (B, C, D, H, W)
# -----------------------------------------
class MPFA3D(nn.Module):
    def __init__(self, channels: int, use_softmax: bool = True):
        super().__init__()
        self.align = nn.Sequential(                     # C(·) + BN + ReLU  (Eq.2/3 中的 C, BN, ReLU)
            nn.Conv3d(2 * channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(channels),
            nn.ReLU(inplace=True)
        )
        self.transconv = TransConv3D(channels)          # 论文中的 TransConv 子模块 (Eq.4)
        # 生成两路权重的 1x1 heads（共享 backbone 特征）
        self.head = nn.Conv3d(channels, 2 * channels, kernel_size=1, bias=True)
        self.use_softmax = use_softmax
        if not use_softmax:
            self.sigmoid = nn.Sigmoid()

    def forward(self, F_ap, F_vp):
        """
        F_ap: (B, C, D, H, W)  动脉期特征
        F_vp: (B, C, D, H, W)  静脉期特征
        return F_vap: (B, C, D, H, W)  融合特征
        """
        # Eq.(2)/(3): Concat → 3x3x3 → BN → ReLU → TransConv
        x = torch.cat([F_vp, F_ap], dim=1)              # Concat(F^vp, F^ap)
        x = self.align(x)
        x = self.transconv(x)                           # TransConv(ReLU(BN(C(Concat(...)))))

        # 生成两路权重
        logits = self.head(x)                           # (B, 2C, D, H, W)
        B, _, D, H, W = logits.shape
        logits = logits.view(B, 2, -1, D, H, W)         # (B, 2, C, D, H, W)

        if self.use_softmax:
            # 在“相位维”做 softmax，确保 W_ap + W_vp = 1
            weights = torch.softmax(logits, dim=1)      # (B, 2, C, D, H, W)
            W_vp, W_ap = weights[:,0], weights[:,1]     # 各自 (B, C, D, H, W)
        else:
            # 按原文“Sigmoid”的风格（不严格保证和为1）
            W_vp = self.sigmoid(logits[:,0])
            W_ap = self.sigmoid(logits[:,1])

        # Eq.(1): F^vap = W^ap ⊙ F^ap + W^vp ⊙ F^vp
        F_vap = W_ap * F_ap + W_vp * F_vp
        return F_vap



class BiCR_3D(nn.Module):
    """
    BiCR-3D (Bi-Stream Channel Recalibration) Block
    Dual-Stream 3D Channel Recalibration with 1x1x1 Conv (SE-style)
    输入:  x1, x2 形状均为 [B, C, D, H, W]
    过程:  全局池化 -> 1x1x1 Conv3d 降维 -> ReLU -> 1x1x1 Conv3d 升维到 2C -> Sigmoid -> 分支加权融合
    输出:  [B, C, D, H, W]
    """
    def __init__(self, channel: int, reduction: int = 8):
        super().__init__()
        hidden = max(channel // reduction, 1)

        self.avg_pool = nn.AdaptiveAvgPool3d(1)  # [B, C, 1, 1, 1]
        self.fc = nn.Sequential(
            nn.Conv3d(channel, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden, 2 * channel, kernel_size=1, bias=False)  # 2C -> 两个分支的权重
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        x1, x2: [B, C, D, H, W]
        """
        x_sum = x1 + x2
        y = self.avg_pool(x_sum)              # [B, C, 1, 1, 1]
        y = self.fc(y)                        # [B, 2C, 1, 1, 1]
        w1, w2 = torch.split(y, x1.size(1), dim=1)   # 各 [B, C, 1, 1, 1]
        w1 = torch.sigmoid(w1)
        w2 = torch.sigmoid(w2)
        out = x1 * w1 + x2 * w2
        return out
    


class BiCR3D_MLP(nn.Module):
    """
    BiCR-3D (MLP版)：双分支3D通道重标定（SE风格）
    输入:  x1, x2 形状均为 [B, C, D, H, W]
    步骤:  GAP -> Linear(C -> C/r) -> ReLU -> Linear(C/r -> 2C) -> Sigmoid -> 分支加权
    输出:  [B, C, D, H, W]
    """
    def __init__(self, channel: int, reduction: int = 8, use_softmax: bool = False):
        super().__init__()
        hidden = max(channel // reduction, 1)
        self.avg_pool = nn.AdaptiveAvgPool3d(1)     # [B, C, 1, 1, 1]

        # 两层 MLP：C -> C/r -> 2C
        self.fc1 = nn.Linear(channel, hidden, bias=False)
        self.fc2 = nn.Linear(hidden, 2 * channel, bias=False)
        self.act = nn.ReLU(inplace=True)

        self.use_softmax = use_softmax  # 若为 True，则在“分支维”做 softmax，保证 w1+w2=1；否则各自 sigmoid

        self._init_weights()

    def _init_weights(self):
        # Kaiming 初始化，匹配 ReLU
        init.kaiming_normal_(self.fc1.weight, mode='fan_in', nonlinearity='relu')
        init.kaiming_normal_(self.fc2.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        B, C, _, _, _ = x1.shape
        x_sum = x1 + x2                                      # 先聚合两分支的全局语义
        y = self.avg_pool(x_sum).view(B, C)                  # [B, C]
        y = self.act(self.fc1(y))                            # [B, C/r]
        y = self.fc2(y)                                      # [B, 2C]

        if self.use_softmax:
            # 在分支维做 softmax：先 reshape 成 [B, 2, C]，沿 dim=1 做 softmax
            weights = torch.softmax(y.view(B, 2, C), dim=1)      # [B, 2, C]
            w1 = weights[:, 0, :].view(B, C, 1, 1, 1)        # [B, C, 1, 1, 1]
            w2 = weights[:, 1, :].view(B, C, 1, 1, 1)
        else:
            # 独立 sigmoid，不强制 w1+w2=1
            w1, w2 = torch.split(y, C, dim=1)                # 两个 [B, C]
            w1 = torch.sigmoid(w1).view(B, C, 1, 1, 1)
            w2 = torch.sigmoid(w2).view(B, C, 1, 1, 1)

        out = x1 * w1 + x2 * w2
        return out


# -----------------------------------------
if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    small_unetr = CDSA_Net(
        in_channels=2,  # ADC and DWI modalities
        out_channels=2,  # Background and lesion
        img_size=(16, 256, 256),
        feature_size=16,
        hidden_size=768,
        mlp_dim=3072,
        num_heads=8,
        norm_name='instance'
    ).to(device)
    # 计算参数量
    total_params = sum(p.numel() for p in small_unetr.parameters())
    trainable_params = sum(p.numel() for p in small_unetr.parameters() if p.requires_grad)
    
    print(f'模型总参数量: {total_params / 1e6:.2f}M')
    print(f'可训练参数量: {trainable_params / 1e6:.2f}M')
    
    small_unetr.eval()
    x = torch.rand(5, 3, 16, 256, 256).to(device)
    out = small_unetr(x)
    print(f'out输出shape: {out.shape}')


