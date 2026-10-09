import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.utils.model_zoo as model_zoo
from monai.networks.blocks.convolutions import Convolution
from monai.networks.layers.factories import Norm


def channel_shuffle_3D(x, groups):
    batchsize, num_channels, depth, height, width = x.size()
    if groups <= 1 or num_channels % groups != 0:
        return x

    channels_per_group = num_channels // groups
    x = x.view(batchsize, groups, channels_per_group, depth, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batchsize, -1, depth, height, width)
    return x


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        hidden_planes = max(in_planes // ratio, 1)
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
           
        self.fc = nn.Sequential(nn.Conv3d(in_planes, hidden_planes, 1, bias=False),
                               nn.ReLU(),
                               nn.Conv3d(hidden_planes, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv3d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class AvgMaxSpatialPsi(nn.Module):
    def __init__(self, spatial_dims: int, dropout: float = 0.0):
        super().__init__()
        self.proj = nn.Sequential(
            Convolution(
                spatial_dims=spatial_dims,
                in_channels=2,
                out_channels=1,
                kernel_size=1,
                strides=1,
                padding=0,
                dropout=dropout,
                conv_only=True,
            ),
            Norm[Norm.BATCH, spatial_dims](1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        return self.proj(x)


class AttentionGate(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        f_int: int,
        f_g: int,
        f_l: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.W_g = nn.Sequential(
            Convolution(
                spatial_dims=spatial_dims,
                in_channels=f_g,
                out_channels=f_int,
                kernel_size=1,
                strides=1,
                padding=0,
                dropout=dropout,
                conv_only=True,
            ),
            Norm[Norm.BATCH, spatial_dims](f_int),
        )

        self.W_x = nn.Sequential(
            Convolution(
                spatial_dims=spatial_dims,
                in_channels=f_l,
                out_channels=f_int,
                kernel_size=1,
                strides=1,
                padding=0,
                dropout=dropout,
                conv_only=True,
            ),
            Norm[Norm.BATCH, spatial_dims](f_int),
        )

        self.psi = AvgMaxSpatialPsi(spatial_dims=spatial_dims, dropout=dropout)

        self.relu = nn.ReLU()

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class MultiScaleDepthwiseBlock(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0,
        activation: str = "relu6",
        shuffle_groups: int = 2,
    ):
        super().__init__()
        if spatial_dims != 3:
            raise ValueError(f"MultiScaleDepthwiseBlock expects spatial_dims=3, got {spatial_dims}.")
        self.shuffle_groups = shuffle_groups

        act = nn.ReLU6(inplace=True) if activation == "relu6" else nn.ReLU(inplace=True)
        self.pconv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm3d(out_channels),
            act,
        )
        self.dwconvs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv3d(
                        out_channels,
                        out_channels,
                        kernel_size=(kernel_size, kernel_size, kernel_size),
                        stride=1,
                        padding=(kernel_size // 2, kernel_size // 2, kernel_size // 2),
                        groups=out_channels,
                        bias=False,
                    ),
                    nn.BatchNorm3d(out_channels),
                    nn.ReLU6(inplace=True) if activation == "relu6" else nn.ReLU(inplace=True),
                )
                for kernel_size in (1, 3, 5)
            ]
        )
        self.dropout = nn.Dropout3d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.dwconvs[0](x)
        for dwconv in self.dwconvs[1:]:
            out = out + dwconv(x)
        out = self.dropout(out)
        out = channel_shuffle_3D(out, self.shuffle_groups)
        
        return out


class MultiScaleAttentionGate(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        f_int: int,
        f_g: int,
        f_l: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.W_g = MultiScaleDepthwiseBlock(
            spatial_dims=spatial_dims,
            in_channels=f_g,
            out_channels=f_int,
            dropout=dropout,
        )
        self.W_x = MultiScaleDepthwiseBlock(
            spatial_dims=spatial_dims,
            in_channels=f_l,
            out_channels=f_int,
            dropout=dropout,
        )

        self.psi = AvgMaxSpatialPsi(spatial_dims=spatial_dims, dropout=dropout)

        self.relu = nn.ReLU()

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class ComplementaryMultiScaleDepthwiseBlock(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0,
        activation: str = "leakyrelu",
    ):
        super().__init__()
        if spatial_dims != 3:
            raise ValueError(
                f"ComplementaryMultiScaleDepthwiseBlock expects spatial_dims=3, got {spatial_dims}."
            )

        def make_activation() -> nn.Module:
            if activation == "leakyrelu":
                return nn.LeakyReLU(negative_slope=0.2, inplace=True)
            if activation == "relu6":
                return nn.ReLU6(inplace=True)
            if activation == "relu":
                return nn.ReLU(inplace=True)
            raise ValueError(f"Unsupported activation: {activation}")

        self.projection = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm3d(out_channels),
            make_activation(),
        )
        kernels = ((1, 3, 3), (1, 5, 5), (3, 1, 1))
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv3d(
                        out_channels,
                        out_channels,
                        kernel_size=kernel,
                        padding=tuple(size // 2 for size in kernel),
                        groups=out_channels,
                        bias=False,
                    ),
                    nn.InstanceNorm3d(out_channels, affine=True),
                    make_activation(),
                )
                for kernel in kernels
            ]
        )
        self.fusion = nn.Sequential(
            nn.Conv3d(3 * out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm3d(out_channels),
        )
        self.activation = make_activation()
        self.dropout = nn.Dropout3d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        projected = self.projection(x)
        branch_features = [branch(projected) for branch in self.branches]
        fused = self.fusion(torch.cat(branch_features, dim=1))
        return self.dropout(self.activation(projected + fused))


class ComplementaryMultiScaleAttentionGate(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        f_int: int,
        f_g: int,
        f_l: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.W_g = nn.Sequential(
            Convolution(
                spatial_dims=spatial_dims,
                in_channels=f_g,
                out_channels=f_int,
                kernel_size=1,
                strides=1,
                padding=0,
                dropout=dropout,
                conv_only=True,
            ),
            Norm[Norm.BATCH, spatial_dims](f_int),
        )
        self.W_x = ComplementaryMultiScaleDepthwiseBlock(
            spatial_dims=spatial_dims,
            in_channels=f_l,
            out_channels=f_int,
            dropout=dropout,
        )
        self.psi = AvgMaxSpatialPsi(spatial_dims=spatial_dims, dropout=dropout)
        self.relu = nn.ReLU()

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.psi(self.relu(g1 + x1))
        return x * psi




class LearnableGaussianFilterBank3D(nn.Module):
    def __init__(self, kernel_size, num_filters, num_channels):
        super().__init__()
        self.kernel_size = kernel_size
        self.num_filters = num_filters
        self.num_channels = num_channels
        self.padding = kernel_size // 2
        self.sigmas = nn.ParameterList(
            [nn.Parameter(torch.tensor([1.0])) for _ in range(num_filters)]
        )

    def forward(self, x):
        filtered_outputs = []
        for sigma in self.sigmas:
            kernel = self._gaussian_kernel(sigma).to(device=x.device, dtype=x.dtype)
            weight = kernel.repeat(self.num_channels, 1, 1, 1, 1)
            x_pad = F.pad(
                x,
                (self.padding, self.padding, self.padding, self.padding, self.padding, self.padding),
                mode='replicate',
            )
            filtered_outputs.append(F.conv3d(x_pad, weight, groups=self.num_channels))
        return torch.cat(filtered_outputs, dim=1)

    def _gaussian_kernel(self, sigma):
        coords = torch.arange(
            self.kernel_size,
            device=sigma.device,
            dtype=sigma.dtype,
        ) - self.padding
        z, y, x = torch.meshgrid(coords, coords, coords, indexing='ij')
        kernel = torch.exp(-(x ** 2 + y ** 2 + z ** 2) / (2 * sigma.clamp_min(1e-6) ** 2))
        kernel = kernel / kernel.sum()
        return kernel.view(1, 1, self.kernel_size, self.kernel_size, self.kernel_size)
    
# if __name__ == '__main__':
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     out = torch.randn(1, 16, 1, 16, 16).to(device)
#     ca = ChannelAttention(16).to(device)
#     sa = SpatialAttention().to(device)
    
#     out = ca(out) * out
#     out = sa(out) * out
    
#     print(out.shape)
