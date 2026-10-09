"""M3-Net-style multi-scale 3D classifier.

Adapted from the public M3-Net idea (macro/meso/micro inputs). Each case is
represented at 32^3, 64^3 and 96^3 resolutions, followed by cross-scale
attention and transformer fusion. The model accepts the repository's three
MRI channels directly and returns binary classification logits.
"""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class _Backbone3D(nn.Module):
    def __init__(self, in_channels: int, dim: int = 128):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv3d(in_channels, 32, 3, 2, 1, bias=False),
            nn.InstanceNorm3d(32, affine=True), nn.GELU(),
            nn.Conv3d(32, 64, 3, 2, 1, bias=False),
            nn.InstanceNorm3d(64, affine=True), nn.GELU(),
            nn.Conv3d(64, 96, 3, 2, 1, bias=False),
            nn.InstanceNorm3d(96, affine=True), nn.GELU(),
            nn.Conv3d(96, dim, 3, 2, 1, bias=False),
            nn.InstanceNorm3d(dim, affine=True), nn.GELU(),
            nn.AdaptiveAvgPool3d(1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x).flatten(1)


class _CrossScaleAttention(nn.Module):
    def __init__(self, dim: int = 128, heads: int = 4):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True, dropout=0.1)
        self.norm1 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, _ = self.attn(x, x, x, need_weights=False)
        x = self.norm1(x + a)
        return self.norm2(x + self.ffn(x))


class M3NetClassifier(nn.Module):
    """Three-scale M3-Net-style classifier for [B, 3, D, H, W] volumes."""
    def __init__(self, in_channels: int = 3, num_classes: int = 2, feature_dim: int = 128):
        super().__init__()
        self.net32 = _Backbone3D(in_channels, feature_dim)
        self.net64 = _Backbone3D(in_channels, feature_dim)
        self.net96 = _Backbone3D(in_channels, feature_dim)
        self.fusion = _CrossScaleAttention(feature_dim)
        self.head = nn.Sequential(
            nn.LayerNorm(feature_dim), nn.Linear(feature_dim, feature_dim // 2),
            nn.GELU(), nn.Dropout(0.3), nn.Linear(feature_dim // 2, num_classes)
        )

    def forward(self, img32: torch.Tensor, img64: torch.Tensor | None = None,
                img96: torch.Tensor | None = None) -> torch.Tensor:
        # A single volume is accepted for convenience; the training dataset
        # supplies precomputed scale tensors to avoid repeated interpolation.
        if img64 is None or img96 is None:
            img64 = F.interpolate(img32, size=(64, 64, 64), mode="trilinear", align_corners=False)
            img96 = F.interpolate(img32, size=(96, 96, 96), mode="trilinear", align_corners=False)
            img32 = F.interpolate(img32, size=(32, 32, 32), mode="trilinear", align_corners=False)
        features = torch.stack([self.net32(img32), self.net64(img64), self.net96(img96)], dim=1)
        fused = self.fusion(features).mean(dim=1)
        return self.head(fused)
