"""Dependency-light TomoGraphView-style 3D classifier.

The paper implementation extracts 2D foundation-model features from
omnidirectional views and aggregates them with a spherical GNN.  This
standalone variant keeps the view-graph idea while learning a compact 2D
slice encoder end-to-end, so it can use the existing Stage-1 dataset.
"""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class _ViewEncoder(nn.Module):
    def __init__(self, in_channels: int, feature_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, 5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(32), nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.GELU(),
            nn.Conv2d(64, feature_dim, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_dim), nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).flatten(1)


class _GraphSAGE(nn.Module):
    """Small complete-graph message-passing block without torch_geometric."""
    def __init__(self, dim: int):
        super().__init__()
        self.self_proj = nn.Linear(dim, dim)
        self.neighbor_proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, views, features]. Every view receives all other views.
        n = x.size(1)
        neighbor = (x.sum(dim=1, keepdim=True) - x) / max(n - 1, 1)
        return self.norm(x + F.gelu(self.self_proj(x) + self.neighbor_proj(neighbor)))


class TomoGraphViewClassifier(nn.Module):
    """TomoGraphView-style classifier for [B, C, D, H, W] volumes.

    Six nodes are created from mean/max projections along the three volume
    axes. They are resized to a common 2D canvas, encoded by a shared CNN,
    and combined by two complete-graph message-passing blocks.
    """
    def __init__(self, in_channels: int = 3, num_classes: int = 2,
                 feature_dim: int = 128, image_size: int = 96):
        super().__init__()
        self.image_size = image_size
        self.encoder = _ViewEncoder(in_channels, feature_dim)
        self.graph1 = _GraphSAGE(feature_dim)
        self.graph2 = _GraphSAGE(feature_dim)
        self.head = nn.Sequential(nn.LayerNorm(feature_dim), nn.Linear(feature_dim, num_classes))

    def _views(self, x: torch.Tensor) -> torch.Tensor:
        # [B,C,D,H,W] -> six [B,C,H,W]-like projections.
        projections = [
            x.mean(dim=2), x.amax(dim=2),
            x.mean(dim=3), x.amax(dim=3),
            x.mean(dim=4), x.amax(dim=4),
        ]
        views = []
        for view in projections:
            views.append(F.interpolate(view, size=(self.image_size, self.image_size),
                                       mode="bilinear", align_corners=False))
        return torch.stack(views, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(f"Expected [B,C,D,H,W], got {tuple(x.shape)}")
        views = self._views(x)
        b, v, c, h, w = views.shape
        features = self.encoder(views.reshape(b * v, c, h, w)).reshape(b, v, -1)
        features = self.graph2(self.graph1(features))
        return self.head(features.mean(dim=1))
