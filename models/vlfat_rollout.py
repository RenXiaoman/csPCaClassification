"""VLFATRollout-style volumetric classifier.

Adapted from the public VLFATRollout design: a 2D DeiT feature extractor is
applied independently to each volume slice, followed by a temporal
Transformer over the slice tokens. The current project supplies volumes as
``[B, C, D, H, W]``; the model internally converts them to ``[B, D, C, H, W]``.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class VLFATRolloutClassifier(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 2,
        image_size: int = 224,
        spatial_model: str = "deit_base_patch16_224",
        temporal_depth: int = 12,
        temporal_heads: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if in_channels != 3:
            raise ValueError("VLFATRollout currently expects three MRI modalities")
        if temporal_depth < 1 or temporal_heads < 1:
            raise ValueError("temporal_depth and temporal_heads must be positive")
        import timm

        self.image_size = int(image_size)
        # The upstream model uses DeiT-B/16 with 768-dimensional CLS features.
        self.spatial = timm.create_model(
            spatial_model,
            pretrained=False,
            num_classes=0,
            global_pool="avg",
            in_chans=in_channels,
        )
        dim = int(getattr(self.spatial, "num_features", 768))
        if dim % temporal_heads:
            raise ValueError(f"spatial feature dimension {dim} is not divisible by temporal_heads={temporal_heads}")
        self.temporal_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.temporal_pos = nn.Parameter(torch.zeros(1, 17, dim))
        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=temporal_heads,
            dim_feedforward=4 * dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.temporal = nn.TransformerEncoder(layer, num_layers=temporal_depth)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, num_classes))
        nn.init.normal_(self.temporal_token, std=0.02)
        nn.init.normal_(self.temporal_pos, std=0.02)

    def _positional_encoding(self, length: int) -> torch.Tensor:
        if length == self.temporal_pos.shape[1]:
            return self.temporal_pos
        # Variable-depth volumes are supported in the same way as VLFATRollout.
        return F.interpolate(
            self.temporal_pos.transpose(1, 2), size=length, mode="linear", align_corners=False
        ).transpose(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(f"expected [B,C,D,H,W], got {tuple(x.shape)}")
        b, c, depth, height, width = x.shape
        slices = x.permute(0, 2, 1, 3, 4).reshape(b * depth, c, height, width)
        slices = F.interpolate(slices, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
        spatial = self.spatial(slices).reshape(b, depth, -1)
        cls = self.temporal_token.expand(b, -1, -1)
        sequence = torch.cat((cls, spatial), dim=1)
        sequence = self.dropout(sequence + self._positional_encoding(sequence.shape[1]))
        temporal = self.temporal(sequence)
        return self.head(temporal[:, 0])

