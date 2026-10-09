"""Full PMG-ViT-CLIP model for three-modality prostate MRI classification.

The gland probability map is used as a soft spatial prior.  It guides the
three MRI modalities through a learnable PMG block, while the resulting image
features are fused with frozen BiomedCLIP text features by cross-attention.
"""
from __future__ import annotations

from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from .vit_text_cross_attention import ViTTextCrossAttentionClassifier


class PMG3D(nn.Module):
    """Probability Map Guided attention from PMCNet, adapted to 3 MRI channels."""

    def __init__(self, in_channels: int = 3) -> None:
        super().__init__()
        self.mri_attention = nn.Sequential(
            nn.Conv3d(in_channels, 1, kernel_size=3, padding=1, bias=True),
            nn.Sigmoid(),
        )
        # Independent learnable gains for T2, ADC and DWI.  exp(0)=1 keeps
        # the initial residual enhancement neutral instead of attenuating a
        # modality as sigmoid(0)=0.5 would do.
        self.modality_gain_raw = nn.Parameter(torch.zeros(in_channels))
        self.alpha_raw = nn.Parameter(torch.tensor(0.0))
        self.beta_raw = nn.Parameter(torch.tensor(0.0))

    def forward(
        self,
        image: torch.Tensor,
        gland_probability: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if image.ndim != 5:
            raise ValueError(f"image must have shape [B, 3, D, H, W], got {tuple(image.shape)}")
        if gland_probability.ndim == 4:
            gland_probability = gland_probability.unsqueeze(1)
        if gland_probability.ndim != 5:
            raise ValueError(
                "gland_probability must have shape [B, D, H, W] or [B, 1, D, H, W], "
                f"got {tuple(gland_probability.shape)}"
            )
        if gland_probability.shape[1] != 1:
            raise ValueError(f"gland_probability must have one channel, got {gland_probability.shape[1]}")
        if gland_probability.shape[2:] != image.shape[2:]:
            gland_probability = F.interpolate(
                gland_probability.float(), size=image.shape[2:], mode="trilinear", align_corners=False
            )
        gland_probability = gland_probability.to(device=image.device, dtype=image.dtype).clamp(0.0, 1.0)
        mri_attention = self.mri_attention(image)
        alpha = torch.sigmoid(self.alpha_raw)
        beta = torch.sigmoid(self.beta_raw)
        attention = torch.sigmoid(alpha * gland_probability + beta * mri_attention)
        modality_gain = torch.exp(self.modality_gain_raw).view(1, -1, 1, 1, 1)
        return image * (1.0 + modality_gain * attention), attention


class PMGViTCLIPClassifier(ViTTextCrossAttentionClassifier):
    """PMG-CLIPNet: three-modality MRI + gland PMG + ViT + BiomedCLIP."""

    model_name = "PMG-CLIPNet"

    def __init__(
        self,
        num_classes: int = 2,
        image_hidden_size: int = 192,
        text_feature_dim: int = 512,
        image_size: Sequence[int] = (16, 256, 256),
        patch_size: Sequence[int] = (4, 16, 16),
        num_layers: int = 6,
        num_heads: int = 6,
        mlp_dim: int = 768,
        dropout: float = 0.1,
        text_encoder: nn.Module | None = None,
    ) -> None:
        super().__init__(
            num_classes=num_classes,
            image_hidden_size=image_hidden_size,
            text_feature_dim=text_feature_dim,
            image_size=image_size,
            patch_size=patch_size,
            num_layers=num_layers,
            num_heads=num_heads,
            mlp_dim=mlp_dim,
            dropout=dropout,
            text_encoder=text_encoder,
        )
        self.pmg = PMG3D(in_channels=3)

    def forward(
        self,
        image: torch.Tensor,
        gland_probability: torch.Tensor,
        text_features: torch.Tensor,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        guided_image, pmg_attention = self.pmg(image, gland_probability)
        result = super().forward(guided_image, text_features, return_attention=return_attention)
        if return_attention:
            logits, text_attention = result
            return logits, text_attention, pmg_attention
        return result
