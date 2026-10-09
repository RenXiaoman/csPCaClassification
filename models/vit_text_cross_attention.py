"""3D ViT + BiomedCLIP text cross-attention classifier."""
from __future__ import annotations

from typing import Sequence
from pathlib import Path
import json

import torch
from torch import nn
from monai.networks.nets import ViT


# Generated clinical-text mappings are intentionally fixed to the project
# layout so training does not need another path argument.
CLINICAL_TEXT_PATHS = {
    "PICAI": {
        "train": Path("nnUNet/nnUNet_raw/Dataset141_FullPICAI/clinical_text_train.json"),
        "val": Path("nnUNet/nnUNet_raw/Dataset141_FullPICAI/clinical_text_val.json"),
    },
    "AHCDU": {
        "train": Path("nnUNet/nnUNet_raw/Dataset130_ProstateAHCDU/clinical_text_train.json"),
        "val": Path("nnUNet/nnUNet_raw/Dataset130_ProstateAHCDU/clinical_text_val.json"),
    },
}


def load_clinical_text_map(dataset: str, split: str) -> dict[str, str]:
    """Load the pre-generated case-name -> clinical sentence mapping."""
    try:
        path = CLINICAL_TEXT_PATHS[dataset][split]
    except KeyError as exc:
        raise ValueError(f"Unsupported dataset/split: {dataset}/{split}") from exc
    if not path.exists():
        raise FileNotFoundError(f"Clinical text mapping not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Clinical text mapping must be a JSON object: {path}")
    return {str(k): str(v) for k, v in data.items()}


class BiomedCLIPTextEncoder(nn.Module):
    """Load the local BiomedCLIP checkpoint and expose its text encoder."""

    def __init__(self, local_dir: str, freeze: bool = True) -> None:
        super().__init__()
        import json
        from open_clip import create_model_and_transforms
        from open_clip.factory import _MODEL_CONFIGS

        root = __import__("pathlib").Path(local_dir)
        config_path = root / "open_clip_config.json"
        weight_path = root / "open_clip_pytorch_model.bin"
        
        
        if not config_path.exists() or not weight_path.exists():
            raise FileNotFoundError(
                f"BiomedCLIP requires {config_path} and {weight_path}"
            )
        with config_path.open() as f:
            config = json.load(f)
        model_name = "biomedclip_local_model"
        _MODEL_CONFIGS[model_name] = config["model_cfg"]
        model, _, _ = create_model_and_transforms(
            model_name=model_name,
            pretrained=str(weight_path),
            **{f"image_{k}": v for k, v in config["preprocess_cfg"].items()},
        )
        self.clip = model
        if freeze:
            for parameter in self.clip.parameters():
                parameter.requires_grad = False
            self.clip.eval()

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # OpenCLIP provides the same L2 normalization through its official
        # normalize=True path, used for cosine-similarity image/text matching.
        return self.clip.encode_text(tokens, normalize=True)


class ViTTextCrossAttentionClassifier(nn.Module):
    """Fuse final 3D ViT image tokens with BiomedCLIP text embeddings.

    Args:
        num_classes: Number of output classes.
        image_hidden_size: Hidden size of the 3D ViT image encoder.
        text_feature_dim: Dimension of the BiomedCLIP text embedding.
        image_size: Input volume spatial size (D, H, W).
        patch_size: 3D ViT patch size.
        num_layers: Number of image ViT transformer blocks.
    """

    def __init__(
        self,
        num_classes: int = 4,
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
        super().__init__()
        self.image_encoder = ViT(
            in_channels=3,
            img_size=tuple(image_size),
            patch_size=tuple(patch_size),
            hidden_size=image_hidden_size,
            mlp_dim=mlp_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            classification=False,
            dropout_rate=dropout,
            spatial_dims=3,
        )
        self.text_encoder = text_encoder
        
        
        self.image_projection = nn.Sequential(
            nn.LayerNorm(image_hidden_size),
            nn.Linear(image_hidden_size, image_hidden_size),
        )
        self.text_projection = nn.Sequential(
            nn.LayerNorm(text_feature_dim),
            nn.Linear(text_feature_dim, image_hidden_size),
        )
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=image_hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.cross_norm = nn.LayerNorm(image_hidden_size)
        self.classifier = nn.Sequential(
            nn.LayerNorm(image_hidden_size * 2),
            nn.Linear(image_hidden_size * 2, image_hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(image_hidden_size, num_classes),
        )

    def encode_image_tokens(self, image: torch.Tensor) -> torch.Tensor:
        """Return the final ViT block tokens with shape [B, N, image_hidden_size]."""
        _, hidden_states = self.image_encoder(image)
        return hidden_states[-1]

    def forward(
        self,
        image: torch.Tensor,
        text_features: torch.Tensor,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        image_tokens = self.encode_image_tokens(image)
        if self.text_encoder is not None and text_features.dtype in (torch.int8, torch.int16, torch.int32, torch.int64):
            text_features = self.text_encoder(text_features)
        # ViT returns [B, N, C] tokens rather than a CNN feature map. Mean
        # pooling is the token analogue of global pooling before fusion.
        image_summary = self.image_projection(image_tokens.mean(dim=1)).unsqueeze(1)
        if text_features.ndim == 2:
            text_token = self.text_projection(text_features).unsqueeze(1)
        elif text_features.ndim == 3:
            text_token = self.text_projection(text_features)
        else:
            raise ValueError(
                f"text_features must have shape [B, F] or [B, L, F], got {tuple(text_features.shape)}"
            )
        fused_tokens, attention = self.cross_attention(
            query=image_summary,
            key=text_token,
            value=text_token,
            need_weights=return_attention,
        )
        fused_tokens = self.cross_norm(image_summary + fused_tokens)
        image_summary = fused_tokens.squeeze(1)
        text_summary = text_token.mean(dim=1)
        logits = self.classifier(torch.cat([image_summary, text_summary], dim=1))
        if return_attention:
            return logits, attention
        return logits

    def freeze_text_projection(self) -> None:
        """Freeze the trainable projection if text features are kept fixed."""
        for parameter in self.text_projection.parameters():
            parameter.requires_grad = False
