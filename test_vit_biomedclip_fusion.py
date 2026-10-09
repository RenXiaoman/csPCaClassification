"""Smoke test for local BiomedCLIP text + 3D ViT cross-attention fusion."""
from __future__ import annotations

from pathlib import Path

import torch
from open_clip import get_tokenizer

from models import BiomedCLIPTextEncoder, ViTTextCrossAttentionClassifier


def main() -> None:
    project = Path(__file__).resolve().parent
    candidates = sorted(project.glob(
        "BiomedCLIP/models--microsoft--BiomedCLIP-PubMedBERT_256-vit_base_patch16_224/snapshots/*"
    ))
    local_dir = next((p for p in candidates
                      if (p / "open_clip_config.json").exists()
                      and (p / "open_clip_pytorch_model.bin").exists()), None)
    if local_dir is None:
        raise FileNotFoundError("Local BiomedCLIP snapshot was not found under ./BiomedCLIP")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    text_encoder = BiomedCLIPTextEncoder(str(local_dir), freeze=True).to(device)
    tokenizer = get_tokenizer("biomedclip_local_model")
    fusion = ViTTextCrossAttentionClassifier(
        num_classes=4, text_feature_dim=512, text_encoder=text_encoder
    ).to(device).eval()

    text = ["Patient age is 68 years. PSAD is 0.42 ng/mL^2. Total PSA is 18.6 ng/mL."]
    tokens = tokenizer(text, context_length=256).to(device)
    image = torch.randn(1, 3, 16, 256, 256, device=device)
    with torch.no_grad():
        logits, attention = fusion(image, tokens, return_attention=True)

    print(f"device={device}")
    print(f"local_biomedclip={local_dir}")
    print(f"image shape={tuple(image.shape)}")
    print(f"token shape={tuple(tokens.shape)}")
    print(f"logits shape={tuple(logits.shape)}")
    print(f"attention shape={tuple(attention.shape)}")
    print(f"logits finite={bool(torch.isfinite(logits).all())}")


if __name__ == "__main__":
    main()
