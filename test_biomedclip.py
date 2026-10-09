"""Minimal BiomedCLIP smoke test for text and optional 2D image encoding.

Install the model-card dependencies first:
  pip install open_clip_torch==2.23.0 transformers==4.35.2 matplotlib
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from open_clip import create_model_and_transforms, get_tokenizer
from open_clip.factory import HF_HUB_PREFIX, _MODEL_CONFIGS


MODEL_ID = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", help="optional local 2D image path")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    project_dir = Path(__file__).resolve().parent
    local_candidates = sorted(project_dir.glob(
        "BiomedCLIP/models--microsoft--BiomedCLIP-PubMedBERT_256-vit_base_patch16_224/snapshots/*"
    ))
    local_dir = next((p for p in local_candidates
                      if (p / "open_clip_config.json").exists()
                      and (p / "open_clip_pytorch_model.bin").exists()), None)
    if local_dir is not None:
        with (local_dir / "open_clip_config.json").open() as f:
            config = json.load(f)
        model_name = "biomedclip_local"
        _MODEL_CONFIGS[model_name] = config["model_cfg"]
        model, _, preprocess = create_model_and_transforms(
            model_name=model_name,
            pretrained=str(local_dir / "open_clip_pytorch_model.bin"),
            **{f"image_{k}": v for k, v in config["preprocess_cfg"].items()},
        )
        tokenizer = get_tokenizer(model_name)
        print(f"model_source={local_dir}")
    else:
        cache_dir = project_dir / "Checkpoint" / "BiomedCLIP"
        cache_dir.mkdir(parents=True, exist_ok=True)
        from open_clip import create_model_from_pretrained
        model, preprocess = create_model_from_pretrained(MODEL_ID, cache_dir=str(cache_dir))
        tokenizer = get_tokenizer(MODEL_ID, cache_dir=str(cache_dir))
        print(f"model_cache={cache_dir}")
    model = model.to(device).eval()

    texts = [
        "Patient age is 68 years. PSAD is 0.42 ng/mL^2. Total PSA is 18.6 ng/mL.",
        "Patient age is 59 years. PSAD is 0.08 ng/mL^2. Total PSA is 6.2 ng/mL.",
    ]
    tokens = tokenizer(texts, context_length=256).to(device)
    with torch.no_grad():
        text_features = model.encode_text(tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    print(f"device={device}")
    print(f"text_features shape={tuple(text_features.shape)}")
    print(f"text_features finite={bool(torch.isfinite(text_features).all())}")
    print(f"text cosine similarity={float(text_features[0] @ text_features[1]):.6f}")

    if args.image:
        from PIL import Image

        image = preprocess(Image.open(args.image).convert("RGB")).unsqueeze(0).to(device)
        with torch.no_grad():
            image_features = model.encode_image(image)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            similarity = model.logit_scale.exp() * image_features @ text_features.T
        print(f"image_features shape={tuple(image_features.shape)}")
        print(f"image_text logits shape={tuple(similarity.shape)}")


if __name__ == "__main__":
    main()
