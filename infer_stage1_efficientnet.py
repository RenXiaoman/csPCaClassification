"""Inference for the MONAI 3D EfficientNet Stage-1 binary classifier."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from train_stage1_binary import Stage1Dataset
from models.efficientnet3d import EfficientNet3DClassifier


DATASET_PATHS = {
    "PICAI": Path("nnUNet/nnUNet_raw/Dataset2302_FullPI-CAI"),
    "AHCDU": Path("nnUNet/nnUNet_raw/Dataset2301_FullAHCDU"),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a MONAI 3D EfficientNet Stage-1 classifier")
    parser.add_argument("--dataset", choices=tuple(DATASET_PATHS), required=True)
    parser.add_argument("--task-name", required=True)
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--checkpoint", choices=("best", "best_auc", "best_recall", "best_loss", "latest"), default="best")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=3)
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args()

    if args.gpu_id >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    device = torch.device("cuda" if args.gpu_id >= 0 and torch.cuda.is_available() else "cpu")
    dataset = Stage1Dataset(DATASET_PATHS[args.dataset], args.split, augment=False)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    output_dir = Path("Checkpoint") / args.task_name
    checkpoint_path = output_dir / f"{args.checkpoint}.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = EfficientNet3DClassifier(num_classes=2).to(device)
    model.load_state_dict(checkpoint.get("model", checkpoint))
    model.eval()
    threshold = float(checkpoint.get("threshold", 0.5)) if args.threshold is None else args.threshold
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"threshold must be between 0 and 1, got {threshold}")

    targets, predictions, probabilities, cases = [], [], [], []
    with torch.no_grad():
        for images, labels, names in tqdm(loader, desc=f"Inference EfficientNet {args.dataset} {args.split}"):
            logits = model(images.to(device))
            positive_probability = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            target = labels.numpy().astype(int)
            prediction = (positive_probability >= threshold).astype(int)
            targets.extend(target.tolist())
            predictions.extend(prediction.tolist())
            probabilities.extend(positive_probability.tolist())
            cases.extend(list(names))

    matrix = confusion_matrix(targets, predictions, labels=[0, 1])
    try:
        auc = float(roc_auc_score(targets, probabilities))
    except ValueError:
        auc = float("nan")
    metrics = {
        "accuracy": float(accuracy_score(targets, predictions)),
        "precision": float(precision_score(targets, predictions, zero_division=0)),
        "recall": float(recall_score(targets, predictions, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(targets, predictions)),
        "auc": auc if np.isfinite(auc) else None,
        "threshold": threshold,
        "confusion_matrix": matrix.tolist(),
        "labels": {"0": "original class 0", "1": "original classes 1/2/3"},
        "support": [int(sum(np.asarray(targets) == i)) for i in (0, 1)],
        "dataset": args.dataset,
        "split": args.split,
        "task_name": args.task_name,
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "model": "efficientnet-b0",
    }
    suffix = f"{args.split}_{args.checkpoint}_threshold_{threshold:g}"
    (output_dir / f"inference_{suffix}.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    (output_dir / f"predictions_{suffix}.json").write_text(
        json.dumps([
            {"case": c, "target": int(t), "prediction": int(p), "positive_probability": round(float(q), 6)}
            for c, t, p, q in zip(cases, targets, predictions, probabilities)
        ], indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
