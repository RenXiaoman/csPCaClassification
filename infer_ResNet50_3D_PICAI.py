"""Inference and evaluation for the MONAI 3D ResNet-50 classifier."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, precision_recall_fscore_support, roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from monai.networks.nets import resnet50
from train_ResNet50_3D import MultimodalClassificationDataset


def main():
    parser = argparse.ArgumentParser(description="Evaluate a MONAI 3D ResNet-50 classifier")
    parser.add_argument("--dataset", choices=("PICAI", "AHCDU"), required=True)
    parser.add_argument("--task-name", required=True)
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--checkpoint", choices=("best", "best_accuracy", "latest"), default="best")
    parser.add_argument("--gpu_id", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    if args.gpu_id >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    device = torch.device("cuda" if args.gpu_id >= 0 and torch.cuda.is_available() else "cpu")
    dataset_paths = {"PICAI": Path("nnUNet/nnUNet_raw/Dataset2302_FullPI-CAI"),
                     "AHCDU": Path("nnUNet/nnUNet_raw/Dataset2301_FullAHCDU")}
    data = MultimodalClassificationDataset(dataset_paths[args.dataset], args.split, augment=False)
    loader = DataLoader(data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                        pin_memory=device.type == "cuda")
    checkpoint_path = Path("Checkpoint") / args.task_name / f"{args.checkpoint}.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state = checkpoint.get("model_state_dict", checkpoint)
    model = resnet50(spatial_dims=3, n_input_channels=3, num_classes=4).to(device)
    model.load_state_dict(state)
    model.eval()

    targets, predictions, probabilities, cases = [], [], [], []
    with torch.no_grad():
        for images, labels, names in tqdm(loader, desc=f"Inference {args.dataset} {args.split}"):
            logits = model(images.to(device))
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            pred = probs.argmax(axis=1)
            targets.extend(labels.numpy().tolist()); predictions.extend(pred.tolist())
            probabilities.extend(probs.tolist()); cases.extend(list(names))
    labels = list(range(4))
    precision, recall, _, support = precision_recall_fscore_support(targets, predictions, labels=labels,
                                                                      average=None, zero_division=0)
    macro_precision, macro_recall, _, _ = precision_recall_fscore_support(
        targets, predictions, labels=labels, average="macro", zero_division=0)
    try:
        auc = roc_auc_score(targets, probabilities, labels=labels, multi_class="ovr", average="macro")
        macro_auc = roc_auc_score(targets, probabilities, labels=labels, multi_class="ovr", average="macro")
    except ValueError:
        auc = macro_auc = float("nan")
    cm = confusion_matrix(targets, predictions, labels=labels)
    metrics = {"accuracy": float(accuracy_score(targets, predictions)),
               "precision": float(macro_precision), "recall": float(macro_recall),
               "auc": float(auc) if np.isfinite(auc) else None,
               "macro_accuracy": float(balanced_accuracy_score(targets, predictions)),
               "macro_precision": float(macro_precision), "macro_recall": float(macro_recall),
               "macro_auc": float(macro_auc) if np.isfinite(macro_auc) else None,
               "per_class_precision": precision.tolist(), "per_class_recall": recall.tolist(),
               "support": support.tolist(), "confusion_matrix": cm.tolist(),
               "dataset": args.dataset, "split": args.split, "task_name": args.task_name,
               "checkpoint": str(checkpoint_path)}
    output_dir = Path("Checkpoint") / args.task_name
    (output_dir / f"inference_{args.split}_{args.checkpoint}.json").write_text(json.dumps(metrics, indent=2) + "\n")
    predictions_out = [{"case": c, "target": int(t), "prediction": int(p),
                        "probabilities": [round(float(x), 6) for x in prob]}
                       for c, t, p, prob in zip(cases, targets, predictions, probabilities)]
    (output_dir / f"predictions_{args.split}_{args.checkpoint}.json").write_text(json.dumps(predictions_out, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
