"""Inference for the Stage-1 PMTNet classifier."""
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

from infer_stage1_diy_clip import DATASET_PATHS, local_biomed_tokenizer
from models import (
    BiomedCLIPTextEncoder,
    PMGViTCLIPClassifier,
    load_clinical_text_map,
)
from train_stage1_binary import PMTStage1Dataset, find_biomedclip_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the Stage-1 PMTNet binary classifier")
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

    text_encoder = BiomedCLIPTextEncoder(str(find_biomedclip_dir()), freeze=True).to(device)
    dataset = PMTStage1Dataset(
        DATASET_PATHS[args.dataset],
        args.split,
        augment=False,
        tokenizer=local_biomed_tokenizer(),
        text_map=load_clinical_text_map(args.dataset, args.split),
        prob_dataset=args.dataset,
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
    )

    output_dir = Path("Checkpoint") / args.task_name
    checkpoint_path = output_dir / f"{args.checkpoint}.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = PMGViTCLIPClassifier(num_classes=2, text_encoder=text_encoder).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    threshold = float(checkpoint.get("threshold", 0.5)) if args.threshold is None else args.threshold
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"threshold must be between 0 and 1, got {threshold}")

    targets, predictions, probabilities, cases = [], [], [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Inference PMTNet {args.dataset} {args.split}"):
            logits = model(batch[0].to(device), batch[4].to(device), batch[3].to(device))
            positive_probability = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            target = batch[1].numpy().astype(int)
            prediction = (positive_probability >= threshold).astype(int)
            targets.extend(target.tolist())
            predictions.extend(prediction.tolist())
            probabilities.extend(positive_probability.tolist())
            cases.extend(list(batch[2]))

    matrix = confusion_matrix(targets, predictions, labels=[0, 1])
    metrics = {
        "accuracy": float(accuracy_score(targets, predictions)),
        "precision": float(precision_score(targets, predictions, zero_division=0)),
        "recall": float(recall_score(targets, predictions, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(targets, predictions)),
        "auc": float(roc_auc_score(targets, probabilities)),
        "threshold": threshold,
        "confusion_matrix": matrix.tolist(),
        "labels": {"0": "original class 0", "1": "original classes 1/2/3"},
        "support": [int(sum(np.asarray(targets) == i)) for i in (0, 1)],
        "dataset": args.dataset,
        "split": args.split,
        "task_name": args.task_name,
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
    }
    suffix = f"{args.split}_{args.checkpoint}_threshold_{threshold:g}"
    (output_dir / f"inference_{suffix}.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / f"predictions_{suffix}.json").write_text(
        json.dumps([
            {"case": c, "target": int(t), "prediction": int(p),
             "positive_probability": round(float(q), 6)}
            for c, t, p, q in zip(cases, targets, predictions, probabilities)
        ], indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
