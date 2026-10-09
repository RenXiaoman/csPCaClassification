"""Inference for the M3-Net-style multi-scale classifier."""
from __future__ import annotations
import argparse, json, os
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, precision_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm
from models.m3net import M3NetClassifier
from train_stage1_binary import M3Stage1Dataset

DATASET_PATHS = {"PICAI": Path("nnUNet/nnUNet_raw/Dataset2302_FullPI-CAI"), "AHCDU": Path("nnUNet/nnUNet_raw/Dataset2301_FullAHCDU")}

def main():
    p = argparse.ArgumentParser(description="Evaluate M3-Net-style classifier")
    p.add_argument("--dataset", choices=tuple(DATASET_PATHS), required=True)
    p.add_argument("--task-name", required=True)
    p.add_argument("--split", choices=("train", "val"), default="val")
    p.add_argument("--checkpoint", choices=("best", "best_auc", "best_recall", "best_loss", "latest"), default="best")
    p.add_argument("--threshold", type=float, default=None); p.add_argument("--gpu_id", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=1); p.add_argument("--num-workers", type=int, default=2)
    args = p.parse_args()
    if args.gpu_id >= 0: os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    device = torch.device("cuda" if args.gpu_id >= 0 and torch.cuda.is_available() else "cpu")
    loader = DataLoader(M3Stage1Dataset(DATASET_PATHS[args.dataset], args.split, augment=False), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    out = Path("Checkpoint") / args.task_name; path = out / f"{args.checkpoint}.pt"
    if not path.exists(): raise FileNotFoundError(path)
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = M3NetClassifier().to(device); model.load_state_dict(checkpoint.get("model", checkpoint)); model.eval()
    threshold = float(checkpoint.get("threshold", .5)) if args.threshold is None else args.threshold
    targets, predictions, probabilities, cases = [], [], [], []
    with torch.no_grad():
        for x32, x64, x96, labels, names in tqdm(loader, desc=f"Inference M3Net {args.dataset} {args.split}"):
            prob = torch.softmax(model(x32.to(device), x64.to(device), x96.to(device)), 1)[:, 1].cpu().numpy()
            targets.extend(labels.numpy().astype(int).tolist()); probabilities.extend(prob.tolist()); predictions.extend((prob >= threshold).astype(int).tolist()); cases.extend(list(names))
    auc = roc_auc_score(targets, probabilities) if len(set(targets)) > 1 else float("nan")
    metrics = {"accuracy": float(accuracy_score(targets, predictions)), "precision": float(precision_score(targets, predictions, zero_division=0)), "recall": float(recall_score(targets, predictions, zero_division=0)), "balanced_accuracy": float(balanced_accuracy_score(targets, predictions)), "auc": None if not np.isfinite(auc) else float(auc), "threshold": threshold, "confusion_matrix": confusion_matrix(targets, predictions, labels=[0,1]).tolist(), "support": [int(sum(np.asarray(targets)==i)) for i in (0,1)], "dataset": args.dataset, "split": args.split, "task_name": args.task_name, "checkpoint": str(path), "checkpoint_epoch": int(checkpoint.get("epoch", -1)), "model": "m3net-compatible"}
    suffix = f"{args.split}_{args.checkpoint}_threshold_{threshold:g}"
    (out / f"inference_{suffix}.json").write_text(json.dumps(metrics, indent=2) + "\n")
    (out / f"predictions_{suffix}.json").write_text(json.dumps([{"case": c, "target": int(t), "prediction": int(q), "positive_probability": round(float(r), 6)} for c,t,q,r in zip(cases, targets, predictions, probabilities)], indent=2) + "\n")
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__": main()
