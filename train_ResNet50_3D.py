"""Train a MONAI 3D ResNet-50 on the nnU-Net multimodal cases.

Labels are read from labelsTr/A_labels.json and labelsTs/A_labels.json.
The JSON ``value`` field is used as the class index (0..3).
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

from monai.networks.nets import resnet50
from Options.Options_ResNet50_3D import parse_classification_options
from dataset_nnunet import Lits_DataSet
from classification_sampling import ClassBalancedSampler


class MultimodalClassificationDataset(Lits_DataSet):
    def __init__(self, dataset_dir: Path, split: str, augment: bool = False):
        self.dataset_dir = Path(dataset_dir)
        image_dir_name = "imagesTr" if split == "train" else "imagesTs"
        label_dir_name = "labelsTr" if split == "train" else "labelsTs"
        # Reuse Lits_DataSet's clipping, z-score normalization and nnU-Net
        # style intensity/spatial augmentation implementation.
        super().__init__(self.dataset_dir, image_dir_name, label_dir_name,
                         enable_augmentation=augment)
        self.image_dir = self.dataset_dir / image_dir_name
        label_dir = self.dataset_dir / label_dir_name
        label_file = label_dir / "A_labels.json"
        # Keep compatibility with the pre-existing 2301 training annotation name.
        if not label_file.exists():
            legacy_file = label_dir / "labels.json"
            if legacy_file.exists():
                label_file = legacy_file
            else:
                raise FileNotFoundError(f"Missing classification labels: {label_file}")
        records = json.loads(label_file.read_text())
        self.records = []
        for record in records:
            case = str(record["case"])
            value = int(record["value"])
            if value < 0 or value > 3:
                raise ValueError(f"Class value must be 0..3, got {value} for {case}")
            paths = [self.image_dir / f"{case}_{channel:04d}.nii.gz" for channel in range(3)]
            missing = [str(p) for p in paths if not p.exists()]
            if missing:
                raise FileNotFoundError(f"Missing modality for {case}: {missing}")
            self.records.append((case, value, paths))
        if not self.records:
            raise RuntimeError(f"No cases found in {label_file}")

    def __getitem__(self, index: int):
        case, value, paths = self.records[index]
        channels = []
        for path in paths:
            raw = self.load(path).astype(np.float32)
            channels.append(self.z_score_normalization(self.clip(raw)))
        image = np.stack(channels, axis=0).astype(np.float32)
        if self.enable_augmentation:
            image = np.stack([self._apply_intensity_transforms(channel) for channel in image], axis=0)
            data_dict = self.transforms(data=image[None, ...], seg=np.zeros((1, 1, *image.shape[1:]), dtype=np.float32))
            image = data_dict["data"][0]
        return torch.from_numpy(np.ascontiguousarray(image)), torch.tensor(value, dtype=torch.long), case


def evaluate(model, loader, device, num_classes=4, criterion=None):
    model.eval()
    targets_all, predictions_all, probabilities_all = [], [], []
    total_loss, batches = 0.0, 0
    with torch.no_grad():
        for images, targets, _ in loader:
            logits = model(images.to(device))
            if criterion is not None:
                total_loss += float(criterion(logits, targets.to(device)).item())
                batches += 1
            probabilities = torch.softmax(logits, dim=1).cpu()
            predictions = probabilities.argmax(1)
            targets = targets.cpu()
            targets_all.extend(targets.tolist())
            predictions_all.extend(predictions.tolist())
            probabilities_all.extend(probabilities.tolist())
    precision, recall, _, _ = precision_recall_fscore_support(
        targets_all, predictions_all, labels=list(range(num_classes)), average="macro", zero_division=0
    )
    accuracy = accuracy_score(targets_all, predictions_all)
    try:
        auc = roc_auc_score(targets_all, probabilities_all, labels=list(range(num_classes)),
                            multi_class="ovr", average="macro")
    except ValueError:
        auc = float("nan")
    confusion = torch.zeros((num_classes, num_classes), dtype=torch.long)
    for target, prediction in zip(targets_all, predictions_all):
        confusion[target, prediction] += 1
    return {"accuracy": float(accuracy), "precision": float(precision), "recall": float(recall), "auc": float(auc)}, confusion, total_loss / max(batches, 1)


def plot_metric(history, metric, output_dir):
    """Save one train/validation curve; train is red and validation is blue."""
    epochs = history["epoch"]
    plt.figure(figsize=(9, 5))
    plt.plot(epochs, history[f"train_{metric}"], "r-", label="train")
    plt.plot(epochs, history[f"val_{metric}"], "b-", label="val")
    plt.xlabel("Epoch")
    plt.ylabel(metric.upper())
    plt.title(f"{metric.upper()} curve")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f"{metric}.png", dpi=150)
    plt.close()


def rounded_metric(value):
    """Round persisted metrics while retaining NaN for undefined AUC."""
    return round(float(value), 4) if np.isfinite(value) else None


def main():
    args = parse_classification_options()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)

    if args.gpu_id >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    device = torch.device("cuda" if args.gpu_id >= 0 and torch.cuda.is_available() else "cpu")
    dataset_paths = {
        "PICAI": Path("nnUNet/nnUNet_raw/Dataset2302_FullPI-CAI"),
        "AHCDU": Path("nnUNet/nnUNet_raw/Dataset2301_FullAHCDU"),
    }
    args.datapath = dataset_paths[args.dataset]
    train_set = MultimodalClassificationDataset(args.datapath, "train", augment=True)
    train_eval_set = MultimodalClassificationDataset(args.datapath, "train", augment=False)
    val_set = MultimodalClassificationDataset(args.datapath, "val", augment=False)
    if args.sample:
        train_sampler = ClassBalancedSampler([value for _, value, _ in train_set.records], args.dataset)
        print("training inverse-frequency sampler counts=", train_sampler.class_counts,
              "samples_per_epoch=", len(train_sampler))
        train_loader = DataLoader(train_set, args.batch_size, sampler=train_sampler,
                                  num_workers=args.num_workers, pin_memory=device.type == "cuda")
    else:
        print("training sampler=disabled; using original class distribution")
        train_loader = DataLoader(train_set, args.batch_size, shuffle=True,
                                  num_workers=args.num_workers, pin_memory=device.type == "cuda")
    train_eval_loader = DataLoader(train_eval_set, args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_set, args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=device.type == "cuda")

    model = resnet50(spatial_dims=3, n_input_channels=3, num_classes=args.num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    args.output_dir = Path("Checkpoint") / args.task_name
    args.output_dir.mkdir(parents=True, exist_ok=True)
    start_epoch = 1
    best_key = (-1.0, -1.0)  # primary macro-AUC, secondary accuracy
    best_accuracy = -1.0
    history = {"epoch": [], "train_loss": [], "val_loss": [],
               "train_accuracy": [], "val_accuracy": [],
               "train_precision": [], "val_precision": [],
               "train_recall": [], "val_recall": [],
               "train_auc": [], "val_auc": []}
    latest = args.output_dir / "latest.pt"
    if args.resume:
        if not latest.exists():
            raise FileNotFoundError(f"--resume requested but checkpoint does not exist: {latest}")
        checkpoint = torch.load(latest, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scaler_state_dict" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        start_epoch = int(checkpoint["epoch"]) + 1
        saved_key = checkpoint.get("best_key")
        best_key = tuple(saved_key) if saved_key is not None else best_key
        best_accuracy = checkpoint.get("best_accuracy", best_accuracy)
        metrics_file = args.output_dir / "metrics.jsonl"
        if metrics_file.exists():
            for line in metrics_file.read_text().splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                def history_value(key):
                    value = row.get(key, float("nan"))
                    return float(value) if value is not None else float("nan")
                history["epoch"].append(row["epoch"])
                history["train_loss"].append(history_value("train_loss"))
                history["val_loss"].append(history_value("val_loss"))
                for metric in ("accuracy", "precision", "recall", "auc"):
                    history[f"train_{metric}"].append(history_value(f"train_{metric}"))
                    history[f"val_{metric}"].append(history_value(f"val_{metric}"))
        print(f"resuming from {latest} at epoch {start_epoch}")
    (args.output_dir / "options.json").write_text(json.dumps(vars(args), default=str, indent=2) + "\n")

    print(f"device={device} train_cases={len(train_set)} val_cases={len(val_set)} classes=4")
    for epoch in range(start_epoch, args.epochs + 1):
        model.train(); running_loss = 0.0
        for images, targets, _ in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}"):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                loss = criterion(model(images.to(device)), targets.to(device))
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            running_loss += loss.item()
        train_metrics, _, _ = evaluate(model, train_eval_loader, device, args.num_classes, criterion)
        metrics, confusion, val_loss = evaluate(model, val_loader, device, args.num_classes, criterion)
        train_loss = running_loss / max(len(train_loader), 1)
        history["epoch"].append(epoch); history["train_loss"].append(train_loss); history["val_loss"].append(val_loss)
        for metric in ("accuracy", "precision", "recall", "auc"):
            history[f"train_{metric}"].append(train_metrics[metric])
            history[f"val_{metric}"].append(metrics[metric])
            plot_metric(history, metric, args.output_dir)
        plot_metric(history, "loss", args.output_dir)
        print(f"epoch={epoch} train_loss={running_loss / max(len(train_loader), 1):.5f} "
              f"train_accuracy={train_metrics['accuracy']:.4f} train_precision={train_metrics['precision']:.4f} "
              f"train_recall={train_metrics['recall']:.4f} train_auc={train_metrics['auc']:.4f} "
              f"val_accuracy={metrics['accuracy']:.4f} val_precision={metrics['precision']:.4f} "
              f"val_recall={metrics['recall']:.4f} val_auc={metrics['auc']:.4f}")
        auc_key = metrics["auc"] if np.isfinite(metrics["auc"]) else -1.0
        key = (auc_key, metrics["accuracy"])
        improved_auc = key > best_key
        improved_accuracy = metrics["accuracy"] > best_accuracy
        if improved_auc:
            best_key = key
        if improved_accuracy:
            best_accuracy = metrics["accuracy"]
        checkpoint = {"epoch": epoch, "model_state_dict": model.state_dict(),
                      "optimizer_state_dict": optimizer.state_dict(), "metrics": metrics,
                      "scaler_state_dict": scaler.state_dict(),
                      "best_key": best_key, "best_accuracy": best_accuracy,
                      "options": vars(args)}
        torch.save(checkpoint, args.output_dir / "latest.pt")
        with (args.output_dir / "metrics.jsonl").open("a") as f:
            row = {"epoch": epoch, "train_loss": rounded_metric(train_loss), "val_loss": rounded_metric(val_loss),
                   **{f"train_{k}": rounded_metric(v) for k, v in train_metrics.items()},
                   **{f"val_{k}": rounded_metric(v) for k, v in metrics.items()}}
            f.write(json.dumps(row, allow_nan=False) + "\n")
        if improved_auc:
            torch.save(checkpoint, args.output_dir / "best.pt")
            print("new best (macro-AUC, accuracy)=", key, "\nconfusion_matrix=\n", confusion.numpy())
        if improved_accuracy:
            torch.save(checkpoint, args.output_dir / "best_accuracy.pt")


if __name__ == "__main__":
    main()
