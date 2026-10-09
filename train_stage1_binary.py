"""Stage-1 binary classifier: original class 0 versus original classes 1/2/3."""
from __future__ import annotations

import os

# Some hosted environments export OMP/MKL thread counts as 0. OpenMP treats
# that as invalid and can fail before the first training log is printed.
def _sanitize_thread_env() -> None:
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        value = os.environ.get(name)
        try:
            valid = value is not None and int(value) > 0
        except (TypeError, ValueError):
            valid = False
        if not valid:
            os.environ[name] = "1"


_sanitize_thread_env()

import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

from train_ResNet50_3D import MultimodalClassificationDataset
from classification_sampling import ClassBalancedSampler
from Options.Options_Swin3D import parse_swin_options


BIOMEDCLIP_LOCAL = Path(
    "BiomedCLIP/models--microsoft--BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
)


def find_biomedclip_dir():
    candidates = sorted(BIOMEDCLIP_LOCAL.glob("snapshots/*"))
    for path in candidates:
        if (path / "open_clip_config.json").exists() and (path / "open_clip_pytorch_model.bin").exists():
            return path
    raise FileNotFoundError("Local BiomedCLIP snapshot not found under ./BiomedCLIP")


def clinical_text_map(dataset, split):
    """Load the split-specific mapping defined by the fusion model module."""
    from models import load_clinical_text_map
    return load_clinical_text_map(dataset, split)


class Stage1Dataset(MultimodalClassificationDataset):
    def __getitem__(self, index):
        image, label, case = super().__getitem__(index)
        return image, torch.tensor(int(label > 0), dtype=torch.long), case


class M3Stage1Dataset(Stage1Dataset):
    """Return the same case at M3-Net's 32/64/96 cubic scales."""
    def __getitem__(self, index):
        image, label, case = super().__getitem__(index)
        image = image.unsqueeze(0)
        scales = [torch.nn.functional.interpolate(
            image, size=(size, size, size), mode="trilinear", align_corners=False
        ).squeeze(0) for size in (32, 64, 96)]
        return scales[0], scales[1], scales[2], label, case


class DiyStage1Dataset(Stage1Dataset):
    def __init__(self, *args, tokenizer, text_map, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokenizer = tokenizer
        self.text_map = text_map

    def __getitem__(self, index):
        image, label, case = super().__getitem__(index)
        text = self.text_map.get(case, f"Patient case {case}. Clinical values are missing.")
        tokens = self.tokenizer([text], context_length=256)[0]
        return image, label, case, tokens


class PMTStage1Dataset(DiyStage1Dataset):
    """DIY-CLIP samples with a precomputed gland probability map."""

    def __init__(self, *args, prob_dataset: str, **kwargs):
        super().__init__(*args, **kwargs)
        prob_id = "130" if prob_dataset == "AHCDU" else "141"
        split = "labelsTr" if self.enable_augmentation else "labelsTs"
        # The refreshed exports are nested under Prob/{id}/{id}; retain the
        # older location as a fallback for PICAI/backward compatibility.
        nested_root = Path("Prob") / prob_id / prob_id
        legacy_root = Path("Prob") / prob_id
        self.prob_dirs = [nested_root / split, legacy_root / split]
        # Some exported probability maps are in the opposite split directory;
        # searching both keeps case-name matching explicit and deterministic.
        opposite = "labelsTs" if split == "labelsTr" else "labelsTr"
        self.prob_dirs.extend([nested_root / opposite, legacy_root / opposite])
        self.missing_probability_cases = []

    def _probability_path(self, case: str) -> Path | None:
        for directory in self.prob_dirs:
            path = directory / f"{case}.npz"
            if path.exists():
                return path
        return None

    def __getitem__(self, index):
        image, label, case, tokens = super().__getitem__(index)
        path = self._probability_path(case)
        if path is None:
            self.missing_probability_cases.append(case)
            probability = torch.zeros(image.shape[1:], dtype=torch.float32)
        else:
            with np.load(path) as data:
                probabilities = data["probabilities"]
            if probabilities.ndim != 4 or probabilities.shape[0] < 2:
                raise ValueError(f"Invalid probability map {path}: shape={probabilities.shape}")
            probability = torch.from_numpy(np.asarray(probabilities[1], dtype=np.float32))
        return image, label, case, tokens, probability.clamp(0.0, 1.0)


def evaluate(model, loader, device, criterion, text_mode=False, pmt_mode=False, m3_mode=False):
    model.eval(); ys = []; probs = []; total = 0.0
    with torch.no_grad():
        for batch in loader:
            if m3_mode:
                x32, x64, x96, y = batch[0], batch[1], batch[2], batch[3]
                logits = model(x32.to(device), x64.to(device), x96.to(device))
            else:
                x, y = batch[0], batch[1]
            if m3_mode:
                pass
            elif pmt_mode:
                logits = model(x.to(device), batch[4].to(device), batch[3].to(device))
            else:
                logits = model(x.to(device), batch[3].to(device)) if text_mode else model(x.to(device))
            target = y.to(device)
            total += criterion(logits, target).item()
            probs.extend(torch.softmax(logits, 1)[:, 1].cpu().tolist())
            ys.extend(y.tolist())
    pred = [int(p >= 0.5) for p in probs]
    try: auc = roc_auc_score(ys, probs)
    except ValueError: auc = float("nan")
    return {
        "loss": total / max(len(loader), 1),
        "accuracy": accuracy_score(ys, pred),
        "precision": precision_score(ys, pred, zero_division=0),
        "recall": recall_score(ys, pred, zero_division=0),
        "auc": auc,
    }


def plot_metric(history, metric, output_dir, best_epoch, best_value):
    plt.figure(figsize=(9, 5))
    plt.plot(history["epoch"], history[f"train_{metric}"], "r-", label="train")
    plt.plot(history["epoch"], history[f"val_{metric}"], "b-", label="val")
    value = "nan" if not np.isfinite(best_value) else f"{best_value:.4f}"
    plt.title(f"{metric.upper()} (best_{metric}.pt: epoch {best_epoch}, value {value})")
    plt.xlabel("Epoch"); plt.ylabel(metric.upper()); plt.grid(True, alpha=0.3)
    plt.legend(); plt.tight_layout(); plt.savefig(output_dir / f"{metric}.png", dpi=150); plt.close()


def metric_value(metrics, metric):
    return float(metrics[metric])


def main():
    a = parse_swin_options()
    device = torch.device("cuda" if a.gpu_id >= 0 and torch.cuda.is_available() else "cpu")
    if a.gpu_id >= 0:  # set before CUDA context is created in normal invocation
        os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu_id)
    paths = {"PICAI": Path("nnUNet/nnUNet_raw/Dataset2302_FullPI-CAI"),
             "AHCDU": Path("nnUNet/nnUNet_raw/Dataset2301_FullAHCDU")}
    tokenizer = text_map = text_encoder = None
    if a.model in ("diy_clip", "pmt_net"):
        from open_clip import get_tokenizer
        from models import BiomedCLIPTextEncoder
        text_encoder_path = find_biomedclip_dir()
        text_encoder = BiomedCLIPTextEncoder(str(text_encoder_path), freeze=True).to(device)
        tokenizer = get_tokenizer("biomedclip_local_model")
        train_text_map = clinical_text_map(a.dataset, "train")
        val_text_map = clinical_text_map(a.dataset, "val")
        dataset_cls = PMTStage1Dataset if a.model == "pmt_net" else DiyStage1Dataset
        dataset_kwargs = {"prob_dataset": a.dataset} if a.model == "pmt_net" else {}
        train = dataset_cls(paths[a.dataset], "train", augment=True, tokenizer=tokenizer, text_map=train_text_map, **dataset_kwargs)
        val = dataset_cls(paths[a.dataset], "val", augment=False, tokenizer=tokenizer, text_map=val_text_map, **dataset_kwargs)
        print(a.model, "text encoder frozen; mapped clinical texts=", len(train_text_map), len(val_text_map))
        if a.model == "pmt_net":
            print("PMTNet missing probability maps fall back to zero maps")
    elif a.model == "m3net":
        train = M3Stage1Dataset(paths[a.dataset], "train", augment=True)
        val = M3Stage1Dataset(paths[a.dataset], "val", augment=False)
    else:
        train = Stage1Dataset(paths[a.dataset], "train", augment=True)
        val = Stage1Dataset(paths[a.dataset], "val", augment=False)
    labels = [int(v > 0) for _, v, _ in train.records]
    if a.sample:
        sampler = ClassBalancedSampler(labels, a.dataset)
        loader = DataLoader(train, a.batch_size, sampler=sampler, num_workers=a.num_workers)
        print("stage1 sampler=enabled counts=", sampler.class_counts)
    else:
        loader = DataLoader(train, a.batch_size, shuffle=True, num_workers=a.num_workers)
        print("stage1 sampler=disabled")
    vloader = DataLoader(val, a.batch_size, shuffle=False, num_workers=a.num_workers)
    if a.model == "lmttm":
        from models.lmttm_vmi import LMTTMVMI
        model = LMTTMVMI(in_channels=3, num_classes=2).to(device)
    elif a.model == "vlfat":
        from models.vlfat_rollout import VLFATRolloutClassifier
        model = VLFATRolloutClassifier(in_channels=3, num_classes=2).to(device)
    elif a.model == "tomographview":
        from models.tomographview import TomoGraphViewClassifier
        model = TomoGraphViewClassifier(in_channels=3, num_classes=2).to(device)
    elif a.model == "m3net":
        from models.m3net import M3NetClassifier
        model = M3NetClassifier(in_channels=3, num_classes=2).to(device)
    elif a.model == "mobibrainnet":
        from models.mobibrainnet import MobiBrainNet3D
        model = MobiBrainNet3D(in_channels=3, num_classes=2).to(device)
    elif a.model == "densenet121":
        from models.densenet3d import DenseNet1213DClassifier
        model = DenseNet1213DClassifier(num_classes=2).to(device)
    elif a.model == "efficientnet":
        from models.efficientnet3d import EfficientNet3DClassifier
        model = EfficientNet3DClassifier(num_classes=2).to(device)
    elif a.model == "diy_clip":
        from models import ViTTextCrossAttentionClassifier
        model = ViTTextCrossAttentionClassifier(num_classes=2, text_encoder=text_encoder).to(device)
    elif a.model == "pmt_net":
        from models import PMGViTCLIPClassifier
        model = PMGViTCLIPClassifier(num_classes=2, text_encoder=text_encoder).to(device)
    elif a.model == "swin":
        from train_SwinT import SwinClassifier
        model = SwinClassifier(2).to(device)
    elif a.model == "vit":
        from train_ViT import ViTClassifier
        model = ViTClassifier(2).to(device)
    else:
        from monai.networks.nets import resnet50
        model = resnet50(spatial_dims=3, n_input_channels=3, num_classes=2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    criterion = nn.CrossEntropyLoss()
    out = Path("Checkpoint") / a.task_name; out.mkdir(parents=True, exist_ok=True)
    # Track best validation value for every plotted metric. Loss is minimized;
    # the remaining metrics are maximized.
    best = {"auc": -float("inf"), "recall": -float("inf"),
            "accuracy": -float("inf"), "precision": -float("inf"),
            "loss": float("inf")}
    best_epochs = {key: -1 for key in best}
    history = {"epoch": []}
    for metric in ("loss", "accuracy", "precision", "recall", "auc"):
        history[f"train_{metric}"] = []; history[f"val_{metric}"] = []
    latest = out / "latest.pt"
    if a.resume:
        if not latest.exists():
            raise FileNotFoundError(f"--resume requested but checkpoint does not exist: {latest}")
        previous = torch.load(latest, map_location=device, weights_only=False)
        model.load_state_dict(previous["model"])
        optimizer.load_state_dict(previous["optimizer"])
        start_epoch = int(previous["epoch"]) + 1
        saved_history = previous.get("history")
        if saved_history and saved_history.get("epoch"):
            history = saved_history
        elif (out / "metrics.jsonl").exists():
            for line_number, line in enumerate((out / "metrics.jsonl").read_text().splitlines(), 1):
                if not line.strip(): continue
                row = json.loads(line)
                # Older stage-1 logs contained only validation metrics and no epoch field.
                history["epoch"].append(int(row.get("epoch", line_number)))
                for metric in ("loss", "accuracy", "precision", "recall", "auc"):
                    for split in ("train", "val"):
                        value = row.get(f"{split}_{metric}")
                        if split == "val" and value is None:
                            value = row.get(metric)
                        history[f"{split}_{metric}"].append(
                            float(value) if value is not None else float("nan"))
        best.update({k: float(v) for k, v in previous.get("best_values", {}).items() if k in best})
        best_epochs.update({k: int(v) for k, v in previous.get("best_epochs", {}).items() if k in best_epochs})
        # Recover best values from legacy best.pt files when metadata is absent.
        legacy_best = out / "best.pt"
        if legacy_best.exists() and not previous.get("best_values"):
            old_checkpoint = torch.load(legacy_best, map_location="cpu", weights_only=False)
            old_metrics = old_checkpoint.get("metrics", {})
            for metric in best:
                value = old_metrics.get(metric)
                if value is not None and np.isfinite(value):
                    best[metric] = float(value)
                    best_epochs[metric] = int(old_checkpoint.get("epoch", -1))
        print(f"resuming from {latest} at epoch {start_epoch}")
    else:
        start_epoch = 1
    for epoch in range(start_epoch, a.epochs + 1):
        model.train()
        train_ys, train_probs, train_loss = [], [], 0.0
        for batch in tqdm(loader, desc=f"Epoch {epoch}/{a.epochs}"):
            optimizer.zero_grad(set_to_none=True)
            if a.model == "m3net":
                logits = model(batch[0].to(device), batch[1].to(device), batch[2].to(device))
                y = batch[3]
            else:
                x, y = batch[0], batch[1]
            if a.model == "m3net":
                pass
            elif a.model == "pmt_net":
                logits = model(x.to(device), batch[4].to(device), batch[3].to(device))
            else:
                logits = model(x.to(device), batch[3].to(device)) if a.model == "diy_clip" else model(x.to(device))
            loss = criterion(logits, y.to(device))
            loss.backward(); optimizer.step()
            train_loss += float(loss.item())
            train_probs.extend(torch.softmax(logits.detach(), 1)[:, 1].cpu().tolist())
            train_ys.extend(y.tolist())
        train_pred = [int(p >= 0.5) for p in train_probs]
        try: train_auc = roc_auc_score(train_ys, train_probs)
        except ValueError: train_auc = float("nan")
        train_metrics = {"loss": train_loss / max(len(loader), 1),
                         "accuracy": accuracy_score(train_ys, train_pred),
                         "precision": precision_score(train_ys, train_pred, zero_division=0),
                         "recall": recall_score(train_ys, train_pred, zero_division=0),
                         "auc": train_auc}
        metrics = evaluate(model, vloader, device, criterion,
                           text_mode=(a.model == "diy_clip"), pmt_mode=(a.model == "pmt_net"),
                           m3_mode=(a.model == "m3net"))
        history["epoch"].append(epoch)
        for metric in ("loss", "accuracy", "precision", "recall", "auc"):
            history[f"train_{metric}"].append(float(train_metrics[metric]))
            history[f"val_{metric}"].append(float(metrics[metric]))
            value = metric_value(metrics, metric)
            valid = np.isfinite(value)
            improved = (valid and value < best[metric]) if metric == "loss" else (valid and value > best[metric])
            if improved:
                best[metric] = value; best_epochs[metric] = epoch
            plot_metric(history, metric, out, best_epochs[metric], best[metric])
        with (out / "metrics.jsonl").open("a") as f:
            row = {"epoch": epoch}
            row.update({f"train_{k}": None if not np.isfinite(v) else round(float(v), 4)
                        for k, v in train_metrics.items()})
            row.update({f"val_{k}": None if not np.isfinite(v) else round(float(v), 4)
                        for k, v in metrics.items()})
            f.write(json.dumps(row, allow_nan=False) + "\n")
        checkpoint = {"epoch": epoch, "model": model.state_dict(),
                      "optimizer": optimizer.state_dict(), "metrics": metrics,
                      "train_metrics": train_metrics, "threshold": 0.5,
                      "task": "0_vs_123", "options": vars(a), "history": history,
                      "best_epochs": best_epochs, "best_values": best}
        torch.save(checkpoint, out / "latest.pt")
        for metric in ("auc", "recall", "loss"):
            if best_epochs[metric] == epoch:
                torch.save(checkpoint, out / f"best_{metric}.pt")
        # best.pt remains the AUC-selected compatibility checkpoint.
        if best_epochs["auc"] == epoch:
            torch.save(checkpoint, out / "best.pt")
        print({"train": train_metrics, "val": metrics})
    (out / "threshold.json").write_text(json.dumps({"threshold": 0.5}, indent=2) + "\n")


if __name__ == "__main__": main()
