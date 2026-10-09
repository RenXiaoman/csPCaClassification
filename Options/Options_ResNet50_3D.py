from __future__ import annotations

import argparse
from pathlib import Path


def parse_classification_options():
    parser = argparse.ArgumentParser(description="MONAI 3D ResNet-50 classification")
    parser.add_argument("--dataset", choices=("PICAI", "AHCDU"), required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-classes", type=int, default=4)
    parser.add_argument("--task-name", type=str, default="ResNet50_3D_Classification")
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--resume", action="store_true", help="resume from Checkpoint/<task-name>/latest.pt")
    parser.add_argument("--sample", action="store_true",
                        help="enable inverse-frequency class-balanced sampling")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()
