from __future__ import annotations
import argparse
from pathlib import Path

def parse_swin_options():
    p = argparse.ArgumentParser(description="MONAI 3D Swin Transformer classification")
    p.add_argument("--model", choices=("resnet50", "efficientnet", "densenet121", "lmttm", "vlfat", "tomographview", "m3net", "mobibrainnet", "vit", "swin", "diy_clip", "pmt_net"), default="swin")
    p.add_argument("--dataset", choices=("PICAI", "AHCDU"), required=True)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=2)
    # Keep startup reliable for large 3D volumes and local BiomedCLIP loading.
    # Users with a stable multiprocessing setup can increase this explicitly.
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--num-classes", type=int, default=4)
    p.add_argument("--task-name", type=str, default="Swin3D_Classification")
    p.add_argument("--gpu_id", type=int, default=0)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--sample", action="store_true",
                   help="enable inverse-frequency class-balanced sampling")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()
