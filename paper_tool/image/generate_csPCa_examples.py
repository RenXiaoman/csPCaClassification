"""Generate title-free five-column csPCa visualization panels.

Rows are T2W, DWI, ADC, prostate gland mask, and lesion mask.
Dataset mapping is intentional:
  Dataset130/141  -> prostate gland masks
  Dataset2301/2302 -> lesion masks and multimodal images
"""
from pathlib import Path
import json

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "paper_tool" / "image"


DATASETS = {
    "PICAI": {
        "image": ROOT / "nnUNet/nnUNet_raw/Dataset2302_FullPI-CAI",
        "gland": ROOT / "nnUNet/nnUNet_raw/Dataset141_FullPICAI",
        "lesion": ROOT / "nnUNet/nnUNet_raw/Dataset2302_FullPI-CAI",
        "output": OUTPUT_DIR / "csPCa_examples_PICAI.png",
    },
    "AHCDU": {
        "image": ROOT / "nnUNet/nnUNet_raw/Dataset2301_FullAHCDU",
        "gland": ROOT / "nnUNet/nnUNet_raw/Dataset130_ProstateAHCDU",
        "lesion": ROOT / "nnUNet/nnUNet_raw/Dataset2301_FullAHCDU",
        "output": OUTPUT_DIR / "csPCa_examples_AHCDU.png",
    },
}


def select_cases(spec):
    records = json.loads((spec["lesion"] / "labelsTr/A_labels.json").read_text())
    positive = [str(row["case"]) for row in records if int(row["value"]) == 1]
    valid = []
    for case in positive:
        image_paths = [
            spec["image"] / "imagesTr" / f"{case}_{suffix}.nii.gz"
            for suffix in ("0000", "0001", "0002")
        ]
        gland_path = spec["gland"] / "labelsTr" / f"{case}.nii.gz"
        lesion_path = spec["lesion"] / "labelsTr" / f"{case}.nii.gz"
        if not all(path.exists() for path in image_paths):
            continue
        if not gland_path.exists() or not lesion_path.exists():
            continue
        gland = np.asanyarray(nib.load(gland_path).dataobj)
        lesion = np.asanyarray(nib.load(lesion_path).dataobj)
        if gland.shape != lesion.shape or gland.shape != (256, 256, 16):
            continue
        lesion_size = int(np.count_nonzero(lesion))
        if lesion_size > 0 and np.count_nonzero(gland) > 0:
            valid.append((case, lesion_size))
    valid.sort(key=lambda item: (-item[1], item[0]))
    return [case for case, _ in valid[:5]]


def draw_panel(name, spec):
    cases = select_cases(spec)
    print(f"{name}: {cases}")
    figure, axes = plt.subplots(5, 5, figsize=(12.5, 12.5), dpi=200, squeeze=False)

    for column, case in enumerate(cases):
        # Stored channel order is T2W=0000, ADC=0001, DWI=0002;
        # display order requested by the paper figure is T2W, DWI, ADC.
        volumes = [
            np.nan_to_num(
                np.asanyarray(
                    nib.load(spec["image"] / "imagesTr" / f"{case}_{suffix}.nii.gz").dataobj
                ).astype(np.float32)
            )
            for suffix in ("0000", "0002", "0001")
        ]
        gland = np.asanyarray(
            nib.load(spec["gland"] / "labelsTr" / f"{case}.nii.gz").dataobj
        )
        lesion = np.asanyarray(
            nib.load(spec["lesion"] / "labelsTr" / f"{case}.nii.gz").dataobj
        )
        slice_index = int(np.argmax((lesion > 0).sum(axis=(0, 1))))

        for row, volume in enumerate(volumes):
            image = volume[:, :, slice_index].T
            low, high = np.percentile(image, [1, 99])
            if high <= low:
                low, high = float(image.min()), float(image.max() + 1e-6)
            image = np.clip((image - low) / (high - low), 0, 1)
            axes[row, column].imshow(image, cmap="gray", vmin=0, vmax=1, interpolation="nearest")

        for row, mask in ((3, gland), (4, lesion)):
            axes[row, column].imshow(
                (mask[:, :, slice_index] > 0).T,
                cmap="gray",
                vmin=0,
                vmax=1,
                interpolation="nearest",
            )

    for axis in axes.ravel():
        axis.set_axis_off()
    figure.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0.015, hspace=0.015)
    spec["output"].parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(spec["output"], dpi=200, bbox_inches="tight", pad_inches=0)
    plt.close(figure)
    print(f"saved: {spec['output']}")


if __name__ == "__main__":
    for dataset_name, dataset_spec in DATASETS.items():
        draw_panel(dataset_name, dataset_spec)
