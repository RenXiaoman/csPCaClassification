from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy import ndimage


def read_mask(path: Path) -> tuple[np.ndarray, tuple[float, ...]]:
    image = sitk.ReadImage(str(path))
    array = sitk.GetArrayFromImage(image)
    spacing_xyz = image.GetSpacing()
    spacing_zyx = tuple(float(i) for i in spacing_xyz[::-1])
    return array > 0, spacing_zyx


def dice(pred: np.ndarray, gt: np.ndarray) -> float:
    pred_sum = int(pred.sum())
    gt_sum = int(gt.sum())
    if pred_sum == 0 and gt_sum == 0:
        return 1.0
    if pred_sum + gt_sum == 0:
        return 0.0
    return 2.0 * float(np.logical_and(pred, gt).sum()) / float(pred_sum + gt_sum)


def iou(pred: np.ndarray, gt: np.ndarray) -> float:
    union = int(np.logical_or(pred, gt).sum())
    if union == 0:
        return 1.0
    return float(np.logical_and(pred, gt).sum()) / float(union)


def surface_distances(source: np.ndarray, target: np.ndarray, spacing: tuple[float, ...]) -> np.ndarray:
    structure = ndimage.generate_binary_structure(source.ndim, 1)
    source_surface = np.logical_xor(source, ndimage.binary_erosion(source, structure=structure, border_value=0))
    target_surface = np.logical_xor(target, ndimage.binary_erosion(target, structure=structure, border_value=0))
    distance_map = ndimage.distance_transform_edt(~target_surface, sampling=spacing)
    return distance_map[source_surface]


def symmetric_surface_distances(pred: np.ndarray, gt: np.ndarray, spacing: tuple[float, ...]) -> np.ndarray | None:
    pred_empty = not np.any(pred)
    gt_empty = not np.any(gt)
    if pred_empty and gt_empty:
        return np.asarray([0.0], dtype=np.float64)
    if pred_empty or gt_empty:
        return None
    return np.concatenate(
        [
            surface_distances(pred, gt, spacing),
            surface_distances(gt, pred, spacing),
        ]
    )


def hd95(pred: np.ndarray, gt: np.ndarray, spacing: tuple[float, ...]) -> float | None:
    distances = symmetric_surface_distances(pred, gt, spacing)
    if distances is None:
        return None
    if distances.size == 0:
        return 0.0
    return float(np.percentile(distances, 95))


def assd(pred: np.ndarray, gt: np.ndarray, spacing: tuple[float, ...]) -> float | None:
    distances = symmetric_surface_distances(pred, gt, spacing)
    if distances is None:
        return None
    if distances.size == 0:
        return 0.0
    return float(distances.mean())


def sensitivity(pred: np.ndarray, gt: np.ndarray) -> float:
    tp = int(np.logical_and(pred, gt).sum())
    fn = int(np.logical_and(~pred, gt).sum())
    if tp + fn == 0:
        return 1.0
    return float(tp) / float(tp + fn)


def precision(pred: np.ndarray, gt: np.ndarray) -> float:
    tp = int(np.logical_and(pred, gt).sum())
    fp = int(np.logical_and(pred, ~gt).sum())
    if tp + fp == 0:
        return 1.0
    return float(tp) / float(tp + fp)


def mean_std(values: list[float]) -> tuple[float | None, float | None]:
    finite = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=np.float64)
    if finite.size == 0:
        return None, None
    return float(finite.mean()), float(finite.std(ddof=0))


def fmt_percent(mean: float | None, std: float | None) -> str:
    if mean is None or std is None:
        return "nan±nan"
    return f"{mean * 100:.2f}±{std * 100:.2f}"


def fmt_mm(mean: float | None, std: float | None) -> str:
    if mean is None or std is None:
        return "nan±nan"
    return f"{mean:.2f}±{std:.2f}"


def round_or_none(value: float | None, digits: int = 4) -> float | None:
    if value is None or not np.isfinite(value):
        return None
    return round(float(value), digits)


def evaluate(pred_dir: Path, gt_dir: Path, output_name: str) -> dict:
    pred_files = sorted(pred_dir.glob("*.nii.gz"))
    cases = []
    missing_gt = []

    for pred_file in pred_files:
        gt_file = gt_dir / pred_file.name
        if not gt_file.is_file():
            missing_gt.append(pred_file.name)
            continue

        pred, pred_spacing = read_mask(pred_file)
        gt, gt_spacing = read_mask(gt_file)
        if pred.shape != gt.shape:
            raise ValueError(f"Shape mismatch for {pred_file.name}: pred {pred.shape}, gt {gt.shape}")

        case_dice = dice(pred, gt)
        case_iou = iou(pred, gt)
        case_hd95 = hd95(pred, gt, gt_spacing)
        case_assd = assd(pred, gt, gt_spacing)
        case_sensitivity = sensitivity(pred, gt)
        case_precision = precision(pred, gt)
        cases.append(
            {
                "case": pred_file.stem.removesuffix(".nii"),
                "prediction": str(pred_file),
                "ground_truth": str(gt_file),
                "dice": round_or_none(case_dice),
                "iou": round_or_none(case_iou),
                "hd95_mm": round_or_none(case_hd95),
                "assd_mm": round_or_none(case_assd),
                "sensitivity": round_or_none(case_sensitivity),
                "precision": round_or_none(case_precision),
                "spacing_zyx": list(gt_spacing),
            }
        )

    if missing_gt:
        raise FileNotFoundError(f"Missing ground-truth files for {len(missing_gt)} predictions: {missing_gt[:10]}")

    dice_mean, dice_std = mean_std([case["dice"] for case in cases])
    iou_mean, iou_std = mean_std([case["iou"] for case in cases])
    hd95_mean, hd95_std = mean_std([case["hd95_mm"] for case in cases])
    assd_mean, assd_std = mean_std([case["assd_mm"] for case in cases])
    sensitivity_mean, sensitivity_std = mean_std([case["sensitivity"] for case in cases])
    precision_mean, precision_std = mean_std([case["precision"] for case in cases])

    result = {
        "prediction_dir": str(pred_dir),
        "ground_truth_dir": str(gt_dir),
        "num_cases": len(cases),
        "summary": {
            "dice": {
                "mean": round_or_none(dice_mean),
                "std": round_or_none(dice_std),
                "mean_std_percent": fmt_percent(dice_mean, dice_std),
            },
            "iou": {
                "mean": round_or_none(iou_mean),
                "std": round_or_none(iou_std),
                "mean_std_percent": fmt_percent(iou_mean, iou_std),
            },
            "hd95_mm": {
                "mean": round_or_none(hd95_mean),
                "std": round_or_none(hd95_std),
                "mean_std": fmt_mm(hd95_mean, hd95_std),
                "note": "None values are excluded from mean/std; they occur when only one of prediction/ground truth is empty.",
            },
            "assd_mm": {
                "mean": round_or_none(assd_mean),
                "std": round_or_none(assd_std),
                "mean_std": fmt_mm(assd_mean, assd_std),
                "note": "None values are excluded from mean/std; they occur when only one of prediction/ground truth is empty.",
            },
            "sensitivity": {
                "mean": round_or_none(sensitivity_mean),
                "std": round_or_none(sensitivity_std),
                "mean_std_percent": fmt_percent(sensitivity_mean, sensitivity_std),
            },
            "precision": {
                "mean": round_or_none(precision_mean),
                "std": round_or_none(precision_std),
                "mean_std_percent": fmt_percent(precision_mean, precision_std),
            },
        },
        "cases": cases,
    }

    output_file = pred_dir / output_name
    output_file.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pred_dir",
        type=Path,
        # default=Path("nnUNet/infer/Dataset131_DIYBoundaryMSAGTrainer/1"),
        default=Path("nnUNet/infer/Dataset130_DIYBoundaryMSAGTrainer/2"),
    )
    parser.add_argument(
        "--gt_dir",
        type=Path,
        # default=Path("nnUNet/nnUNet_raw/Dataset131_ProstatePI-CAI/labelsTs"),
        default=Path("nnUNet/nnUNet_raw/Dataset130_ProstateAHCDU/labelsTs"),
    )
    parser.add_argument("--output_name", default="1A.json")
    args = parser.parse_args()

    result = evaluate(args.pred_dir, args.gt_dir, args.output_name)
    print(f"Wrote {args.pred_dir / args.output_name}")
    print(f"Dice: {result['summary']['dice']['mean_std_percent']}")
    print(f"IoU: {result['summary']['iou']['mean_std_percent']}")
    print(f"HD95(mm): {result['summary']['hd95_mm']['mean_std']}")
    print(f"ASSD(mm): {result['summary']['assd_mm']['mean_std']}")
    print(f"Sensitivity: {result['summary']['sensitivity']['mean_std_percent']}")
    print(f"Precision: {result['summary']['precision']['mean_std_percent']}")


if __name__ == "__main__":
    main()
