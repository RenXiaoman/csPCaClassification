"""Restore AHCDU raw archives into the project's Dataset2301 layout."""
from __future__ import annotations

import argparse
import json
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import SimpleITK as sitk
from tqdm import tqdm


TARGET_SIZE = (16, 256, 256)       # z, y, x
TARGET_SPACING = (3.0, 0.5, 0.5)  # z, y, x


def case_id(name: str) -> str:
    parts = re.findall(r"[A-Za-z0-9]+", name)
    return "".join(p[:1].upper() + p[1:].lower() for p in parts)


def find_series(patient: Path) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for directory in patient.iterdir():
        if not directory.is_dir():
            continue
        lower = directory.name.lower()
        images = [p for p in directory.glob("*.nii.gz") if "mask" not in p.name.lower()]
        if not images:
            continue
        if "adc" in lower:
            result.setdefault("adc", images[0])
        elif "localizer_t2" in lower or lower.startswith("t2"):
            result.setdefault("t2", images[0])
        elif "f_tra_b0_b800" in lower or "dwi" in lower:
            result.setdefault("dwi", images[0])
    return result


def find_masks(patient: Path) -> tuple[Path | None, Path | None]:
    t2_masks: list[Path] = []
    gland_masks: list[Path] = []
    for p in patient.glob("*/"): 
        for mask in p.glob("*.nii.gz"):
            name = mask.name.lower()
            if "mask" not in name:
                continue
            if "prostate(mr)" in name or "prostate" in name:
                gland_masks.append(mask)
            elif "t2" in name:
                t2_masks.append(mask)
    return (t2_masks[0] if t2_masks else (gland_masks[0] if gland_masks else None),
            gland_masks[0] if gland_masks else None)


def crop_pad(image: sitk.Image, size_zyx: tuple[int, int, int], pad_value: float, crop_only: bool = False) -> sitk.Image:
    target = list(size_zyx)[::-1]
    current = list(image.GetSize())
    lower_crop, upper_crop, lower_pad, upper_pad = [], [], [], []
    for cur, tar in zip(current, target):
        if cur > tar:
            diff = cur - tar
            lo = diff // 2
            lower_crop.append(lo); upper_crop.append(diff - lo)
            lower_pad.append(0); upper_pad.append(0)
        else:
            diff = tar - cur
            lo = diff // 2
            lower_crop.append(0); upper_crop.append(0)
            lower_pad.append(lo); upper_pad.append(diff - lo)
    if any(lower_crop) and not crop_only:
        image = sitk.Crop(image, lower_crop, upper_crop)
    if any(lower_pad):
        image = sitk.ConstantPad(image, lower_pad, upper_pad, pad_value)
    return image


def resample(image: sitk.Image, spacing_zyx, interpolator, default_value=0.0) -> sitk.Image:
    out_spacing = list(spacing_zyx)[::-1]
    out_size = [int(round(size * sin / sout)) for size, sin, sout in zip(
        image.GetSize(), image.GetSpacing(), out_spacing)]
    f = sitk.ResampleImageFilter()
    f.SetOutputSpacing(out_spacing); f.SetSize(out_size)
    f.SetOutputDirection(image.GetDirection()); f.SetOutputOrigin(image.GetOrigin())
    f.SetTransform(sitk.Transform()); f.SetInterpolator(interpolator)
    f.SetDefaultPixelValue(default_value)
    return f.Execute(image)


def prepare_case(job: tuple[str, str, str, str]) -> str:
    patient_text, output_text, split, root_text = job
    patient = Path(patient_text); output_root = Path(output_text)
    # Reuse the already-validated registration implementation from PICAI.
    sys.path.insert(0, str(Path("nnUNet/nnUNet_raw/Dataset2302_FullPI-CAI").resolve()))
    from deal_with import Case, PreprocessingSettings
    cid = case_id(patient.name)
    image_dir = output_root / f"images{split}"; label_dir = output_root / f"labels{split}"
    gland_dir = output_root / f"labels{split}_gland_Bosma22b"
    if all((image_dir / f"{cid}_{c:04d}.nii.gz").exists() for c in range(3)) and (label_dir / f"{cid}.nii.gz").exists():
        return cid
    series = find_series(patient)
    if set(series) != {"t2", "adc", "dwi"}:
        return f"SKIP {patient.name}: missing modalities {sorted(set(('t2','adc','dwi')) - set(series))}"
    lesion_path, gland_path = find_masks(patient)
    if lesion_path is None:
        return f"SKIP {patient.name}: no segmentation mask"

    fixed = sitk.ReadImage(str(series["t2"]), sitk.sitkFloat32)
    adc = sitk.ReadImage(str(series["adc"]), sitk.sitkFloat32)
    dwi = sitk.ReadImage(str(series["dwi"]), sitk.sitkFloat32)
    lesion = sitk.ReadImage(str(lesion_path), sitk.sitkUInt8)
    gland = sitk.ReadImage(str(gland_path), sitk.sitkUInt8) if gland_path else lesion

    fixed = crop_pad(resample(fixed, TARGET_SPACING, sitk.sitkBSpline), TARGET_SIZE, 0.0)
    adc = crop_pad(resample(adc, TARGET_SPACING, sitk.sitkBSpline), TARGET_SIZE, 0.0)
    dwi = crop_pad(resample(dwi, TARGET_SPACING, sitk.sitkBSpline), TARGET_SIZE, 0.0)
    lesion = crop_pad(resample(lesion, TARGET_SPACING, sitk.sitkNearestNeighbor), TARGET_SIZE, 0)
    gland = crop_pad(resample(gland, TARGET_SPACING, sitk.sitkNearestNeighbor), TARGET_SIZE, 0)

    helper = object.__new__(Case)
    helper.settings = PreprocessingSettings(matrix_size=TARGET_SIZE, spacing=TARGET_SPACING)
    fallback = []
    try:
        adc = helper.register_image_pair(fixed, adc, to_float32=True, verbose=False)[0]
    except RuntimeError:
        fallback.append("adc")
    try:
        dwi = helper.register_image_pair(fixed, dwi, to_float32=True, verbose=False)[0]
    except RuntimeError:
        fallback.append("dwi")
    for image in (adc, dwi, lesion, gland):
        image.CopyInformation(fixed)

    image_dir.mkdir(parents=True, exist_ok=True); label_dir.mkdir(parents=True, exist_ok=True); gland_dir.mkdir(parents=True, exist_ok=True)
    for channel, image in enumerate((fixed, adc, dwi)):
        sitk.WriteImage(image, str(image_dir / f"{cid}_{channel:04d}.nii.gz"), useCompression=True)
    sitk.WriteImage(lesion, str(label_dir / f"{cid}.nii.gz"), useCompression=True)
    sitk.WriteImage(gland, str(gland_dir / f"{cid}.nii.gz"), useCompression=True)
    return f"FALLBACK {cid}: {','.join(fallback)}" if fallback else cid


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("nnUNet/nnUNet_raw/Dataset2301_FullAHCDU/source"))
    parser.add_argument("--output", type=Path, default=Path("nnUNet/nnUNet_raw/Dataset2301_FullAHCDU"))
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    jobs = []
    for folder, split in (("建模", "Tr"), ("验证", "Ts")):
        for patient in sorted((args.source / folder).iterdir()):
            if patient.is_dir() and not patient.name.startswith("."):
                jobs.append((str(patient), str(args.output), split, str(Path("nnUNet/nnUNet_raw/Dataset2301_FullAHCDU"))))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(tqdm(pool.map(prepare_case, jobs), total=len(jobs), desc="Preparing AHCDU"))
    train = sorted(p.name[:-12] for p in (args.output / "imagesTr").glob("*_0000.nii.gz"))
    test = sorted(p.name[:-12] for p in (args.output / "imagesTs").glob("*_0000.nii.gz"))
    (args.output / "dataset.json").write_text(json.dumps({
        "channel_names": {"0": "T2W", "1": "ADC", "2": "DWI"},
        "labels": {"background": 0, "lesion": 1}, "numTraining": len(train),
        "file_ending": ".nii.gz", "tensorImageSize": "4D"
    }, indent=2) + "\n")
    print(f"Prepared train={len(train)}, val={len(test)}")
    skipped = [r for r in results if r.startswith("SKIP ")]
    fallbacks = [r for r in results if r.startswith("FALLBACK ")]
    if skipped:
        (args.output / "skipped_cases.txt").write_text("\n".join(skipped) + "\n")
        print(f"Skipped {len(skipped)} incomplete cases; see skipped_cases.txt")
    if fallbacks:
        (args.output / "registration_fallbacks.txt").write_text("\n".join(fallbacks) + "\n")
        print(f"Registration fallback for {len(fallbacks)} cases; see registration_fallbacks.txt")


if __name__ == "__main__": main()
