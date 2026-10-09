"""Register PI-CAI ADC/DWI volumes to the T2W reference.

The uploaded PI-CAI export already has the project target geometry:
16 slices x 256 x 256 at (3.0, 0.5, 0.5) mm in z,y,x order.  This script
therefore only performs the modality registration from ``deal_with.py`` and
does not resample or crop the volumes again.
"""
from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import SimpleITK as sitk
from tqdm import tqdm


ROOT = Path("nnUNet/nnUNet_raw/Dataset2302_FullPI-CAI")


def register_one(fixed: sitk.Image, moving: sitk.Image) -> sitk.Image:
    # deal_with.py computes registrations on gradient features and uses the
    # B-spline/Mattes-MI settings in reg_lib.py.
    sys.path.insert(0, str(ROOT.resolve()))
    from deal_with import Case, PreprocessingSettings

    helper = object.__new__(Case)
    helper.settings = PreprocessingSettings(
        matrix_size=(16, 256, 256),
        spacing=(3.0, 0.5, 0.5),
    )
    registered, _, _ = helper.register_image_pair(
        fixed,
        moving,
        to_float32=True,
        verbose=False,
    )
    return registered


def process_case(args: tuple[str, str, str]) -> str:
    case, source_text, output_text = args
    source = Path(source_text)
    output = Path(output_text)
    fixed_path = source / f"{case}_0000.nii.gz"
    fixed = sitk.ReadImage(str(fixed_path), sitk.sitkFloat32)
    sitk.WriteImage(fixed, str(output / fixed_path.name), useCompression=True)
    for channel in (1, 2):
        out_path = output / f"{case}_{channel:04d}.nii.gz"
        if out_path.exists():
            continue
        moving = sitk.ReadImage(str(source / f"{case}_{channel:04d}.nii.gz"), sitk.sitkFloat32)
        registered = register_one(fixed, moving)
        registered.CopyInformation(fixed)
        sitk.WriteImage(registered, str(out_path), useCompression=True)
    return case


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel registration workers; use OMP_NUM_THREADS=1.")
    parser.add_argument("--replace", action="store_true",
                        help="After success, keep imagesTr as imagesTr_unregistered and use registered output as imagesTr.")
    args = parser.parse_args()

    root = args.root
    source = root / "imagesTr"
    output = args.output or (root / "imagesTr_registered")
    output.mkdir(parents=True, exist_ok=True)

    cases = sorted(p.name[:-12] for p in source.glob("*_0000.nii.gz"))
    if args.limit is not None:
        cases = cases[:args.limit]
    if not cases:
        raise RuntimeError(f"No *_0000.nii.gz files found in {source}")

    jobs = [(case, str(source), str(output)) for case in cases]
    if args.workers <= 1:
        iterator = (process_case(job) for job in jobs)
        for _ in tqdm(iterator, total=len(jobs), desc="Registering PI-CAI modalities"):
            pass
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            iterator = pool.map(process_case, jobs)
            for _ in tqdm(iterator, total=len(jobs), desc=f"Registering PI-CAI ({args.workers} workers)"):
                pass

    if len(list(output.glob("*_0000.nii.gz"))) != len(cases):
        raise RuntimeError("Registration output is incomplete; refusing to replace imagesTr")
    if args.replace:
        backup = root / "imagesTr_unregistered"
        if backup.exists():
            raise FileExistsError(f"Refusing to overwrite existing backup: {backup}")
        source.rename(backup)
        output.rename(source)
        print(f"Original images kept at: {backup}")
        print(f"Registered images active at: {source}")


if __name__ == "__main__":
    main()
