"""Restore classification labels and the PICAI train/validation layout."""
from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import pandas as pd


def norm(value: object) -> str:
    return "".join(re.findall(r"[a-z0-9]+", str(value).lower()))


def write_records(cases: list[str], mapping: dict[str, int], path: Path) -> None:
    missing = [case for case in cases if case not in mapping]
    if missing:
        raise RuntimeError(f"Missing labels for {len(missing)} cases in {path}: {missing[:10]}")
    records = [{"case": case, "value": int(mapping[case])} for case in cases]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records, indent=2) + "\n")


def split_picai(root: Path) -> tuple[list[str], list[str]]:
    split_file = root / "splits.json"
    splits = json.loads(split_file.read_text())[0]
    train = list(splits["train"])
    val = list(splits["val"])
    train_set, val_set = set(train), set(val)
    if train_set & val_set or len(train_set) + len(val_set) != len(set(train + val)):
        raise RuntimeError("Invalid PICAI split file")

    # The uploaded export places all cases in imagesTr. Restore the project's
    # 1194/300 train/validation directory convention using splits.json.
    if not any((root / "imagesTs").glob("*_0000.nii.gz")):
        for case in val:
            for directory, pattern in (
                ("imagesTr", f"{case}_*.nii.gz"),
                ("labelsTr", f"{case}.nii.gz"),
                ("labelsTr_gland_Bosma22b", f"{case}.nii.gz"),
            ):
                source_dir = root / directory
                target_dir = root / directory.replace("Tr", "Ts", 1)
                target_dir.mkdir(parents=True, exist_ok=True)
                for source in source_dir.glob(pattern):
                    shutil.move(str(source), str(target_dir / source.name))
    cases_train = sorted(p.name[:-12] for p in (root / "imagesTr").glob("*_0000.nii.gz"))
    cases_val = sorted(p.name[:-12] for p in (root / "imagesTs").glob("*_0000.nii.gz"))
    if set(cases_train) != train_set or set(cases_val) != val_set:
        raise RuntimeError(f"PICAI split mismatch: got {len(cases_train)}/{len(cases_val)}")
    return cases_train, cases_val


def main() -> None:
    picai = Path("nnUNet/nnUNet_raw/Dataset2302_FullPI-CAI")
    ahcdu = Path("nnUNet/nnUNet_raw/Dataset2301_FullAHCDU")

    cases_train, cases_val = split_picai(picai)
    csv = pd.read_csv(picai / "marksheet (1).csv", encoding="utf-8-sig")
    picai_map = {
        f"{int(row.patient_id)}_{int(row.study_id)}": 1 if str(row.case_csPCa).strip().upper() == "YES" else 0
        for row in csv.itertuples(index=False)
    }
    write_records(cases_train, picai_map, picai / "labelsTr" / "A_labels.json")
    write_records(cases_val, picai_map, picai / "labelsTs" / "A_labels.json")

    def ahcdu_records(file_name: str, split: str, column: str) -> None:
        frame = pd.read_excel(Path("nnUNet/nnUNet_raw/Dataset130_ProstateAHCDU") / file_name,
                              sheet_name="检查级别", header=1)
        labels = {norm(row["PatientName"]): int(row[column]) for _, row in frame.iterrows()}
        cases = sorted(p.name[:-12] for p in (ahcdu / f"images{split}").glob("*_0000.nii.gz"))
        mapping = {case: labels[norm(case)] for case in cases if norm(case) in labels}
        write_records(cases, mapping, ahcdu / f"labels{split}" / "A_labels.json")

    ahcdu_records("建模临床信息表.xlsx", "Tr", "STUDY->CLINICAL->是否为CSPCa")
    ahcdu_records("验证临床信息.xlsx", "Ts", "STUDY->CLINICAL->csPca")
    print(f"PICAI labels: train={len(cases_train)}, val={len(cases_val)}")
    print("AHCDU labels: train=462, val=104")


if __name__ == "__main__":
    main()
