from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

from PIL import Image


def run_command(args: list[str], cwd: Path) -> subprocess.CompletedProcess:
    result = subprocess.run(
        args,
        cwd=cwd,
        text=True,
        capture_output=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, (
        f"Command failed: {' '.join(args)}\n"
        f"STDOUT:\n{result.stdout}\n"
        f"STDERR:\n{result.stderr}"
    )
    return result


def create_image(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 32), color).save(path)


def test_prepare_breakhis_dataset_extracts_metadata(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[1]
    source_dir = tmp_path / "breakhis"
    output_dir = tmp_path / "raw"

    create_image(
        source_dir / "benign" / "40X" / "SOB_B_A-14-11111AB-40-001.png",
        (90, 160, 220),
    )
    create_image(
        source_dir / "malignant" / "100X" / "SOB_M_DC-15-22222CD-100-001.png",
        (220, 90, 120),
    )

    result = run_command(
        [
            sys.executable,
            "scripts/prepare_breakhis_dataset.py",
            "--input",
            str(source_dir),
            "--output",
            str(output_dir),
        ],
        cwd=project_root,
    )

    assert "benign: copied 1 images" in result.stdout
    assert "malignant: copied 1 images" in result.stdout
    assert (output_dir / "benign").exists()
    assert (output_dir / "malignant").exists()

    with (output_dir / "metadata.csv").open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))

    assert len(rows) == 2
    assert {row["class_name"] for row in rows} == {"benign", "malignant"}
    assert {row["magnification"] for row in rows} == {"40X", "100X"}
    assert {row["patient_id"] for row in rows} == {"14-11111AB", "15-22222CD"}


def test_classic_split_and_patient_aware_split(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[1]
    raw_dir = tmp_path / "raw"
    classic_output = tmp_path / "processed_classic"
    patient_output = tmp_path / "processed_patient"

    rows: list[dict[str, str]] = []
    for class_name, color in {
        "benign": (80, 180, 220),
        "malignant": (220, 80, 120),
    }.items():
        for patient_index in range(4):
            image_path = raw_dir / class_name / f"{class_name}_patient_{patient_index}.png"
            create_image(image_path, color)
            rows.append(
                {
                    "class_name": class_name,
                    "relative_path": str(image_path.relative_to(raw_dir)),
                    "output_path": str(image_path),
                    "source_path": str(image_path),
                    "magnification": "40X",
                    "patient_id": f"{class_name}_{patient_index}",
                }
            )

    metadata_path = raw_dir / "metadata.csv"
    with metadata_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    run_command(
        [
            sys.executable,
            "scripts/split_image_folder.py",
            "--input",
            str(raw_dir),
            "--output",
            str(classic_output),
            "--train-ratio",
            "0.5",
            "--val-ratio",
            "0.25",
            "--test-ratio",
            "0.25",
        ],
        cwd=project_root,
    )
    assert len(list((classic_output / "train" / "benign").glob("*.png"))) == 2
    assert len(list((classic_output / "val" / "benign").glob("*.png"))) == 1
    assert len(list((classic_output / "test" / "benign").glob("*.png"))) == 1

    run_command(
        [
            sys.executable,
            "scripts/split_image_folder.py",
            "--input",
            str(raw_dir),
            "--output",
            str(patient_output),
            "--metadata",
            str(metadata_path),
            "--patient-aware",
            "--train-ratio",
            "0.5",
            "--val-ratio",
            "0.25",
            "--test-ratio",
            "0.25",
        ],
        cwd=project_root,
    )

    patients_by_split: dict[str, set[str]] = {}
    for split in ("train", "val", "test"):
        patients_by_split[split] = {
            path.stem
            for path in (patient_output / split).glob("*/*.png")
        }

    assert patients_by_split["train"].isdisjoint(patients_by_split["val"])
    assert patients_by_split["train"].isdisjoint(patients_by_split["test"])
    assert patients_by_split["val"].isdisjoint(patients_by_split["test"])
