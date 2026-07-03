"""Create train/val/test folders for R.C.C.I.A Breast.

The script supports a classic image-level split and a patient-aware split when
metadata.csv contains a usable patient_id column.
"""

from __future__ import annotations

import argparse
import csv
import random
import shutil
from collections import defaultdict
from pathlib import Path


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split a Breast ImageFolder dataset.")
    parser.add_argument("--input", "--input-dir", dest="input_dir", type=Path, required=True)
    parser.add_argument("--output", "--output-dir", dest="output_dir", type=Path, default=Path("data"))
    parser.add_argument("--metadata", type=Path, default=None)
    parser.add_argument("--patient-aware", action="store_true")
    parser.add_argument("--train-ratio", type=float, default=None)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def collect_images(class_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in class_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def split_items(
    items: list,
    train_ratio: float,
    val_ratio: float,
) -> dict[str, list]:
    train_end = int(len(items) * train_ratio)
    val_end = train_end + int(len(items) * val_ratio)
    return {
        "train": items[:train_end],
        "val": items[train_end:val_end],
        "test": items[val_end:],
    }


def copy_images(images: list[Path], destination_dir: Path) -> None:
    destination_dir.mkdir(parents=True, exist_ok=True)
    for source in images:
        target = destination_dir / source.name
        if target.exists():
            target = destination_dir / f"{source.stem}_{abs(hash(source))}{source.suffix}"
        shutil.copy2(source, target)


def validate_output_path(input_dir: Path, output_dir: Path) -> None:
    resolved_input = input_dir.resolve()
    resolved_output = output_dir.resolve()

    if resolved_input == resolved_output:
        raise ValueError("Output folder must be different from input folder.")
    if resolved_output.is_relative_to(resolved_input):
        raise ValueError("Output folder must not be inside the input folder.")
    if len(resolved_output.parts) <= 2:
        raise ValueError(f"Refusing to write to a top-level folder: {resolved_output}")


def output_has_payload(output_dir: Path) -> bool:
    return any(path.name != ".gitkeep" for path in output_dir.iterdir())


def resolve_metadata_image(input_dir: Path, row: dict[str, str]) -> Path:
    if row.get("relative_path"):
        candidate = input_dir / row["relative_path"]
        if candidate.exists():
            return candidate

    if row.get("output_path"):
        candidate = Path(row["output_path"])
        if candidate.exists():
            return candidate
        fallback = input_dir / row.get("class_name", "") / candidate.name
        if fallback.exists():
            return fallback

    raise FileNotFoundError(f"Could not resolve metadata image row: {row}")


def read_patient_metadata(input_dir: Path, metadata_path: Path) -> list[dict[str, str | Path]]:
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    rows: list[dict[str, str | Path]] = []
    with metadata_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        required_columns = {"class_name", "patient_id"}
        missing_columns = required_columns.difference(reader.fieldnames or [])
        if missing_columns:
            raise ValueError(
                f"Metadata is missing columns {sorted(missing_columns)}. "
                "Patient-aware split requires class_name and patient_id."
            )

        for row in reader:
            patient_id = (row.get("patient_id") or "").strip()
            class_name = (row.get("class_name") or "").strip()
            if not patient_id:
                raise ValueError(
                    "Patient-aware split requested, but at least one metadata row has no "
                    "patient_id. Use classic split or provide complete metadata."
                )
            if not class_name:
                raise ValueError("Metadata row has no class_name.")
            image_path = resolve_metadata_image(input_dir, row)
            rows.append({"class_name": class_name, "patient_id": patient_id, "path": image_path})

    if not rows:
        raise ValueError(f"No rows found in metadata file: {metadata_path}")
    return rows


def split_classic(
    input_dir: Path,
    output_dir: Path,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> None:
    rng = random.Random(seed)
    class_dirs = sorted(path for path in input_dir.iterdir() if path.is_dir())
    if not class_dirs:
        raise ValueError(f"No class folders found in {input_dir}.")

    for class_dir in class_dirs:
        images = collect_images(class_dir)
        if not images:
            raise ValueError(f"No images found for class {class_dir.name}.")

        rng.shuffle(images)
        splits = split_items(images, train_ratio, val_ratio)
        for split_name, split_images_for_class in splits.items():
            copy_images(split_images_for_class, output_dir / split_name / class_dir.name)
        print(
            f"{class_dir.name}: train={len(splits['train'])} "
            f"val={len(splits['val'])} test={len(splits['test'])}"
        )


def split_patient_aware(
    input_dir: Path,
    output_dir: Path,
    metadata_path: Path,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> None:
    rows = read_patient_metadata(input_dir, metadata_path)
    by_class_and_patient: dict[str, dict[str, list[Path]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        by_class_and_patient[str(row["class_name"])][str(row["patient_id"])].append(
            Path(row["path"])
        )

    rng = random.Random(seed)
    used_patients_by_split: dict[str, set[str]] = {"train": set(), "val": set(), "test": set()}

    for class_name, patients in sorted(by_class_and_patient.items()):
        patient_groups = list(patients.items())
        if len(patient_groups) < 3:
            raise ValueError(
                f"Patient-aware split needs at least 3 patients for class {class_name}, "
                f"got {len(patient_groups)}."
            )

        rng.shuffle(patient_groups)
        splits = split_items(patient_groups, train_ratio, val_ratio)
        for split_name, groups in splits.items():
            images: list[Path] = []
            for patient_id, patient_images in groups:
                images.extend(patient_images)
                used_patients_by_split[split_name].add(patient_id)
            copy_images(images, output_dir / split_name / class_name)
        print(
            f"{class_name}: train={sum(len(v) for _, v in splits['train'])} "
            f"val={sum(len(v) for _, v in splits['val'])} "
            f"test={sum(len(v) for _, v in splits['test'])}"
        )

    overlaps = (
        used_patients_by_split["train"] & used_patients_by_split["val"]
        or used_patients_by_split["train"] & used_patients_by_split["test"]
        or used_patients_by_split["val"] & used_patients_by_split["test"]
    )
    if overlaps:
        raise RuntimeError(f"Patient leakage detected across splits: {sorted(overlaps)}")

    print(
        "patient_aware_split: "
        f"train_patients={len(used_patients_by_split['train'])} "
        f"val_patients={len(used_patients_by_split['val'])} "
        f"test_patients={len(used_patients_by_split['test'])}"
    )


def main() -> None:
    args = parse_args()

    if not args.input_dir.exists():
        raise FileNotFoundError(f"Input folder not found: {args.input_dir}")

    validate_output_path(args.input_dir, args.output_dir)

    if args.output_dir.exists() and args.overwrite:
        shutil.rmtree(args.output_dir)
    elif args.output_dir.exists() and output_has_payload(args.output_dir):
        raise FileExistsError(
            f"Output folder already exists and is not empty: {args.output_dir}. "
            "Use --overwrite to recreate it."
        )

    train_ratio = args.train_ratio
    if train_ratio is None:
        train_ratio = 1 - args.val_ratio - args.test_ratio
    if train_ratio <= 0 or args.val_ratio <= 0 or args.test_ratio <= 0:
        raise ValueError("Ratios must be positive.")
    if abs(train_ratio + args.val_ratio + args.test_ratio - 1) > 1e-6:
        raise ValueError("Ratios must add up to 1.")

    if args.patient_aware:
        if args.metadata is None:
            raise ValueError("Patient-aware split requires --metadata.")
        split_patient_aware(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            metadata_path=args.metadata,
            train_ratio=train_ratio,
            val_ratio=args.val_ratio,
            seed=args.seed,
        )
    else:
        split_classic(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            train_ratio=train_ratio,
            val_ratio=args.val_ratio,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
