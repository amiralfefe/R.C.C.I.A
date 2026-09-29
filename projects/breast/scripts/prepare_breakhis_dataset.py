"""Prepare BreakHis into the normalized R.C.C.I.A Breast ImageFolder layout."""

from __future__ import annotations

import argparse
import sys
import csv
import re
import shutil
from collections import Counter
from pathlib import Path

PROJECTS_DIR = Path(__file__).resolve().parents[2]
if str(PROJECTS_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECTS_DIR))
from rccia_common.dataset_paths import validate_dataset_paths


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
CLASS_ALIASES = {
    "benign": {
        "benign",
        "b",
        "adenosis",
        "fibroadenoma",
        "phyllodes_tumor",
        "tubular_adenoma",
    },
    "malignant": {
        "malignant",
        "m",
        "carcinoma",
        "ductal_carcinoma",
        "lobular_carcinoma",
        "mucinous_carcinoma",
        "papillary_carcinoma",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare BreakHis for R.C.C.I.A Breast.")
    parser.add_argument("--input", "--source", dest="source", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/raw"))
    parser.add_argument("--max-per-class", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def normalize_name(value: str) -> str:
    return value.lower().replace("-", "_").replace(" ", "_")


def collect_images(source: Path) -> list[Path]:
    return sorted(
        path
        for path in source.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def detect_class(path: Path, source: Path) -> str | None:
    relative_parts = path.relative_to(source).parts
    normalized_parts = [normalize_name(part) for part in relative_parts]

    for class_name, aliases in CLASS_ALIASES.items():
        if any(part in aliases for part in normalized_parts):
            return class_name

    filename = normalize_name(path.name)
    if filename.startswith("sob_b") or "_b_" in filename:
        return "benign"
    if filename.startswith("sob_m") or "_m_" in filename:
        return "malignant"
    return None


def extract_magnification(path: Path) -> str | None:
    text = " ".join([path.name, *path.parts])
    for pattern in (
        r"(?<!\d)(40|100|200|400)\s*[xX](?!\w)",
        r"[-_](40|100|200|400)[-_]",
    ):
        match = re.search(pattern, text)
        if match:
            return f"{match.group(1)}X"
    return None


def extract_patient_id(path: Path) -> str | None:
    stem = path.stem
    breakhis_match = re.search(r"SOB_[BM]_[A-Z]+-([0-9]+-[A-Za-z0-9]+)-", stem, re.I)
    if breakhis_match:
        return breakhis_match.group(1)

    generic_match = re.search(r"(?:patient|case|pt)[-_ ]?([A-Za-z0-9]+)", str(path), re.I)
    if generic_match:
        return generic_match.group(1)
    return None


def prepare_output(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists() and overwrite:
        shutil.rmtree(output_dir)
    elif output_dir.exists() and any(path.name != ".gitkeep" for path in output_dir.iterdir()):
        raise FileExistsError(
            f"Output folder already exists and is not empty: {output_dir}. "
            "Use --overwrite to recreate it."
        )
    output_dir.mkdir(parents=True, exist_ok=True)


def unique_target(destination_dir: Path, source: Path, index: int) -> Path:
    safe_stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", source.stem)
    target = destination_dir / f"{destination_dir.name}_{index:05d}_{safe_stem}{source.suffix.lower()}"
    if not target.exists():
        return target
    return destination_dir / f"{destination_dir.name}_{index:05d}_{abs(hash(source))}{source.suffix.lower()}"


def write_metadata(rows: list[dict[str, str]], output_path: Path) -> None:
    fieldnames = [
        "class_name",
        "relative_path",
        "output_path",
        "source_path",
        "magnification",
        "patient_id",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    validate_dataset_paths(args.source, args.output)
    if args.max_per_class is not None and args.max_per_class <= 0:
        raise ValueError("--max-per-class must be positive.")
    if not args.source.exists():
        raise FileNotFoundError(f"Input folder not found: {args.source}")

    images = collect_images(args.source)
    if not images:
        raise ValueError(f"No supported images found under {args.source}.")

    selected_by_class: dict[str, list[Path]] = {"benign": [], "malignant": []}
    skipped = 0
    for image_path in images:
        class_name = detect_class(image_path, args.source)
        if class_name is None:
            skipped += 1
            continue
        selected_by_class[class_name].append(image_path)

    if any(not images for images in selected_by_class.values()):
        raise ValueError("Both benign and malignant classes must contain images.")
    prepare_output(args.output, overwrite=args.overwrite)
    rows: list[dict[str, str]] = []
    magnifications: Counter[str] = Counter()
    patient_ids: set[str] = set()

    for class_name in ("benign", "malignant"):
        class_images = selected_by_class[class_name]
        if args.max_per_class is not None:
            class_images = class_images[: args.max_per_class]
        if not class_images:
            raise ValueError(
                f"No images detected for class {class_name}. "
                "Use a source folder that contains benign/malignant labels in paths or filenames."
            )

        destination_dir = args.output / class_name
        destination_dir.mkdir(parents=True, exist_ok=True)
        for index, source_path in enumerate(class_images):
            target = unique_target(destination_dir, source_path, index)
            shutil.copy2(source_path, target)

            magnification = extract_magnification(source_path) or ""
            patient_id = extract_patient_id(source_path) or ""
            if magnification:
                magnifications[magnification] += 1
            if patient_id:
                patient_ids.add(patient_id)

            rows.append(
                {
                    "class_name": class_name,
                    "relative_path": str(target.relative_to(args.output)),
                    "output_path": str(target),
                    "source_path": str(source_path),
                    "magnification": magnification,
                    "patient_id": patient_id,
                }
            )

        print(f"{class_name}: copied {len(class_images)} images")

    write_metadata(rows, args.output / "metadata.csv")
    print(f"metadata: {args.output / 'metadata.csv'}")
    print(f"skipped_unlabeled_images: {skipped}")
    print(f"magnifications_detected: {dict(sorted(magnifications.items()))}")
    if patient_ids:
        print(f"patients_detected: {len(patient_ids)}")
    else:
        print("warning: no patient_id detected; use classic split or provide metadata manually.")


if __name__ == "__main__":
    main()
