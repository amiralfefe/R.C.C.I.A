"""Prepare a PCam-like dataset into the R.C.C.I.A Metastasis ImageFolder layout."""

from __future__ import annotations

import argparse
import csv
import re
import shutil
from collections import Counter
from pathlib import Path


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
HDF5_EXTENSIONS = {".h5", ".hdf5"}
CLASS_ALIASES = {
    "non_metastatic": {
        "non_metastatic",
        "normal",
        "negative",
        "nonmetastatic",
        "no_metastasis",
        "no_tumor",
        "0",
    },
    "metastatic": {
        "metastatic",
        "metastasis",
        "positive",
        "tumor",
        "tumour",
        "cancer",
        "1",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a PCam-like metastasis dataset.")
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


def detect_hdf5_files(source: Path) -> list[Path]:
    return sorted(
        path
        for path in source.rglob("*")
        if path.is_file() and path.suffix.lower() in HDF5_EXTENSIONS
    )


def detect_class(path: Path, source: Path) -> str | None:
    relative_parts = path.relative_to(source).parts
    normalized_parts = [normalize_name(part) for part in relative_parts]

    for class_name, aliases in CLASS_ALIASES.items():
        if any(part in aliases for part in normalized_parts):
            return class_name

    filename = normalize_name(path.name)
    for class_name, aliases in CLASS_ALIASES.items():
        if any(re.search(rf"(^|[_\-.]){re.escape(alias)}([_\-.]|$)", filename) for alias in aliases):
            return class_name
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
    fieldnames = ["class_name", "relative_path", "output_path", "source_path"]
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if not args.source.exists():
        raise FileNotFoundError(f"Input folder not found: {args.source}")

    images = collect_images(args.source)
    if not images:
        hdf5_files = detect_hdf5_files(args.source)
        if hdf5_files:
            raise ValueError(
                "HDF5 files were found, but this V1 script only supports image folders. "
                "Export PCam patches to class folders first, or add a dedicated HDF5 converter."
            )
        raise ValueError(f"No supported images found under {args.source}.")

    prepare_output(args.output, overwrite=args.overwrite)

    selected_by_class: dict[str, list[Path]] = {"non_metastatic": [], "metastatic": []}
    skipped = 0
    for image_path in images:
        class_name = detect_class(image_path, args.source)
        if class_name is None:
            skipped += 1
            continue
        selected_by_class[class_name].append(image_path)

    rows: list[dict[str, str]] = []
    counts: Counter[str] = Counter()
    for class_name in ("non_metastatic", "metastatic"):
        class_images = selected_by_class[class_name]
        if args.max_per_class is not None:
            class_images = class_images[: args.max_per_class]
        if not class_images:
            raise ValueError(
                f"No images detected for class {class_name}. Expected class folders or filenames "
                "with aliases such as non_metastatic/normal/negative or metastatic/tumor/positive."
            )

        destination_dir = args.output / class_name
        destination_dir.mkdir(parents=True, exist_ok=True)
        for index, source_path in enumerate(class_images):
            target = unique_target(destination_dir, source_path, index)
            shutil.copy2(source_path, target)
            counts[class_name] += 1
            rows.append(
                {
                    "class_name": class_name,
                    "relative_path": str(target.relative_to(args.output)),
                    "output_path": str(target),
                    "source_path": str(source_path),
                }
            )

        print(f"{class_name}: copied {len(class_images)} images")

    write_metadata(rows, args.output / "metadata.csv")
    print(f"metadata: {args.output / 'metadata.csv'}")
    print(f"skipped_unlabeled_images: {skipped}")
    print(f"class_counts: {dict(sorted(counts.items()))}")


if __name__ == "__main__":
    main()

