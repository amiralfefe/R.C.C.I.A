"""Prepare a PCam-like dataset into the R.C.C.I.A Metastasis ImageFolder layout."""

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

import numpy as np
from PIL import Image

IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
HDF5_EXTENSIONS = {".h5", ".hdf5", ".gz"}
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
    parser.add_argument("--input", "--source", dest="source", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("data/raw"))
    parser.add_argument("--max-per-class", type=int, default=None)
    parser.add_argument("--x-h5", type=Path, default=None, help="PCam HDF5 image file, uncompressed.")
    parser.add_argument("--y-h5", type=Path, default=None, help="PCam HDF5 label file, uncompressed.")
    parser.add_argument("--hdf5-key-x", default="x")
    parser.add_argument("--hdf5-key-y", default="y")
    parser.add_argument("--split-name", default="train")
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
    base_fields = ["class_name", "relative_path", "output_path", "source_path", "split", "source_index"]
    fieldnames = [field for field in base_fields if any(field in row for row in rows)]
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def label_to_class(label: object) -> str:
    value = int(np.asarray(label).squeeze())
    if value == 0:
        return "non_metastatic"
    if value == 1:
        return "metastatic"
    raise ValueError(f"Unsupported binary label in HDF5 file: {value}")


def image_array_to_pil(image_array: object) -> Image.Image:
    image = np.asarray(image_array)
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    if image.ndim != 3 or image.shape[-1] not in {1, 3, 4}:
        raise ValueError(f"Unsupported HDF5 image shape: {image.shape}")
    if image.shape[-1] == 1:
        image = image.squeeze(axis=-1)
    return Image.fromarray(image).convert("RGB")


def prepare_hdf5_dataset(
    x_h5: Path,
    y_h5: Path,
    output_dir: Path,
    split_name: str,
    hdf5_key_x: str,
    hdf5_key_y: str,
    max_per_class: int | None,
) -> None:
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError(
            "HDF5 conversion requires h5py. Install dependencies with "
            "`python -m pip install -r requirements.txt`."
        ) from exc

    if not x_h5.exists():
        raise FileNotFoundError(f"HDF5 image file not found: {x_h5}")
    if not y_h5.exists():
        raise FileNotFoundError(f"HDF5 label file not found: {y_h5}")
    if x_h5.suffix.lower() == ".gz" or y_h5.suffix.lower() == ".gz":
        raise ValueError(
            "Compressed .h5.gz files must be decompressed before conversion. "
            "Expected uncompressed .h5 files."
        )

    rows: list[dict[str, str]] = []
    counts: Counter[str] = Counter()
    with h5py.File(x_h5, "r") as images_file, h5py.File(y_h5, "r") as labels_file:
        if hdf5_key_x not in images_file:
            raise KeyError(f"Image key '{hdf5_key_x}' not found in {x_h5}.")
        if hdf5_key_y not in labels_file:
            raise KeyError(f"Label key '{hdf5_key_y}' not found in {y_h5}.")

        images = images_file[hdf5_key_x]
        labels = labels_file[hdf5_key_y]
        if len(images) != len(labels):
            raise ValueError(f"HDF5 image/label count mismatch: {len(images)} vs {len(labels)}")

        for index in range(len(images)):
            class_name = label_to_class(labels[index])
            if max_per_class is not None and counts[class_name] >= max_per_class:
                if all(counts[name] >= max_per_class for name in ("non_metastatic", "metastatic")):
                    break
                continue

            destination_dir = output_dir / class_name
            destination_dir.mkdir(parents=True, exist_ok=True)
            target = destination_dir / f"{split_name}_{class_name}_{counts[class_name]:06d}.png"
            image = image_array_to_pil(images[index])
            image.save(target)
            counts[class_name] += 1
            rows.append(
                {
                    "class_name": class_name,
                    "relative_path": str(target.relative_to(output_dir)),
                    "output_path": str(target),
                    "source_path": str(x_h5),
                    "split": split_name,
                    "source_index": str(index),
                }
            )

    if not rows:
        raise ValueError("No images were exported from HDF5 files.")

    write_metadata(rows, output_dir / "metadata.csv")
    print(f"metadata: {output_dir / 'metadata.csv'}")
    print(f"class_counts: {dict(sorted(counts.items()))}")


def prepare_image_folder_dataset(source: Path, output_dir: Path, max_per_class: int | None) -> None:
    if not source.exists():
        raise FileNotFoundError(f"Input folder not found: {source}")

    images = collect_images(source)
    if not images:
        hdf5_files = detect_hdf5_files(source)
        if hdf5_files:
            raise ValueError(
                "HDF5 files were found. Use --x-h5 and --y-h5 with uncompressed PCam .h5 files "
                "to convert them, or export patches to class folders first."
            )
        raise ValueError(f"No supported images found under {source}.")

    selected_by_class: dict[str, list[Path]] = {"non_metastatic": [], "metastatic": []}
    skipped = 0
    for image_path in images:
        class_name = detect_class(image_path, source)
        if class_name is None:
            skipped += 1
            continue
        selected_by_class[class_name].append(image_path)

    rows: list[dict[str, str]] = []
    counts: Counter[str] = Counter()
    for class_name in ("non_metastatic", "metastatic"):
        class_images = selected_by_class[class_name]
        if max_per_class is not None:
            class_images = class_images[:max_per_class]
        if not class_images:
            raise ValueError(
                f"No images detected for class {class_name}. Expected class folders or filenames "
                "with aliases such as non_metastatic/normal/negative or metastatic/tumor/positive."
            )

        destination_dir = output_dir / class_name
        destination_dir.mkdir(parents=True, exist_ok=True)
        for index, source_path in enumerate(class_images):
            target = unique_target(destination_dir, source_path, index)
            shutil.copy2(source_path, target)
            counts[class_name] += 1
            rows.append(
                {
                    "class_name": class_name,
                    "relative_path": str(target.relative_to(output_dir)),
                    "output_path": str(target),
                    "source_path": str(source_path),
                }
            )

        print(f"{class_name}: copied {len(class_images)} images")

    write_metadata(rows, output_dir / "metadata.csv")
    print(f"metadata: {output_dir / 'metadata.csv'}")
    print(f"skipped_unlabeled_images: {skipped}")
    print(f"class_counts: {dict(sorted(counts.items()))}")


def main() -> None:
    args = parse_args()
    if args.max_per_class is not None and args.max_per_class <= 0:
        raise ValueError("--max-per-class must be positive.")

    if args.x_h5 is not None or args.y_h5 is not None:
        if args.x_h5 is None or args.y_h5 is None:
            raise ValueError("HDF5 conversion requires both --x-h5 and --y-h5.")
        for source in (args.x_h5, args.y_h5):
            validate_dataset_paths(source, args.output)
            if source.suffix.lower() == ".gz":
                raise ValueError("Decompress .h5.gz files before conversion.")
        import h5py

        with h5py.File(args.x_h5, "r") as x_file, h5py.File(args.y_h5, "r") as y_file:
            images, labels = x_file[args.hdf5_key_x], y_file[args.hdf5_key_y]
            if len(images) == 0 or len(images) != len(labels):
                raise ValueError("HDF5 image/label count mismatch or empty dataset.")
            if len(images.shape) != 4 or images.shape[-1] not in {1, 3, 4}:
                raise ValueError("Unsupported HDF5 image shape.")
            if labels.size != len(images) or not np.isin(labels[:], [0, 1]).all():
                raise ValueError("Expected one binary label per HDF5 image.")
        prepare_output(args.output, overwrite=args.overwrite)
        prepare_hdf5_dataset(
            x_h5=args.x_h5,
            y_h5=args.y_h5,
            output_dir=args.output,
            split_name=args.split_name,
            hdf5_key_x=args.hdf5_key_x,
            hdf5_key_y=args.hdf5_key_y,
            max_per_class=args.max_per_class,
        )
        return

    if args.source is None:
        raise ValueError("Image-folder preparation requires --input, or use --x-h5 and --y-h5.")
    validate_dataset_paths(args.source, args.output)
    detected = {detect_class(image, args.source) for image in collect_images(args.source)}
    if not {"non_metastatic", "metastatic"}.issubset(detected):
        raise ValueError("Both classes must contain supported, labeled images.")
    prepare_output(args.output, overwrite=args.overwrite)
    prepare_image_folder_dataset(
        source=args.source,
        output_dir=args.output,
        max_per_class=args.max_per_class,
    )


if __name__ == "__main__":
    main()
