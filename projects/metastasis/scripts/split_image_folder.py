"""Create train/val/test folders for a Metastasis ImageFolder dataset."""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split a Metastasis ImageFolder dataset.")
    parser.add_argument("--input", "--input-dir", dest="input_dir", type=Path, required=True)
    parser.add_argument("--output", "--output-dir", dest="output_dir", type=Path, required=True)
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


def split_items(items: list[Path], train_ratio: float, val_ratio: float) -> dict[str, list[Path]]:
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

    rng = random.Random(args.seed)
    class_dirs = sorted(path for path in args.input_dir.iterdir() if path.is_dir())
    if not class_dirs:
        raise ValueError(f"No class folders found in {args.input_dir}.")

    for class_dir in class_dirs:
        images = collect_images(class_dir)
        if not images:
            raise ValueError(f"No images found for class {class_dir.name}.")
        rng.shuffle(images)
        splits = split_items(images, train_ratio, args.val_ratio)
        for split_name, split_images_for_class in splits.items():
            copy_images(split_images_for_class, args.output_dir / split_name / class_dir.name)
        print(
            f"{class_dir.name}: train={len(splits['train'])} "
            f"val={len(splits['val'])} test={len(splits['test'])}"
        )


if __name__ == "__main__":
    main()

