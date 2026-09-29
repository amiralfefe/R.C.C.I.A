"""Create train/val/test folders from a raw image folder.

Expected input:

raw_data/
  normal/
  cancer/

Output:

data/
  train/
  val/
  test/
"""

from __future__ import annotations

import argparse
import math
import sys
import random
import shutil
from pathlib import Path

PROJECTS_DIR = Path(__file__).resolve().parents[2]
if str(PROJECTS_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECTS_DIR))
from rccia_common.dataset_paths import validate_dataset_paths


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split an ImageFolder dataset.")
    parser.add_argument("--input", "--input-dir", dest="input_dir", type=Path, required=True)
    parser.add_argument("--output", "--output-dir", dest="output_dir", type=Path, default=Path("data"))
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


def split_images(
    images: list[Path],
    train_ratio: float,
    val_ratio: float,
    _test_ratio: float,
) -> dict[str, list[Path]]:
    train_end = int(len(images) * train_ratio)
    val_end = train_end + int(len(images) * val_ratio)
    return {
        "train": images[:train_end],
        "val": images[train_end:val_end],
        "test": images[val_end:],
    }


def copy_images(images: list[Path], destination_dir: Path) -> None:
    destination_dir.mkdir(parents=True, exist_ok=True)
    for source in images:
        target = destination_dir / source.name
        if target.exists():
            target = destination_dir / f"{source.stem}_{abs(hash(source))}{source.suffix}"
        shutil.copy2(source, target)


def validate_output_path(input_dir: Path, output_dir: Path) -> None:
    validate_dataset_paths(input_dir, output_dir)
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

    train_ratio = args.train_ratio
    if train_ratio is None:
        train_ratio = 1 - args.val_ratio - args.test_ratio

    if any(not math.isfinite(r) or r <= 0 for r in (train_ratio, args.val_ratio, args.test_ratio)):
        raise ValueError("Ratios must be positive.")
    if abs(train_ratio + args.val_ratio + args.test_ratio - 1) > 1e-6:
        raise ValueError("Ratios must add up to 1.")

    rng = random.Random(args.seed)
    class_dirs = sorted(path for path in args.input_dir.iterdir() if path.is_dir())
    if not class_dirs:
        raise ValueError(f"No class folders found in {args.input_dir}.")

    images_by_class = {class_dir: collect_images(class_dir) for class_dir in class_dirs}
    if any(not images for images in images_by_class.values()):
        raise ValueError("Every source class must contain supported images.")
    if args.output_dir.exists() and args.overwrite:
        shutil.rmtree(args.output_dir)
    elif args.output_dir.exists() and output_has_payload(args.output_dir):
        raise FileExistsError("Output folder is not empty; use --overwrite to recreate it.")

    for class_dir in class_dirs:
        images = images_by_class[class_dir]
        if not images:
            raise ValueError(f"No images found for class {class_dir.name}.")

        rng.shuffle(images)
        splits = split_images(images, train_ratio, args.val_ratio, args.test_ratio)
        for split_name, split_images_for_class in splits.items():
            copy_images(
                split_images_for_class,
                args.output_dir / split_name / class_dir.name,
            )
        print(
            f"{class_dir.name}: "
            f"train={len(splits['train'])} val={len(splits['val'])} test={len(splits['test'])}"
        )


if __name__ == "__main__":
    main()
