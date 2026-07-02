"""Prepare LC25000 into the normalized Lung + Colon ImageFolder layout."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}

CLASS_MAPPING = {
    "colon_adenocarcinoma": ("colon_aca", "colon_adenocarcinoma", "colonaca"),
    "colon_benign": ("colon_n", "colon_benign", "colon_normal", "colonn"),
    "lung_adenocarcinoma": ("lung_aca", "lung_adenocarcinoma", "lungaca"),
    "lung_benign": ("lung_n", "lung_benign", "lung_normal", "lungn"),
    "lung_squamous_cell_carcinoma": (
        "lung_scc",
        "lung_squamous_cell_carcinoma",
        "lung_squamous",
        "lungscc",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare LC25000 for Lung + Colon V1.")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/raw"))
    parser.add_argument("--max-per-class", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def normalize_name(value: str) -> str:
    return value.lower().replace("-", "_").replace(" ", "_")


def collect_images(folder: Path) -> list[Path]:
    return sorted(
        path
        for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def find_class_dir(source: Path, candidates: tuple[str, ...]) -> Path:
    normalized_candidates = {normalize_name(candidate) for candidate in candidates}
    matches = [
        path
        for path in source.rglob("*")
        if path.is_dir() and normalize_name(path.name) in normalized_candidates
    ]
    if not matches:
        raise FileNotFoundError(
            f"Could not find a folder matching {sorted(candidates)} under {source}."
        )
    return sorted(matches, key=lambda path: len(path.parts))[0]


def prepare_output(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists() and overwrite:
        shutil.rmtree(output_dir)
    elif output_dir.exists() and any(path.name != ".gitkeep" for path in output_dir.iterdir()):
        raise FileExistsError(
            f"Output folder already exists and is not empty: {output_dir}. "
            "Use --overwrite to recreate it."
        )
    output_dir.mkdir(parents=True, exist_ok=True)


def copy_images(images: list[Path], destination_dir: Path, max_per_class: int | None) -> int:
    destination_dir.mkdir(parents=True, exist_ok=True)
    selected_images = images[:max_per_class] if max_per_class else images
    for index, source in enumerate(selected_images):
        target = destination_dir / f"{destination_dir.name}_{index:05d}{source.suffix.lower()}"
        shutil.copy2(source, target)
    return len(selected_images)


def main() -> None:
    args = parse_args()
    if not args.source.exists():
        raise FileNotFoundError(f"Source folder not found: {args.source}")

    prepare_output(args.output, overwrite=args.overwrite)

    for target_class, candidates in CLASS_MAPPING.items():
        class_dir = find_class_dir(args.source, candidates)
        images = collect_images(class_dir)
        if not images:
            raise ValueError(f"No supported images found in {class_dir}.")
        count = copy_images(
            images=images,
            destination_dir=args.output / target_class,
            max_per_class=args.max_per_class,
        )
        print(f"{target_class}: copied {count} images from {class_dir}")


if __name__ == "__main__":
    main()
