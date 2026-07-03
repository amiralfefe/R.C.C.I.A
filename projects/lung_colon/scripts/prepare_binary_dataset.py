"""Create a benign/malignant ImageFolder dataset from LC25000 splits."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rccia_lung_colon.binary import (  # noqa: E402
    BINARY_CLASSES,
    BINARY_CLASS_MAPPING,
    SOURCE_CLASSES_BY_BINARY,
    binary_label_for_source,
)
from rccia_lung_colon.data import IMAGE_EXTENSIONS  # noqa: E402


SPLITS = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare a binary benign/malignant dataset from LC25000 ImageFolder splits."
    )
    parser.add_argument("--input", "--input-dir", dest="input_dir", type=Path, required=True)
    parser.add_argument("--output", "--output-dir", dest="output_dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def collect_images(class_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in class_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def output_has_payload(output_dir: Path) -> bool:
    return any(path.name != ".gitkeep" for path in output_dir.iterdir())


def prepare_output(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists() and overwrite:
        shutil.rmtree(output_dir)
    elif output_dir.exists() and output_has_payload(output_dir):
        raise FileExistsError(
            f"Output folder already exists and is not empty: {output_dir}. "
            "Use --overwrite to recreate it."
        )
    output_dir.mkdir(parents=True, exist_ok=True)


def validate_input(input_dir: Path) -> None:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input dataset not found: {input_dir}")

    for split in SPLITS:
        split_dir = input_dir / split
        if not split_dir.is_dir():
            raise FileNotFoundError(f"Missing split folder: {split_dir}")

        missing_classes = [
            source_class
            for source_class in BINARY_CLASS_MAPPING
            if not (split_dir / source_class).is_dir()
        ]
        if missing_classes:
            raise FileNotFoundError(
                f"Missing source class folders in {split_dir}: {missing_classes}"
            )


def copy_binary_split(input_dir: Path, output_dir: Path, split: str) -> dict[str, int]:
    counts = {binary_class: 0 for binary_class in BINARY_CLASSES}
    split_dir = input_dir / split

    for source_class in sorted(BINARY_CLASS_MAPPING):
        binary_class = binary_label_for_source(source_class)
        source_dir = split_dir / source_class
        destination_dir = output_dir / split / binary_class
        destination_dir.mkdir(parents=True, exist_ok=True)

        images = collect_images(source_dir)
        if not images:
            raise ValueError(f"No supported images found in {source_dir}.")

        for image in images:
            target = destination_dir / f"{source_class}_{image.name}"
            shutil.copy2(image, target)
        counts[binary_class] += len(images)

    return counts


def prepare_binary_dataset(
    input_dir: Path,
    output_dir: Path,
    overwrite: bool = False,
) -> dict[str, dict[str, int]]:
    validate_input(input_dir)
    prepare_output(output_dir, overwrite=overwrite)

    summary = {split: copy_binary_split(input_dir, output_dir, split) for split in SPLITS}
    summary_path = output_dir / "binary_dataset_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "source_classes_by_binary_label": SOURCE_CLASSES_BY_BINARY,
                "splits": summary,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return summary


def print_summary(summary: dict[str, dict[str, int]]) -> None:
    print("Source classes used:")
    for binary_class in BINARY_CLASSES:
        sources = ", ".join(SOURCE_CLASSES_BY_BINARY[binary_class])
        print(f"- {binary_class}: {sources}")

    print("Binary dataset counts:")
    for split in SPLITS:
        counts = summary[split]
        print(
            f"- {split}: "
            f"benign={counts['benign']} malignant={counts['malignant']}"
        )


def main() -> None:
    args = parse_args()
    summary = prepare_binary_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
    )
    print_summary(summary)


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, FileExistsError, ValueError) as exc:
        raise SystemExit(f"Error: {exc}") from exc
