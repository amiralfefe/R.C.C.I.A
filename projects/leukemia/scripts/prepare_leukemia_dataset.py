"""Prepare a binary leukemia dataset for Cancer Cell Vision.

The script copies images from a downloaded dataset into:

data/raw/normal/
data/raw/leukemia_blast/

It never modifies the original downloaded files.
"""

from __future__ import annotations

import argparse
import sys
import random
import shutil
from collections import Counter
from pathlib import Path

PROJECTS_DIR = Path(__file__).resolve().parents[2]
if str(PROJECTS_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECTS_DIR))
from rccia_common.dataset_paths import validate_dataset_paths


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
NORMAL_NAMES = {"hem", "healthy", "normal", "non-cancer", "non_cancer", "benign"}
BLAST_NAMES = {"all", "blast", "blasts", "cancer", "leukemia", "leukaemia", "malignant"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare leukemia raw folders.")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/raw"))
    parser.add_argument("--normal-dir", type=Path, action="append", default=[])
    parser.add_argument("--blast-dir", type=Path, action="append", default=[])
    parser.add_argument("--max-per-class", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_dir(source_dir: Path, value: Path) -> Path:
    path = value if value.is_absolute() else source_dir / value
    if not path.exists() or not path.is_dir():
        raise FileNotFoundError(f"Class folder not found: {path}")
    return path


def folder_tokens(path: Path) -> set[str]:
    raw_tokens = path.name.lower().replace("-", "_").split("_")
    return {token for token in raw_tokens if token}


def infer_class_dirs(source_dir: Path, names: set[str]) -> list[Path]:
    matches: list[Path] = []
    for path in sorted(source_dir.rglob("*")):
        if not path.is_dir():
            continue
        tokens = folder_tokens(path)
        if path.name.lower() in names or tokens.intersection(names):
            matches.append(path)
    return matches


def collect_images(class_dirs: list[Path]) -> tuple[list[Path], Counter[str], int]:
    images: list[Path] = []
    seen: set[Path] = set()
    ignored_files = 0
    extensions: Counter[str] = Counter()

    for class_dir in class_dirs:
        for path in sorted(class_dir.rglob("*")):
            if not path.is_file():
                continue
            suffix = path.suffix.lower()
            if suffix in IMAGE_EXTENSIONS:
                resolved = path.resolve()
                if resolved not in seen:
                    images.append(path)
                    extensions[suffix] += 1
                    seen.add(resolved)
            else:
                ignored_files += 1

    return images, extensions, ignored_files


def prepare_output_dir(output_dir: Path, class_name: str, overwrite: bool) -> Path:
    target_dir = output_dir / class_name
    if target_dir.exists() and overwrite:
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    existing_payload = [path for path in target_dir.iterdir() if path.name != ".gitkeep"]
    if existing_payload and not overwrite:
        raise FileExistsError(
            f"{target_dir} already contains files. Use --overwrite or choose another output folder."
        )
    return target_dir


def copy_images(
    images: list[Path],
    target_dir: Path,
    class_name: str,
    max_per_class: int | None,
    seed: int,
) -> list[Path]:
    selected = list(images)
    if max_per_class is not None and len(selected) > max_per_class:
        rng = random.Random(seed)
        rng.shuffle(selected)
        selected = sorted(selected[:max_per_class])

    copied: list[Path] = []
    for index, source in enumerate(selected, start=1):
        target = target_dir / f"{class_name}_{index:06d}{source.suffix.lower()}"
        shutil.copy2(source, target)
        copied.append(target)
    return copied


def build_class_sources(
    source_dir: Path,
    explicit_dirs: list[Path],
    class_names: set[str],
    label: str,
) -> list[Path]:
    if explicit_dirs:
        return [resolve_dir(source_dir, path) for path in explicit_dirs]

    inferred = infer_class_dirs(source_dir, class_names)
    if not inferred:
        raise ValueError(
            f"Could not infer folders for {label}. Use --{label}-dir with one or more paths."
        )
    return inferred


def summarize(label: str, source_dirs: list[Path], copied: list[Path], extensions: Counter[str], ignored: int) -> None:
    print(f"{label}:")
    print(f"  source folders: {len(source_dirs)}")
    for source_dir in source_dirs:
        print(f"    - {source_dir}")
    print(f"  copied images: {len(copied)}")
    print(f"  extensions: {dict(sorted(extensions.items()))}")
    print(f"  ignored non-image files: {ignored}")


def main() -> None:
    args = parse_args()
    source_dir = args.source.resolve()
    output_dir = args.output.resolve()
    validate_dataset_paths(source_dir, output_dir)

    if not source_dir.exists() or not source_dir.is_dir():
        raise FileNotFoundError(f"Source dataset folder not found: {source_dir}")
    if output_dir == source_dir or output_dir.is_relative_to(source_dir):
        raise ValueError("Output folder must not be the source folder or inside the source folder.")
    if args.max_per_class is not None and args.max_per_class <= 0:
        raise ValueError("--max-per-class must be positive.")

    normal_dirs = build_class_sources(source_dir, args.normal_dir, NORMAL_NAMES, "normal")
    blast_dirs = build_class_sources(source_dir, args.blast_dir, BLAST_NAMES, "blast")
    for class_source in normal_dirs + blast_dirs:
        validate_dataset_paths(class_source, output_dir)

    overlap = set(normal_dirs).intersection(blast_dirs)
    if overlap:
        raise ValueError(f"Some folders matched both classes: {sorted(str(path) for path in overlap)}")

    normal_images, normal_extensions, normal_ignored = collect_images(normal_dirs)
    blast_images, blast_extensions, blast_ignored = collect_images(blast_dirs)
    if not normal_images:
        raise ValueError("No normal images found.")
    if not blast_images:
        raise ValueError("No leukemia blast images found.")

    normal_target = prepare_output_dir(output_dir, "normal", args.overwrite)
    blast_target = prepare_output_dir(output_dir, "leukemia_blast", args.overwrite)

    normal_copied = copy_images(
        normal_images,
        normal_target,
        "normal",
        args.max_per_class,
        args.seed,
    )
    blast_copied = copy_images(
        blast_images,
        blast_target,
        "leukemia_blast",
        args.max_per_class,
        args.seed,
    )

    summarize("normal", normal_dirs, normal_copied, normal_extensions, normal_ignored)
    summarize("leukemia_blast", blast_dirs, blast_copied, blast_extensions, blast_ignored)
    print(f"Prepared raw dataset: {output_dir}")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"Error: {exc}") from exc
