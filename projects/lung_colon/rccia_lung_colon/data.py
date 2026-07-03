"""Dataset and dataloader helpers."""

from __future__ import annotations

from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from .config import IMAGENET_MEAN, IMAGENET_STD

IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def build_transforms(image_size: int, train: bool) -> transforms.Compose:
    if train:
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )

    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def build_image_transform(image_size: int) -> transforms.Compose:
    return build_transforms(image_size=image_size, train=False)


def validate_split_dir(split_dir: Path) -> None:
    if not split_dir.exists():
        raise FileNotFoundError(
            f"Dataset split not found: {split_dir}. Expected folders like "
            f"{split_dir / 'colon_adenocarcinoma'} and {split_dir / 'lung_benign'}."
        )
    if not split_dir.is_dir():
        raise ValueError(f"Dataset split is not a directory: {split_dir}")

    class_dirs = sorted(path for path in split_dir.iterdir() if path.is_dir())
    if not class_dirs:
        raise ValueError(
            f"No class folders found in {split_dir}. Expected one subfolder per class."
        )

    empty_classes = [
        class_dir.name
        for class_dir in class_dirs
        if not any(
            path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            for path in class_dir.rglob("*")
        )
    ]
    if empty_classes:
        raise ValueError(
            f"No supported images found for class folders {empty_classes} in {split_dir}. "
            f"Supported extensions: {sorted(IMAGE_EXTENSIONS)}."
        )


def build_dataset(data_dir: Path, split: str, image_size: int) -> datasets.ImageFolder:
    split_dir = data_dir / split
    validate_split_dir(split_dir)

    return datasets.ImageFolder(
        root=split_dir,
        transform=build_transforms(image_size=image_size, train=split == "train"),
    )


def build_dataloaders(
    data_dir: Path,
    image_size: int,
    batch_size: int,
    num_workers: int = 0,
) -> tuple[dict[str, DataLoader], list[str]]:
    if not data_dir.exists():
        raise FileNotFoundError(
            f"Dataset folder not found: {data_dir}. Run scripts/split_image_folder.py first "
            "or point --data-dir to a folder with train, val and test splits."
        )

    datasets_by_split = {
        split: build_dataset(data_dir=data_dir, split=split, image_size=image_size)
        for split in ("train", "val", "test")
    }
    class_names = datasets_by_split["train"].classes
    if len(class_names) < 2:
        raise ValueError(
            f"At least two class folders are required in {data_dir / 'train'}, got {class_names}."
        )

    for split, dataset in datasets_by_split.items():
        if dataset.classes != class_names:
            raise ValueError(
                f"Class folders differ in {split}. Expected {class_names}, got {dataset.classes}."
            )

    loaders = {
        "train": DataLoader(
            datasets_by_split["train"],
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        ),
        "val": DataLoader(
            datasets_by_split["val"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        ),
        "test": DataLoader(
            datasets_by_split["test"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        ),
    }
    return loaders, class_names


def class_counts(dataset: datasets.ImageFolder) -> dict[str, int]:
    counts = {class_name: 0 for class_name in dataset.classes}
    for _, class_index in dataset.samples:
        counts[dataset.classes[class_index]] += 1
    return counts
