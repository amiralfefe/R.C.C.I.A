"""Shared project configuration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TrainConfig:
    data_dir: Path
    output_dir: Path = Path("outputs")
    report_dir: Path = Path("outputs/train")
    model_name: str = "resnet18"
    image_size: int = 224
    epochs: int = 10
    batch_size: int = 16
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    num_workers: int = 0
    pretrained: bool = True
    seed: int = 42


BREAST_CLASSES = ("benign", "malignant")
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
