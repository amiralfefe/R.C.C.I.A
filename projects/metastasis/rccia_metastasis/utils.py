"""Small utility helpers for Metastasis Vision."""

from __future__ import annotations

from pathlib import Path

import torch


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path
