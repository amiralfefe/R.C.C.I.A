"""Leukemia adapter configuration using the shared Torchvision implementation."""

from pathlib import Path

from .torchvision_adapter import TorchvisionImageAdapter


class LeukemiaAdapter(TorchvisionImageAdapter):
    """Expose the completed Leukemia project through the common V1 contract."""

    PROJECT_ID = "leukemia"
    PROJECT_FOLDER = "leukemia"
    PACKAGE_NAME = "rccia_leukemia"
    DEFAULT_CHECKPOINT_CANDIDATES = (
        Path("projects") / "leukemia" / "outputs" / "best_model.pt",
    )
