"""Metastasis adapter configuration using the shared Torchvision implementation."""

from pathlib import Path

from .torchvision_adapter import TorchvisionImageAdapter


class MetastasisAdapter(TorchvisionImageAdapter):
    """Expose the PCam subset pipeline while preserving raw model inference."""

    PROJECT_ID = "metastasis"
    PROJECT_FOLDER = "metastasis"
    PACKAGE_NAME = "rccia_metastasis"
    DEFAULT_CHECKPOINT_CANDIDATES = (
        Path("projects")
        / "metastasis"
        / "outputs"
        / "model_comparison"
        / "efficientnet_b0"
        / "best_model.pt",
    )
