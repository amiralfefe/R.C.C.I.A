"""Breast adapter configuration using the shared Torchvision implementation."""

from pathlib import Path

from .torchvision_adapter import TorchvisionImageAdapter


class BreastAdapter(TorchvisionImageAdapter):
    """Expose the patient-aware BreakHis pipeline through the common contract."""

    PROJECT_ID = "breast"
    PROJECT_FOLDER = "breast"
    PACKAGE_NAME = "rccia_breast"
    DEFAULT_CHECKPOINT_CANDIDATES = (
        Path("projects")
        / "breast"
        / "outputs"
        / "model_comparison"
        / "efficientnet_b0"
        / "best_model.pt",
    )
