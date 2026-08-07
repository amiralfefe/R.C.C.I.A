"""Shared metadata primitives for the R.C.C.I.A MultiCancer hub."""

from .model_manager import ModelManager
from .registry import PROJECT_REGISTRY, PROJECTS, get_project
from .schemas import (
    GLOBAL_DISCLAIMER,
    AdapterModeMetadata,
    CheckpointStatus,
    ExplanationResult,
    PredictionResult,
    ProjectMetadata,
    ThresholdDecision,
)
from .thresholds import apply_binary_threshold

__all__ = [
    "GLOBAL_DISCLAIMER",
    "AdapterModeMetadata",
    "CheckpointStatus",
    "ExplanationResult",
    "ModelManager",
    "PROJECT_REGISTRY",
    "PROJECTS",
    "PredictionResult",
    "ProjectMetadata",
    "ThresholdDecision",
    "apply_binary_threshold",
    "get_project",
]
