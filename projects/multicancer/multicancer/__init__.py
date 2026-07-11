"""Shared metadata primitives for the R.C.C.I.A MultiCancer hub."""

from .model_manager import ModelManager
from .registry import PROJECT_REGISTRY, PROJECTS, get_project
from .schemas import (
    GLOBAL_DISCLAIMER,
    CheckpointStatus,
    ExplanationResult,
    PredictionResult,
    ProjectMetadata,
)

__all__ = [
    "GLOBAL_DISCLAIMER",
    "CheckpointStatus",
    "ExplanationResult",
    "ModelManager",
    "PROJECT_REGISTRY",
    "PROJECTS",
    "PredictionResult",
    "ProjectMetadata",
    "get_project",
]
