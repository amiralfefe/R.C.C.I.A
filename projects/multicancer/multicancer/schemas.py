"""Common metadata, checkpoint, prediction, and explanation schemas."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping


GLOBAL_DISCLAIMER = (
    "Educational portfolio demonstrator only. No clinical validation, medical diagnosis, "
    "medical recommendation, or support for healthcare decisions."
)


@dataclass(frozen=True)
class ProjectMetadata:
    """Stable, checkpoint-independent metadata for one specialized project."""

    project_id: str
    display_name: str
    modality: str
    task: str
    dataset: str
    classes: tuple[str, ...]
    image_size: int
    model_name: str
    status: str
    adapter_status: str
    integrated: bool
    supports_prediction: bool
    supports_gradcam: bool
    primary_metrics: tuple[str, ...]
    limitations: tuple[str, ...]
    methodological_note: str
    disclaimer: str = GLOBAL_DISCLAIMER


CheckpointState = Literal["available", "missing", "incompatible", "unchecked"]


@dataclass(frozen=True)
class CheckpointStatus:
    """Normalized state of a local, non-versioned model checkpoint."""

    status: CheckpointState
    checkpoint_path: Path | None
    message: str


@dataclass(frozen=True)
class PredictionResult:
    """Normalized prediction returned by a specialized adapter."""

    project_id: str
    predicted_class: str
    predicted_index: int
    confidence: float
    class_probabilities: Mapping[str, float]
    model_name: str
    image_size: int
    explanation_path: str | None = None
    warnings: tuple[str, ...] = ()
    disclaimer: str = GLOBAL_DISCLAIMER
    raw_metadata: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class ExplanationResult:
    """Optional visual explanation returned without persisting uploaded images."""

    available: bool
    image: object | None
    image_path: str | None
    class_index: int | None
    class_name: str | None
    message: str
