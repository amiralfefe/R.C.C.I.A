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
    supports_threshold_exploration: bool
    supports_modes: bool
    modes: tuple[str, ...]
    primary_metrics: tuple[str, ...]
    limitations: tuple[str, ...]
    methodological_note: str
    disclaimer: str = GLOBAL_DISCLAIMER


@dataclass(frozen=True)
class AdapterModeMetadata:
    """Checkpoint-independent configuration for one explicit adapter mode."""

    mode_id: str
    display_name: str
    classes: tuple[str, ...]
    model_name: str
    image_size: int
    task: str
    checkpoint_description: str
    primary_metrics: tuple[str, ...]
    limitations: tuple[str, ...]
    supports_gradcam: bool


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


@dataclass(frozen=True)
class ThresholdDecision:
    """Exploratory binary decision derived from an immutable model prediction."""

    positive_class: str
    negative_class: str
    positive_probability: float
    threshold: float
    thresholded_class: str
    is_threshold_override: bool
    default_threshold: float
    educational_warning: str
