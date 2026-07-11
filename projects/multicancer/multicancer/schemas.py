"""Minimal common schemas for the MultiCancer V0 registry and future adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


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
    image_size: str
    status: str
    supports_gradcam: bool
    primary_metrics: tuple[str, ...]
    methodological_note: str


@dataclass(frozen=True)
class PredictionResult:
    """Conceptual normalized output returned by a future V1 adapter."""

    project_id: str
    predicted_class: str
    confidence: float
    class_probabilities: Mapping[str, float]
    model_name: str
    image_size: int
    explanation_path: str | None = None
    warnings: tuple[str, ...] = ()
    disclaimer: str = GLOBAL_DISCLAIMER
