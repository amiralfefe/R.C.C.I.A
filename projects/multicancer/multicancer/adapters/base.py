"""Common lifecycle contract for specialized MultiCancer adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod

from PIL import Image

from ..schemas import CheckpointStatus, ExplanationResult, PredictionResult, ProjectMetadata


class BaseAdapter(ABC):
    """Contract implemented by every prediction-capable project adapter."""

    @abstractmethod
    def metadata(self) -> ProjectMetadata:
        """Return stable project metadata without loading a model."""

    @abstractmethod
    def checkpoint_status(self) -> CheckpointStatus:
        """Return the current local checkpoint state."""

    @abstractmethod
    def load(self) -> None:
        """Load the specialized model lazily and idempotently."""

    @abstractmethod
    def predict(self, image: Image.Image) -> PredictionResult:
        """Return a normalized prediction for a validated image."""

    @abstractmethod
    def explain(
        self,
        image: Image.Image,
        class_index: int | None = None,
    ) -> ExplanationResult:
        """Return an optional visual explanation for one prediction."""

    @abstractmethod
    def unload(self) -> None:
        """Release model references and accelerator cache if applicable."""

    @property
    @abstractmethod
    def is_loaded(self) -> bool:
        """Whether this adapter currently owns a loaded model."""
