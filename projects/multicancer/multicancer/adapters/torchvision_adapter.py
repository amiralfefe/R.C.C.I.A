"""Reusable lazy adapter for the existing Torchvision image-classification projects."""

from __future__ import annotations

import gc
import importlib
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from PIL import Image

from ..exceptions import (
    CheckpointIncompatibleError,
    CheckpointMissingError,
    ExplanationUnavailableError,
    InvalidImageError,
    ModelNotLoadedError,
)
from ..registry import get_project
from ..schemas import CheckpointStatus, ExplanationResult, PredictionResult, ProjectMetadata
from .base import BaseAdapter


class TorchvisionImageAdapter(BaseAdapter):
    """Common implementation that delegates ML behavior to a specialized package."""

    PROJECT_ID = ""
    PROJECT_FOLDER = ""
    PACKAGE_NAME = ""
    DEFAULT_CHECKPOINT_CANDIDATES: tuple[Path, ...] = ()

    def __init__(self, checkpoint_path: Path | str | None = None) -> None:
        if not all((self.PROJECT_ID, self.PROJECT_FOLDER, self.PACKAGE_NAME)):
            raise TypeError("Specialized adapter metadata is incomplete.")
        if not self.DEFAULT_CHECKPOINT_CANDIDATES and checkpoint_path is None:
            raise TypeError("At least one checkpoint candidate is required.")

        self._repo_root = Path(__file__).resolve().parents[4]
        self._project_dir = self._repo_root / "projects" / self.PROJECT_FOLDER
        self._configured_checkpoint = Path(checkpoint_path) if checkpoint_path else None
        self._model: Any | None = None
        self._checkpoint: dict[str, Any] | None = None
        self._device: Any | None = None
        self._backend: SimpleNamespace | None = None
        self._checkpoint_error: str | None = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None and self._checkpoint is not None

    def metadata(self) -> ProjectMetadata:
        return get_project(self.PROJECT_ID)

    def _resolve_path(self, path: Path) -> Path:
        return path if path.is_absolute() else self._repo_root / path

    def _checkpoint_candidates(self) -> tuple[Path, ...]:
        if self._configured_checkpoint is not None:
            return (self._resolve_path(self._configured_checkpoint),)
        return tuple(self._resolve_path(path) for path in self.DEFAULT_CHECKPOINT_CANDIDATES)

    def _selected_checkpoint_path(self) -> Path:
        candidates = self._checkpoint_candidates()
        return next((path for path in candidates if path.is_file()), candidates[0])

    def checkpoint_status(self) -> CheckpointStatus:
        checkpoint_path = self._selected_checkpoint_path()
        display_name = self.metadata().display_name
        if self._checkpoint_error:
            return CheckpointStatus(
                status="incompatible",
                checkpoint_path=checkpoint_path,
                message=self._checkpoint_error,
            )
        if checkpoint_path.is_file():
            return CheckpointStatus(
                status="available",
                checkpoint_path=checkpoint_path,
                message=(
                    f"Checkpoint {display_name} local detecte. Il sera charge uniquement "
                    "a la demande."
                ),
            )
        relative_hint = self.DEFAULT_CHECKPOINT_CANDIDATES[0].as_posix()
        return CheckpointStatus(
            status="missing",
            checkpoint_path=checkpoint_path,
            message=(
                f"Checkpoint {display_name} absent. Le hub reste consultable, mais la "
                f"prediction necessite {relative_hint}."
            ),
        )

    def _import_backend(self) -> SimpleNamespace:
        if self._backend is not None:
            return self._backend

        project_path = str(self._project_dir)
        if project_path not in sys.path:
            sys.path.insert(0, project_path)

        gradcam = importlib.import_module(f"{self.PACKAGE_NAME}.gradcam")
        model = importlib.import_module(f"{self.PACKAGE_NAME}.model")
        utils = importlib.import_module(f"{self.PACKAGE_NAME}.utils")
        self._backend = SimpleNamespace(
            GradCAM=gradcam.GradCAM,
            denormalize_image=gradcam.denormalize_image,
            get_device=utils.get_device,
            get_gradcam_target_layer=gradcam.get_gradcam_target_layer,
            image_to_tensor=gradcam.image_to_tensor,
            load_checkpoint=model.load_checkpoint,
            overlay_cam=gradcam.overlay_cam,
            predict_image=model.predict_image,
        )
        return self._backend

    def load(self) -> None:
        if self.is_loaded:
            return

        status = self.checkpoint_status()
        if status.status == "missing" or status.checkpoint_path is None:
            raise CheckpointMissingError(status.message)
        if status.status == "incompatible":
            raise CheckpointIncompatibleError(status.message)

        backend = self._import_backend()
        try:
            device = backend.get_device()
            model, checkpoint = backend.load_checkpoint(status.checkpoint_path, device=device)
            model.eval()
        except FileNotFoundError as exc:
            raise CheckpointMissingError(
                f"Checkpoint {self.metadata().display_name} introuvable : {exc}"
            ) from exc
        except (
            EOFError,
            KeyError,
            OSError,
            pickle.UnpicklingError,
            RuntimeError,
            ValueError,
        ) as exc:
            self._checkpoint_error = (
                f"Checkpoint {self.metadata().display_name} incompatible : {exc}"
            )
            raise CheckpointIncompatibleError(self._checkpoint_error) from exc

        self._device = device
        self._model = model
        self._checkpoint = checkpoint
        self._checkpoint_error = None

    @staticmethod
    def _validate_image(image: Image.Image) -> Image.Image:
        if not isinstance(image, Image.Image):
            raise InvalidImageError("Le contenu fourni n'est pas une image PIL valide.")
        if image.width <= 0 or image.height <= 0:
            raise InvalidImageError("L'image est vide ou possede des dimensions invalides.")
        try:
            return image.convert("RGB")
        except (OSError, ValueError) as exc:
            raise InvalidImageError(f"Impossible de convertir l'image en RGB : {exc}") from exc

    def _require_loaded(self) -> tuple[Any, dict[str, Any], Any, SimpleNamespace]:
        if not self.is_loaded or self._checkpoint is None:
            raise ModelNotLoadedError(
                f"Le modele {self.metadata().display_name} n'est pas charge. "
                "Chargez-le avant la prediction."
            )
        return self._model, self._checkpoint, self._device, self._import_backend()

    def predict(self, image: Image.Image) -> PredictionResult:
        model, checkpoint, device, backend = self._require_loaded()
        validated_image = self._validate_image(image)
        class_names = list(checkpoint["class_names"])
        image_size = int(checkpoint.get("image_size", 224))

        try:
            result = backend.predict_image(
                image=validated_image,
                model=model,
                class_names=class_names,
                image_size=image_size,
                device=device,
            )
        except (OSError, RuntimeError, ValueError) as exc:
            raise InvalidImageError(f"Prediction impossible pour cette image : {exc}") from exc

        probabilities = [float(value) for value in result["probabilities"]]
        if len(probabilities) != len(class_names):
            raise CheckpointIncompatibleError(
                "Le checkpoint retourne un nombre de probabilites incompatible avec ses classes."
            )

        predicted_class = str(result["class_name"])
        try:
            predicted_index = class_names.index(predicted_class)
        except ValueError as exc:
            raise CheckpointIncompatibleError(
                f"Classe predite absente du checkpoint : {predicted_class}."
            ) from exc

        metrics = checkpoint.get("metrics")
        raw_metadata = {"checkpoint_metrics": dict(metrics)} if isinstance(metrics, dict) else None
        return PredictionResult(
            project_id=self.PROJECT_ID,
            predicted_class=predicted_class,
            predicted_index=predicted_index,
            confidence=float(result["confidence"]),
            class_probabilities=dict(zip(class_names, probabilities, strict=True)),
            model_name=str(checkpoint.get("model_name", self.metadata().model_name)),
            image_size=image_size,
            warnings=("Resultat experimental sans validation clinique.",),
            disclaimer=self.metadata().disclaimer,
            raw_metadata=raw_metadata,
        )

    def explain(
        self,
        image: Image.Image,
        class_index: int | None = None,
    ) -> ExplanationResult:
        model, checkpoint, device, backend = self._require_loaded()
        validated_image = self._validate_image(image)
        class_names = list(checkpoint["class_names"])
        image_size = int(checkpoint.get("image_size", 224))

        if class_index is not None and not 0 <= class_index < len(class_names):
            raise ExplanationUnavailableError("Indice de classe Grad-CAM invalide.")

        try:
            target_layer = backend.get_gradcam_target_layer(model)
            tensor = backend.image_to_tensor(validated_image, image_size=image_size, device=device)
            gradcam = backend.GradCAM(model=model, target_layer=target_layer)
            try:
                cam, resolved_index = gradcam.generate(tensor, target_index=class_index)
            finally:
                gradcam.remove_hooks()
            overlay = backend.overlay_cam(backend.denormalize_image(tensor), cam)
        except (LookupError, OSError, RuntimeError, TypeError, ValueError) as exc:
            raise ExplanationUnavailableError(f"Grad-CAM indisponible : {exc}") from exc

        return ExplanationResult(
            available=True,
            image=overlay,
            image_path=None,
            class_index=int(resolved_index),
            class_name=class_names[int(resolved_index)],
            message=(
                f"Grad-CAM genere avec le pipeline {self.metadata().display_name} existant."
            ),
        )

    def unload(self) -> None:
        self._model = None
        self._checkpoint = None
        self._device = None
        gc.collect()

        if self._backend is not None:
            try:
                torch = importlib.import_module("torch")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except (ImportError, RuntimeError):
                pass
