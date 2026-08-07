"""Mode-aware LungColon adapter backed by the existing specialized pipeline."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Mapping

from PIL import Image

from ..exceptions import AdapterError, CheckpointIncompatibleError
from ..schemas import AdapterModeMetadata, CheckpointStatus, ExplanationResult, PredictionResult
from .torchvision_adapter import TorchvisionImageAdapter


MULTICLASS_MODE = AdapterModeMetadata(
    mode_id="multiclass",
    display_name="5 classes - organe et sous-type histologique",
    classes=(
        "colon_adenocarcinoma",
        "colon_benign",
        "lung_adenocarcinoma",
        "lung_benign",
        "lung_squamous_cell_carcinoma",
    ),
    model_name="efficientnet_b0",
    image_size=224,
    task="Classification LC25000 en cinq classes pulmonaires et coliques",
    checkpoint_description="Meilleur checkpoint EfficientNet-B0 du benchmark V2",
    primary_metrics=("accuracy=0.9992", "macro_f1=0.9992", "test_images=3750"),
    limitations=(
        "Trois erreurs locales, toutes entre sous-types malins pulmonaires.",
        "Les performances tres elevees peuvent refleter les caracteristiques de LC25000.",
        "Aucune validation externe ou clinique.",
    ),
    supports_gradcam=True,
)

BINARY_MODE = AdapterModeMetadata(
    mode_id="binary",
    display_name="Binaire - benign / malignant",
    classes=("benign", "malignant"),
    model_name="resnet18",
    image_size=224,
    task="Classification LC25000 binaire avec regroupement benign et malignant",
    checkpoint_description="Checkpoint ResNet18 specialise du mode binaire V2.2",
    primary_metrics=("accuracy=1.0000", "recall_malignant=1.0000"),
    limitations=(
        "Le score parfait concerne uniquement le split local controle.",
        "Le regroupement binaire simplifie la tache par rapport au mode cinq classes.",
        "Aucune preuve de generalisation, validation externe ou validation clinique.",
    ),
    supports_gradcam=True,
)

LUNG_COLON_MODES = {
    MULTICLASS_MODE.mode_id: MULTICLASS_MODE,
    BINARY_MODE.mode_id: BINARY_MODE,
}

DEFAULT_CHECKPOINTS = {
    "multiclass": (
        Path("projects")
        / "lung_colon"
        / "outputs"
        / "model_comparison"
        / "efficientnet_b0"
        / "best_model.pt"
    ),
    "binary": (
        Path("projects")
        / "lung_colon"
        / "outputs"
        / "binary_resnet18"
        / "best_model.pt"
    ),
}


class LungColonAdapter(TorchvisionImageAdapter):
    """Expose two explicit LungColon checkpoints without automatic mode inference."""

    PROJECT_ID = "lung_colon"
    PROJECT_FOLDER = "lung_colon"
    PACKAGE_NAME = "rccia_lung_colon"
    DEFAULT_CHECKPOINT_CANDIDATES = (DEFAULT_CHECKPOINTS["multiclass"],)

    def __init__(
        self,
        mode_id: str = "multiclass",
        checkpoint_paths: Mapping[str, Path | str] | None = None,
    ) -> None:
        super().__init__()
        self._checkpoint_paths = {
            key: Path(value)
            for key, value in (checkpoint_paths or DEFAULT_CHECKPOINTS).items()
        }
        self._checkpoint_errors: dict[str, str] = {}
        self._loaded_mode: str | None = None
        self._mode_id = self._validate_mode(mode_id)

    @staticmethod
    def _validate_mode(mode_id: str) -> str:
        if mode_id not in LUNG_COLON_MODES:
            available = ", ".join(LUNG_COLON_MODES)
            raise AdapterError(
                f"Mode LungColon inconnu '{mode_id}'. Modes disponibles : {available}."
            )
        return mode_id

    @property
    def current_mode(self) -> str:
        return self._mode_id

    @property
    def loaded_mode(self) -> str | None:
        return self._loaded_mode

    @property
    def is_loaded(self) -> bool:
        return super().is_loaded and self._loaded_mode == self._mode_id

    def available_modes(self) -> tuple[AdapterModeMetadata, ...]:
        return tuple(LUNG_COLON_MODES.values())

    def mode_metadata(self, mode_id: str | None = None) -> AdapterModeMetadata:
        return LUNG_COLON_MODES[self._validate_mode(mode_id or self._mode_id)]

    def set_mode(self, mode_id: str) -> None:
        validated_mode = self._validate_mode(mode_id)
        if validated_mode == self._mode_id:
            return
        self.unload()
        self._mode_id = validated_mode
        self._checkpoint_error = self._checkpoint_errors.get(validated_mode)

    def _checkpoint_path_for_mode(self, mode_id: str) -> Path:
        configured = self._checkpoint_paths.get(mode_id, DEFAULT_CHECKPOINTS[mode_id])
        return self._resolve_path(configured)

    def _selected_checkpoint_path(self) -> Path:
        return self._checkpoint_path_for_mode(self._mode_id)

    def checkpoint_status_for_mode(self, mode_id: str) -> CheckpointStatus:
        mode = self.mode_metadata(mode_id)
        path = self._checkpoint_path_for_mode(mode.mode_id)
        error = self._checkpoint_errors.get(mode.mode_id)
        if error:
            return CheckpointStatus("incompatible", path, error)
        if path.is_file():
            return CheckpointStatus(
                "available",
                path,
                f"Checkpoint Lung + Colon {mode.display_name} detecte : "
                f"{mode.model_name}, {mode.image_size}x{mode.image_size}.",
            )
        return CheckpointStatus(
            "missing",
            path,
            f"Checkpoint Lung + Colon absent pour le mode {mode.display_name} : {path}.",
        )

    def checkpoint_status(self) -> CheckpointStatus:
        return self.checkpoint_status_for_mode(self._mode_id)

    def load(self) -> None:
        if self.is_loaded:
            return
        mode = self.mode_metadata()
        self._checkpoint_error = self._checkpoint_errors.get(mode.mode_id)
        try:
            super().load()
        except CheckpointIncompatibleError as exc:
            self._checkpoint_errors[mode.mode_id] = str(exc)
            raise

        checkpoint = self._checkpoint or {}
        actual_classes = tuple(checkpoint.get("class_names", ()))
        actual_model = str(checkpoint.get("model_name", ""))
        actual_size = int(checkpoint.get("image_size", 0))
        if (
            actual_classes != mode.classes
            or actual_model != mode.model_name
            or actual_size != mode.image_size
        ):
            message = (
                f"Checkpoint Lung + Colon incompatible pour le mode {mode.display_name}. "
                f"Attendu : {mode.model_name}, {mode.image_size}px, {mode.classes}. "
                f"Recu : {actual_model}, {actual_size}px, {actual_classes}."
            )
            self._checkpoint_errors[mode.mode_id] = message
            self._checkpoint_error = message
            super().unload()
            self._loaded_mode = None
            raise CheckpointIncompatibleError(message)

        self._loaded_mode = mode.mode_id
        self._checkpoint_errors.pop(mode.mode_id, None)
        self._checkpoint_error = None

    def predict(self, image: Image.Image) -> PredictionResult:
        mode = self.mode_metadata()
        result = super().predict(image)
        if tuple(result.class_probabilities) != mode.classes:
            raise CheckpointIncompatibleError(
                f"Ordre des classes incompatible pour le mode {mode.display_name}."
            )
        metadata = dict(result.raw_metadata or {})
        metadata.update(
            {
                "mode_id": mode.mode_id,
                "mode_display_name": mode.display_name,
            }
        )
        return replace(result, raw_metadata=metadata)

    def explain(
        self,
        image: Image.Image,
        class_index: int | None = None,
    ) -> ExplanationResult:
        mode = self.mode_metadata()
        result = super().explain(image, class_index=class_index)
        return replace(result, message=f"{result.message} Mode actif : {mode.display_name}.")

    def unload(self) -> None:
        self._loaded_mode = None
        super().unload()
