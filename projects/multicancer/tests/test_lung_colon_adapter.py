from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from multicancer.adapters.lung_colon_adapter import LungColonAdapter
from multicancer.exceptions import (
    AdapterError,
    CheckpointIncompatibleError,
    CheckpointMissingError,
    ExplanationUnavailableError,
    InvalidImageError,
)


class FakeModel:
    def __init__(self) -> None:
        self.eval_calls = 0

    def eval(self) -> None:
        self.eval_calls += 1


class FakeGradCAM:
    def __init__(self, model: FakeModel, target_layer: object) -> None:
        self.model = model
        self.target_layer = target_layer

    def generate(
        self,
        _tensor: object,
        target_index: int | None = None,
    ) -> tuple[np.ndarray, int]:
        return np.ones((32, 32), dtype=np.float32), 0 if target_index is None else target_index

    def remove_hooks(self) -> None:
        return None


def checkpoint_payload(path: Path) -> dict:
    if "binary" in path.as_posix():
        return {
            "class_names": ["benign", "malignant"],
            "image_size": 224,
            "model_name": "resnet18",
            "metrics": {"accuracy": 1.0},
        }
    return {
        "class_names": [
            "colon_adenocarcinoma",
            "colon_benign",
            "lung_adenocarcinoma",
            "lung_benign",
            "lung_squamous_cell_carcinoma",
        ],
        "image_size": 224,
        "model_name": "efficientnet_b0",
        "metrics": {"accuracy": 0.9992},
    }


def build_fake_backend() -> SimpleNamespace:
    def load_checkpoint(path: Path, device: str) -> tuple[FakeModel, dict]:
        assert device == "cpu"
        return FakeModel(), checkpoint_payload(path)

    def predict_image(**kwargs: object) -> dict:
        class_names = kwargs["class_names"]
        if len(class_names) == 2:
            probabilities = [0.2, 0.8]
            predicted_index = 1
        else:
            probabilities = [0.01, 0.02, 0.03, 0.04, 0.90]
            predicted_index = 4
        return {
            "class_name": class_names[predicted_index],
            "confidence": probabilities[predicted_index],
            "probabilities": probabilities,
        }

    return SimpleNamespace(
        GradCAM=FakeGradCAM,
        denormalize_image=lambda _tensor: np.zeros((32, 32, 3), dtype=np.float32),
        get_device=lambda: "cpu",
        get_gradcam_target_layer=lambda _model: object(),
        image_to_tensor=lambda _image, image_size, device: (image_size, device),
        load_checkpoint=load_checkpoint,
        overlay_cam=lambda image, _cam: image,
        predict_image=predict_image,
    )


def adapter_with_checkpoints(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> LungColonAdapter:
    multiclass = tmp_path / "multiclass" / "best_model.pt"
    binary = tmp_path / "binary" / "best_model.pt"
    multiclass.parent.mkdir()
    binary.parent.mkdir()
    multiclass.write_bytes(b"multiclass")
    binary.write_bytes(b"binary")
    adapter = LungColonAdapter(
        checkpoint_paths={"multiclass": multiclass, "binary": binary}
    )
    monkeypatch.setattr(adapter, "_import_backend", build_fake_backend)
    return adapter


def test_modes_are_explicit_and_invalid_mode_is_rejected() -> None:
    adapter = LungColonAdapter()

    assert adapter.current_mode == "multiclass"
    assert tuple(mode.mode_id for mode in adapter.available_modes()) == (
        "multiclass",
        "binary",
    )
    assert len(adapter.mode_metadata("multiclass").classes) == 5
    assert adapter.mode_metadata("binary").classes == ("benign", "malignant")

    with pytest.raises(AdapterError, match="Mode LungColon inconnu"):
        adapter.set_mode("automatic")


def test_checkpoint_status_is_independent_for_each_mode(tmp_path: Path) -> None:
    multiclass = tmp_path / "multiclass.pt"
    multiclass.write_bytes(b"present")
    adapter = LungColonAdapter(
        checkpoint_paths={
            "multiclass": multiclass,
            "binary": tmp_path / "missing-binary.pt",
        }
    )

    assert adapter.checkpoint_status_for_mode("multiclass").status == "available"
    assert adapter.checkpoint_status_for_mode("binary").status == "missing"
    assert adapter.current_mode == "multiclass"


def test_missing_active_checkpoint_raises_common_error(tmp_path: Path) -> None:
    adapter = LungColonAdapter(
        checkpoint_paths={
            "multiclass": tmp_path / "missing-multiclass.pt",
            "binary": tmp_path / "missing-binary.pt",
        }
    )

    with pytest.raises(CheckpointMissingError, match="absent pour le mode"):
        adapter.load()


@pytest.mark.parametrize(
    ("mode_id", "expected_count", "expected_class", "expected_model"),
    [
        ("multiclass", 5, "lung_squamous_cell_carcinoma", "efficientnet_b0"),
        ("binary", 2, "malignant", "resnet18"),
    ],
)
def test_prediction_is_normalized_for_each_mode(
    mode_id: str,
    expected_count: int,
    expected_class: str,
    expected_model: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = adapter_with_checkpoints(tmp_path, monkeypatch)
    adapter.set_mode(mode_id)
    adapter.load()

    result = adapter.predict(Image.new("RGB", (32, 32), color="white"))

    assert result.predicted_class == expected_class
    assert result.model_name == expected_model
    assert len(result.class_probabilities) == expected_count
    assert sum(result.class_probabilities.values()) == pytest.approx(1.0)
    assert result.raw_metadata["mode_id"] == mode_id
    assert adapter.loaded_mode == mode_id


def test_mode_switch_unloads_previous_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = adapter_with_checkpoints(tmp_path, monkeypatch)
    adapter.load()
    multiclass_model = adapter._model

    adapter.set_mode("binary")

    assert not adapter.is_loaded
    assert adapter.loaded_mode is None
    assert adapter._model is None

    adapter.load()
    binary_model = adapter._model
    adapter.set_mode("multiclass")

    assert multiclass_model is not binary_model
    assert not adapter.is_loaded
    assert adapter.loaded_mode is None


def test_same_mode_does_not_unload_loaded_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = adapter_with_checkpoints(tmp_path, monkeypatch)
    adapter.load()
    loaded_model = adapter._model

    adapter.set_mode("multiclass")

    assert adapter.is_loaded
    assert adapter._model is loaded_model


def test_incompatible_mode_checkpoint_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = adapter_with_checkpoints(tmp_path, monkeypatch)
    backend = build_fake_backend()
    backend.load_checkpoint = lambda _path, device: (
        FakeModel(),
        checkpoint_payload(Path("binary/best_model.pt")),
    )
    monkeypatch.setattr(adapter, "_import_backend", lambda: backend)

    with pytest.raises(CheckpointIncompatibleError, match="mode 5 classes"):
        adapter.load()

    assert not adapter.is_loaded
    assert adapter.checkpoint_status().status == "incompatible"


def test_invalid_image_and_gradcam_index_are_controlled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = adapter_with_checkpoints(tmp_path, monkeypatch)
    adapter.load()

    with pytest.raises(InvalidImageError, match="image PIL"):
        adapter.predict(object())  # type: ignore[arg-type]
    with pytest.raises(ExplanationUnavailableError, match="Indice de classe"):
        adapter.explain(Image.new("RGB", (32, 32)), class_index=5)


@pytest.mark.parametrize("mode_id", ["multiclass", "binary"])
def test_gradcam_uses_active_mode(
    mode_id: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = adapter_with_checkpoints(tmp_path, monkeypatch)
    adapter.set_mode(mode_id)
    adapter.load()

    explanation = adapter.explain(Image.new("RGB", (32, 32)), class_index=0)

    assert explanation.available
    assert explanation.class_name == adapter.mode_metadata().classes[0]
    assert np.asarray(explanation.image).shape == (32, 32, 3)
    assert adapter.mode_metadata().display_name in explanation.message


def test_unload_is_idempotent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = adapter_with_checkpoints(tmp_path, monkeypatch)
    adapter.load()

    adapter.unload()
    adapter.unload()

    assert not adapter.is_loaded
    assert adapter.loaded_mode is None
