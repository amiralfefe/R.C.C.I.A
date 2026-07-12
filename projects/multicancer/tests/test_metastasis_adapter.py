from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from multicancer.adapters.metastasis_adapter import MetastasisAdapter
from multicancer.exceptions import (
    CheckpointIncompatibleError,
    CheckpointMissingError,
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
        return np.ones((24, 24), dtype=np.float32), 1 if target_index is None else target_index

    def remove_hooks(self) -> None:
        return None


def build_fake_backend(model: FakeModel) -> SimpleNamespace:
    def load_checkpoint(_path: Path, device: str) -> tuple[FakeModel, dict]:
        assert device == "cpu"
        return model, {
            "class_names": ["metastatic", "non_metastatic"],
            "image_size": 96,
            "model_name": "efficientnet_b0",
            "metrics": {"roc_auc": 0.9762},
        }

    def predict_image(**_kwargs: object) -> dict:
        return {
            "class_name": "non_metastatic",
            "confidence": 0.53,
            "probabilities": [0.47, 0.53],
            "prob_metastatic": 0.47,
        }

    return SimpleNamespace(
        GradCAM=FakeGradCAM,
        denormalize_image=lambda _tensor: np.zeros((24, 24, 3), dtype=np.float32),
        get_device=lambda: "cpu",
        get_gradcam_target_layer=lambda _model: object(),
        image_to_tensor=lambda _image, image_size, device: (image_size, device),
        load_checkpoint=load_checkpoint,
        overlay_cam=lambda image, _cam: image,
        predict_image=predict_image,
    )


def loaded_fake_adapter(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> MetastasisAdapter:
    checkpoint_path = tmp_path / "best_model.pt"
    checkpoint_path.write_bytes(b"synthetic-checkpoint-placeholder")
    adapter = MetastasisAdapter(checkpoint_path=checkpoint_path)
    backend = build_fake_backend(FakeModel())
    monkeypatch.setattr(adapter, "_import_backend", lambda: backend)
    adapter.load()
    return adapter


def test_missing_checkpoint_is_reported_without_loading(tmp_path: Path) -> None:
    adapter = MetastasisAdapter(checkpoint_path=tmp_path / "missing.pt")

    assert adapter.checkpoint_status().status == "missing"
    assert not adapter.is_loaded


def test_load_missing_checkpoint_raises_common_error(tmp_path: Path) -> None:
    adapter = MetastasisAdapter(checkpoint_path=tmp_path / "missing.pt")

    with pytest.raises(CheckpointMissingError, match="Checkpoint Metastasis absent"):
        adapter.load()


def test_incompatible_checkpoint_raises_common_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint_path = tmp_path / "best_model.pt"
    checkpoint_path.write_bytes(b"invalid")
    adapter = MetastasisAdapter(checkpoint_path=checkpoint_path)
    backend = build_fake_backend(FakeModel())
    backend.load_checkpoint = lambda _path, device: (_ for _ in ()).throw(
        RuntimeError(f"invalid state dict on {device}")
    )
    monkeypatch.setattr(adapter, "_import_backend", lambda: backend)

    with pytest.raises(CheckpointIncompatibleError, match="Checkpoint Metastasis incompatible"):
        adapter.load()


def test_prediction_preserves_raw_argmax_without_threshold(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = loaded_fake_adapter(tmp_path, monkeypatch)

    result = adapter.predict(Image.new("RGB", (24, 24), color="white"))

    assert result.project_id == "metastasis"
    assert result.predicted_class == "non_metastatic"
    assert result.predicted_index == 1
    assert result.image_size == 96
    assert result.model_name == "efficientnet_b0"
    assert result.confidence == pytest.approx(0.53)
    assert sum(result.class_probabilities.values()) == pytest.approx(1.0)
    assert result.class_probabilities["metastatic"] == pytest.approx(0.47)
    assert not hasattr(result, "threshold")
    assert not hasattr(result, "thresholded_class")


def test_invalid_image_raises_common_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = loaded_fake_adapter(tmp_path, monkeypatch)

    with pytest.raises(InvalidImageError, match="image PIL"):
        adapter.predict(object())  # type: ignore[arg-type]


def test_gradcam_result_is_normalized(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = loaded_fake_adapter(tmp_path, monkeypatch)

    explanation = adapter.explain(Image.new("RGB", (24, 24)), class_index=0)

    assert explanation.available
    assert explanation.class_index == 0
    assert explanation.class_name == "metastatic"
    assert isinstance(explanation.image, np.ndarray)
    assert explanation.image.shape == (24, 24, 3)


def test_unload_is_idempotent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = loaded_fake_adapter(tmp_path, monkeypatch)

    adapter.unload()
    adapter.unload()

    assert not adapter.is_loaded
