from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from multicancer.adapters.leukemia_adapter import LeukemiaAdapter
from multicancer.exceptions import (
    CheckpointMissingError,
    ExplanationUnavailableError,
    InvalidImageError,
)


class FakeModel:
    def __init__(self) -> None:
        self.eval_calls = 0

    def eval(self) -> None:
        self.eval_calls += 1


def build_fake_backend(model: FakeModel) -> SimpleNamespace:
    def load_checkpoint(_path: Path, device: str) -> tuple[FakeModel, dict]:
        assert device == "cpu"
        return model, {
            "class_names": ["normal", "leukemia_blast"],
            "image_size": 224,
            "model_name": "resnet18",
            "metrics": {"accuracy": 0.9},
        }

    def predict_image(**_kwargs: object) -> dict:
        return {
            "class_name": "leukemia_blast",
            "confidence": 0.8,
            "probabilities": [0.2, 0.8],
        }

    return SimpleNamespace(
        get_device=lambda: "cpu",
        load_checkpoint=load_checkpoint,
        predict_image=predict_image,
        get_gradcam_target_layer=lambda _model: (_ for _ in ()).throw(
            ValueError("unsupported test layer")
        ),
    )


def loaded_fake_adapter(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> LeukemiaAdapter:
    checkpoint_path = tmp_path / "best_model.pt"
    checkpoint_path.write_bytes(b"synthetic-checkpoint-placeholder")
    adapter = LeukemiaAdapter(checkpoint_path=checkpoint_path)
    backend = build_fake_backend(FakeModel())
    monkeypatch.setattr(adapter, "_import_backend", lambda: backend)
    adapter.load()
    return adapter


def test_missing_checkpoint_is_reported_without_loading(tmp_path: Path) -> None:
    adapter = LeukemiaAdapter(checkpoint_path=tmp_path / "missing.pt")

    status = adapter.checkpoint_status()

    assert status.status == "missing"
    assert not adapter.is_loaded


def test_load_missing_checkpoint_raises_common_error(tmp_path: Path) -> None:
    adapter = LeukemiaAdapter(checkpoint_path=tmp_path / "missing.pt")

    with pytest.raises(CheckpointMissingError, match="Checkpoint Leukemia absent"):
        adapter.load()


def test_load_is_lazy_and_idempotent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    checkpoint_path = tmp_path / "best_model.pt"
    checkpoint_path.write_bytes(b"synthetic-checkpoint-placeholder")
    model = FakeModel()
    backend = build_fake_backend(model)
    original_load_checkpoint = backend.load_checkpoint
    load_calls = 0

    def load_checkpoint(path: Path, device: str) -> tuple[FakeModel, dict]:
        nonlocal load_calls
        load_calls += 1
        return original_load_checkpoint(path, device)

    backend.load_checkpoint = load_checkpoint
    adapter = LeukemiaAdapter(checkpoint_path=checkpoint_path)
    monkeypatch.setattr(adapter, "_import_backend", lambda: backend)

    assert not adapter.is_loaded
    adapter.load()
    adapter.load()

    assert adapter.is_loaded
    assert load_calls == 1
    assert model.eval_calls == 1


def test_prediction_is_normalized(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = loaded_fake_adapter(tmp_path, monkeypatch)

    result = adapter.predict(Image.new("RGB", (32, 32), color="white"))

    assert result.project_id == "leukemia"
    assert result.predicted_class in adapter.metadata().classes
    assert result.predicted_index == 1
    assert result.model_name == "resnet18"
    assert sum(result.class_probabilities.values()) == pytest.approx(1.0)
    assert result.class_probabilities["leukemia_blast"] == pytest.approx(0.8)


def test_invalid_image_raises_common_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = loaded_fake_adapter(tmp_path, monkeypatch)

    with pytest.raises(InvalidImageError, match="image PIL"):
        adapter.predict(object())  # type: ignore[arg-type]


def test_gradcam_failure_is_controlled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = loaded_fake_adapter(tmp_path, monkeypatch)

    with pytest.raises(ExplanationUnavailableError, match="Grad-CAM indisponible"):
        adapter.explain(Image.new("RGB", (32, 32)), class_index=1)


def test_unload_is_idempotent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = loaded_fake_adapter(tmp_path, monkeypatch)

    adapter.unload()
    adapter.unload()

    assert not adapter.is_loaded
