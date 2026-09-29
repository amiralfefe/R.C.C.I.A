"""Optional end-to-end AppTest checks using local checkpoints and sample images."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from PIL import Image
from streamlit.testing.v1 import AppTest


REPO_ROOT = Path(__file__).resolve().parents[3]
APP_PATH = Path(__file__).resolve().parents[1] / "app.py"

LOCAL_CASES = (
    (
        "leukemia",
        None,
        REPO_ROOT / "projects/leukemia/outputs/best_model.pt",
        REPO_ROOT
        / "projects/leukemia/data/processed/test/leukemia_blast/leukemia_blast_000004.bmp",
    ),
    (
        "breast",
        None,
        REPO_ROOT
        / "projects/breast/outputs/model_comparison/efficientnet_b0/best_model.pt",
        REPO_ROOT
        / "projects/breast/data/processed/test/malignant/"
        "malignant_00000_SOB_M_DC-14-10926-100-001.png",
    ),
    (
        "metastasis",
        None,
        REPO_ROOT
        / "projects/metastasis/outputs/model_comparison/efficientnet_b0/best_model.pt",
        REPO_ROOT
        / "projects/metastasis/data/processed/test/metastatic/valid_metastatic_000002.png",
    ),
    (
        "lung_colon",
        "multiclass",
        REPO_ROOT
        / "projects/lung_colon/outputs/model_comparison/efficientnet_b0/best_model.pt",
        REPO_ROOT
        / "projects/lung_colon/data/processed/test/lung_adenocarcinoma/"
        "lung_adenocarcinoma_00030.jpeg",
    ),
    (
        "lung_colon",
        "binary",
        REPO_ROOT / "projects/lung_colon/outputs/binary_resnet18/best_model.pt",
        REPO_ROOT
        / "projects/lung_colon/data/binary_processed/test/malignant/"
        "colon_adenocarcinoma_colon_adenocarcinoma_00003.jpeg",
    ),
)


def _require_local_assets(checkpoint: Path, image_path: Path) -> None:
    missing = [path for path in (checkpoint, image_path) if not path.is_file()]
    if missing:
        pytest.skip(f"Local QA assets unavailable: {', '.join(map(str, missing))}")


def _uploadable_image(image_path: Path, tmp_path: Path) -> Path:
    if image_path.suffix.lower() != ".bmp":
        return image_path
    converted = tmp_path / f"{image_path.stem}.png"
    with Image.open(image_path) as source:
        source.convert("RGB").save(converted)
    return converted


@pytest.mark.parametrize(
    ("project_id", "mode_id", "checkpoint", "image_path"),
    LOCAL_CASES,
)
def test_real_streamlit_prediction_and_gradcam(
    project_id: str,
    mode_id: str | None,
    checkpoint: Path,
    image_path: Path,
    tmp_path: Path,
) -> None:
    """Exercise each real adapter while keeping absent local assets CI-safe."""

    _require_local_assets(checkpoint, image_path)
    upload_path = _uploadable_image(image_path, tmp_path)

    app = AppTest.from_file(str(APP_PATH)).run(timeout=30)
    app.selectbox[0].select(project_id).run(timeout=30)
    if mode_id is not None:
        app.radio[0].set_value(mode_id).run(timeout=30)

    app.get("file_uploader")[0].upload(
        upload_path.name,
        upload_path.read_bytes(),
        f"image/{'jpeg' if upload_path.suffix.lower() in {'.jpg', '.jpeg'} else 'png'}",
    ).run(timeout=30)
    app.button(key=f"analyze_{project_id}_{mode_id or 'single'}").click().run(timeout=90)

    assert not app.exception
    prediction = app.session_state["multicancer_last_prediction"]
    assert prediction.project_id == project_id
    assert sum(prediction.class_probabilities.values()) == pytest.approx(1.0, abs=1e-5)
    assert prediction.disclaimer
    assert prediction.warnings
    assert any("Modele en memoire : oui" in item.value for item in app.caption)

    app.toggle[0].set_value(True).run(timeout=90)
    assert not app.exception
    assert any("Grad-CAM genere" in item.value for item in app.caption)


def test_metastasis_slider_reuses_prediction_and_loaded_model() -> None:
    """Changing the educational threshold must not trigger a new inference."""

    project_id, _, checkpoint, image_path = LOCAL_CASES[2]
    _require_local_assets(checkpoint, image_path)

    app = AppTest.from_file(str(APP_PATH)).run(timeout=30)
    app.selectbox[0].select(project_id).run(timeout=30)
    app.get("file_uploader")[0].upload(
        image_path.name,
        image_path.read_bytes(),
        "image/png",
    ).run(timeout=30)
    app.button(key="analyze_metastasis_single").click().run(timeout=90)

    prediction = app.session_state["multicancer_last_prediction"]
    adapter = app.session_state["multicancer_model_manager"].current_adapter
    model_id = id(adapter._model)
    with patch.object(adapter, "explain", wraps=adapter.explain) as explain:
        app.toggle[0].set_value(True).run(timeout=90)
        app.slider[0].set_value(0.70).run(timeout=30)
        assert explain.call_count == 1

    assert not app.exception
    assert app.session_state["multicancer_last_prediction"] is prediction
    assert id(app.session_state["multicancer_model_manager"].current_adapter._model) == model_id


def test_changed_upload_clears_prediction_and_gradcam() -> None:
    _, _, checkpoint, image_path = LOCAL_CASES[2]
    _require_local_assets(checkpoint, image_path)
    app = AppTest.from_file(str(APP_PATH)).run(timeout=30)
    app.selectbox[0].select("metastasis").run()
    uploader = app.get("file_uploader")[0]
    uploader.upload("first.png", image_path.read_bytes(), "image/png").run()
    app.button(key="analyze_metastasis_single").click().run(timeout=90)
    app.toggle[0].set_value(True).run(timeout=90)
    assert "multicancer_last_explanation" in app.session_state
    app.get("file_uploader")[0].upload("second.png", image_path.read_bytes(), "image/png").run()
    assert not app.exception
    for key in ("multicancer_last_prediction", "multicancer_last_image", "multicancer_last_explanation"):
        assert key not in app.session_state


def test_failed_prediction_does_not_display_previous_result() -> None:
    from multicancer.exceptions import AdapterError

    _, _, checkpoint, image_path = LOCAL_CASES[2]
    _require_local_assets(checkpoint, image_path)
    app = AppTest.from_file(str(APP_PATH)).run(timeout=30)
    app.selectbox[0].select("metastasis").run()
    app.get("file_uploader")[0].upload("sample.png", image_path.read_bytes(), "image/png").run()
    app.button(key="analyze_metastasis_single").click().run(timeout=90)
    adapter = app.session_state["multicancer_model_manager"].current_adapter
    with patch.object(adapter, "predict", side_effect=AdapterError("Controlled test failure")):
        app.button(key="analyze_metastasis_single").click().run()
    assert not app.exception
    assert "multicancer_last_prediction" not in app.session_state
    assert any("Controlled test failure" in error.value for error in app.error)
