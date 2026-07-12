from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from multicancer.registry import PROJECT_REGISTRY, PROJECTS, get_project
from multicancer.schemas import PredictionResult
from streamlit.testing.v1 import AppTest


EXPECTED_PROJECT_IDS = {"leukemia", "lung_colon", "breast", "metastasis"}


def test_registry_contains_all_specialized_projects() -> None:
    assert set(PROJECT_REGISTRY) == EXPECTED_PROJECT_IDS
    assert len(PROJECTS) == 4


def test_project_ids_are_unique() -> None:
    project_ids = [project.project_id for project in PROJECTS]
    assert len(project_ids) == len(set(project_ids))


def test_required_metadata_is_present() -> None:
    for project in PROJECTS:
        assert project.classes
        assert project.dataset
        assert project.task
        assert project.methodological_note
        assert project.primary_metrics
        assert project.limitations


def test_registry_has_no_absolute_windows_path() -> None:
    for project in PROJECTS:
        serialized = repr(asdict(project))
        assert ":\\" not in serialized
        assert ":/" not in serialized


def test_registry_loads_without_checkpoint() -> None:
    project = get_project("metastasis")
    assert project.project_id == "metastasis"
    assert not hasattr(project, "checkpoint")


def test_prediction_result_can_be_instantiated() -> None:
    result = PredictionResult(
        project_id="leukemia",
        predicted_class="normal",
        predicted_index=0,
        confidence=0.75,
        class_probabilities={"normal": 0.75, "leukemia_blast": 0.25},
        model_name="resnet18",
        image_size=224,
    )

    assert result.predicted_class == "normal"
    assert result.explanation_path is None
    assert result.disclaimer


def test_registry_marks_three_adapters_as_integrated() -> None:
    integrated = {project.project_id for project in PROJECTS if project.integrated}
    prediction_enabled = {
        project.project_id for project in PROJECTS if project.supports_prediction
    }

    assert integrated == {"leukemia", "breast", "metastasis"}
    assert prediction_enabled == {"leukemia", "breast", "metastasis"}
    assert not get_project("lung_colon").integrated


def test_streamlit_app_renders_without_loading_checkpoint() -> None:
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    app = AppTest.from_file(str(app_path)).run(timeout=20)

    assert not app.exception
    assert len(app.dataframe) == 1
    assert len(app.selectbox) == 1


def test_streamlit_non_integrated_project_stays_informational() -> None:
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    app = AppTest.from_file(str(app_path)).run(timeout=20)

    app.selectbox[0].select("lung_colon").run(timeout=20)

    assert not app.exception
    assert any("adaptateur Lung + Colon est prevu" in item.value for item in app.info)
    assert len(app.get("file_uploader")) == 0


def test_streamlit_breast_context_and_upload_are_available() -> None:
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    app = AppTest.from_file(str(app_path)).run(timeout=20)

    app.selectbox[0].select("breast").run(timeout=20)

    assert not app.exception
    assert any("Contexte methodologique Breast" in item.value for item in app.subheader)
    assert len(app.get("file_uploader")) == 1
    assert any("patient 14-16184CD" in item.value for item in app.warning)


def test_streamlit_metastasis_reference_table_is_visible_before_prediction() -> None:
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    app = AppTest.from_file(str(app_path)).run(timeout=20)

    app.selectbox[0].select("metastasis").run(timeout=20)

    assert not app.exception
    assert any("Contexte methodologique Metastasis" in item.value for item in app.subheader)
    assert any("Exploration du seuil de decision" in item.value for item in app.subheader)
    assert len(app.slider) == 0
    assert len(app.get("file_uploader")) == 1
    assert any("Aucun seuil n'est recommande" in item.value for item in app.warning)


def test_streamlit_rejects_invalid_image_without_crashing() -> None:
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    app = AppTest.from_file(str(app_path)).run(timeout=20)

    app.get("file_uploader")[0].upload(
        "invalid.png",
        b"not-an-image",
        "image/png",
    ).run(timeout=20)

    assert not app.exception
    assert any("ne contient pas une image" in item.value for item in app.error)
