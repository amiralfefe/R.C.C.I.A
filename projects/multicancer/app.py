from __future__ import annotations

import io
import sys
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image, UnidentifiedImageError


APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from multicancer.exceptions import (  # noqa: E402
    AdapterError,
    CheckpointIncompatibleError,
    CheckpointMissingError,
    ExplanationUnavailableError,
    InvalidImageError,
)
from multicancer.model_manager import ModelManager  # noqa: E402
from multicancer.registry import PROJECTS, get_project  # noqa: E402
from multicancer.schemas import GLOBAL_DISCLAIMER, PredictionResult  # noqa: E402


SESSION_MANAGER_KEY = "multicancer_model_manager"
SESSION_PROJECT_KEY = "multicancer_selected_project"
SESSION_PREDICTION_KEY = "multicancer_last_prediction"
SESSION_IMAGE_KEY = "multicancer_last_image"


def project_rows() -> list[dict[str, str]]:
    """Build display rows without loading models, checkpoints, or datasets."""

    rows: list[dict[str, str]] = []
    for project in PROJECTS:
        values = asdict(project)
        rows.append(
            {
                "Project": values["display_name"],
                "Dataset": values["dataset"],
                "Task": values["task"],
                "Classes": ", ".join(values["classes"]),
                "Input": values["image_size"],
                "Adapter": values["adapter_status"],
                "Prediction": "yes" if values["supports_prediction"] else "planned",
                "Grad-CAM": "yes" if values["supports_gradcam"] else "no",
            }
        )
    return rows


def decode_uploaded_image(content: bytes) -> Image.Image:
    """Validate PNG/JPEG bytes and return an in-memory RGB image."""

    if not content:
        raise InvalidImageError("Le fichier image est vide.")

    try:
        with Image.open(io.BytesIO(content)) as candidate:
            detected_format = candidate.format
            candidate.verify()
        if detected_format not in {"PNG", "JPEG"}:
            raise InvalidImageError(
                f"Format image non supporte : {detected_format or 'inconnu'}. Utilisez PNG ou JPEG."
            )
        with Image.open(io.BytesIO(content)) as candidate:
            image = candidate.convert("RGB")
            image.load()
    except InvalidImageError:
        raise
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise InvalidImageError("Le fichier ne contient pas une image PNG/JPEG valide.") from exc

    if image.width <= 0 or image.height <= 0:
        raise InvalidImageError("L'image possede des dimensions invalides.")
    return image


def get_model_manager() -> ModelManager:
    manager = st.session_state.get(SESSION_MANAGER_KEY)
    if not isinstance(manager, ModelManager):
        manager = ModelManager()
        st.session_state[SESSION_MANAGER_KEY] = manager
    return manager


def clear_prediction_state() -> None:
    st.session_state.pop(SESSION_PREDICTION_KEY, None)
    st.session_state.pop(SESSION_IMAGE_KEY, None)


def synchronize_project(manager: ModelManager, project_id: str) -> None:
    previous_project = st.session_state.get(SESSION_PROJECT_KEY)
    if previous_project != project_id:
        manager.unload_current()
        clear_prediction_state()
        st.session_state[SESSION_PROJECT_KEY] = project_id


def render_project_metadata(project_id: str) -> None:
    project = get_project(project_id)
    summary = st.columns(4)
    summary[0].metric("Projet", project.display_name)
    summary[1].metric("Modele", project.model_name)
    summary[2].metric("Resolution", project.image_size)
    summary[3].metric("Adaptateur", project.adapter_status)

    st.write(f"**Tache :** {project.task}")
    st.write(f"**Dataset :** {project.dataset}")
    st.write(f"**Classes :** {', '.join(project.classes)}")
    st.write(f"**Metriques documentees :** {', '.join(project.primary_metrics)}")

    st.subheader("Limites du projet")
    for limitation in project.limitations:
        st.write(f"- {limitation}")
    st.warning(project.methodological_note)


def render_prediction(result: PredictionResult) -> None:
    st.subheader("Prediction normalisee")
    prediction_columns = st.columns(4)
    prediction_columns[0].metric("Classe predite", result.predicted_class)
    prediction_columns[1].metric("Confiance", f"{result.confidence:.2%}")
    prediction_columns[2].metric("Modele", result.model_name)
    prediction_columns[3].metric("Resolution", f"{result.image_size}x{result.image_size}")

    probability_rows = pd.DataFrame(
        {
            "classe": list(result.class_probabilities),
            "probabilite": list(result.class_probabilities.values()),
        }
    )
    st.dataframe(probability_rows, hide_index=True, width="stretch")
    st.bar_chart(probability_rows.set_index("classe"))
    for warning in result.warnings:
        st.caption(warning)


def render_leukemia_flow(manager: ModelManager) -> None:
    try:
        adapter = manager.activate("leukemia")
    except AdapterError as exc:
        st.error(str(exc))
        return

    checkpoint = adapter.checkpoint_status()
    st.subheader("Checkpoint Leukemia")
    if checkpoint.status == "available":
        st.success(checkpoint.message)
    elif checkpoint.status == "missing":
        st.warning(checkpoint.message)
    else:
        st.error(checkpoint.message)
    if checkpoint.checkpoint_path is not None:
        try:
            display_path = checkpoint.checkpoint_path.relative_to(APP_DIR.parents[1])
        except ValueError:
            display_path = checkpoint.checkpoint_path
        st.caption(f"Chemin local attendu : `{display_path}`")

    if st.button(
        "Charger le modele Leukemia",
        disabled=checkpoint.status != "available" or adapter.is_loaded,
        type="secondary",
    ):
        try:
            adapter.load()
        except (CheckpointMissingError, CheckpointIncompatibleError) as exc:
            st.error(str(exc))
        else:
            st.success("Modele Leukemia charge a la demande.")

    st.caption(f"Modele en memoire : {'oui' if adapter.is_loaded else 'non'}")
    uploaded_file = st.file_uploader(
        "Image de cellule sanguine",
        type=["png", "jpg", "jpeg"],
        help="Le fichier reste en memoire pendant la session et n'est pas enregistre.",
    )

    uploaded_image: Image.Image | None = None
    if uploaded_file is not None:
        try:
            uploaded_image = decode_uploaded_image(uploaded_file.getvalue())
        except InvalidImageError as exc:
            st.error(str(exc))
            clear_prediction_state()
        else:
            st.image(uploaded_image, caption="Image chargee", width=420)

    if st.button("Analyser l'image", type="primary", disabled=uploaded_image is None):
        if uploaded_image is None:
            st.error("Chargez une image PNG ou JPEG valide avant l'analyse.")
        else:
            try:
                adapter.load()
                prediction = adapter.predict(uploaded_image)
            except (AdapterError, OSError, RuntimeError, ValueError) as exc:
                st.error(str(exc))
            else:
                st.session_state[SESSION_PREDICTION_KEY] = prediction
                st.session_state[SESSION_IMAGE_KEY] = uploaded_image.copy()
                st.success("Prediction Leukemia terminee.")

    prediction = st.session_state.get(SESSION_PREDICTION_KEY)
    if isinstance(prediction, PredictionResult) and prediction.project_id == "leukemia":
        render_prediction(prediction)
        show_gradcam = st.toggle("Afficher Grad-CAM", value=False)
        if show_gradcam:
            explanation_image = st.session_state.get(SESSION_IMAGE_KEY)
            if not isinstance(explanation_image, Image.Image):
                st.info("Rechargez l'image pour generer Grad-CAM.")
            else:
                try:
                    explanation = adapter.explain(
                        explanation_image,
                        class_index=prediction.predicted_index,
                    )
                except ExplanationUnavailableError as exc:
                    st.info(str(exc))
                else:
                    st.image(
                        explanation.image,
                        caption=f"Grad-CAM - classe {explanation.class_name}",
                        width=520,
                    )
                    st.caption(explanation.message)


def render_app() -> None:
    st.set_page_config(page_title="R.C.C.I.A MultiCancer", layout="wide")
    st.title("R.C.C.I.A MultiCancer")
    st.caption("Hub de pipelines specialises - V1.1 Leukemia Adapter")
    st.error(
        "Demonstrateur educatif / portfolio uniquement. Aucune validation clinique, "
        "aucun diagnostic medical et aucune recommandation medicale."
    )

    manager = get_model_manager()
    with st.sidebar:
        st.header("Projet")
        selected_project_id = st.selectbox(
            "Pipeline specialise",
            options=[project.project_id for project in PROJECTS],
            format_func=lambda value: get_project(value).display_name,
        )

    synchronize_project(manager, selected_project_id)
    selected_project = get_project(selected_project_id)

    with st.sidebar:
        st.caption(f"Integration : {selected_project.adapter_status}")
        st.caption(
            "Prediction : "
            f"{'disponible' if selected_project.supports_prediction else 'prochaine phase'}"
        )
        st.caption(
            "Modele actif : "
            f"{manager.active_project_id if manager.active_project_id else 'aucun'}"
        )

    st.subheader("Portefeuille specialise")
    st.dataframe(project_rows(), hide_index=True, width="stretch")
    render_project_metadata(selected_project_id)

    if selected_project.integrated and selected_project.supports_prediction:
        render_leukemia_flow(manager)
    else:
        manager.unload_current()
        st.info(
            f"L'adaptateur {selected_project.display_name} est prevu dans une prochaine phase. "
            "Aucun checkpoint n'est charge pour ce projet."
        )

    st.divider()
    st.subheader("Cadre d'utilisation")
    st.write(
        "Le projet est choisi explicitement. Le hub ne detecte pas automatiquement une "
        "modalite et ne melange ni datasets ni modeles. Un seul adaptateur peut conserver "
        "un modele charge pendant la session."
    )
    st.warning(selected_project.disclaimer)
    st.caption(GLOBAL_DISCLAIMER)


if __name__ == "__main__":
    render_app()
