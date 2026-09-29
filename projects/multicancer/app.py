from __future__ import annotations

import sys
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image


APP_DIR = Path(__file__).resolve().parent
PROJECTS_DIR = APP_DIR.parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))
if str(PROJECTS_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECTS_DIR))

from rccia_common.image_uploads import (  # noqa: E402
    HUB_IMAGE_FORMATS,
    ImageUploadError,
    decode_uploaded_image,
)

from multicancer.exceptions import (  # noqa: E402
    AdapterError,
    CheckpointIncompatibleError,
    CheckpointMissingError,
    ExplanationUnavailableError,
)
from multicancer.model_manager import ModelManager  # noqa: E402
from multicancer.adapters.lung_colon_adapter import LungColonAdapter  # noqa: E402
from multicancer.registry import PROJECTS, get_project  # noqa: E402
from multicancer.schemas import (  # noqa: E402
    GLOBAL_DISCLAIMER,
    AdapterModeMetadata,
    PredictionResult,
)
from multicancer.thresholds import apply_binary_threshold  # noqa: E402


SESSION_MANAGER_KEY = "multicancer_model_manager"
SESSION_PROJECT_KEY = "multicancer_selected_project"
SESSION_PREDICTION_KEY = "multicancer_last_prediction"
SESSION_IMAGE_KEY = "multicancer_last_image"
SESSION_EXPLANATION_KEY = "multicancer_last_explanation"
SESSION_LUNG_COLON_MODE_KEY = "multicancer_lung_colon_mode"

UPLOAD_LABELS = {
    "leukemia": "Image de cellule sanguine",
    "breast": "Image histopathologique mammaire",
    "metastasis": "Patch histopathologique PCam",
    "lung_colon": "Image histopathologique LC25000",
}

METASTASIS_THRESHOLD_ROWS = [
    {
        "threshold": 0.30,
        "accuracy": 0.8907,
        "precision_metastatic": 0.8399,
        "recall_metastatic": 0.9653,
        "f1_metastatic": 0.8983,
        "FP": 69,
        "FN": 13,
    },
    {
        "threshold": 0.40,
        "accuracy": 0.9160,
        "precision_metastatic": 0.8861,
        "recall_metastatic": 0.9547,
        "f1_metastatic": 0.9191,
        "FP": 46,
        "FN": 17,
    },
    {
        "threshold": 0.50,
        "accuracy": 0.9320,
        "precision_metastatic": 0.9355,
        "recall_metastatic": 0.9280,
        "f1_metastatic": 0.9317,
        "FP": 24,
        "FN": 27,
    },
    {
        "threshold": 0.60,
        "accuracy": 0.9187,
        "precision_metastatic": 0.9460,
        "recall_metastatic": 0.8880,
        "f1_metastatic": 0.9161,
        "FP": 19,
        "FN": 42,
    },
    {
        "threshold": 0.70,
        "accuracy": 0.9120,
        "precision_metastatic": 0.9668,
        "recall_metastatic": 0.8533,
        "f1_metastatic": 0.9065,
        "FP": 11,
        "FN": 55,
    },
]


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
                "Input": f"{values['image_size']}x{values['image_size']}",
                "Adapter": values["adapter_status"],
                "Prediction": "yes" if values["supports_prediction"] else "planned",
                "Grad-CAM": "yes" if values["supports_gradcam"] else "no",
                "Threshold": (
                    "exploratory" if values["supports_threshold_exploration"] else "no"
                ),
                "Modes": ", ".join(values["modes"]) or "single",
            }
        )
    return rows


def get_model_manager() -> ModelManager:
    manager = st.session_state.get(SESSION_MANAGER_KEY)
    if not isinstance(manager, ModelManager):
        manager = ModelManager()
        st.session_state[SESSION_MANAGER_KEY] = manager
    return manager


def clear_result_state() -> None:
    st.session_state.pop(SESSION_PREDICTION_KEY, None)
    st.session_state.pop(SESSION_IMAGE_KEY, None)
    st.session_state.pop(SESSION_EXPLANATION_KEY, None)


def clear_prediction_state() -> None:
    clear_result_state()
    st.session_state.pop("upload_lung_colon", None)
    st.session_state.pop("gradcam_lung_colon", None)


def synchronize_project(manager: ModelManager, project_id: str) -> None:
    previous_project = st.session_state.get(SESSION_PROJECT_KEY)
    if previous_project != project_id:
        manager.unload_current()
        clear_prediction_state()
        st.session_state[SESSION_PROJECT_KEY] = project_id


def render_project_metadata(
    project_id: str,
    mode: AdapterModeMetadata | None = None,
) -> None:
    project = get_project(project_id)
    model_name = mode.model_name if mode else project.model_name
    image_size = mode.image_size if mode else project.image_size
    task = mode.task if mode else project.task
    classes = mode.classes if mode else project.classes
    metrics = mode.primary_metrics if mode else project.primary_metrics
    limitations = mode.limitations if mode else project.limitations
    summary = st.columns(4)
    summary[0].metric("Projet", project.display_name)
    summary[1].metric("Modele", model_name)
    summary[2].metric("Resolution", f"{image_size}x{image_size}")
    summary[3].metric("Adaptateur", project.adapter_status)

    if mode:
        st.write(f"**Mode explicite :** {mode.display_name}")
    st.write(f"**Tache :** {task}")
    st.write(f"**Dataset :** {project.dataset}")
    st.write(f"**Classes :** {', '.join(classes)}")
    st.write(f"**Metriques documentees :** {', '.join(metrics)}")

    st.subheader("Limites du projet")
    for limitation in limitations:
        st.write(f"- {limitation}")
    st.warning(project.methodological_note)


def configure_lung_colon_mode(manager: ModelManager) -> AdapterModeMetadata:
    adapter = manager.current_adapter
    if not isinstance(adapter, LungColonAdapter):
        raise AdapterError("L'adaptateur LungColon actif est indisponible.")

    modes = adapter.available_modes()
    selected_mode = st.radio(
        "Mode de classification",
        options=[mode.mode_id for mode in modes],
        format_func=lambda value: next(
            mode.display_name for mode in modes if mode.mode_id == value
        ),
        key=SESSION_LUNG_COLON_MODE_KEY,
    )
    if selected_mode != adapter.current_mode:
        manager.set_active_mode(selected_mode)
        clear_prediction_state()
    st.caption("Choix explicite : aucun mode n'est deduit automatiquement de l'image.")
    return adapter.mode_metadata()


def render_lung_colon_methodology(mode: AdapterModeMetadata) -> None:
    st.subheader("Contexte methodologique Lung + Colon")
    if mode.mode_id == "multiclass":
        context = st.columns(4)
        context[0].metric("Images test", "3 750")
        context[1].metric("Accuracy", "0.9992")
        context[2].metric("Macro F1", "0.9992")
        context[3].metric("Erreurs", "3")
        st.caption(
            "Les trois erreurs concernent uniquement la distinction entre deux sous-types "
            "malins pulmonaires. Aucune seconde decision par organe n'est derivee."
        )
    else:
        context = st.columns(3)
        context[0].metric("Accuracy locale", "1.0000")
        context[1].metric("Recall malignant", "1.0000")
        context[2].metric("Modele distinct", "ResNet18")
        st.caption(
            "La sortie binaire vient de son checkpoint specialise ; elle n'est jamais "
            "derivee des probabilites du modele cinq classes."
        )
    st.warning(
        "LC25000 est un benchmark public aux performances souvent tres elevees. Les scores "
        "dependent du split et du protocole local, ne sont pas directement comparables aux "
        "autres projets et ne prouvent aucune generalisation clinique."
    )


def render_breast_methodology() -> None:
    st.subheader("Contexte methodologique Breast")
    context = st.columns(4)
    context[0].metric("Patients detectes", "81")
    context[1].metric("Overlap patient", "0")
    context[2].metric("Images test", "1 481")
    context[3].metric("Erreurs V2.1", "130")
    st.write(
        "Le split est patient-aware : les images d'un meme patient ne sont pas reparties "
        "entre train, validation et test. Les resultats restent issus d'un dataset public."
    )
    magnification_rows = pd.DataFrame(
        [
            {"grossissement": "40X", "accuracy": 0.8757},
            {"grossissement": "100X", "accuracy": 0.9194},
            {"grossissement": "200X", "accuracy": 0.9467},
            {"grossissement": "400X", "accuracy": 0.9063},
        ]
    )
    st.dataframe(magnification_rows, hide_index=True, width="stretch")
    st.caption(
        "V2.1 : 88 faux positifs benign -> malignant, 42 faux negatifs malignant -> "
        "benign. Confiance moyenne : 0.9700 sur les predictions correctes et 0.8464 "
        "sur les erreurs."
    )
    st.warning(
        "Le patient 14-16184CD concentre 76 erreurs. Cette observation illustre une "
        "variabilite patient et ne constitue pas une conclusion clinique. Les scores ne "
        "sont pas directement comparables a ceux des autres projets."
    )


def render_metastasis_methodology() -> None:
    st.subheader("Contexte methodologique Metastasis")
    context = st.columns(4)
    context[0].metric("Images test", "750")
    context[1].metric("Accuracy", "0.9320")
    context[2].metric("ROC-AUC", "0.9762")
    context[3].metric("PR-AUC", "0.9780")
    st.write(
        "Le benchmark utilise un subset PCam equilibre de 5 000 patches, dont 750 dans "
        "le test set local. Les metriques dependent de ce split et du protocole local."
    )
    st.caption(
        "Au seuil 0.50 : 699 predictions correctes, 51 erreurs, 24 faux positifs et "
        "27 faux negatifs. Confiance moyenne : 0.9071 sur les predictions correctes et "
        "0.7107 sur les erreurs."
    )


def render_threshold_exploration(prediction: PredictionResult | None) -> None:
    st.subheader("Exploration du seuil de decision")
    st.warning(
        "Aucun seuil n'est recommande medicalement. Cette interface illustre uniquement "
        "l'impact d'un seuil sur les faux positifs et les faux negatifs dans un benchmark "
        "educatif."
    )

    if prediction is None:
        st.info(
            "Lancez une prediction Metastasis pour explorer une decision derivee des "
            "probabilites brutes. Le tableau de reference reste consultable ci-dessous."
        )
    else:
        threshold = st.slider(
            "Seuil exploratoire metastatic",
            min_value=0.30,
            max_value=0.70,
            value=0.50,
            step=0.01,
            key="metastasis_threshold",
        )
        decision = apply_binary_threshold(
            prediction,
            positive_class="metastatic",
            threshold=threshold,
        )
        original = st.columns(2)
        original[0].metric("Argmax original", prediction.predicted_class)
        original[1].metric(
            "Probabilite metastatic",
            f"{decision.positive_probability:.2%}",
        )
        derived = st.columns(2)
        derived[0].metric("Seuil applique", f"{decision.threshold:.2f}")
        derived[1].metric("Decision au seuil", decision.thresholded_class)
        if decision.is_threshold_override:
            st.info("Override exploratoire actif : le seuil par defaut est 0.50.")
        else:
            st.caption("Seuil par defaut 0.50 utilise.")
        st.write(
            "L'argmax et la decision au seuil peuvent differer. Le modele n'est pas "
            "recalcule : seule une regle distincte est appliquee a la probabilite "
            "`metastatic`."
        )
        st.caption(decision.educational_warning)

    st.markdown("**Resultats agreges V2.1 sur les 750 images test locales**")
    st.dataframe(pd.DataFrame(METASTASIS_THRESHOLD_ROWS), hide_index=True, width="stretch")
    st.caption(
        "Seuil plus bas : recall metastatic generalement plus eleve et davantage de faux "
        "positifs. Seuil plus haut : moins de faux positifs et davantage de faux negatifs."
    )


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
    mode_display_name = (result.raw_metadata or {}).get("mode_display_name")
    if mode_display_name:
        st.caption(f"Mode LungColon utilise : {mode_display_name}.")
        if (result.raw_metadata or {}).get("mode_id") == "binary":
            st.info(
                "Cette sortie utilise le checkpoint binaire specialise, distinct du "
                "modele 5 classes."
            )
    for warning in result.warnings:
        st.caption(warning)


def render_integrated_flow(manager: ModelManager, project_id: str) -> None:
    project = get_project(project_id)
    try:
        adapter = manager.activate(project_id)
    except AdapterError as exc:
        st.error(str(exc))
        return

    checkpoint = adapter.checkpoint_status()
    st.subheader(f"Checkpoint {project.display_name}")
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

    mode_id = getattr(adapter, "current_mode", "single")
    mode_label = ""
    if isinstance(adapter, LungColonAdapter):
        mode_label = f" - {adapter.mode_metadata().display_name}"

    memory_status = st.empty()
    memory_status.caption(f"Modele en memoire : {'oui' if adapter.is_loaded else 'non'}")

    if st.button(
        f"Charger le modele {project.display_name}{mode_label}",
        disabled=checkpoint.status != "available" or adapter.is_loaded,
        type="secondary",
        key=f"load_{project_id}_{mode_id}",
    ):
        try:
            adapter.load()
        except (CheckpointMissingError, CheckpointIncompatibleError) as exc:
            st.error(str(exc))
        else:
            st.success(f"Modele {project.display_name}{mode_label} charge a la demande.")
            memory_status.caption("Modele en memoire : oui")
    uploaded_file = st.file_uploader(
        UPLOAD_LABELS.get(project_id, "Image a analyser"),
        type=["png", "jpg", "jpeg"],
        help="Le fichier reste en memoire pendant la session et n'est pas enregistre.",
        key=f"upload_{project_id}",
        on_change=clear_result_state,
    )
    st.caption(
        "Utilisez uniquement des images publiques de demonstration, sans donnees "
        "personnelles ni images medicales privees. L'image est traitee sur le serveur "
        "et conservee temporairement en memoire pour cette session, sans sauvegarde "
        "applicative ni reutilisation pour l'entrainement."
    )

    uploaded_image: Image.Image | None = None
    if uploaded_file is not None:
        try:
            uploaded_image = decode_uploaded_image(
                uploaded_file,
                allowed_formats=HUB_IMAGE_FORMATS,
            )
        except ImageUploadError as exc:
            st.error(str(exc))
            clear_result_state()
        else:
            with st.container(width=420, border=False):
                st.image(uploaded_image, caption="Image chargee", width="stretch")

    if st.button(
        "Analyser l'image",
        type="primary",
        disabled=uploaded_image is None,
        key=f"analyze_{project_id}_{mode_id}",
    ):
        if uploaded_image is None:
            st.error("Chargez une image PNG ou JPEG valide avant l'analyse.")
        else:
            clear_result_state()
            try:
                adapter.load()
                prediction = adapter.predict(uploaded_image)
            except (AdapterError, OSError, RuntimeError, ValueError) as exc:
                st.error(str(exc))
            else:
                st.session_state[SESSION_PREDICTION_KEY] = prediction
                st.session_state[SESSION_IMAGE_KEY] = uploaded_image.copy()
                st.success(f"Prediction {project.display_name} terminee.")
                memory_status.caption("Modele en memoire : oui")

    stored_prediction = st.session_state.get(SESSION_PREDICTION_KEY)
    prediction = (
        stored_prediction
        if isinstance(stored_prediction, PredictionResult)
        and stored_prediction.project_id == project_id
        else None
    )
    if prediction is not None:
        render_prediction(prediction)

    if project_id == "metastasis":
        render_threshold_exploration(prediction)

    if prediction is not None:
        show_gradcam = st.toggle(
            "Afficher Grad-CAM",
            value=False,
            key=f"gradcam_{project_id}",
        )
        if show_gradcam:
            explanation_image = st.session_state.get(SESSION_IMAGE_KEY)
            if not isinstance(explanation_image, Image.Image):
                st.info("Rechargez l'image pour generer Grad-CAM.")
            else:
                try:
                    explanation = st.session_state.get(SESSION_EXPLANATION_KEY)
                    if explanation is None:
                        explanation = adapter.explain(
                            explanation_image,
                            class_index=prediction.predicted_index,
                        )
                        st.session_state[SESSION_EXPLANATION_KEY] = explanation
                except ExplanationUnavailableError as exc:
                    st.info(str(exc))
                else:
                    with st.container(width=520, border=False):
                        st.image(
                            explanation.image,
                            caption=f"Grad-CAM - classe {explanation.class_name}",
                            width="stretch",
                        )
                    st.caption(explanation.message)


def render_app() -> None:
    st.set_page_config(page_title="R.C.C.I.A MultiCancer", layout="wide")
    st.title("R.C.C.I.A MultiCancer")
    st.caption("Hub de pipelines specialises - MultiCancer V1 portfolio-ready")
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
    if selected_project.integrated and selected_project.supports_prediction:
        try:
            manager.activate(selected_project_id)
        except AdapterError as exc:
            st.error(str(exc))
    else:
        manager.unload_current()

    selected_mode_metadata: AdapterModeMetadata | None = None
    with st.sidebar:
        if selected_project_id == "lung_colon" and manager.active_project_id == "lung_colon":
            try:
                selected_mode_metadata = configure_lung_colon_mode(manager)
            except AdapterError as exc:
                st.error(str(exc))
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
    render_project_metadata(selected_project_id, mode=selected_mode_metadata)
    if selected_project_id == "breast":
        render_breast_methodology()
    elif selected_project_id == "metastasis":
        render_metastasis_methodology()
    elif selected_project_id == "lung_colon" and selected_mode_metadata is not None:
        render_lung_colon_methodology(selected_mode_metadata)

    if selected_project.integrated and selected_project.supports_prediction:
        render_integrated_flow(manager, selected_project_id)
    else:
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
