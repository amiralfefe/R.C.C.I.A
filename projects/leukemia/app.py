from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image, UnidentifiedImageError

from rccia_leukemia.gradcam import (
    GradCAM,
    denormalize_image,
    get_gradcam_target_layer,
    image_to_tensor,
    overlay_cam,
)
from rccia_leukemia.error_analysis import load_error_analysis_artifacts
from rccia_leukemia.model import load_checkpoint, predict_image
from rccia_leukemia.utils import get_device


DEFAULT_CHECKPOINT = "outputs/best_model.pt"
DEFAULT_ERROR_ANALYSIS_DIR = Path("outputs/error_analysis")

V1_CLASS_RESULTS = pd.DataFrame(
    [
        {
            "classe": "leukemia_blast",
            "precision": 0.9419,
            "recall": 0.9359,
            "f1": 0.9389,
        },
        {
            "classe": "normal",
            "precision": 0.8643,
            "recall": 0.8762,
            "f1": 0.8702,
        },
    ]
)

V1_ACCURACY = 0.9169


st.set_page_config(page_title="Cancer Cell Vision", layout="wide")


@st.cache_resource
def load_model(checkpoint_path: str):
    device = get_device()
    model, checkpoint = load_checkpoint(Path(checkpoint_path), device=device)
    return model, checkpoint, device


def format_percent(value: float) -> str:
    return f"{value:.2%}"


def format_optional_percent(value: float | None) -> str:
    if value is None:
        return "n/a"
    return format_percent(float(value))


st.title("Cancer Cell Vision")
st.caption("Demo portfolio IA/data pour classifier des images microscopiques.")
st.warning(
    "Demonstrateur educatif uniquement. Cette application n'est pas un outil medical, "
    "ne fournit pas de diagnostic et ne doit pas orienter une decision de sante."
)

checkpoint_path = st.sidebar.text_input("Checkpoint", value=DEFAULT_CHECKPOINT)
show_gradcam = st.sidebar.toggle("Afficher Grad-CAM", value=True)
checkpoint = Path(checkpoint_path)

model = None
checkpoint_data = None
device = None
class_names: list[str] = []
image_size = 224
model_error = None

if checkpoint.exists():
    try:
        model, checkpoint_data, device = load_model(checkpoint_path)
        class_names = list(checkpoint_data["class_names"])
        image_size = int(checkpoint_data.get("image_size", 224))
    except Exception as exc:  # pragma: no cover - visible Streamlit state.
        model_error = exc

st.sidebar.subheader("Etat du modele")
if not checkpoint.exists():
    st.sidebar.error("Checkpoint absent")
    st.sidebar.caption(
        "Chemin attendu par defaut : `outputs/best_model.pt`. "
        "Lance l'entrainement avant la demo."
    )
elif model_error is not None:
    st.sidebar.error("Checkpoint trouve mais impossible a charger")
    st.sidebar.caption(str(model_error))
else:
    st.sidebar.success("Modele charge")
    st.sidebar.caption(f"Device : `{device}`")
    st.sidebar.caption(f"Classes : `{', '.join(class_names)}`")

st.sidebar.divider()
st.sidebar.subheader("Resultats V1")
st.sidebar.metric("Accuracy test", format_percent(V1_ACCURACY))
st.sidebar.dataframe(
    V1_CLASS_RESULTS,
    hide_index=True,
    use_container_width=True,
)

st.sidebar.divider()
st.sidebar.subheader("Limites")
st.sidebar.markdown(
    """
- Demonstrateur educatif.
- Pas un diagnostic medical.
- Dataset public Kaggle.
- Baseline courte entrainee sur CPU.
"""
)

left, right = st.columns([0.46, 0.54], gap="large")

image = None
with left:
    st.subheader("Image")
    uploaded_file = st.file_uploader("Image microscopique", type=["png", "jpg", "jpeg", "bmp"])
    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, use_container_width=True)
        except (UnidentifiedImageError, OSError):
            st.error(
                "Image invalide ou illisible. Utilise un fichier PNG, JPG, JPEG ou BMP "
                "exporte correctement."
            )
    else:
        st.info("Charge une image du test set ou une image microscopique compatible.")

with right:
    st.subheader("Prediction")
    if not checkpoint.exists():
        st.error(
            "Checkpoint absent : impossible de lancer la prediction. "
            "Entraine d'abord le modele ou indique un autre chemin de checkpoint."
        )
    elif model_error is not None:
        st.error(f"Checkpoint impossible a charger : {model_error}")
    elif uploaded_file is None:
        st.info("Ajoute une image pour lancer une prediction.")
    elif image is None:
        st.info("Charge une image valide pour lancer la prediction.")
    else:
        try:
            result = predict_image(
                image=image,
                model=model,
                class_names=class_names,
                image_size=image_size,
                device=device,
            )
        except Exception as exc:
            st.error(f"Prediction impossible : {exc}")
            st.stop()

        metric_left, metric_right = st.columns(2)
        metric_left.metric("Classe predite", result["class_name"])
        metric_right.metric("Confiance", format_percent(result["confidence"]))

        probabilities = pd.DataFrame(
            {
                "classe": class_names,
                "probabilite": result["probabilities"],
            }
        )
        probabilities["probabilite_pct"] = probabilities["probabilite"].map(format_percent)

        st.markdown("**Probabilites par classe**")
        st.dataframe(
            probabilities[["classe", "probabilite_pct"]],
            hide_index=True,
            use_container_width=True,
        )
        st.bar_chart(probabilities.set_index("classe")["probabilite"])

        if show_gradcam:
            st.markdown("**Grad-CAM**")
            gradcam = None
            try:
                tensor = image_to_tensor(image, image_size=image_size, device=device)
                gradcam = GradCAM(model, get_gradcam_target_layer(model))
                cam, _ = gradcam.generate(tensor)
                overlay = overlay_cam(denormalize_image(tensor), cam)
                st.image(overlay, caption="Grad-CAM", use_container_width=True)
            except Exception as exc:
                st.warning(
                    "Grad-CAM indisponible pour cette prediction. "
                    f"Cause technique : {exc}"
                )
            finally:
                if gradcam is not None:
                    gradcam.remove_hooks()
        else:
            st.info("Grad-CAM desactive dans la sidebar.")

st.divider()
st.subheader("Resultats experimentaux V1")

summary_left, summary_right = st.columns([0.35, 0.65], gap="large")
summary_left.metric("Accuracy test", format_percent(V1_ACCURACY))
summary_right.dataframe(
    V1_CLASS_RESULTS,
    hide_index=True,
    use_container_width=True,
)

with st.expander("Limites du demonstrateur"):
    st.write(
        "Les resultats dependent du dataset public, du split local, du preprocessing et "
        "du protocole d'entrainement. La baseline a ete entrainee 5 epochs sur CPU. "
        "Aucune validation clinique n'est realisee dans ce projet portfolio."
    )

st.divider()
st.subheader("Analyse des erreurs V2.2")
st.caption(
    "Analyse qualitative locale du test set : bonnes predictions, erreurs, "
    "false positives, false negatives et exemples exportes."
)

with st.expander("Definitions", expanded=False):
    st.write(
        "False positive : image `normal` predite `leukemia_blast`. "
        "False negative : image `leukemia_blast` predite `normal`. "
        "Ces categories servent a analyser le comportement du modele, pas a poser un diagnostic."
    )

analysis_state = load_error_analysis_artifacts(DEFAULT_ERROR_ANALYSIS_DIR)
if not analysis_state["available"]:
    st.info(
        "Analyse V2.2 non generee localement. Lance "
        "`.\\.venv\\Scripts\\python.exe scripts\\analyze_errors.py --data-dir data\\processed "
        "--checkpoint outputs\\best_model.pt --model resnet18 --output-dir outputs\\error_analysis "
        "--max-examples 12` pour creer les rapports."
    )
else:
    summary = analysis_state["summary"]
    metric_cols = st.columns(5)
    metric_cols[0].metric("Images test", int(summary["total_images"]))
    metric_cols[1].metric("Erreurs", int(summary["error_count"]))
    metric_cols[2].metric("False positives", int(summary["false_positive_count"]))
    metric_cols[3].metric("False negatives", int(summary["false_negative_count"]))
    metric_cols[4].metric("Accuracy", format_percent(float(summary["accuracy"])))

    confidence_frame = pd.DataFrame(
        [
            {
                "groupe": "predictions correctes",
                "confiance moyenne": format_optional_percent(
                    summary.get("average_confidence_correct")
                ),
            },
            {
                "groupe": "erreurs",
                "confiance moyenne": format_optional_percent(
                    summary.get("average_confidence_errors")
                ),
            },
        ]
    )
    st.dataframe(confidence_frame, hide_index=True, use_container_width=True)
    st.caption(f"Checkpoint analyse : `{summary.get('checkpoint_path', 'n/a')}`")

    examples = [
        path
        for path in analysis_state["examples"]
        if path.name.endswith("_annotated.png") or path.name.endswith("_gradcam.png")
    ]
    if not examples:
        st.info("Aucun exemple visuel exporte dans `outputs/error_analysis/examples`.")
    else:
        st.markdown("**Exemples exportes localement**")
        columns = st.columns(3)
        for index, image_path in enumerate(examples[:12]):
            with columns[index % 3]:
                st.image(str(image_path), caption=image_path.name, use_container_width=True)
