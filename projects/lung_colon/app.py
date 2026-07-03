from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image, UnidentifiedImageError

from rccia_lung_colon.error_analysis import load_summary
from rccia_lung_colon.gradcam import (
    GradCAM,
    denormalize_image,
    get_gradcam_target_layer,
    image_to_tensor,
    overlay_cam,
)
from rccia_lung_colon.model import load_checkpoint, predict_image
from rccia_lung_colon.utils import get_device


DEFAULT_CHECKPOINT = "outputs/best_model.pt"
ERROR_ANALYSIS_DIR = Path("outputs/error_analysis")
APP_DIR = Path(__file__).resolve().parent


def format_percent(value: float) -> str:
    return f"{value:.2%}"


def resolve_artifact_path(path: Path) -> Path:
    if path.is_absolute() or path.exists():
        return path
    app_relative_path = APP_DIR / path
    return app_relative_path if app_relative_path.exists() else path


def render_error_analysis_section() -> None:
    st.divider()
    st.subheader("Analyse des erreurs V2.1")

    error_analysis_dir = resolve_artifact_path(ERROR_ANALYSIS_DIR)
    summary_path = error_analysis_dir / "summary.json"
    summary = load_summary(summary_path)
    if summary is None:
        st.info(
            "Aucune analyse d'erreurs locale detectee. Lance la commande ci-dessous pour "
            "generer `outputs/error_analysis/summary.json` et les exemples visuels."
        )
        st.code(
            "..\\..\\.venv\\Scripts\\python.exe scripts\\analyze_errors.py "
            "--data-dir data\\processed "
            "--checkpoint outputs\\model_comparison\\efficientnet_b0\\best_model.pt "
            "--model efficientnet_b0 "
            "--output-dir outputs\\error_analysis "
            "--max-examples 15",
            language="powershell",
        )
        return

    metrics = st.columns(4)
    metrics[0].metric("Images test", int(summary.get("total_images", 0)))
    metrics[1].metric("Erreurs", int(summary.get("error_count", 0)))
    metrics[2].metric("Accuracy", format_percent(float(summary.get("accuracy", 0))))

    avg_correct = summary.get("average_confidence_correct")
    avg_errors = summary.get("average_confidence_errors")
    metrics[3].metric(
        "Confiance erreurs",
        "n/a" if avg_errors is None else format_percent(float(avg_errors)),
    )
    st.caption(
        "Confiance moyenne correctes : "
        f"{'n/a' if avg_correct is None else format_percent(float(avg_correct))}"
    )

    confusion_pairs = summary.get("confusion_pairs", {})
    if confusion_pairs:
        st.write("Confusions principales")
        st.dataframe(
            pd.DataFrame(
                [{"confusion": pair, "count": count} for pair, count in confusion_pairs.items()]
            ),
            hide_index=True,
            use_container_width=True,
        )
    else:
        st.success("Aucune confusion detectee dans le summary local.")

    error_types = summary.get("error_types", {})
    if error_types:
        st.write("Types d'erreurs")
        st.dataframe(
            pd.DataFrame(
                [{"type": error_type, "count": count} for error_type, count in error_types.items()]
            ),
            hide_index=True,
            use_container_width=True,
        )

    exported_examples = summary.get("exported_examples", [])
    visible_examples = [
        example
        for example in exported_examples
        if example.get("image_path") and Path(example["image_path"]).exists()
    ][:6]
    if visible_examples:
        st.write("Exemples exportes")
        for example in visible_examples:
            with st.expander(
                f"{example['kind']} - {example['true_label']} -> {example['predicted_label']}"
            ):
                cols = st.columns(2)
                cols[0].image(example["image_path"], caption="Image", use_container_width=True)
                gradcam_path = example.get("gradcam_path")
                if gradcam_path and Path(gradcam_path).exists():
                    cols[1].image(gradcam_path, caption="Grad-CAM", use_container_width=True)
                else:
                    cols[1].info("Grad-CAM indisponible pour cet exemple.")


st.set_page_config(page_title="Lung + Colon Vision", layout="wide")
st.title("Lung + Colon Vision")
st.caption("Demo portfolio IA/data pour classifier des images histopathologiques LC25000.")
st.warning(
    "Demonstrateur educatif uniquement : ce projet ne fournit pas de diagnostic medical "
    "et ne doit jamais orienter une decision de sante."
)

with st.sidebar:
    st.header("Modele")
    checkpoint_path = Path(st.text_input("Checkpoint", value=DEFAULT_CHECKPOINT))
    st.caption("Chemin relatif attendu depuis `projects/lung_colon`.")

device = get_device()
model = None
checkpoint = None
model_error = None

try:
    model, checkpoint = load_checkpoint(checkpoint_path, device=device)
except (FileNotFoundError, ValueError, RuntimeError) as exc:
    model_error = str(exc)

if model is None or checkpoint is None:
    st.error("Checkpoint indisponible.")
    st.info(
        "Entraine d'abord une baseline avec "
        "`..\\..\\.venv\\Scripts\\python.exe -m rccia_lung_colon.train "
        "--data-dir data\\processed --epochs 5 --batch-size 16 --output-dir outputs`."
    )
    if model_error:
        st.code(model_error)
else:
    class_names = checkpoint["class_names"]
    image_size = int(checkpoint.get("image_size", 224))
    model_name = checkpoint.get("model_name", "resnet18")

    st.success(f"Modele charge : `{model_name}` sur `{device}`")
    st.caption(f"Classes : {', '.join(class_names)}")

    uploaded_file = st.file_uploader(
        "Image histopathologique",
        type=["bmp", "jpeg", "jpg", "png", "tif", "tiff", "webp"],
    )

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file).convert("RGB")
        except (UnidentifiedImageError, OSError) as exc:
            st.error(f"Image invalide : {exc}")
        else:
            left, right = st.columns([1, 1])
            with left:
                st.image(image, caption="Image chargee", use_container_width=True)

            prediction = predict_image(
                image=image,
                model=model,
                class_names=class_names,
                image_size=image_size,
                device=device,
            )
            probabilities = pd.DataFrame(
                {
                    "classe": class_names,
                    "probabilite": prediction["probabilities"],
                }
            )

            with right:
                st.metric("Classe predite", str(prediction["class_name"]))
                st.metric("Confiance", format_percent(float(prediction["confidence"])))
                st.dataframe(probabilities, hide_index=True, use_container_width=True)
                st.bar_chart(probabilities.set_index("classe"))

            with st.expander("Grad-CAM", expanded=True):
                try:
                    target_layer = get_gradcam_target_layer(model)
                    tensor = image_to_tensor(image, image_size=image_size, device=device)
                    gradcam = GradCAM(model=model, target_layer=target_layer)
                    try:
                        cam, _target_index = gradcam.generate(tensor)
                    finally:
                        gradcam.remove_hooks()
                    overlay = overlay_cam(denormalize_image(tensor), cam)
                    st.image(overlay, caption="Visualisation Grad-CAM", use_container_width=True)
                except (LookupError, RuntimeError, ValueError) as exc:
                    st.info(f"Grad-CAM indisponible pour cette prediction : {exc}")

st.divider()
st.subheader("Limites")
st.write(
    "Cette demo est un projet portfolio educatif. Les resultats dependent du dataset "
    "prepare, du split, du preprocessing et du protocole d'entrainement. Aucun resultat ne "
    "constitue une validation medicale."
)

render_error_analysis_section()
