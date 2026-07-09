from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
from PIL import Image, UnidentifiedImageError

from rccia_metastasis.error_analysis import load_summary
from rccia_metastasis.gradcam import (
    GradCAM,
    denormalize_image,
    get_gradcam_target_layer,
    image_to_tensor,
    overlay_cam,
)
from rccia_metastasis.model import load_checkpoint, predict_image
from rccia_metastasis.utils import get_device


APP_DIR = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = Path("outputs/best_model.pt")
DEFAULT_ERROR_ANALYSIS_DIR = Path("outputs/error_analysis")


def format_percent(value: float) -> str:
    return f"{value:.2%}"


def format_optional_percent(value: float | None) -> str:
    if value is None:
        return "n/a"
    return format_percent(float(value))


def resolve_artifact_path(path: Path) -> Path:
    if path.is_absolute() or path.exists():
        return path
    app_relative_path = APP_DIR / path
    return app_relative_path if app_relative_path.exists() else path


def load_checkpoint_safely(checkpoint_path: Path, device: Any) -> tuple[Any, dict | None, str | None]:
    try:
        model, checkpoint = load_checkpoint(checkpoint_path, device=device)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        return None, None, str(exc)
    return model, checkpoint, None


def render_checkpoint_error(model_error: str | None) -> None:
    st.error("Checkpoint indisponible.")
    st.info(
        "L'application reste consultable sans modele entraine. Pour obtenir un checkpoint "
        "local, prepare un dataset type PCam, lance le split, puis entraine une baseline."
    )
    st.code(
        "..\\..\\.venv\\Scripts\\python.exe -m rccia_metastasis.train "
        "--data-dir data\\processed --model resnet18 --epochs 3 "
        "--batch-size 16 --image-size 224 --output-dir outputs",
        language="powershell",
    )
    if model_error:
        st.code(model_error)


def load_error_analysis_summary(summary_path: Path) -> dict[str, Any] | None:
    try:
        return load_summary(summary_path)
    except (OSError, ValueError):
        return None


def render_error_analysis_section() -> None:
    st.divider()
    st.subheader("Analyse des erreurs V2.1")

    error_analysis_dir = resolve_artifact_path(DEFAULT_ERROR_ANALYSIS_DIR)
    summary_path = error_analysis_dir / "error_summary.json"
    threshold_path = error_analysis_dir / "threshold_analysis.csv"
    threshold_plot_path = error_analysis_dir / "threshold_analysis.png"
    examples_dir = error_analysis_dir / "examples"

    summary = load_error_analysis_summary(summary_path)
    if summary is None:
        st.info("Analyse V2.1 locale non trouvee. Genere-la avec la commande ci-dessous.")
        st.code(
            "..\\..\\.venv\\Scripts\\python.exe scripts\\analyze_errors.py "
            "--data-dir data\\processed "
            "--checkpoint outputs\\model_comparison\\efficientnet_b0\\best_model.pt "
            "--model efficientnet_b0 "
            "--image-size 96 "
            "--output-dir outputs\\error_analysis "
            "--thresholds 0.30 0.40 0.50 0.60 0.70 "
            "--max-examples 20",
            language="powershell",
        )
        return

    st.caption(f"Resume charge : `{summary_path}`")
    metrics = st.columns(5)
    metrics[0].metric("Images test", int(summary.get("total_images", 0)))
    metrics[1].metric("Erreurs", int(summary.get("error_count", 0)))
    metrics[2].metric("False positives", int(summary.get("false_positive_count", 0)))
    metrics[3].metric("False negatives", int(summary.get("false_negative_count", 0)))
    metrics[4].metric("Accuracy @0.50", format_optional_percent(summary.get("accuracy_at_0_50")))

    confidence_cols = st.columns(2)
    confidence_cols[0].metric(
        "Confiance moyenne correctes",
        format_optional_percent(summary.get("average_confidence_correct")),
    )
    confidence_cols[1].metric(
        "Confiance moyenne erreurs",
        format_optional_percent(summary.get("average_confidence_errors")),
    )
    st.caption(
        "ROC-AUC : "
        f"{summary.get('roc_auc', 'n/a')} | PR-AUC : {summary.get('pr_auc', 'n/a')}"
    )

    if threshold_path.exists():
        st.markdown("**Analyse des seuils**")
        threshold_rows = pd.read_csv(threshold_path)
        st.dataframe(threshold_rows, hide_index=True, use_container_width=True)

    if threshold_plot_path.exists():
        st.image(str(threshold_plot_path), caption="Precision / recall / F1 selon le seuil", use_container_width=True)

    exported_examples = summary.get("exported_examples") or []
    visible_examples = []
    for example in exported_examples:
        image_path_value = example.get("image_path")
        if not image_path_value:
            continue
        image_path = resolve_artifact_path(Path(image_path_value))
        if not image_path.exists() and examples_dir.exists():
            image_path = examples_dir / Path(image_path_value).name
        if image_path.exists():
            visible_examples.append((example, image_path))

    if visible_examples:
        st.markdown("**Exemples exportes**")
        for example, image_path in visible_examples[:8]:
            label = (
                f"{example.get('kind', 'example')} - "
                f"{example.get('true_label')} -> {example.get('predicted_label')}"
            )
            with st.expander(label):
                cols = st.columns(2)
                cols[0].image(str(image_path), caption="Image", use_container_width=True)
                gradcam_path_value = example.get("gradcam_path")
                gradcam_path = (
                    resolve_artifact_path(Path(gradcam_path_value))
                    if gradcam_path_value
                    else None
                )
                if gradcam_path and not gradcam_path.exists() and examples_dir.exists():
                    gradcam_path = examples_dir / Path(gradcam_path_value).name
                if gradcam_path and gradcam_path.exists():
                    cols[1].image(str(gradcam_path), caption="Grad-CAM", use_container_width=True)
                else:
                    cols[1].info("Grad-CAM indisponible pour cet exemple.")


def render_app() -> None:
    st.set_page_config(page_title="R.C.C.I.A Metastasis", layout="wide")
    st.title("R.C.C.I.A Metastasis")
    st.caption("Demo portfolio IA/data pour classification de patches metastatic vs non_metastatic.")
    st.warning(
        "Demonstrateur educatif uniquement : ce projet ne fournit pas de diagnostic medical "
        "et ne doit jamais orienter une decision de sante."
    )

    with st.sidebar:
        st.header("Modele")
        checkpoint_path = Path(st.text_input("Checkpoint", value=str(DEFAULT_CHECKPOINT)))
        resolved_checkpoint_path = resolve_artifact_path(checkpoint_path)
        st.caption("Chemin relatif attendu depuis `projects/metastasis` ou depuis la racine.")
        st.caption(f"Chemin resolu : `{resolved_checkpoint_path}`")
        st.header("Seuil")
        threshold = st.slider("Seuil metastatic", 0.0, 1.0, 0.5, 0.01)
        st.caption(
            "V1 utilise surtout l'argmax. Une prochaine version pourra comparer plusieurs seuils."
        )

    device = get_device()
    model, checkpoint, model_error = load_checkpoint_safely(resolved_checkpoint_path, device)

    if model is None or checkpoint is None:
        render_checkpoint_error(model_error)
    else:
        class_names = checkpoint["class_names"]
        image_size = int(checkpoint.get("image_size", 224))
        model_name = checkpoint.get("model_name", "resnet18")
        metastatic_index = class_names.index("metastatic") if "metastatic" in class_names else None

        st.success(f"Modele charge : `{model_name}` sur `{device}`")
        st.caption(f"Classes : {', '.join(class_names)}")

        uploaded_file = st.file_uploader(
            "Patch histopathologique",
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
                    if metastatic_index is not None:
                        prob_metastatic = float(prediction["probabilities"][metastatic_index])
                        st.metric("Probabilite metastatic", format_percent(prob_metastatic))
                        st.caption(
                            "Decision seuil 0.5 ajustable : "
                            f"{'metastatic' if prob_metastatic >= threshold else 'non_metastatic'}"
                        )
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
                        st.image(
                            overlay,
                            caption="Visualisation Grad-CAM",
                            use_container_width=True,
                        )
                    except (LookupError, RuntimeError, ValueError) as exc:
                        st.info(f"Grad-CAM indisponible pour cette prediction : {exc}")

    st.divider()
    st.subheader("Metriques V1")
    st.write(
        "Ce projet est oriente benchmark : accuracy, precision, recall, F1, ROC-AUC, "
        "PR-AUC, matrice de confusion, courbe ROC et courbe precision/recall. "
        "L'analyse de seuil est prevue pour une phase suivante."
    )

    render_error_analysis_section()

    st.subheader("Limites")
    st.write(
        "Cette demo est un projet portfolio educatif. Les resultats dependront du dataset, "
        "du split, du preprocessing et du seuil de decision. Aucun resultat ne constitue "
        "une validation medicale."
    )


if __name__ == "__main__":
    render_app()
