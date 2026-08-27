from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st


APP_DIR = Path(__file__).resolve().parent
PROJECTS_DIR = APP_DIR.parent
if str(PROJECTS_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECTS_DIR))

from rccia_common.image_uploads import (  # noqa: E402
    STANDALONE_IMAGE_FORMATS,
    ImageUploadError,
    decode_uploaded_image,
)
from rccia_breast.error_analysis import load_summary  # noqa: E402
from rccia_breast.gradcam import (  # noqa: E402
    GradCAM,
    denormalize_image,
    get_gradcam_target_layer,
    image_to_tensor,
    overlay_cam,
)
from rccia_breast.model import load_checkpoint, predict_image  # noqa: E402
from rccia_breast.utils import get_device  # noqa: E402


DEFAULT_CHECKPOINT = Path("outputs/best_model.pt")
DEFAULT_ERROR_ANALYSIS_SUMMARY = Path("outputs/error_analysis/summary.json")


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
        "L'application reste consultable sans modele entraine. Pour obtenir un checkpoint local, "
        "prepare le dataset BreakHis, lance le split, puis entraine une baseline."
    )
    st.code(
        "..\\..\\.venv\\Scripts\\python.exe -m rccia_breast.train "
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

    summary_path = resolve_artifact_path(DEFAULT_ERROR_ANALYSIS_SUMMARY)
    summary = load_error_analysis_summary(summary_path)
    if summary is None:
        st.info("Analyse locale non trouvee. Genere-la avec la commande ci-dessous.")
        st.code(
            "..\\..\\.venv\\Scripts\\python.exe scripts\\analyze_errors.py "
            "--data-dir data\\processed "
            "--checkpoint outputs\\model_comparison\\efficientnet_b0\\best_model.pt "
            "--model efficientnet_b0 "
            "--output-dir outputs\\error_analysis "
            "--metadata data\\raw\\metadata.csv "
            "--max-examples 15",
            language="powershell",
        )
        return

    st.caption(f"Resume charge : `{summary_path}`")
    cols = st.columns(5)
    cols[0].metric("Images test", int(summary.get("total_images", 0)))
    cols[1].metric("Erreurs", int(summary.get("error_count", 0)))
    cols[2].metric("False positives", int(summary.get("false_positive_count", 0)))
    cols[3].metric("False negatives", int(summary.get("false_negative_count", 0)))
    cols[4].metric("Accuracy", format_optional_percent(summary.get("accuracy")))

    confidence_cols = st.columns(2)
    confidence_cols[0].metric(
        "Confiance moyenne correctes",
        format_optional_percent(summary.get("average_confidence_correct")),
    )
    confidence_cols[1].metric(
        "Confiance moyenne erreurs",
        format_optional_percent(summary.get("average_confidence_errors")),
    )

    accuracy_by_magnification = summary.get("accuracy_by_magnification") or {}
    if accuracy_by_magnification:
        st.markdown("**Par grossissement**")
        rows = []
        errors_by_magnification = summary.get("errors_by_magnification") or {}
        for magnification, payload in accuracy_by_magnification.items():
            rows.append(
                {
                    "magnification": magnification,
                    "images": payload.get("total", 0),
                    "errors": errors_by_magnification.get(magnification, 0),
                    "accuracy": payload.get("accuracy", 0.0),
                }
            )
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    top_error_patients = summary.get("top_error_patients") or []
    if top_error_patients:
        st.markdown("**Patients avec erreurs**")
        st.dataframe(pd.DataFrame(top_error_patients), hide_index=True, use_container_width=True)

    exported_examples = summary.get("exported_examples") or []
    visible_examples = []
    for example in exported_examples:
        image_path = resolve_artifact_path(Path(example.get("image_path", "")))
        gradcam_path_value = example.get("gradcam_path")
        gradcam_path = (
            resolve_artifact_path(Path(gradcam_path_value)) if gradcam_path_value else None
        )
        if image_path.exists():
            visible_examples.append((example, image_path, gradcam_path))

    if visible_examples:
        st.markdown("**Exemples locaux**")
        for example, image_path, gradcam_path in visible_examples[:8]:
            left, right = st.columns(2)
            caption = (
                f"{example.get('kind')} | true={example.get('true_label')} | "
                f"pred={example.get('predicted_label')} | "
                f"conf={format_optional_percent(example.get('confidence'))}"
            )
            left.image(str(image_path), caption=caption, use_container_width=True)
            if gradcam_path and gradcam_path.exists():
                right.image(str(gradcam_path), caption="Grad-CAM", use_container_width=True)
            else:
                right.info("Grad-CAM indisponible pour cet exemple.")

    gradcam_errors = summary.get("gradcam_errors") or []
    if gradcam_errors:
        st.info("Certains Grad-CAM n'ont pas pu etre generes.")
        st.code("\n".join(str(error) for error in gradcam_errors[:5]))


def render_app() -> None:
    st.set_page_config(page_title="R.C.C.I.A Breast", layout="wide")
    st.title("R.C.C.I.A Breast")
    st.caption("Demo portfolio IA/data pour classification BreakHis benign vs malignant.")
    st.warning(
        "Demonstrateur educatif uniquement : ce projet ne fournit pas de diagnostic medical "
        "et ne doit jamais orienter une decision de sante."
    )

    with st.sidebar:
        st.header("Modele")
        checkpoint_path = Path(st.text_input("Checkpoint", value=str(DEFAULT_CHECKPOINT)))
        resolved_checkpoint_path = resolve_artifact_path(checkpoint_path)
        st.caption("Chemin relatif attendu depuis `projects/breast` ou depuis la racine.")
        st.caption(f"Chemin resolu : `{resolved_checkpoint_path}`")

    device = get_device()
    model, checkpoint, model_error = load_checkpoint_safely(resolved_checkpoint_path, device)

    if model is None or checkpoint is None:
        render_checkpoint_error(model_error)
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
                image = decode_uploaded_image(
                    uploaded_file,
                    allowed_formats=STANDALONE_IMAGE_FORMATS,
                )
            except ImageUploadError as exc:
                st.error(str(exc))
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
                        st.image(
                            overlay,
                            caption="Visualisation Grad-CAM",
                            use_container_width=True,
                        )
                    except (LookupError, RuntimeError, ValueError) as exc:
                        st.info(f"Grad-CAM indisponible pour cette prediction : {exc}")

    render_error_analysis_section()

    st.divider()
    st.subheader("Limites")
    st.write(
        "Cette demo est un projet portfolio educatif. Les resultats dependront du dataset "
        "prepare, du split, du preprocessing et du protocole d'entrainement. Aucun resultat "
        "ne constitue une validation medicale."
    )


if __name__ == "__main__":
    render_app()
