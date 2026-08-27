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
from rccia_lung_colon.binary import BINARY_CLASSES, SOURCE_CLASSES_BY_BINARY  # noqa: E402
from rccia_lung_colon.error_analysis import load_summary  # noqa: E402
from rccia_lung_colon.gradcam import (  # noqa: E402
    GradCAM,
    denormalize_image,
    get_gradcam_target_layer,
    image_to_tensor,
    overlay_cam,
)
from rccia_lung_colon.model import load_checkpoint, predict_image  # noqa: E402
from rccia_lung_colon.utils import get_device  # noqa: E402


MULTICLASS_MODE = "Mode 5 classes"
BINARY_MODE = "Mode binaire benign/malignant"

CHECKPOINTS_BY_MODE = {
    MULTICLASS_MODE: Path("outputs/best_model.pt"),
    BINARY_MODE: Path("outputs/binary_resnet18/best_model.pt"),
}

ERROR_ANALYSIS_DIR = Path("outputs/error_analysis")


def format_percent(value: float) -> str:
    return f"{value:.2%}"


def resolve_artifact_path(path: Path) -> Path:
    if path.is_absolute() or path.exists():
        return path
    app_relative_path = APP_DIR / path
    return app_relative_path if app_relative_path.exists() else path


def get_default_checkpoint(mode: str) -> Path:
    return CHECKPOINTS_BY_MODE.get(mode, CHECKPOINTS_BY_MODE[MULTICLASS_MODE])


def load_checkpoint_safely(checkpoint_path: Path, device: Any) -> tuple[Any, dict | None, str | None]:
    try:
        model, checkpoint = load_checkpoint(checkpoint_path, device=device)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        return None, None, str(exc)
    return model, checkpoint, None


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


def render_binary_mapping() -> None:
    mapping_rows = [
        {"mode binaire": binary_class, "classes sources": ", ".join(source_classes)}
        for binary_class, source_classes in SOURCE_CLASSES_BY_BINARY.items()
    ]
    st.dataframe(pd.DataFrame(mapping_rows), hide_index=True, use_container_width=True)


def render_checkpoint_error(selected_mode: str, model_error: str | None) -> None:
    st.error("Checkpoint indisponible.")
    if selected_mode == BINARY_MODE:
        st.info(
            "Le mode binaire reste disponible dans l'interface, mais il faut d'abord "
            "entrainer un checkpoint local `outputs/binary_resnet18/best_model.pt`."
        )
        st.code(
            "..\\..\\.venv\\Scripts\\python.exe -m rccia_lung_colon.train "
            "--data-dir data\\binary_processed --model resnet18 --epochs 3 "
            "--batch-size 16 --image-size 224 --output-dir outputs\\binary_resnet18",
            language="powershell",
        )
    else:
        st.info(
            "Entraine d'abord une baseline avec "
            "`..\\..\\.venv\\Scripts\\python.exe -m rccia_lung_colon.train "
            "--data-dir data\\processed --epochs 3 --batch-size 16 --output-dir outputs`."
        )
    if model_error:
        st.code(model_error)


def render_app() -> None:
    st.set_page_config(page_title="Lung + Colon Vision", layout="wide")
    st.title("Lung + Colon Vision")
    st.caption("Demo portfolio IA/data pour classifier des images histopathologiques LC25000.")
    st.warning(
        "Demonstrateur educatif uniquement : ce projet ne fournit pas de diagnostic medical "
        "et ne doit jamais orienter une decision de sante."
    )

    with st.sidebar:
        st.header("Modele")
        selected_mode = st.radio(
            "Mode de classification",
            options=[MULTICLASS_MODE, BINARY_MODE],
            index=0,
        )
        default_checkpoint = get_default_checkpoint(selected_mode)
        checkpoint_path = Path(
            st.text_input(
                "Checkpoint",
                value=str(default_checkpoint),
                key=f"checkpoint_{selected_mode}",
            )
        )
        resolved_checkpoint_path = resolve_artifact_path(checkpoint_path)
        st.caption("Chemin relatif attendu depuis `projects/lung_colon` ou depuis la racine.")
        st.caption(f"Chemin resolu : `{resolved_checkpoint_path}`")

    if selected_mode == BINARY_MODE:
        st.subheader("Mapping binaire")
        render_binary_mapping()

    device = get_device()
    model, checkpoint, model_error = load_checkpoint_safely(resolved_checkpoint_path, device)

    if model is None or checkpoint is None:
        render_checkpoint_error(selected_mode, model_error)
    else:
        class_names = checkpoint["class_names"]
        image_size = int(checkpoint.get("image_size", 224))
        model_name = checkpoint.get("model_name", "resnet18")

        st.success(f"Modele charge : `{model_name}` sur `{device}`")
        st.caption(f"Classes : {', '.join(class_names)}")

        if selected_mode == BINARY_MODE and tuple(class_names) != BINARY_CLASSES:
            st.warning(
                "Le mode binaire est selectionne, mais le checkpoint charge ne contient pas "
                "les classes attendues `benign` et `malignant`."
            )
        if selected_mode == MULTICLASS_MODE and len(class_names) == 2:
            st.warning(
                "Le mode 5 classes est selectionne, mais le checkpoint charge semble binaire."
            )

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

    st.divider()
    st.subheader("Limites")
    st.write(
        "Cette demo est un projet portfolio educatif. Les resultats dependent du dataset "
        "prepare, du split, du preprocessing et du protocole d'entrainement. Aucun resultat ne "
        "constitue une validation medicale."
    )

    if selected_mode == MULTICLASS_MODE:
        render_error_analysis_section()


if __name__ == "__main__":
    render_app()
