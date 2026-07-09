from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
from PIL import Image, UnidentifiedImageError

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


def format_percent(value: float) -> str:
    return f"{value:.2%}"


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

    st.subheader("Limites")
    st.write(
        "Cette demo est un projet portfolio educatif. Les resultats dependront du dataset, "
        "du split, du preprocessing et du seuil de decision. Aucun resultat ne constitue "
        "une validation medicale."
    )


if __name__ == "__main__":
    render_app()
