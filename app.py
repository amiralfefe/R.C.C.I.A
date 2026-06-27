from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image, UnidentifiedImageError

from cancer_cell_vision.gradcam import (
    GradCAM,
    denormalize_image,
    get_resnet_target_layer,
    image_to_tensor,
    overlay_cam,
)
from cancer_cell_vision.model import load_checkpoint, predict_image
from cancer_cell_vision.utils import get_device


st.set_page_config(page_title="Cancer Cell Vision", layout="wide")


@st.cache_resource
def load_model(checkpoint_path: str):
    device = get_device()
    model, checkpoint = load_checkpoint(Path(checkpoint_path), device=device)
    return model, checkpoint, device


st.title("Cancer Cell Vision")
st.caption("Demonstrateur IA educatif. Non destine au diagnostic medical.")

checkpoint_path = st.sidebar.text_input("Modele", value="outputs/best_model.pt")
show_gradcam = st.sidebar.toggle("Grad-CAM", value=True)
checkpoint = Path(checkpoint_path)

if not checkpoint.exists():
    st.sidebar.warning("Aucun modele entraine trouve.")

left, right = st.columns([0.48, 0.52], gap="large")

image = None
with left:
    uploaded_file = st.file_uploader("Image microscopique", type=["png", "jpg", "jpeg", "bmp"])
    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, use_container_width=True)
        except (UnidentifiedImageError, OSError):
            st.error("Image invalide ou illisible. Essaie un fichier PNG, JPG, JPEG ou BMP.")

with right:
    if not checkpoint.exists():
        st.warning(
            "L'app est prete, mais aucun checkpoint n'est disponible. "
            "Entraine d'abord un modele avec python -m cancer_cell_vision.train."
        )
    elif uploaded_file is None:
        st.info("Ajoute une image pour lancer une prediction.")
    elif image is None:
        st.info("Charge une image valide pour lancer la prediction.")
    else:
        try:
            model, checkpoint_data, device = load_model(checkpoint_path)
            class_names = checkpoint_data["class_names"]
            image_size = int(checkpoint_data.get("image_size", 224))
        except Exception as exc:
            st.error(f"Impossible de charger le modele : {exc}")
            st.stop()

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
        metric_right.metric("Confiance", f"{result['confidence']:.1%}")

        probabilities = pd.DataFrame(
            {
                "classe": class_names,
                "probabilite": result["probabilities"],
            }
        ).set_index("classe")
        st.bar_chart(probabilities)

        if show_gradcam:
            try:
                tensor = image_to_tensor(image, image_size=image_size, device=device)
                gradcam = GradCAM(model, get_resnet_target_layer(model))
                cam, _ = gradcam.generate(tensor)
                overlay = overlay_cam(denormalize_image(tensor), cam)
                st.image(overlay, caption="Grad-CAM", use_container_width=True)
            except Exception as exc:
                st.warning(f"Grad-CAM indisponible pour cette prediction : {exc}")
            finally:
                if "gradcam" in locals():
                    gradcam.remove_hooks()

with st.expander("Limites"):
    st.write(
        "Les resultats dependent du dataset, du protocole d'entrainement et de validation. "
        "Cette application n'est pas un dispositif medical."
    )
