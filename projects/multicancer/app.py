from __future__ import annotations

import sys
from dataclasses import asdict
from pathlib import Path

import streamlit as st


APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from multicancer.registry import PROJECTS  # noqa: E402
from multicancer.schemas import GLOBAL_DISCLAIMER  # noqa: E402


def project_rows() -> list[dict[str, str]]:
    """Build display-only rows without loading models, checkpoints, or datasets."""

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
                "Status": values["status"],
                "Grad-CAM": "yes" if values["supports_gradcam"] else "no",
            }
        )
    return rows


def render_app() -> None:
    st.set_page_config(page_title="R.C.C.I.A MultiCancer", layout="wide")
    st.title("R.C.C.I.A MultiCancer")
    st.caption("Specialized cancer imaging portfolio hub - V0 scope and architecture")
    st.error(
        "Demonstrateur educatif / portfolio uniquement. Aucune validation clinique, "
        "aucun diagnostic medical et aucune recommandation medicale."
    )

    st.subheader("Statut V0")
    st.info(
        "Cette page presente les quatre pipelines specialises. Les predictions, "
        "adaptateurs et chargements de modeles seront integres en V1."
    )

    st.subheader("Projets specialises")
    st.dataframe(project_rows(), hide_index=True, width="stretch")

    selected_project_id = st.selectbox(
        "Explorer les metadonnees d'un projet",
        options=[project.project_id for project in PROJECTS],
        format_func=lambda value: next(
            project.display_name for project in PROJECTS if project.project_id == value
        ),
    )
    selected = next(project for project in PROJECTS if project.project_id == selected_project_id)

    details = st.columns(3)
    details[0].metric("Projet", selected.display_name)
    details[1].metric("Resolution", selected.image_size)
    details[2].metric("Grad-CAM", "Disponible" if selected.supports_gradcam else "Indisponible")
    st.write(f"**Tache :** {selected.task}")
    st.write(f"**Classes :** {', '.join(selected.classes)}")
    st.write(f"**Metriques principales :** {', '.join(selected.primary_metrics)}")
    st.warning(selected.methodological_note)

    st.subheader("Pourquoi plusieurs modeles ?")
    st.write(
        "Les modalites, resolutions, classes et protocoles different. MultiCancer route "
        "explicitement vers un pipeline specialise au lieu de presenter un modele universel."
    )

    st.caption(GLOBAL_DISCLAIMER)


if __name__ == "__main__":
    render_app()
