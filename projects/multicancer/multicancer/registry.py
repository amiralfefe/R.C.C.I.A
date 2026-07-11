"""Checkpoint-independent registry for the four specialized R.C.C.I.A projects."""

from __future__ import annotations

from .schemas import ProjectMetadata


PROJECTS: tuple[ProjectMetadata, ...] = (
    ProjectMetadata(
        project_id="leukemia",
        display_name="Leukemia",
        modality="Microscopic blood cell images",
        task="Binary classification of normal cells and leukemia blasts",
        dataset="Kaggle andrewmvd/leukemia-classification",
        classes=("normal", "leukemia_blast"),
        image_size="224x224",
        status="complete",
        supports_gradcam=True,
        primary_metrics=("accuracy", "precision", "recall", "f1"),
        methodological_note=(
            "Public educational dataset with class imbalance and no documented "
            "patient-aware split."
        ),
    ),
    ProjectMetadata(
        project_id="lung_colon",
        display_name="Lung + Colon",
        modality="Lung and colon histopathology images",
        task="Five-class tissue classification with an additional binary mode",
        dataset="LC25000",
        classes=(
            "colon_adenocarcinoma",
            "colon_benign",
            "lung_adenocarcinoma",
            "lung_benign",
            "lung_squamous_cell_carcinoma",
        ),
        image_size="224x224",
        status="complete",
        supports_gradcam=True,
        primary_metrics=("accuracy", "macro_f1", "per_class_f1", "train_time"),
        methodological_note=(
            "Very high scores on a relatively easy public benchmark must be interpreted "
            "with caution."
        ),
    ),
    ProjectMetadata(
        project_id="breast",
        display_name="Breast",
        modality="Breast histopathology at 40X, 100X, 200X, and 400X",
        task="Binary classification of benign and malignant tissue",
        dataset="BreakHis",
        classes=("benign", "malignant"),
        image_size="224x224",
        status="complete",
        supports_gradcam=True,
        primary_metrics=("accuracy", "macro_f1", "recall_malignant", "per_magnification"),
        methodological_note=(
            "Patient-aware split prevents patient overlap; results still depend on a "
            "single public dataset and local protocol."
        ),
    ),
    ProjectMetadata(
        project_id="metastasis",
        display_name="Metastasis",
        modality="Histopathology patches",
        task="Binary classification of non-metastatic and metastatic patches",
        dataset="Kaggle tyson04/pcam-validate (PCam subset)",
        classes=("non_metastatic", "metastatic"),
        image_size="96x96",
        status="complete",
        supports_gradcam=True,
        primary_metrics=("accuracy", "f1", "roc_auc", "pr_auc", "threshold_analysis"),
        methodological_note=(
            "Balanced 5,000-patch subset; threshold trade-offs are educational and not "
            "medical decision thresholds."
        ),
    ),
)

PROJECT_REGISTRY: dict[str, ProjectMetadata] = {
    project.project_id: project for project in PROJECTS
}


def get_project(project_id: str) -> ProjectMetadata:
    """Return one registered project or raise a clear lookup error."""

    try:
        return PROJECT_REGISTRY[project_id]
    except KeyError as exc:
        available = ", ".join(PROJECT_REGISTRY)
        raise KeyError(f"Unknown project '{project_id}'. Available projects: {available}.") from exc
