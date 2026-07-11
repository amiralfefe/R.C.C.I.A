"""Checkpoint-independent registry for the four specialized R.C.C.I.A projects."""

from __future__ import annotations

from .schemas import GLOBAL_DISCLAIMER, ProjectMetadata


PROJECTS: tuple[ProjectMetadata, ...] = (
    ProjectMetadata(
        project_id="leukemia",
        display_name="Leukemia",
        modality="Microscopic blood cell images",
        task="Binary classification of normal cells and leukemia blasts",
        dataset="Kaggle andrewmvd/leukemia-classification",
        classes=("normal", "leukemia_blast"),
        image_size="224x224",
        model_name="resnet18",
        status="complete",
        adapter_status="integrated",
        integrated=True,
        supports_prediction=True,
        supports_gradcam=True,
        primary_metrics=(
            "accuracy=0.9169",
            "f1_normal=0.8702",
            "f1_leukemia_blast=0.9389",
        ),
        limitations=(
            "Public educational dataset without external clinical validation.",
            "Class imbalance and no documented patient-aware split.",
            "Grad-CAM is exploratory and not medical evidence.",
        ),
        methodological_note=(
            "Public educational dataset with class imbalance and no documented "
            "patient-aware split."
        ),
        disclaimer=GLOBAL_DISCLAIMER,
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
        model_name="efficientnet_b0",
        status="complete",
        adapter_status="planned",
        integrated=False,
        supports_prediction=False,
        supports_gradcam=True,
        primary_metrics=("accuracy=0.9992", "macro_f1=0.9992", "train_time"),
        limitations=(
            "Very high scores on a relatively easy public benchmark.",
            "No external or multi-center clinical validation.",
        ),
        methodological_note=(
            "Very high scores on a relatively easy public benchmark must be interpreted "
            "with caution."
        ),
        disclaimer=GLOBAL_DISCLAIMER,
    ),
    ProjectMetadata(
        project_id="breast",
        display_name="Breast",
        modality="Breast histopathology at 40X, 100X, 200X, and 400X",
        task="Binary classification of benign and malignant tissue",
        dataset="BreakHis",
        classes=("benign", "malignant"),
        image_size="224x224",
        model_name="efficientnet_b0",
        status="complete",
        adapter_status="planned",
        integrated=False,
        supports_prediction=False,
        supports_gradcam=True,
        primary_metrics=(
            "accuracy=0.9122",
            "macro_f1=0.9004",
            "recall_malignant=0.9568",
        ),
        limitations=(
            "Results depend on one public dataset and one local patient-aware split.",
            "Important patient and magnification variability remains.",
        ),
        methodological_note=(
            "Patient-aware split prevents patient overlap; results still depend on a "
            "single public dataset and local protocol."
        ),
        disclaimer=GLOBAL_DISCLAIMER,
    ),
    ProjectMetadata(
        project_id="metastasis",
        display_name="Metastasis",
        modality="Histopathology patches",
        task="Binary classification of non-metastatic and metastatic patches",
        dataset="Kaggle tyson04/pcam-validate (PCam subset)",
        classes=("non_metastatic", "metastatic"),
        image_size="96x96",
        model_name="efficientnet_b0",
        status="complete",
        adapter_status="planned",
        integrated=False,
        supports_prediction=False,
        supports_gradcam=True,
        primary_metrics=(
            "accuracy=0.9320",
            "roc_auc=0.9762",
            "pr_auc=0.9780",
        ),
        limitations=(
            "Benchmark uses a balanced 5,000-patch subset rather than the full corpus.",
            "Threshold trade-offs are educational, not medical decision thresholds.",
        ),
        methodological_note=(
            "Balanced 5,000-patch subset; threshold trade-offs are educational and not "
            "medical decision thresholds."
        ),
        disclaimer=GLOBAL_DISCLAIMER,
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
