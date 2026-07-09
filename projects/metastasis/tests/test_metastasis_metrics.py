from __future__ import annotations

from rccia_metastasis.metrics import compute_binary_metrics, positive_class_index


def test_metastasis_metrics_include_roc_auc_and_pr_auc() -> None:
    class_names = ["metastatic", "non_metastatic"]
    y_true = [0, 0, 1, 1]
    y_pred = [0, 0, 1, 1]
    positive_scores = [0.95, 0.80, 0.25, 0.10]

    metrics = compute_binary_metrics(
        y_true=y_true,
        y_pred=y_pred,
        positive_scores=positive_scores,
        class_names=class_names,
        positive_class="metastatic",
    )

    assert positive_class_index(class_names) == 0
    assert metrics["accuracy"] == 1.0
    assert metrics["roc_auc"] == 1.0
    assert metrics["pr_auc"] == 1.0
    assert metrics["per_class"]["metastatic"]["recall"] == 1.0


def test_metastasis_metrics_handle_single_class_auc_gracefully() -> None:
    class_names = ["metastatic", "non_metastatic"]
    metrics = compute_binary_metrics(
        y_true=[0, 0],
        y_pred=[0, 0],
        positive_scores=[0.9, 0.8],
        class_names=class_names,
    )

    assert metrics["accuracy"] == 1.0
    assert metrics["roc_auc"] is None

