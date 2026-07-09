"""Metrics and report helpers for Metastasis Vision."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)


def positive_class_index(class_names: list[str], positive_class: str = "metastatic") -> int:
    if positive_class in class_names:
        return class_names.index(positive_class)
    return len(class_names) - 1


def safe_float(value: float | np.floating | None) -> float | None:
    if value is None:
        return None
    if np.isnan(value):
        return None
    return float(value)


def compute_binary_metrics(
    y_true: list[int],
    y_pred: list[int],
    positive_scores: list[float],
    class_names: list[str],
    positive_class: str = "metastatic",
) -> dict[str, Any]:
    positive_index = positive_class_index(class_names, positive_class=positive_class)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        zero_division=0,
    )
    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    binary_true = [1 if label == positive_index else 0 for label in y_true]

    roc_auc: float | None
    pr_auc: float | None
    if len(set(binary_true)) < 2:
        roc_auc = None
        pr_auc = None
    else:
        roc_auc = float(roc_auc_score(binary_true, positive_scores))
        pr_auc = float(average_precision_score(binary_true, positive_scores))

    per_class = {
        class_name: {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
        }
        for index, class_name in enumerate(class_names)
    }

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "roc_auc": safe_float(roc_auc),
        "pr_auc": safe_float(pr_auc),
        "positive_class": positive_class,
        "positive_class_index": positive_index,
        "classification_report": report,
        "per_class": per_class,
        "confusion_matrix": confusion_matrix(
            y_true,
            y_pred,
            labels=list(range(len(class_names))),
        ).tolist(),
    }


def save_json(payload: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)
        file.write("\n")


def save_classification_report_csv(report: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for label, values in report.items():
        if isinstance(values, dict):
            rows.append({"label": label, **values})
        else:
            rows.append({"label": label, "value": values})

    fieldnames = sorted({key for row in rows for key in row})
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_confusion_matrix(
    y_true: list[int],
    y_pred: list[int],
    class_names: list[str],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=class_names)
    _, ax = plt.subplots(figsize=(5, 4))
    display.plot(ax=ax, cmap="Blues", values_format="d", colorbar=False)
    ax.set_title("Confusion matrix")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_roc_curve(
    y_true: list[int],
    positive_scores: list[float],
    positive_index: int,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    binary_true = np.array([1 if label == positive_index else 0 for label in y_true])
    if len(np.unique(binary_true)) < 2:
        return
    fpr, tpr, _thresholds = roc_curve(binary_true, positive_scores)
    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, label="ROC")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("ROC curve")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_precision_recall_curve(
    y_true: list[int],
    positive_scores: list[float],
    positive_index: int,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    binary_true = np.array([1 if label == positive_index else 0 for label in y_true])
    if len(np.unique(binary_true)) < 2:
        return
    precision, recall, _thresholds = precision_recall_curve(binary_true, positive_scores)
    plt.figure(figsize=(5, 4))
    plt.plot(recall, precision, label="PR")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision/Recall curve")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
