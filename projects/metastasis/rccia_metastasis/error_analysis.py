"""Error and threshold-analysis helpers for Metastasis Vision."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    precision_recall_fscore_support,
    roc_auc_score,
)


ERROR_TYPES = ("correct", "false_positive", "false_negative")
NEGATIVE_CLASS = "non_metastatic"
POSITIVE_CLASS = "metastatic"


def classify_error(true_label: str, predicted_label: str) -> str:
    if true_label == predicted_label:
        return "correct"
    if true_label == NEGATIVE_CLASS and predicted_label == POSITIVE_CLASS:
        return "false_positive"
    if true_label == POSITIVE_CLASS and predicted_label == NEGATIVE_CLASS:
        return "false_negative"
    raise ValueError(f"Unsupported Metastasis labels: {true_label=} {predicted_label=}.")


def label_from_threshold(prob_metastatic: float, threshold: float) -> str:
    return POSITIVE_CLASS if float(prob_metastatic) >= float(threshold) else NEGATIVE_CLASS


def prediction_row(
    image_path: str,
    true_label: str,
    prob_non_metastatic: float,
    prob_metastatic: float,
    threshold: float = 0.5,
) -> dict[str, Any]:
    predicted_label = label_from_threshold(prob_metastatic, threshold)
    confidence = prob_metastatic if predicted_label == POSITIVE_CLASS else prob_non_metastatic
    return {
        "image_path": image_path,
        "true_label": true_label,
        "predicted_label": predicted_label,
        "prob_non_metastatic": float(prob_non_metastatic),
        "prob_metastatic": float(prob_metastatic),
        "confidence": float(confidence),
        "is_correct": true_label == predicted_label,
        "error_type": classify_error(true_label, predicted_label),
    }


def compact_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "image_path": row["image_path"],
        "true_label": row["true_label"],
        "predicted_label": row["predicted_label"],
        "prob_non_metastatic": row["prob_non_metastatic"],
        "prob_metastatic": row["prob_metastatic"],
        "confidence": row["confidence"],
        "error_type": row["error_type"],
    }


def average_confidence(rows: list[dict[str, Any]]) -> float | None:
    if not rows:
        return None
    return sum(float(row["confidence"]) for row in rows) / len(rows)


def y_true_binary(rows: list[dict[str, Any]]) -> list[int]:
    return [1 if row["true_label"] == POSITIVE_CLASS else 0 for row in rows]


def y_pred_binary(rows: list[dict[str, Any]], threshold: float = 0.5) -> list[int]:
    return [1 if float(row["prob_metastatic"]) >= threshold else 0 for row in rows]


def roc_auc(rows: list[dict[str, Any]]) -> float | None:
    y_true = y_true_binary(rows)
    if len(set(y_true)) < 2:
        return None
    return float(roc_auc_score(y_true, [float(row["prob_metastatic"]) for row in rows]))


def pr_auc(rows: list[dict[str, Any]]) -> float | None:
    y_true = y_true_binary(rows)
    if len(set(y_true)) < 2:
        return None
    return float(average_precision_score(y_true, [float(row["prob_metastatic"]) for row in rows]))


def threshold_metrics(rows: list[dict[str, Any]], threshold: float) -> dict[str, float | int]:
    true_labels = y_true_binary(rows)
    predicted_labels = y_pred_binary(rows, threshold=threshold)
    precision, recall, f1, _support = precision_recall_fscore_support(
        true_labels,
        predicted_labels,
        labels=[0, 1],
        zero_division=0,
    )

    false_positive_count = sum(
        1 for true, predicted in zip(true_labels, predicted_labels, strict=True) if true == 0 and predicted == 1
    )
    false_negative_count = sum(
        1 for true, predicted in zip(true_labels, predicted_labels, strict=True) if true == 1 and predicted == 0
    )
    predicted_positive_count = sum(predicted_labels)
    predicted_negative_count = len(predicted_labels) - predicted_positive_count

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(true_labels, predicted_labels)) if rows else 0.0,
        "precision_metastatic": float(precision[1]),
        "recall_metastatic": float(recall[1]),
        "f1_metastatic": float(f1[1]),
        "precision_non_metastatic": float(precision[0]),
        "recall_non_metastatic": float(recall[0]),
        "f1_non_metastatic": float(f1[0]),
        "false_positive_count": int(false_positive_count),
        "false_negative_count": int(false_negative_count),
        "predicted_positive_count": int(predicted_positive_count),
        "predicted_negative_count": int(predicted_negative_count),
    }


def compute_threshold_analysis(
    rows: list[dict[str, Any]],
    thresholds: list[float],
) -> list[dict[str, float | int]]:
    return [threshold_metrics(rows, threshold) for threshold in thresholds]


def summarize_predictions(
    rows: list[dict[str, Any]],
    thresholds: list[float],
    max_examples: int = 20,
) -> dict[str, Any]:
    correct_rows = [row for row in rows if row["is_correct"]]
    error_rows = [row for row in rows if not row["is_correct"]]
    error_types = Counter(row["error_type"] for row in error_rows)
    threshold_rows = compute_threshold_analysis(rows, thresholds)

    most_confident_errors = sorted(
        error_rows,
        key=lambda row: float(row["confidence"]),
        reverse=True,
    )[:max_examples]
    least_confident_correct = sorted(
        correct_rows,
        key=lambda row: float(row["confidence"]),
    )[:max_examples]

    threshold_050 = min(
        threshold_rows,
        key=lambda row: abs(float(row["threshold"]) - 0.5),
    ) if threshold_rows else threshold_metrics(rows, 0.5)

    total_images = len(rows)
    correct_count = len(correct_rows)
    error_count = len(error_rows)
    return {
        "total_images": total_images,
        "correct_count": correct_count,
        "error_count": error_count,
        "accuracy_at_0_50": correct_count / total_images if total_images else 0.0,
        "false_positive_count": error_types.get("false_positive", 0),
        "false_negative_count": error_types.get("false_negative", 0),
        "average_confidence_correct": average_confidence(correct_rows),
        "average_confidence_errors": average_confidence(error_rows),
        "most_confident_errors": [compact_row(row) for row in most_confident_errors],
        "least_confident_correct": [compact_row(row) for row in least_confident_correct],
        "roc_auc": roc_auc(rows),
        "pr_auc": pr_auc(rows),
        "threshold_default_note": (
            "The default 0.50 threshold predicts metastatic when prob_metastatic >= 0.50. "
            "This is an educational threshold analysis, not a medical operating point."
        ),
        "threshold_at_0_50": threshold_050,
    }


def save_rows_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        output_path.write_text("", encoding="utf-8")
        return
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_json(payload: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)
        file.write("\n")


def load_summary(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def plot_threshold_analysis(rows: list[dict[str, float | int]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    thresholds = [float(row["threshold"]) for row in rows]
    plt.figure(figsize=(6, 4))
    plt.plot(thresholds, [float(row["precision_metastatic"]) for row in rows], marker="o", label="precision metastatic")
    plt.plot(thresholds, [float(row["recall_metastatic"]) for row in rows], marker="o", label="recall metastatic")
    plt.plot(thresholds, [float(row["f1_metastatic"]) for row in rows], marker="o", label="F1 metastatic")
    plt.xlabel("Threshold metastatic")
    plt.ylabel("Score")
    plt.ylim(0, 1.05)
    plt.title("Threshold analysis")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
