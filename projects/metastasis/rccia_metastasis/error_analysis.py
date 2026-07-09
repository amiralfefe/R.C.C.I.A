"""Small error-analysis helpers for Metastasis Vision."""

from __future__ import annotations

from collections import Counter
from typing import Any


ERROR_TYPES = ("correct", "false_positive", "false_negative")


def classify_error(true_label: str, predicted_label: str) -> str:
    if true_label == predicted_label:
        return "correct"
    if true_label == "non_metastatic" and predicted_label == "metastatic":
        return "false_positive"
    if true_label == "metastatic" and predicted_label == "non_metastatic":
        return "false_negative"
    return "false_positive"


def prediction_row(
    image_path: str,
    true_label: str,
    predicted_label: str,
    confidence: float,
    prob_metastatic: float,
) -> dict[str, Any]:
    return {
        "image_path": image_path,
        "true_label": true_label,
        "predicted_label": predicted_label,
        "confidence": float(confidence),
        "prob_metastatic": float(prob_metastatic),
        "is_correct": true_label == predicted_label,
        "error_type": classify_error(true_label, predicted_label),
    }


def summarize_predictions(rows: list[dict[str, Any]]) -> dict[str, Any]:
    correct_rows = [row for row in rows if row["is_correct"]]
    error_rows = [row for row in rows if not row["is_correct"]]
    error_types = Counter(row["error_type"] for row in error_rows)

    total = len(rows)
    return {
        "total_images": total,
        "correct_count": len(correct_rows),
        "error_count": len(error_rows),
        "accuracy": len(correct_rows) / total if total else 0.0,
        "false_positive_count": error_types.get("false_positive", 0),
        "false_negative_count": error_types.get("false_negative", 0),
        "errors_by_true_class": dict(Counter(row["true_label"] for row in error_rows)),
        "errors_by_predicted_class": dict(Counter(row["predicted_label"] for row in error_rows)),
    }
