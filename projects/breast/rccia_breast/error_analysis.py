"""Error analysis helpers for binary Breast Vision."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ERROR_TYPES = ("correct", "false_positive", "false_negative")


def classify_error(true_label: str, predicted_label: str) -> str:
    """Return the binary error category used in Breast V2.1."""
    if true_label == predicted_label:
        return "correct"
    if true_label == "benign" and predicted_label == "malignant":
        return "false_positive"
    if true_label == "malignant" and predicted_label == "benign":
        return "false_negative"
    return "false_positive"


def prediction_row(
    image_path: str,
    true_label: str,
    predicted_label: str,
    confidence: float,
    probabilities: list[float],
    class_names: list[str],
    patient_id: str | None = None,
    magnification: str | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "image_path": image_path,
        "true_label": true_label,
        "predicted_label": predicted_label,
        "confidence": float(confidence),
        "is_correct": true_label == predicted_label,
        "error_type": classify_error(true_label, predicted_label),
        "patient_id": patient_id or "",
        "magnification": magnification or "",
    }
    for class_name, probability in zip(class_names, probabilities, strict=True):
        row[f"prob_{class_name}"] = float(probability)
    return row


def compact_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "image_path": row["image_path"],
        "true_label": row["true_label"],
        "predicted_label": row["predicted_label"],
        "confidence": row["confidence"],
        "error_type": row["error_type"],
        "patient_id": row.get("patient_id", ""),
        "magnification": row.get("magnification", ""),
    }


def average_confidence(rows: list[dict[str, Any]]) -> float | None:
    if not rows:
        return None
    return sum(float(row["confidence"]) for row in rows) / len(rows)


def _sorted_counter(values: list[str]) -> dict[str, int]:
    counter = Counter(value or "unknown" for value in values)
    return dict(sorted(counter.items()))


def _accuracy_by_field(rows: list[dict[str, Any]], field_name: str) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get(field_name) or "unknown")].append(row)

    summary: dict[str, dict[str, Any]] = {}
    for group_name, group_rows in sorted(groups.items()):
        total = len(group_rows)
        correct = sum(1 for row in group_rows if row["is_correct"])
        error_count = total - correct
        summary[group_name] = {
            "total": total,
            "correct": correct,
            "errors": error_count,
            "accuracy": correct / total if total else 0.0,
        }
    return summary


def _patient_error_summary(rows: list[dict[str, Any]], max_examples: int) -> list[dict[str, Any]]:
    rows_with_patient = [row for row in rows if row.get("patient_id")]
    grouped = _accuracy_by_field(rows_with_patient, "patient_id")
    false_positive_counts = Counter(
        row["patient_id"]
        for row in rows_with_patient
        if row["error_type"] == "false_positive"
    )
    false_negative_counts = Counter(
        row["patient_id"]
        for row in rows_with_patient
        if row["error_type"] == "false_negative"
    )

    patient_rows = [
        {
            "patient_id": patient_id,
            "total": payload["total"],
            "correct": payload["correct"],
            "errors": payload["errors"],
            "accuracy": payload["accuracy"],
            "false_positive_count": false_positive_counts.get(patient_id, 0),
            "false_negative_count": false_negative_counts.get(patient_id, 0),
        }
        for patient_id, payload in grouped.items()
        if payload["errors"] > 0
    ]
    return sorted(
        patient_rows,
        key=lambda row: (int(row["errors"]), int(row["total"])),
        reverse=True,
    )[:max_examples]


def summarize_predictions(
    rows: list[dict[str, Any]],
    max_examples: int = 15,
) -> dict[str, Any]:
    correct_rows = [row for row in rows if row["is_correct"]]
    error_rows = [row for row in rows if not row["is_correct"]]

    errors_by_true_class = Counter(row["true_label"] for row in error_rows)
    errors_by_predicted_class = Counter(row["predicted_label"] for row in error_rows)
    error_types = Counter(row["error_type"] for row in error_rows)
    confusion_pairs = Counter(
        f"{row['true_label']} -> {row['predicted_label']}" for row in error_rows
    )
    most_confident_errors = sorted(
        error_rows,
        key=lambda row: float(row["confidence"]),
        reverse=True,
    )[:max_examples]
    least_confident_correct = sorted(
        correct_rows,
        key=lambda row: float(row["confidence"]),
    )[:max_examples]

    total_images = len(rows)
    correct_count = len(correct_rows)
    error_count = len(error_rows)

    rows_with_patient = [row for row in rows if row.get("patient_id")]
    rows_with_magnification = [row for row in rows if row.get("magnification")]

    return {
        "total_images": total_images,
        "correct_count": correct_count,
        "error_count": error_count,
        "accuracy": correct_count / total_images if total_images else 0.0,
        "false_positive_count": error_types.get("false_positive", 0),
        "false_negative_count": error_types.get("false_negative", 0),
        "average_confidence_correct": average_confidence(correct_rows),
        "average_confidence_errors": average_confidence(error_rows),
        "errors_by_true_class": dict(sorted(errors_by_true_class.items())),
        "errors_by_predicted_class": dict(sorted(errors_by_predicted_class.items())),
        "error_types": {error_type: error_types.get(error_type, 0) for error_type in ERROR_TYPES},
        "confusion_pairs": dict(confusion_pairs.most_common()),
        "errors_by_magnification": _sorted_counter(
            [str(row.get("magnification") or "unknown") for row in error_rows]
        ),
        "accuracy_by_magnification": _accuracy_by_field(rows_with_magnification, "magnification"),
        "errors_by_patient": _sorted_counter(
            [str(row.get("patient_id") or "unknown") for row in error_rows]
        )
        if rows_with_patient
        else {},
        "top_error_patients": _patient_error_summary(rows, max_examples=max_examples)
        if rows_with_patient
        else [],
        "most_confident_errors": [compact_row(row) for row in most_confident_errors],
        "least_confident_correct": [compact_row(row) for row in least_confident_correct],
    }


def save_predictions_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        output_path.write_text("", encoding="utf-8")
        return

    fieldnames = list(rows[0].keys())
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_summary_json(summary: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
        file.write("\n")


def load_summary(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)
