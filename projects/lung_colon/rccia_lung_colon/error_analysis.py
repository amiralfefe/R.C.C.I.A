"""Error analysis helpers for Lung + Colon Vision."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


CLASS_METADATA = {
    "colon_adenocarcinoma": {"organ": "colon", "malignancy": "malignant"},
    "colon_benign": {"organ": "colon", "malignancy": "benign"},
    "lung_adenocarcinoma": {"organ": "lung", "malignancy": "malignant"},
    "lung_benign": {"organ": "lung", "malignancy": "benign"},
    "lung_squamous_cell_carcinoma": {"organ": "lung", "malignancy": "malignant"},
}

ERROR_TYPES = (
    "correct",
    "same_organ_confusion",
    "lung_colon_confusion",
    "benign_malignant_confusion",
    "cancer_subtype_confusion",
)


def class_organ(class_name: str) -> str:
    metadata = CLASS_METADATA.get(class_name)
    if metadata:
        return metadata["organ"]
    return class_name.split("_", maxsplit=1)[0]


def class_malignancy(class_name: str) -> str:
    metadata = CLASS_METADATA.get(class_name)
    if metadata:
        return metadata["malignancy"]
    return "benign" if "benign" in class_name else "malignant"


def classify_error(true_label: str, predicted_label: str) -> str:
    if true_label == predicted_label:
        return "correct"

    true_organ = class_organ(true_label)
    predicted_organ = class_organ(predicted_label)
    true_malignancy = class_malignancy(true_label)
    predicted_malignancy = class_malignancy(predicted_label)

    if (
        true_malignancy == "malignant"
        and predicted_malignancy == "malignant"
        and true_organ == predicted_organ
    ):
        return "cancer_subtype_confusion"

    if true_malignancy != predicted_malignancy:
        return "benign_malignant_confusion"

    if true_organ != predicted_organ:
        return "lung_colon_confusion"

    return "same_organ_confusion"


def is_benign_malignant_error(true_label: str, predicted_label: str) -> bool:
    return true_label != predicted_label and class_malignancy(true_label) != class_malignancy(
        predicted_label
    )


def prediction_row(
    image_path: str,
    true_label: str,
    predicted_label: str,
    confidence: float,
    probabilities: list[float],
    class_names: list[str],
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "image_path": image_path,
        "true_label": true_label,
        "predicted_label": predicted_label,
        "confidence": float(confidence),
        "is_correct": true_label == predicted_label,
        "error_type": classify_error(true_label, predicted_label),
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
    }


def average_confidence(rows: list[dict[str, Any]]) -> float | None:
    if not rows:
        return None
    return sum(float(row["confidence"]) for row in rows) / len(rows)


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
    benign_malignant_errors = [
        row
        for row in error_rows
        if is_benign_malignant_error(row["true_label"], row["predicted_label"])
    ]

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

    return {
        "total_images": total_images,
        "correct_count": correct_count,
        "error_count": error_count,
        "accuracy": correct_count / total_images if total_images else 0.0,
        "average_confidence_correct": average_confidence(correct_rows),
        "average_confidence_errors": average_confidence(error_rows),
        "errors_by_true_class": dict(sorted(errors_by_true_class.items())),
        "errors_by_predicted_class": dict(sorted(errors_by_predicted_class.items())),
        "error_types": {error_type: error_types.get(error_type, 0) for error_type in ERROR_TYPES},
        "confusion_pairs": dict(confusion_pairs.most_common()),
        "benign_malignant_errors": len(benign_malignant_errors),
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
