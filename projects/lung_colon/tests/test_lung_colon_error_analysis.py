from __future__ import annotations

import csv
from pathlib import Path

from rccia_lung_colon.error_analysis import (
    classify_error,
    load_summary,
    prediction_row,
    save_predictions_csv,
    summarize_predictions,
)


CLASSES = [
    "colon_adenocarcinoma",
    "colon_benign",
    "lung_adenocarcinoma",
    "lung_benign",
    "lung_squamous_cell_carcinoma",
]


def test_error_type_classification_rules() -> None:
    assert classify_error("colon_benign", "colon_benign") == "correct"
    assert (
        classify_error("colon_benign", "colon_adenocarcinoma")
        == "benign_malignant_confusion"
    )
    assert (
        classify_error("lung_adenocarcinoma", "lung_squamous_cell_carcinoma")
        == "cancer_subtype_confusion"
    )
    assert classify_error("colon_adenocarcinoma", "lung_adenocarcinoma") == "lung_colon_confusion"


def test_summary_counts_confidence_and_confusions() -> None:
    rows = [
        prediction_row(
            image_path="a.png",
            true_label="colon_benign",
            predicted_label="colon_benign",
            confidence=0.90,
            probabilities=[0.01, 0.90, 0.03, 0.04, 0.02],
            class_names=CLASSES,
        ),
        prediction_row(
            image_path="b.png",
            true_label="lung_adenocarcinoma",
            predicted_label="lung_squamous_cell_carcinoma",
            confidence=0.80,
            probabilities=[0.01, 0.02, 0.15, 0.02, 0.80],
            class_names=CLASSES,
        ),
        prediction_row(
            image_path="c.png",
            true_label="lung_benign",
            predicted_label="lung_adenocarcinoma",
            confidence=0.70,
            probabilities=[0.01, 0.02, 0.70, 0.20, 0.07],
            class_names=CLASSES,
        ),
    ]

    summary = summarize_predictions(rows, max_examples=2)

    assert summary["total_images"] == 3
    assert summary["correct_count"] == 1
    assert summary["error_count"] == 2
    assert summary["accuracy"] == 1 / 3
    assert summary["average_confidence_correct"] == 0.90
    assert summary["average_confidence_errors"] == 0.75
    assert summary["errors_by_true_class"] == {
        "lung_adenocarcinoma": 1,
        "lung_benign": 1,
    }
    assert summary["error_types"]["cancer_subtype_confusion"] == 1
    assert summary["error_types"]["benign_malignant_confusion"] == 1
    assert summary["benign_malignant_errors"] == 1
    assert summary["confusion_pairs"]["lung_adenocarcinoma -> lung_squamous_cell_carcinoma"] == 1


def test_predictions_csv_and_missing_summary_are_safe(tmp_path: Path) -> None:
    row = prediction_row(
        image_path="a.png",
        true_label="colon_benign",
        predicted_label="colon_benign",
        confidence=0.95,
        probabilities=[0.01, 0.95, 0.01, 0.02, 0.01],
        class_names=CLASSES,
    )

    output_path = tmp_path / "predictions.csv"
    save_predictions_csv([row], output_path)

    with output_path.open("r", encoding="utf-8", newline="") as file:
        loaded = list(csv.DictReader(file))

    assert loaded[0]["true_label"] == "colon_benign"
    assert loaded[0]["prob_lung_squamous_cell_carcinoma"] == "0.01"
    assert load_summary(tmp_path / "missing_summary.json") is None
