from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

from rccia_breast.error_analysis import (
    classify_error,
    prediction_row,
    save_predictions_csv,
    save_summary_json,
    summarize_predictions,
)


def test_breast_error_types_are_binary_clinical_tradeoffs() -> None:
    assert classify_error("benign", "benign") == "correct"
    assert classify_error("malignant", "malignant") == "correct"
    assert classify_error("benign", "malignant") == "false_positive"
    assert classify_error("malignant", "benign") == "false_negative"


def test_breast_error_summary_includes_patient_and_magnification() -> None:
    class_names = ["benign", "malignant"]
    rows = [
        prediction_row(
            image_path="test/benign/a.png",
            true_label="benign",
            predicted_label="benign",
            confidence=0.91,
            probabilities=[0.91, 0.09],
            class_names=class_names,
            patient_id="patient-a",
            magnification="40X",
        ),
        prediction_row(
            image_path="test/benign/b.png",
            true_label="benign",
            predicted_label="malignant",
            confidence=0.88,
            probabilities=[0.12, 0.88],
            class_names=class_names,
            patient_id="patient-b",
            magnification="100X",
        ),
        prediction_row(
            image_path="test/malignant/c.png",
            true_label="malignant",
            predicted_label="benign",
            confidence=0.73,
            probabilities=[0.73, 0.27],
            class_names=class_names,
            patient_id="patient-c",
            magnification="100X",
        ),
    ]

    summary = summarize_predictions(rows, max_examples=2)

    assert summary["total_images"] == 3
    assert summary["correct_count"] == 1
    assert summary["error_count"] == 2
    assert summary["false_positive_count"] == 1
    assert summary["false_negative_count"] == 1
    assert summary["errors_by_magnification"] == {"100X": 2}
    assert summary["accuracy_by_magnification"]["40X"]["accuracy"] == 1.0
    assert summary["accuracy_by_magnification"]["100X"]["errors"] == 2
    assert summary["errors_by_patient"] == {"patient-b": 1, "patient-c": 1}
    assert summary["top_error_patients"][0]["errors"] == 1
    assert summary["most_confident_errors"][0]["error_type"] == "false_positive"
    assert summary["least_confident_correct"][0]["patient_id"] == "patient-a"


def test_breast_error_analysis_exports_csv_and_summary(tmp_path: Path) -> None:
    rows = [
        prediction_row(
            image_path="test/benign/a.png",
            true_label="benign",
            predicted_label="malignant",
            confidence=0.80,
            probabilities=[0.20, 0.80],
            class_names=["benign", "malignant"],
            patient_id="patient-a",
            magnification="200X",
        )
    ]
    summary = summarize_predictions(rows)
    predictions_path = tmp_path / "predictions.csv"
    summary_path = tmp_path / "summary.json"

    save_predictions_csv(rows, predictions_path)
    save_summary_json(summary, summary_path)

    with predictions_path.open("r", encoding="utf-8", newline="") as file:
        exported_rows = list(csv.DictReader(file))
    exported_summary = json.loads(summary_path.read_text(encoding="utf-8"))

    assert exported_rows[0]["error_type"] == "false_positive"
    assert exported_rows[0]["patient_id"] == "patient-a"
    assert exported_rows[0]["magnification"] == "200X"
    assert "prob_benign" in exported_rows[0]
    assert "prob_malignant" in exported_rows[0]
    assert exported_summary["false_positive_count"] == 1


def test_breast_app_error_analysis_loader_handles_missing_summary(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("breast_streamlit_app_v21", project_root / "app.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module.load_error_analysis_summary(tmp_path / "missing_summary.json") is None
