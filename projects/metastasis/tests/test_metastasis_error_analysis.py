from __future__ import annotations

import importlib.util
from pathlib import Path

from rccia_metastasis.error_analysis import (
    classify_error,
    compute_threshold_analysis,
    prediction_row,
    summarize_predictions,
)


def test_metastasis_error_types_are_binary_tradeoffs() -> None:
    assert classify_error("non_metastatic", "non_metastatic") == "correct"
    assert classify_error("metastatic", "metastatic") == "correct"
    assert classify_error("non_metastatic", "metastatic") == "false_positive"
    assert classify_error("metastatic", "non_metastatic") == "false_negative"


def test_metastasis_threshold_analysis_counts_fp_and_fn() -> None:
    rows = [
        prediction_row("a.png", "metastatic", prob_non_metastatic=0.20, prob_metastatic=0.80),
        prediction_row("b.png", "metastatic", prob_non_metastatic=0.55, prob_metastatic=0.45),
        prediction_row("c.png", "non_metastatic", prob_non_metastatic=0.60, prob_metastatic=0.40),
        prediction_row("d.png", "non_metastatic", prob_non_metastatic=0.25, prob_metastatic=0.75),
    ]

    thresholds = compute_threshold_analysis(rows, thresholds=[0.3, 0.5, 0.7])

    at_050 = next(row for row in thresholds if row["threshold"] == 0.5)
    assert at_050["accuracy"] == 0.5
    assert at_050["false_positive_count"] == 1
    assert at_050["false_negative_count"] == 1
    assert at_050["predicted_positive_count"] == 2
    assert at_050["predicted_negative_count"] == 2

    at_030 = next(row for row in thresholds if row["threshold"] == 0.3)
    assert at_030["false_negative_count"] == 0
    assert at_030["false_positive_count"] == 2

    at_070 = next(row for row in thresholds if row["threshold"] == 0.7)
    assert at_070["false_negative_count"] == 1
    assert at_070["false_positive_count"] == 1


def test_metastasis_error_summary_contains_required_metrics() -> None:
    rows = [
        prediction_row("a.png", "metastatic", prob_non_metastatic=0.10, prob_metastatic=0.90),
        prediction_row("b.png", "metastatic", prob_non_metastatic=0.70, prob_metastatic=0.30),
        prediction_row("c.png", "non_metastatic", prob_non_metastatic=0.95, prob_metastatic=0.05),
        prediction_row("d.png", "non_metastatic", prob_non_metastatic=0.20, prob_metastatic=0.80),
    ]

    summary = summarize_predictions(rows, thresholds=[0.3, 0.5, 0.7], max_examples=2)

    assert summary["total_images"] == 4
    assert summary["correct_count"] == 2
    assert summary["error_count"] == 2
    assert summary["accuracy_at_0_50"] == 0.5
    assert summary["false_positive_count"] == 1
    assert summary["false_negative_count"] == 1
    assert summary["average_confidence_correct"] is not None
    assert summary["average_confidence_errors"] is not None
    assert summary["roc_auc"] is not None
    assert summary["pr_auc"] is not None
    assert summary["most_confident_errors"][0]["error_type"] == "false_positive"
    assert summary["least_confident_correct"][0]["true_label"] == "metastatic"
    assert "threshold_at_0_50" in summary


def test_metastasis_app_error_analysis_loader_handles_missing_summary(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "metastasis_streamlit_app_v21",
        project_root / "app.py",
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module.load_error_analysis_summary(tmp_path / "missing_summary.json") is None
