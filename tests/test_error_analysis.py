from __future__ import annotations

from pathlib import Path

from cancer_cell_vision.error_analysis import (
    build_summary,
    load_error_analysis_artifacts,
)


def test_build_summary_counts_error_types() -> None:
    rows = [
        {
            "image_path": "normal_1.png",
            "true_label": "normal",
            "predicted_label": "leukemia_blast",
            "confidence": 0.91,
            "prob_normal": 0.09,
            "prob_leukemia_blast": 0.91,
            "is_correct": False,
            "error_type": "false_positive",
        },
        {
            "image_path": "blast_1.png",
            "true_label": "leukemia_blast",
            "predicted_label": "leukemia_blast",
            "confidence": 0.88,
            "prob_normal": 0.12,
            "prob_leukemia_blast": 0.88,
            "is_correct": True,
            "error_type": "correct",
        },
    ]

    summary = build_summary(rows)

    assert summary["total_images"] == 2
    assert summary["correct_count"] == 1
    assert summary["error_count"] == 1
    assert summary["false_positive_count"] == 1
    assert summary["false_negative_count"] == 0
    assert summary["accuracy"] == 0.5


def test_load_error_analysis_artifacts_handles_missing_output(tmp_path: Path) -> None:
    state = load_error_analysis_artifacts(tmp_path / "missing")

    assert state["available"] is False
    assert state["summary"] is None
    assert state["examples"] == []
