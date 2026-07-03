from __future__ import annotations

import importlib.util
from pathlib import Path


def load_lung_colon_model_comparison_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "run_model_comparison.py"
    spec = importlib.util.spec_from_file_location("lung_colon_model_comparison", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_model_comparison_summary_row_contains_class_metrics() -> None:
    report = {
        "colon_benign": {
            "precision": 0.9,
            "recall": 0.8,
            "f1-score": 0.85,
        },
        "lung_benign": {
            "precision": 0.7,
            "recall": 0.6,
            "f1-score": 0.65,
        },
        "accuracy": 0.75,
        "macro avg": {
            "precision": 0.8,
            "recall": 0.7,
            "f1-score": 0.75,
        },
        "weighted avg": {
            "precision": 0.81,
            "recall": 0.71,
            "f1-score": 0.76,
        },
    }

    module = load_lung_colon_model_comparison_module()
    row = module.build_summary_row(
        model_name="resnet18",
        report=report,
        class_names=["colon_benign", "lung_benign"],
        checkpoint_path=Path("outputs/model_comparison/resnet18/best_model.pt"),
        train_time_seconds=12.345,
        eval_time_seconds=1.25,
        test_image_count=10,
    )

    assert row["model"] == "resnet18"
    assert row["accuracy"] == 0.75
    assert row["macro_f1"] == 0.75
    assert row["f1_colon_benign"] == 0.85
    assert row["recall_lung_benign"] == 0.6
    assert row["train_time_seconds"] == 12.35
    assert row["inference_time_ms_per_image"] == 125
    assert row["checkpoint_path"] == str(
        Path("outputs/model_comparison/resnet18/best_model.pt")
    )
