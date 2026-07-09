from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw


CLASSES = ("non_metastatic", "metastatic")


def load_metastasis_model_comparison_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "run_model_comparison.py"
    spec = importlib.util.spec_from_file_location("metastasis_model_comparison", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_command(args: list[str], cwd: Path, timeout: int = 240) -> subprocess.CompletedProcess:
    result = subprocess.run(
        args,
        cwd=cwd,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    assert result.returncode == 0, (
        f"Command failed: {' '.join(args)}\n"
        f"STDOUT:\n{result.stdout}\n"
        f"STDERR:\n{result.stderr}"
    )
    return result


def create_synthetic_image(path: Path, class_index: int, image_index: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    base = 80 + class_index * 90
    image = Image.new("RGB", (64, 64), (base, 145, 175 - class_index * 70))
    draw = ImageDraw.Draw(image)
    offset = 6 + image_index
    draw.rectangle((offset, offset, 62 - offset, 62 - offset), outline=(255, 255, 255), width=2)
    draw.line((8, 16 + class_index * 22, 56, 16 + class_index * 22), fill=(20, 20, 20), width=2)
    image.save(path)


def create_processed_dataset(base_dir: Path) -> Path:
    processed_dir = base_dir / "processed"
    split_counts = {"train": 4, "val": 2, "test": 2}
    for class_index, class_name in enumerate(CLASSES):
        for split, count in split_counts.items():
            for image_index in range(count):
                create_synthetic_image(
                    processed_dir / split / class_name / f"{class_name}_{split}_{image_index}.png",
                    class_index,
                    image_index,
                )
    return processed_dir


def test_model_comparison_summary_row_contains_metastasis_metrics() -> None:
    metrics = {
        "accuracy": 0.9,
        "roc_auc": 0.96,
        "pr_auc": 0.97,
        "classification_report": {
            "metastatic": {"precision": 0.91, "recall": 0.92, "f1-score": 0.915},
            "non_metastatic": {"precision": 0.89, "recall": 0.88, "f1-score": 0.885},
            "macro avg": {"precision": 0.9, "recall": 0.9, "f1-score": 0.9},
            "weighted avg": {"precision": 0.9, "recall": 0.9, "f1-score": 0.9},
        },
        "per_class": {
            "metastatic": {"precision": 0.91, "recall": 0.92, "f1": 0.915},
            "non_metastatic": {"precision": 0.89, "recall": 0.88, "f1": 0.885},
        },
    }

    module = load_metastasis_model_comparison_module()
    row = module.build_summary_row(
        model_name="resnet18",
        metrics=metrics,
        class_names=["metastatic", "non_metastatic"],
        checkpoint_path=Path("outputs/model_comparison/resnet18/best_model.pt"),
        train_time_seconds=12.345,
        eval_time_seconds=1.5,
        test_image_count=10,
    )

    assert row["model"] == "resnet18"
    assert row["accuracy"] == 0.9
    assert row["macro_f1"] == 0.9
    assert row["roc_auc"] == 0.96
    assert row["pr_auc"] == 0.97
    assert row["recall_metastatic"] == 0.92
    assert row["f1_non_metastatic"] == 0.885
    assert row["train_time_seconds"] == 12.35
    assert row["inference_time_ms_per_image"] == 150


def test_metastasis_model_comparison_smoke_with_synthetic_dataset(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[1]
    processed_dir = create_processed_dataset(tmp_path)
    output_dir = tmp_path / "outputs" / "model_comparison"

    run_command(
        [
            sys.executable,
            "scripts/run_model_comparison.py",
            "--data-dir",
            str(processed_dir),
            "--models",
            "resnet18",
            "--epochs",
            "1",
            "--batch-size",
            "4",
            "--image-size",
            "32",
            "--output-dir",
            str(output_dir),
            "--no-pretrained",
            "--num-workers",
            "0",
        ],
        cwd=project_root,
        timeout=240,
    )

    summary_path = output_dir / "summary.json"
    assert summary_path.exists()
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    result = payload["results"][0]

    assert result["model"] == "resnet18"
    assert "accuracy" in result
    assert "macro_f1" in result
    assert "roc_auc" in result
    assert "pr_auc" in result
    assert "precision_non_metastatic" in result
    assert "recall_metastatic" in result
    assert (output_dir / "summary.csv").exists()
    assert (output_dir / "resnet18" / "metrics.json").exists()
    assert (output_dir / "resnet18" / "classification_report.json").exists()
    assert (output_dir / "resnet18" / "confusion_matrix.png").exists()
    assert (output_dir / "resnet18" / "training_curves.png").exists()
