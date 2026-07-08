from __future__ import annotations

import csv
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw


CLASSES = ("benign", "malignant")


def load_breast_model_comparison_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "run_model_comparison.py"
    spec = importlib.util.spec_from_file_location("breast_model_comparison", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_command(args: list[str], cwd: Path, timeout: int = 180) -> subprocess.CompletedProcess:
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
    base = 90 + class_index * 80
    image = Image.new("RGB", (64, 64), (base, 160 - class_index * 40, 150))
    draw = ImageDraw.Draw(image)
    offset = 6 + image_index
    draw.rectangle((offset, offset, 62 - offset, 62 - offset), outline=(255, 255, 255), width=2)
    draw.line((8, 18 + class_index * 18, 56, 18 + class_index * 18), fill=(20, 20, 20), width=2)
    image.save(path)


def create_processed_dataset_with_metadata(base_dir: Path) -> tuple[Path, Path]:
    data_root = base_dir / "data"
    processed_dir = data_root / "processed"
    raw_dir = data_root / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    metadata_rows: list[dict[str, str]] = []
    split_counts = {"train": 4, "val": 2, "test": 2}
    magnifications = ("40X", "100X", "200X", "400X")
    for class_index, class_name in enumerate(CLASSES):
        for split, count in split_counts.items():
            for image_index in range(count):
                magnification = magnifications[image_index % len(magnifications)]
                filename = f"{class_name}_{split}_{image_index}_{magnification}.png"
                image_path = processed_dir / split / class_name / filename
                create_synthetic_image(image_path, class_index, image_index)
                metadata_rows.append(
                    {
                        "class_name": class_name,
                        "relative_path": f"{class_name}/{filename}",
                        "output_path": "",
                        "source_path": "",
                        "magnification": magnification,
                        "patient_id": f"{split}_{class_name}_{image_index}",
                    }
                )

    with (raw_dir / "metadata.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(metadata_rows[0].keys()))
        writer.writeheader()
        writer.writerows(metadata_rows)

    return processed_dir, raw_dir / "metadata.csv"


def test_model_comparison_summary_row_contains_breast_metrics() -> None:
    report = {
        "benign": {
            "precision": 0.88,
            "recall": 0.79,
            "f1-score": 0.83,
        },
        "malignant": {
            "precision": 0.90,
            "recall": 0.95,
            "f1-score": 0.92,
        },
        "accuracy": 0.89,
        "macro avg": {
            "precision": 0.89,
            "recall": 0.87,
            "f1-score": 0.88,
        },
        "weighted avg": {
            "precision": 0.89,
            "recall": 0.89,
            "f1-score": 0.89,
        },
    }

    module = load_breast_model_comparison_module()
    row = module.build_summary_row(
        model_name="resnet18",
        report=report,
        class_names=["benign", "malignant"],
        checkpoint_path=Path("outputs/model_comparison/resnet18/best_model.pt"),
        train_time_seconds=12.345,
        eval_time_seconds=1.25,
        test_image_count=10,
        magnification_accuracy={
            "accuracy_40X": 0.8,
            "accuracy_100X": 0.9,
            "accuracy_200X": 1.0,
            "accuracy_400X": "",
        },
    )

    assert row["model"] == "resnet18"
    assert row["accuracy"] == 0.89
    assert row["macro_f1"] == 0.88
    assert row["recall_malignant"] == 0.95
    assert row["f1_benign"] == 0.83
    assert row["accuracy_200X"] == 1.0
    assert row["train_time_seconds"] == 12.35
    assert row["inference_time_ms_per_image"] == 125


def test_breast_model_comparison_smoke_with_synthetic_dataset(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[1]
    processed_dir, metadata_path = create_processed_dataset_with_metadata(tmp_path)
    output_dir = tmp_path / "outputs" / "model_comparison"

    run_command(
        [
            sys.executable,
            "scripts/run_model_comparison.py",
            "--data-dir",
            str(processed_dir),
            "--metadata",
            str(metadata_path),
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
    assert "macro_f1" in result
    assert "recall_malignant" in result
    assert "accuracy_40X" in result
    assert payload["split_verification"]["patient_overlap_count"] == 0
    assert (output_dir / "summary.csv").exists()
    assert (output_dir / "resnet18" / "eval" / "test_classification_report.json").exists()
    assert (output_dir / "resnet18" / "eval" / "test_predictions.csv").exists()
    assert (output_dir / "resnet18" / "train" / "training_loss.png").exists()
