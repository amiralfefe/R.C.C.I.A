"""Run patient-aware model comparisons for R.C.C.I.A Breast."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rccia_breast.data import build_dataset  # noqa: E402
from rccia_breast.metrics import (  # noqa: E402
    build_classification_report,
    save_classification_report_csv,
    save_confusion_matrix,
)
from rccia_breast.model import SUPPORTED_MODEL_NAMES, load_checkpoint  # noqa: E402
from rccia_breast.utils import ensure_dir, get_device, save_json  # noqa: E402


DEFAULT_MODELS = ("resnet18", "mobilenet_v3_small", "efficientnet_b0")
MAGNIFICATIONS = ("40X", "100X", "200X", "400X")
SPLITS = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare Breast model architectures.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--models", nargs="+", choices=SUPPORTED_MODEL_NAMES, default=DEFAULT_MODELS)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/model_comparison"))
    parser.add_argument("--metadata", type=Path, default=None)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-pretrained", action="store_true")
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else (Path.cwd() / path).resolve()


def infer_metadata_path(data_dir: Path) -> Path:
    return data_dir.parent / "raw" / "metadata.csv"


def command_env() -> dict[str, str]:
    env = os.environ.copy()
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        str(PROJECT_ROOT) if not existing else f"{PROJECT_ROOT}{os.pathsep}{existing}"
    )
    return env


def run_command(command: list[str]) -> float:
    started_at = time.perf_counter()
    subprocess.run(command, cwd=REPO_ROOT, env=command_env(), check=True)
    return time.perf_counter() - started_at


def load_metadata(metadata_path: Path | None) -> dict[tuple[str, str], dict[str, str]]:
    if metadata_path is None or not metadata_path.exists():
        return {}

    rows: dict[tuple[str, str], dict[str, str]] = {}
    with metadata_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            class_name = (row.get("class_name") or "").strip()
            relative_path = row.get("relative_path") or row.get("output_path") or ""
            filename = Path(relative_path).name
            if class_name and filename:
                rows[(class_name, filename)] = row
    return rows


def metadata_for_sample(
    sample_path: str | Path,
    class_name: str,
    metadata_rows: dict[tuple[str, str], dict[str, str]],
) -> dict[str, str]:
    return metadata_rows.get((class_name, Path(sample_path).name), {})


def verify_patient_split(
    data_dir: Path,
    metadata_rows: dict[tuple[str, str], dict[str, str]],
) -> dict[str, Any]:
    if not metadata_rows:
        return {
            "metadata_available": False,
            "patient_overlap_count": None,
            "patient_overlap": [],
            "patients_by_split": {},
            "missing_metadata_count": None,
        }

    patients_by_split: dict[str, set[str]] = {split: set() for split in SPLITS}
    missing_metadata_count = 0
    for split in SPLITS:
        split_dir = data_dir / split
        for image_path in sorted(split_dir.glob("*/*")):
            if not image_path.is_file():
                continue
            class_name = image_path.parent.name
            row = metadata_for_sample(image_path, class_name, metadata_rows)
            patient_id = (row.get("patient_id") or "").strip()
            if patient_id:
                patients_by_split[split].add(patient_id)
            else:
                missing_metadata_count += 1

    overlap = sorted(
        (patients_by_split["train"] & patients_by_split["val"])
        | (patients_by_split["train"] & patients_by_split["test"])
        | (patients_by_split["val"] & patients_by_split["test"])
    )
    return {
        "metadata_available": True,
        "patient_overlap_count": len(overlap),
        "patient_overlap": overlap,
        "patients_by_split": {
            split: len(patients) for split, patients in patients_by_split.items()
        },
        "missing_metadata_count": missing_metadata_count,
    }


@torch.inference_mode()
def collect_predictions(
    model: torch.nn.Module,
    loader: DataLoader,
    dataset,
    class_names: list[str],
    device: torch.device,
    metadata_rows: dict[tuple[str, str], dict[str, str]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    sample_offset = 0
    model.eval()

    for inputs, labels in tqdm(loader, leave=False):
        inputs = inputs.to(device)
        logits = model(inputs)
        probabilities = torch.softmax(logits, dim=1).detach().cpu()
        predictions = probabilities.argmax(dim=1)

        for index in range(labels.size(0)):
            sample_path, _class_index = dataset.samples[sample_offset + index]
            true_label = class_names[int(labels[index].item())]
            predicted_label = class_names[int(predictions[index].item())]
            metadata = metadata_for_sample(sample_path, true_label, metadata_rows)
            row = {
                "image_path": str(sample_path),
                "true_label": true_label,
                "predicted_label": predicted_label,
                "correct": true_label == predicted_label,
                "confidence": float(probabilities[index, predictions[index]].item()),
                "magnification": metadata.get("magnification", ""),
                "patient_id": metadata.get("patient_id", ""),
            }
            for class_index, class_name in enumerate(class_names):
                row[f"probability_{class_name}"] = float(probabilities[index, class_index].item())
            rows.append(row)
        sample_offset += labels.size(0)
    return rows


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


def accuracy_by_magnification(rows: list[dict[str, Any]]) -> dict[str, float | str]:
    by_magnification: dict[str, list[bool]] = defaultdict(list)
    for row in rows:
        magnification = str(row.get("magnification") or "")
        if magnification:
            by_magnification[magnification].append(bool(row["correct"]))

    values: dict[str, float | str] = {}
    for magnification in MAGNIFICATIONS:
        correct_flags = by_magnification.get(magnification, [])
        values[f"accuracy_{magnification}"] = (
            sum(correct_flags) / len(correct_flags) if correct_flags else ""
        )
    return values


def evaluate_checkpoint(
    data_dir: Path,
    checkpoint_path: Path,
    output_dir: Path,
    batch_size: int,
    num_workers: int,
    metadata_rows: dict[tuple[str, str], dict[str, str]],
) -> tuple[dict, list[dict[str, Any]], float, int]:
    started_at = time.perf_counter()
    device = get_device()
    model, checkpoint = load_checkpoint(checkpoint_path, device=device)
    class_names = checkpoint["class_names"]
    image_size = int(checkpoint.get("image_size", 224))

    dataset = build_dataset(data_dir=data_dir, split="test", image_size=image_size)
    if dataset.classes != class_names:
        raise ValueError(
            f"Checkpoint classes {class_names} do not match test classes {dataset.classes}."
        )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    prediction_rows = collect_predictions(
        model=model,
        loader=loader,
        dataset=dataset,
        class_names=class_names,
        device=device,
        metadata_rows=metadata_rows,
    )

    y_true = [class_names.index(str(row["true_label"])) for row in prediction_rows]
    y_pred = [class_names.index(str(row["predicted_label"])) for row in prediction_rows]
    report = build_classification_report(y_true, y_pred, class_names)

    save_json(report, output_dir / "test_classification_report.json")
    save_classification_report_csv(report, output_dir / "test_classification_report.csv")
    save_confusion_matrix(
        y_true=y_true,
        y_pred=y_pred,
        class_names=class_names,
        output_path=output_dir / "test_confusion_matrix.png",
    )
    save_predictions_csv(prediction_rows, output_dir / "test_predictions.csv")

    elapsed = time.perf_counter() - started_at
    return report, prediction_rows, elapsed, len(dataset)


def build_summary_row(
    model_name: str,
    report: dict,
    class_names: list[str],
    checkpoint_path: Path,
    train_time_seconds: float,
    eval_time_seconds: float,
    test_image_count: int,
    magnification_accuracy: dict[str, float | str],
) -> dict[str, float | str]:
    row: dict[str, float | str] = {
        "model": model_name,
        "accuracy": float(report["accuracy"]),
        "macro_precision": float(report["macro avg"]["precision"]),
        "macro_recall": float(report["macro avg"]["recall"]),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "weighted_precision": float(report["weighted avg"]["precision"]),
        "weighted_recall": float(report["weighted avg"]["recall"]),
        "weighted_f1": float(report["weighted avg"]["f1-score"]),
        "checkpoint_path": str(checkpoint_path),
        "train_time_seconds": round(train_time_seconds, 2),
        "eval_time_seconds": round(eval_time_seconds, 2),
        "inference_time_ms_per_image": round(
            (eval_time_seconds / max(test_image_count, 1)) * 1000,
            4,
        ),
    }

    for class_name in class_names:
        metrics = report[class_name]
        row[f"precision_{class_name}"] = float(metrics["precision"])
        row[f"recall_{class_name}"] = float(metrics["recall"])
        row[f"f1_{class_name}"] = float(metrics["f1-score"])

    row.update(magnification_accuracy)
    return row


def save_summary_csv(rows: list[dict[str, float | str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        output_path.write_text("", encoding="utf-8")
        return

    fieldnames = list(rows[0].keys())
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_summary_json(
    rows: list[dict[str, float | str]],
    output_path: Path,
    config: dict[str, Any],
    split_verification: dict[str, Any],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "config": config,
                "split_verification": split_verification,
                "results": rows,
            },
            file,
            indent=2,
        )
        file.write("\n")


def save_summary(
    rows: list[dict[str, float | str]],
    output_dir: Path,
    config: dict[str, Any],
    split_verification: dict[str, Any],
) -> None:
    save_summary_csv(rows, output_dir / "summary.csv")
    save_summary_json(rows, output_dir / "summary.json", config, split_verification)


def main() -> None:
    args = parse_args()
    data_dir = resolve_path(args.data_dir)
    output_dir = resolve_path(args.output_dir)
    metadata_path = resolve_path(args.metadata) if args.metadata else infer_metadata_path(data_dir)

    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    ensure_dir(output_dir)
    metadata_rows = load_metadata(metadata_path)
    split_verification = verify_patient_split(data_dir, metadata_rows)
    if split_verification["patient_overlap_count"]:
        raise RuntimeError(
            "Patient leakage detected across splits: "
            f"{split_verification['patient_overlap']}"
        )

    test_dataset = build_dataset(data_dir, split="test", image_size=args.image_size)
    class_names = test_dataset.classes
    test_image_count = len(test_dataset)
    config = {
        "data_dir": str(data_dir),
        "metadata_path": str(metadata_path) if metadata_path.exists() else "",
        "models": list(args.models),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "image_size": args.image_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "num_workers": args.num_workers,
        "pretrained": not args.no_pretrained,
        "seed": args.seed,
        "test_image_count": test_image_count,
    }

    rows: list[dict[str, float | str]] = []
    for model_name in args.models:
        print(f"\n=== {model_name} ===")
        model_output_dir = output_dir / model_name
        checkpoint_path = model_output_dir / "best_model.pt"
        train_report_dir = model_output_dir / "train"
        eval_report_dir = model_output_dir / "eval"

        train_command = [
            sys.executable,
            "-m",
            "rccia_breast.train",
            "--data-dir",
            str(data_dir),
            "--output-dir",
            str(model_output_dir),
            "--report-dir",
            str(train_report_dir),
            "--model",
            model_name,
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
            "--image-size",
            str(args.image_size),
            "--learning-rate",
            str(args.learning_rate),
            "--weight-decay",
            str(args.weight_decay),
            "--num-workers",
            str(args.num_workers),
            "--seed",
            str(args.seed),
        ]
        if args.no_pretrained:
            train_command.append("--no-pretrained")

        train_time_seconds = run_command(train_command)
        report, prediction_rows, eval_time_seconds, evaluated_count = evaluate_checkpoint(
            data_dir=data_dir,
            checkpoint_path=checkpoint_path,
            output_dir=eval_report_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            metadata_rows=metadata_rows,
        )
        row = build_summary_row(
            model_name=model_name,
            report=report,
            class_names=class_names,
            checkpoint_path=checkpoint_path,
            train_time_seconds=train_time_seconds,
            eval_time_seconds=eval_time_seconds,
            test_image_count=evaluated_count,
            magnification_accuracy=accuracy_by_magnification(prediction_rows),
        )
        rows.append(row)
        save_summary(rows, output_dir, config, split_verification)
        print(
            f"{model_name}: accuracy={row['accuracy']:.4f} "
            f"macro_f1={row['macro_f1']:.4f} "
            f"recall_malignant={row.get('recall_malignant', 0):.4f} "
            f"train_time_seconds={row['train_time_seconds']}"
        )

    print(f"\nSaved summary: {output_dir / 'summary.csv'}")
    print(f"Saved summary: {output_dir / 'summary.json'}")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        raise SystemExit(f"Error: {exc}") from exc
