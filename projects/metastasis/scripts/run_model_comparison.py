"""Run repeatable model comparisons for R.C.C.I.A Metastasis."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rccia_metastasis.data import build_dataset  # noqa: E402
from rccia_metastasis.model import SUPPORTED_MODEL_NAMES  # noqa: E402


DEFAULT_MODELS = ("resnet18", "mobilenet_v3_small", "efficientnet_b0")
SUMMARY_FIELDS = (
    "model",
    "accuracy",
    "macro_f1",
    "roc_auc",
    "pr_auc",
    "precision_non_metastatic",
    "recall_non_metastatic",
    "f1_non_metastatic",
    "precision_metastatic",
    "recall_metastatic",
    "f1_metastatic",
    "train_time_seconds",
    "checkpoint_path",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare Metastasis model architectures.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--models", nargs="+", choices=SUPPORTED_MODEL_NAMES, default=DEFAULT_MODELS)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=96)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/model_comparison"))
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--no-pretrained", action="store_true")
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else (Path.cwd() / path).resolve()


def command_env() -> dict[str, str]:
    env = os.environ.copy()
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(PROJECT_ROOT) if not existing else f"{PROJECT_ROOT}{os.pathsep}{existing}"
    return env


def run_command(command: list[str]) -> float:
    started_at = time.perf_counter()
    subprocess.run(command, cwd=REPO_ROOT, env=command_env(), check=True)
    return time.perf_counter() - started_at


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_json(payload: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)
        file.write("\n")


def copy_if_exists(source: Path, destination: Path) -> None:
    if source.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def save_training_curves(history_path: Path, output_path: Path) -> None:
    if not history_path.exists():
        return

    history = load_json(history_path)
    if not history:
        return

    epochs = [int(row["epoch"]) for row in history]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(epochs, [row["train_loss"] for row in history], label="train")
    axes[0].plot(epochs, [row["val_loss"] for row in history], label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(epochs, [row["train_accuracy"] for row in history], label="train")
    axes[1].plot(epochs, [row["val_accuracy"] for row in history], label="val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def normalize_model_outputs(model_output_dir: Path, eval_report_dir: Path) -> None:
    copy_if_exists(eval_report_dir / "test_metrics.json", model_output_dir / "metrics.json")
    copy_if_exists(
        eval_report_dir / "test_classification_report.json",
        model_output_dir / "classification_report.json",
    )
    copy_if_exists(eval_report_dir / "test_confusion_matrix.png", model_output_dir / "confusion_matrix.png")
    save_training_curves(
        model_output_dir / "train" / "training_history.json",
        model_output_dir / "training_curves.png",
    )


def metric_value(value: Any) -> float | str:
    return "" if value is None else float(value)


def build_summary_row(
    model_name: str,
    metrics: dict[str, Any],
    class_names: list[str],
    checkpoint_path: Path,
    train_time_seconds: float,
    eval_time_seconds: float,
    test_image_count: int,
) -> dict[str, float | str]:
    report = metrics["classification_report"]
    per_class = metrics["per_class"]
    row: dict[str, float | str] = {
        "model": model_name,
        "accuracy": float(metrics["accuracy"]),
        "macro_precision": float(report["macro avg"]["precision"]),
        "macro_recall": float(report["macro avg"]["recall"]),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "weighted_precision": float(report["weighted avg"]["precision"]),
        "weighted_recall": float(report["weighted avg"]["recall"]),
        "weighted_f1": float(report["weighted avg"]["f1-score"]),
        "roc_auc": metric_value(metrics.get("roc_auc")),
        "pr_auc": metric_value(metrics.get("pr_auc")),
        "checkpoint_path": str(checkpoint_path),
        "train_time_seconds": round(train_time_seconds, 2),
        "eval_time_seconds": round(eval_time_seconds, 2),
        "inference_time_ms_per_image": round(
            (eval_time_seconds / max(test_image_count, 1)) * 1000,
            4,
        ),
    }

    for class_name in class_names:
        class_metrics = per_class[class_name]
        row[f"precision_{class_name}"] = float(class_metrics["precision"])
        row[f"recall_{class_name}"] = float(class_metrics["recall"])
        row[f"f1_{class_name}"] = float(class_metrics["f1"])

    return row


def save_summary_csv(rows: list[dict[str, float | str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        output_path.write_text("", encoding="utf-8")
        return

    extra_fields = [field for field in rows[0] if field not in SUMMARY_FIELDS]
    fieldnames = [field for field in SUMMARY_FIELDS if field in rows[0]] + extra_fields
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_summary_json(
    rows: list[dict[str, float | str]],
    output_path: Path,
    config: dict[str, Any],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump({"config": config, "results": rows}, file, indent=2)
        file.write("\n")


def save_summary(
    rows: list[dict[str, float | str]],
    output_dir: Path,
    config: dict[str, Any],
) -> None:
    save_summary_csv(rows, output_dir / "summary.csv")
    save_summary_json(rows, output_dir / "summary.json", config=config)


def load_existing_summary(output_dir: Path) -> tuple[list[dict[str, float | str]], dict[str, Any]]:
    summary_path = output_dir / "summary.json"
    if not summary_path.exists():
        return [], {}

    payload = load_json(summary_path)
    rows = payload.get("results", [])
    config = payload.get("config", {})
    if not isinstance(rows, list) or not isinstance(config, dict):
        return [], {}
    return rows, config


def main() -> None:
    args = parse_args()
    data_dir = resolve_path(args.data_dir)
    output_dir = resolve_path(args.output_dir)

    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    test_dataset = build_dataset(data_dir=data_dir, split="test", image_size=args.image_size)
    class_names = test_dataset.classes
    test_image_count = len(test_dataset)
    config = {
        "data_dir": str(data_dir),
        "models": list(args.models),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "image_size": args.image_size,
        "learning_rate": args.learning_rate,
        "num_workers": args.num_workers,
        "pretrained": not args.no_pretrained,
        "test_image_count": test_image_count,
    }

    existing_rows, existing_config = load_existing_summary(output_dir)
    rows: list[dict[str, float | str]] = [
        row for row in existing_rows if str(row.get("model")) not in set(args.models)
    ]
    if existing_config:
        config = {**config, **existing_config}

    for model_name in args.models:
        print(f"\n=== {model_name} ===")
        model_output_dir = output_dir / model_name
        checkpoint_path = model_output_dir / "best_model.pt"
        eval_report_dir = model_output_dir / "eval"

        train_command = [
            sys.executable,
            "-m",
            "rccia_metastasis.train",
            "--data-dir",
            str(data_dir),
            "--output-dir",
            str(model_output_dir),
            "--model",
            model_name,
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
            "--image-size",
            str(args.image_size),
            "--lr",
            str(args.learning_rate),
            "--num-workers",
            str(args.num_workers),
        ]
        if args.no_pretrained:
            train_command.append("--no-pretrained")

        train_time_seconds = run_command(train_command)

        eval_command = [
            sys.executable,
            "-m",
            "rccia_metastasis.evaluate",
            "--data-dir",
            str(data_dir),
            "--checkpoint",
            str(checkpoint_path),
            "--output-dir",
            str(eval_report_dir),
            "--batch-size",
            str(args.batch_size),
            "--num-workers",
            str(args.num_workers),
        ]
        eval_time_seconds = run_command(eval_command)
        metrics = load_json(eval_report_dir / "test_metrics.json")
        normalize_model_outputs(model_output_dir=model_output_dir, eval_report_dir=eval_report_dir)

        row = build_summary_row(
            model_name=model_name,
            metrics=metrics,
            class_names=class_names,
            checkpoint_path=checkpoint_path,
            train_time_seconds=train_time_seconds,
            eval_time_seconds=eval_time_seconds,
            test_image_count=test_image_count,
        )
        rows.append(row)
        config["models"] = [str(existing_row["model"]) for existing_row in rows]
        save_summary(rows, output_dir, config)
        print(
            f"{model_name}: accuracy={row['accuracy']:.4f} "
            f"macro_f1={row['macro_f1']:.4f} "
            f"roc_auc={row['roc_auc']} "
            f"pr_auc={row['pr_auc']} "
            f"train_time_seconds={row['train_time_seconds']}"
        )

    print(f"\nSaved summary: {output_dir / 'summary.csv'}")
    print(f"Saved summary: {output_dir / 'summary.json'}")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError, subprocess.CalledProcessError) as exc:
        raise SystemExit(f"Error: {exc}") from exc
