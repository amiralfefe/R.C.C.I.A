"""Run repeatable model comparisons for Lung + Colon Vision."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rccia_lung_colon.data import build_dataset  # noqa: E402
from rccia_lung_colon.model import SUPPORTED_MODEL_NAMES  # noqa: E402


DEFAULT_MODELS = ("resnet18", "mobilenet_v3_small", "efficientnet_b0")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare Lung + Colon model architectures.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--models", nargs="+", choices=SUPPORTED_MODEL_NAMES, default=DEFAULT_MODELS)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/model_comparison"))
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-pretrained", action="store_true")
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else (Path.cwd() / path).resolve()


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


def load_report(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def count_test_images(data_dir: Path, image_size: int) -> int:
    return len(build_dataset(data_dir, split="test", image_size=image_size))


def build_summary_row(
    model_name: str,
    report: dict,
    class_names: list[str],
    checkpoint_path: Path,
    train_time_seconds: float,
    eval_time_seconds: float,
    test_image_count: int,
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
    config: dict[str, object],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump({"config": config, "results": rows}, file, indent=2)
        file.write("\n")


def save_summary(
    rows: list[dict[str, float | str]],
    output_dir: Path,
    config: dict[str, object],
) -> None:
    save_summary_csv(rows, output_dir / "summary.csv")
    save_summary_json(rows, output_dir / "summary.json", config=config)


def main() -> None:
    args = parse_args()
    data_dir = resolve_path(args.data_dir)
    output_dir = resolve_path(args.output_dir)

    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    test_image_count = count_test_images(data_dir, image_size=args.image_size)
    class_names = build_dataset(data_dir, split="test", image_size=args.image_size).classes
    config = {
        "data_dir": str(data_dir),
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
            "rccia_lung_colon.train",
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

        eval_command = [
            sys.executable,
            "-m",
            "rccia_lung_colon.evaluate",
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
        report = load_report(eval_report_dir / "test_classification_report.json")

        row = build_summary_row(
            model_name=model_name,
            report=report,
            class_names=class_names,
            checkpoint_path=checkpoint_path,
            train_time_seconds=train_time_seconds,
            eval_time_seconds=eval_time_seconds,
            test_image_count=test_image_count,
        )
        rows.append(row)
        save_summary(rows, output_dir, config)
        print(
            f"{model_name}: accuracy={row['accuracy']:.4f} "
            f"macro_f1={row['macro_f1']:.4f} "
            f"train_time_seconds={row['train_time_seconds']}"
        )

    print(f"\nSaved summary: {output_dir / 'summary.csv'}")
    print(f"Saved summary: {output_dir / 'summary.json'}")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError, subprocess.CalledProcessError) as exc:
        raise SystemExit(f"Error: {exc}") from exc
