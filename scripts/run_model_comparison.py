"""Train and evaluate several model architectures on the same processed dataset."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cancer_cell_vision.model import SUPPORTED_MODEL_NAMES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Cancer Cell Vision model comparison.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/model_comparison"))
    parser.add_argument("--models", nargs="+", choices=SUPPORTED_MODEL_NAMES, default=list(SUPPORTED_MODEL_NAMES))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-pretrained", action="store_true")
    return parser.parse_args()


def run_command(command: list[str]) -> None:
    print("Running:", " ".join(command))
    subprocess.run(command, check=True)


def metric(report: dict, class_name: str, key: str) -> float:
    return float(report.get(class_name, {}).get(key, 0.0))


def build_summary_row(
    model_name: str,
    checkpoint_path: Path,
    report_path: Path,
    train_time_seconds: float,
) -> dict[str, str | float]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    return {
        "model": model_name,
        "accuracy": float(report.get("accuracy", 0.0)),
        "precision_normal": metric(report, "normal", "precision"),
        "recall_normal": metric(report, "normal", "recall"),
        "f1_normal": metric(report, "normal", "f1-score"),
        "precision_leukemia_blast": metric(report, "leukemia_blast", "precision"),
        "recall_leukemia_blast": metric(report, "leukemia_blast", "recall"),
        "f1_leukemia_blast": metric(report, "leukemia_blast", "f1-score"),
        "checkpoint_path": str(checkpoint_path),
        "train_time_seconds": round(train_time_seconds, 2),
    }


def save_summary(rows: list[dict[str, str | float]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    columns = [
        "model",
        "accuracy",
        "precision_normal",
        "recall_normal",
        "f1_normal",
        "precision_leukemia_blast",
        "recall_leukemia_blast",
        "f1_leukemia_blast",
        "checkpoint_path",
        "train_time_seconds",
    ]

    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    (output_dir / "summary.json").write_text(
        json.dumps(rows, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str | float]] = []
    for model_name in args.models:
        model_dir = args.output_dir / model_name
        train_dir = model_dir / "train"
        eval_dir = model_dir / "eval"
        checkpoint_path = model_dir / "best_model.pt"

        train_command = [
            sys.executable,
            "-m",
            "cancer_cell_vision.train",
            "--data-dir",
            str(args.data_dir),
            "--output-dir",
            str(model_dir),
            "--report-dir",
            str(train_dir),
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

        start = time.perf_counter()
        run_command(train_command)
        train_time_seconds = time.perf_counter() - start

        run_command(
            [
                sys.executable,
                "-m",
                "cancer_cell_vision.evaluate",
                "--data-dir",
                str(args.data_dir),
                "--checkpoint",
                str(checkpoint_path),
                "--output-dir",
                str(eval_dir),
                "--batch-size",
                str(args.batch_size),
                "--num-workers",
                str(args.num_workers),
            ]
        )

        rows.append(
            build_summary_row(
                model_name=model_name,
                checkpoint_path=checkpoint_path,
                report_path=eval_dir / "test_classification_report.json",
                train_time_seconds=train_time_seconds,
            )
        )
        save_summary(rows, args.output_dir)

    print(f"Saved comparison summary to {args.output_dir / 'summary.csv'}")
    print(f"Saved comparison summary to {args.output_dir / 'summary.json'}")


if __name__ == "__main__":
    try:
        main()
    except (subprocess.CalledProcessError, FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"Error: {exc}") from exc
