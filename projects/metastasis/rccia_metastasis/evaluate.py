"""Evaluate a Metastasis Vision checkpoint."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import build_dataset
from .metrics import (
    compute_binary_metrics,
    plot_confusion_matrix,
    plot_precision_recall_curve,
    plot_roc_curve,
    positive_class_index,
    save_classification_report_csv,
    save_json,
)
from .model import SUPPORTED_MODEL_NAMES, load_checkpoint
from .utils import get_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a Metastasis classifier.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model", choices=SUPPORTED_MODEL_NAMES, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/eval"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


@torch.inference_mode()
def collect_predictions(
    model: torch.nn.Module,
    loader: DataLoader,
    dataset_samples: list[tuple[str, int]],
    class_names: list[str],
    device: torch.device,
) -> tuple[list[int], list[int], list[float], list[dict[str, Any]]]:
    y_true: list[int] = []
    y_pred: list[int] = []
    positive_scores: list[float] = []
    rows: list[dict[str, Any]] = []
    positive_index = positive_class_index(class_names)
    sample_offset = 0

    model.eval()
    for inputs, labels in tqdm(loader, leave=False):
        batch_size = labels.shape[0]
        inputs = inputs.to(device)
        logits = model(inputs)
        probabilities = torch.softmax(logits, dim=1).detach().cpu()
        predictions = probabilities.argmax(dim=1)

        batch_samples = dataset_samples[sample_offset : sample_offset + batch_size]
        for local_index, (image_path, true_index) in enumerate(batch_samples):
            predicted_index = int(predictions[local_index].item())
            score = float(probabilities[local_index, positive_index].item())
            y_true.append(int(true_index))
            y_pred.append(predicted_index)
            positive_scores.append(score)
            rows.append(
                {
                    "image_path": image_path,
                    "true_label": class_names[int(true_index)],
                    "predicted_label": class_names[predicted_index],
                    "prob_metastatic": score,
                    "confidence": float(probabilities[local_index, predicted_index].item()),
                }
            )
        sample_offset += batch_size

    return y_true, y_pred, positive_scores, rows


def save_predictions_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        output_path.write_text("", encoding="utf-8")
        return
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()
    model, checkpoint = load_checkpoint(args.checkpoint, device=device)
    checkpoint_model_name = checkpoint.get("model_name", "resnet18")
    if args.model is not None and args.model != checkpoint_model_name:
        raise ValueError(
            f"--model={args.model} does not match checkpoint model '{checkpoint_model_name}'."
        )

    class_names = list(checkpoint["class_names"])
    image_size = int(checkpoint.get("image_size", 224))
    test_dataset = build_dataset(args.data_dir, split="test", image_size=image_size)
    if test_dataset.classes != class_names:
        raise ValueError(
            f"Checkpoint classes {class_names} do not match test classes {test_dataset.classes}."
        )

    loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    y_true, y_pred, positive_scores, rows = collect_predictions(
        model=model,
        loader=loader,
        dataset_samples=test_dataset.samples,
        class_names=class_names,
        device=device,
    )

    metrics = compute_binary_metrics(
        y_true=y_true,
        y_pred=y_pred,
        positive_scores=positive_scores,
        class_names=class_names,
    )
    metrics.update(
        {
            "checkpoint_path": str(args.checkpoint),
            "model_name": checkpoint_model_name,
            "image_size": image_size,
            "class_names": class_names,
            "test_image_count": len(y_true),
        }
    )

    save_json(metrics, args.output_dir / "test_metrics.json")
    save_json(metrics["classification_report"], args.output_dir / "test_classification_report.json")
    save_classification_report_csv(
        metrics["classification_report"],
        args.output_dir / "test_classification_report.csv",
    )
    save_predictions_csv(rows, args.output_dir / "test_predictions.csv")
    plot_confusion_matrix(y_true, y_pred, class_names, args.output_dir / "test_confusion_matrix.png")
    plot_roc_curve(
        y_true,
        positive_scores,
        int(metrics["positive_class_index"]),
        args.output_dir / "test_roc_curve.png",
    )
    plot_precision_recall_curve(
        y_true,
        positive_scores,
        int(metrics["positive_class_index"]),
        args.output_dir / "test_precision_recall_curve.png",
    )

    print(f"accuracy={metrics['accuracy']:.4f}")
    print(f"roc_auc={metrics['roc_auc']}")
    print(f"pr_auc={metrics['pr_auc']}")
    print(f"metrics={args.output_dir / 'test_metrics.json'}")


if __name__ == "__main__":
    main()

