"""Evaluate a trained checkpoint on the test split."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import build_dataset
from .metrics import (
    build_classification_report,
    save_classification_report_csv,
    save_confusion_matrix,
)
from .model import SUPPORTED_MODEL_NAMES, load_checkpoint
from .utils import ensure_dir, get_device, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Lung Colon Vision.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/best_model.pt"))
    parser.add_argument("--model", choices=SUPPORTED_MODEL_NAMES, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/eval"))
    parser.add_argument("--report-dir", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


@torch.inference_mode()
def predict_loader(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[list[int], list[int]]:
    y_true: list[int] = []
    y_pred: list[int] = []

    model.eval()
    for inputs, labels in tqdm(loader, leave=False):
        inputs = inputs.to(device)
        logits = model(inputs)
        predictions = logits.argmax(dim=1).detach().cpu().tolist()
        y_pred.extend(predictions)
        y_true.extend(labels.tolist())

    return y_true, y_pred


def main() -> None:
    args = parse_args()
    output_dir = args.report_dir or args.output_dir
    ensure_dir(output_dir)

    device = get_device()
    model, checkpoint = load_checkpoint(args.checkpoint, device=device)
    class_names = checkpoint["class_names"]
    image_size = int(checkpoint.get("image_size", 224))
    checkpoint_model_name = checkpoint.get("model_name", "resnet18")
    if args.model is not None and args.model != checkpoint_model_name:
        raise ValueError(
            f"--model {args.model} does not match checkpoint model {checkpoint_model_name}."
        )

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
    y_true, y_pred = predict_loader(model, loader, device)

    report = build_classification_report(y_true, y_pred, class_names)
    save_json(report, output_dir / "test_classification_report.json")
    save_classification_report_csv(report, output_dir / "test_classification_report.csv")
    save_confusion_matrix(
        y_true=y_true,
        y_pred=y_pred,
        class_names=class_names,
        output_path=output_dir / "test_confusion_matrix.png",
    )

    print("Test report")
    for class_name in class_names:
        metrics = report[class_name]
        print(
            f"{class_name}: precision={metrics['precision']:.4f} "
            f"recall={metrics['recall']:.4f} f1={metrics['f1-score']:.4f}"
        )
    print(f"accuracy={report['accuracy']:.4f}")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"Error: {exc}") from exc
