"""Generate prediction rows and a compact error-analysis summary."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from rccia_breast.data import build_dataset
from rccia_breast.error_analysis import (
    prediction_row,
    save_predictions_csv,
    save_summary_json,
    summarize_predictions,
)
from rccia_breast.model import load_checkpoint
from rccia_breast.utils import ensure_dir, get_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze Breast prediction errors.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/best_model.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/error_analysis"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-examples", type=int, default=15)
    return parser.parse_args()


@torch.inference_mode()
def collect_prediction_rows(
    model: torch.nn.Module,
    loader: DataLoader,
    dataset,
    class_names: list[str],
    device: torch.device,
) -> list[dict]:
    rows: list[dict] = []
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
            confidence = float(probabilities[index, predictions[index]].item())
            rows.append(
                prediction_row(
                    image_path=str(sample_path),
                    true_label=true_label,
                    predicted_label=predicted_label,
                    confidence=confidence,
                    probabilities=[float(value) for value in probabilities[index].tolist()],
                    class_names=class_names,
                )
            )
        sample_offset += labels.size(0)
    return rows


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_dir)

    device = get_device()
    model, checkpoint = load_checkpoint(args.checkpoint, device=device)
    class_names = checkpoint["class_names"]
    image_size = int(checkpoint.get("image_size", 224))

    dataset = build_dataset(args.data_dir, split="test", image_size=image_size)
    if dataset.classes != class_names:
        raise ValueError(
            f"Checkpoint classes {class_names} do not match test classes {dataset.classes}."
        )

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    rows = collect_prediction_rows(model, loader, dataset, class_names, device)
    summary = summarize_predictions(rows, max_examples=args.max_examples)

    save_predictions_csv(rows, args.output_dir / "predictions.csv")
    save_summary_json(summary, args.output_dir / "summary.json")

    print(
        f"Analyzed {summary['total_images']} images: "
        f"{summary['correct_count']} correct, {summary['error_count']} errors, "
        f"accuracy={summary['accuracy']:.4f}"
    )


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"Error: {exc}") from exc
