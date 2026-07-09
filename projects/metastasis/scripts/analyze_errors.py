"""Analyze false positives and false negatives for Metastasis Vision."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rccia_metastasis.data import build_dataset  # noqa: E402
from rccia_metastasis.error_analysis import prediction_row, summarize_predictions  # noqa: E402
from rccia_metastasis.metrics import positive_class_index  # noqa: E402
from rccia_metastasis.model import SUPPORTED_MODEL_NAMES, load_checkpoint  # noqa: E402
from rccia_metastasis.utils import get_device  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze Metastasis prediction errors.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model", choices=SUPPORTED_MODEL_NAMES, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/error_analysis"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


@torch.inference_mode()
def collect_rows(
    model: torch.nn.Module,
    loader: DataLoader,
    dataset_samples: list[tuple[str, int]],
    class_names: list[str],
    device: torch.device,
) -> list[dict]:
    rows: list[dict] = []
    sample_offset = 0
    positive_index = positive_class_index(class_names)

    model.eval()
    for inputs, labels in loader:
        batch_size = labels.shape[0]
        inputs = inputs.to(device)
        logits = model(inputs)
        probabilities = torch.softmax(logits, dim=1).detach().cpu()
        predicted_indices = probabilities.argmax(dim=1)
        batch_samples = dataset_samples[sample_offset : sample_offset + batch_size]

        for local_index, (image_path, true_index) in enumerate(batch_samples):
            predicted_index = int(predicted_indices[local_index].item())
            rows.append(
                prediction_row(
                    image_path=image_path,
                    true_label=class_names[int(true_index)],
                    predicted_label=class_names[predicted_index],
                    confidence=float(probabilities[local_index, predicted_index].item()),
                    prob_metastatic=float(probabilities[local_index, positive_index].item()),
                )
            )
        sample_offset += batch_size
    return rows


def save_predictions(rows: list[dict], output_path: Path) -> None:
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
    dataset = build_dataset(args.data_dir, split="test", image_size=image_size)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    rows = collect_rows(model, loader, dataset.samples, class_names, device)
    summary = summarize_predictions(rows)
    summary.update({"model_name": checkpoint_model_name, "checkpoint_path": str(args.checkpoint)})

    save_predictions(rows, args.output_dir / "predictions.csv")
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
        file.write("\n")

    print(f"total_images={summary['total_images']}")
    print(f"errors={summary['error_count']}")
    print(f"false_positives={summary['false_positive_count']}")
    print(f"false_negatives={summary['false_negative_count']}")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError, RuntimeError, OSError) as exc:
        raise SystemExit(f"Error: {exc}") from exc
