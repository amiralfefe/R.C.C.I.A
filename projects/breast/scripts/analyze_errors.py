"""Analyze prediction errors for R.C.C.I.A Breast."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rccia_breast.data import build_dataset  # noqa: E402
from rccia_breast.error_analysis import (  # noqa: E402
    ERROR_TYPES,
    prediction_row,
    save_predictions_csv,
    save_summary_json,
    summarize_predictions,
)
from rccia_breast.gradcam import (  # noqa: E402
    GradCAM,
    denormalize_image,
    get_gradcam_target_layer,
    image_to_tensor,
    overlay_cam,
)
from rccia_breast.model import SUPPORTED_MODEL_NAMES, load_checkpoint  # noqa: E402
from rccia_breast.utils import get_device  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze Breast prediction errors.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model", choices=SUPPORTED_MODEL_NAMES, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/error_analysis"))
    parser.add_argument("--metadata", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-examples", type=int, default=15)
    parser.add_argument("--skip-gradcam", action="store_true")
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else (Path.cwd() / path).resolve()


def _metadata_keys(class_name: str, image_name: str) -> list[str]:
    return [
        f"{class_name}/{image_name}",
        f"{class_name}\\{image_name}",
        image_name,
    ]


def load_metadata_index(metadata_path: Path | None) -> dict[str, dict[str, str]]:
    if metadata_path is None:
        return {}
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    index: dict[str, dict[str, str]] = {}
    with metadata_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            class_name = (row.get("class_name") or "").strip()
            relative_path = (row.get("relative_path") or "").strip()
            output_path = (row.get("output_path") or "").strip()

            candidate_names = [
                relative_path.replace("\\", "/"),
                relative_path.replace("/", "\\"),
            ]
            if output_path:
                candidate_names.append(Path(output_path).name)
            if relative_path:
                candidate_names.append(Path(relative_path).name)

            for key in candidate_names:
                if not key:
                    continue
                index[key] = {
                    "patient_id": (row.get("patient_id") or "").strip(),
                    "magnification": (row.get("magnification") or "").strip(),
                }
                if class_name:
                    index[f"{class_name}/{Path(key).name}"] = index[key]
                    index[f"{class_name}\\{Path(key).name}"] = index[key]
    return index


def metadata_for_sample(
    sample_path: str,
    true_label: str,
    metadata_index: dict[str, dict[str, str]],
) -> dict[str, str]:
    image_name = Path(sample_path).name
    for key in _metadata_keys(true_label, image_name):
        if key in metadata_index:
            return metadata_index[key]
    return {"patient_id": "", "magnification": ""}


@torch.inference_mode()
def collect_prediction_rows(
    model: torch.nn.Module,
    loader: DataLoader,
    dataset_samples: list[tuple[str, int]],
    class_names: list[str],
    device: torch.device,
    metadata_index: dict[str, dict[str, str]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    sample_offset = 0

    model.eval()
    for inputs, labels in tqdm(loader, leave=False):
        batch_size = labels.shape[0]
        inputs = inputs.to(device)
        logits = model(inputs)
        probabilities = torch.softmax(logits, dim=1).detach().cpu()
        predicted_indices = probabilities.argmax(dim=1)

        batch_samples = dataset_samples[sample_offset : sample_offset + batch_size]
        for local_index, (image_path, true_index) in enumerate(batch_samples):
            predicted_index = int(predicted_indices[local_index].item())
            true_label = class_names[int(true_index)]
            predicted_label = class_names[predicted_index]
            probability_values = [float(value) for value in probabilities[local_index].tolist()]
            confidence = probability_values[predicted_index]
            metadata = metadata_for_sample(
                sample_path=image_path,
                true_label=true_label,
                metadata_index=metadata_index,
            )
            rows.append(
                prediction_row(
                    image_path=image_path,
                    true_label=true_label,
                    predicted_label=predicted_label,
                    confidence=confidence,
                    probabilities=probability_values,
                    class_names=class_names,
                    patient_id=metadata.get("patient_id", ""),
                    magnification=metadata.get("magnification", ""),
                )
            )
        sample_offset += batch_size

    return rows


def save_example_image(row: dict[str, Any], output_path: Path, max_size: int = 512) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.open(row["image_path"]).convert("RGB")
    image.thumbnail((max_size, max_size))
    image.save(output_path)


def save_gradcam_image(
    row: dict[str, Any],
    output_path: Path,
    model: torch.nn.Module,
    class_names: list[str],
    image_size: int,
    device: torch.device,
) -> None:
    image = Image.open(row["image_path"]).convert("RGB")
    tensor = image_to_tensor(image, image_size=image_size, device=device)
    target_index = class_names.index(row["predicted_label"])
    target_layer = get_gradcam_target_layer(model)
    gradcam = GradCAM(model=model, target_layer=target_layer)
    try:
        cam, _ = gradcam.generate(tensor, target_index=target_index)
    finally:
        gradcam.remove_hooks()

    overlay = overlay_cam(denormalize_image(tensor), cam)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.uint8(overlay * 255)).save(output_path)


def write_examples(
    rows: list[dict[str, Any]],
    output_dir: Path,
    model: torch.nn.Module,
    class_names: list[str],
    image_size: int,
    device: torch.device,
    max_examples: int,
    skip_gradcam: bool,
) -> dict[str, Any]:
    examples_dir = output_dir / "examples"
    error_rows = [row for row in rows if not row["is_correct"]]
    correct_rows = [row for row in rows if row["is_correct"]]

    selections: list[tuple[str, list[dict[str, Any]]]] = [
        ("false_positive", [row for row in error_rows if row["error_type"] == "false_positive"]),
        ("false_negative", [row for row in error_rows if row["error_type"] == "false_negative"]),
        (
            "most_confident_error",
            sorted(error_rows, key=lambda row: float(row["confidence"]), reverse=True),
        ),
        (
            "correct_low_confidence",
            sorted(correct_rows, key=lambda row: float(row["confidence"])),
        ),
    ]

    exported: list[dict[str, Any]] = []
    gradcam_errors: list[str] = []

    for prefix, selected_rows in selections:
        for index, row in enumerate(selected_rows[:max_examples], start=1):
            filename = f"{prefix}_{index:03d}.png"
            image_path = examples_dir / filename
            save_example_image(row, image_path)

            gradcam_path = examples_dir / f"{prefix}_{index:03d}_gradcam.png"
            gradcam_status = "skipped" if skip_gradcam else "created"
            if not skip_gradcam:
                try:
                    save_gradcam_image(
                        row=row,
                        output_path=gradcam_path,
                        model=model,
                        class_names=class_names,
                        image_size=image_size,
                        device=device,
                    )
                except (RuntimeError, ValueError, LookupError, OSError) as exc:
                    gradcam_status = "failed"
                    gradcam_errors.append(f"{filename}: {exc}")

            exported.append(
                {
                    "kind": prefix,
                    "image_path": str(image_path),
                    "gradcam_path": str(gradcam_path) if gradcam_path.exists() else None,
                    "gradcam_status": gradcam_status,
                    "true_label": row["true_label"],
                    "predicted_label": row["predicted_label"],
                    "confidence": row["confidence"],
                    "error_type": row["error_type"],
                    "patient_id": row.get("patient_id", ""),
                    "magnification": row.get("magnification", ""),
                }
            )

    return {
        "examples_dir": str(examples_dir),
        "exported_examples": exported,
        "gradcam_errors": gradcam_errors,
    }


def main() -> None:
    args = parse_args()
    data_dir = resolve_path(args.data_dir)
    checkpoint_path = resolve_path(args.checkpoint)
    output_dir = resolve_path(args.output_dir)
    metadata_path = resolve_path(args.metadata) if args.metadata is not None else None
    output_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()
    model, checkpoint = load_checkpoint(checkpoint_path, device=device)
    checkpoint_model_name = checkpoint.get("model_name", "resnet18")
    if args.model is not None and args.model != checkpoint_model_name:
        raise ValueError(
            f"--model={args.model} does not match checkpoint model '{checkpoint_model_name}'."
        )

    class_names = list(checkpoint["class_names"])
    image_size = int(checkpoint.get("image_size", 224))
    test_dataset = build_dataset(data_dir, split="test", image_size=image_size)
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
    metadata_index = load_metadata_index(metadata_path)

    rows = collect_prediction_rows(
        model=model,
        loader=loader,
        dataset_samples=test_dataset.samples,
        class_names=class_names,
        device=device,
        metadata_index=metadata_index,
    )
    summary = summarize_predictions(rows, max_examples=args.max_examples)
    summary.update(
        {
            "checkpoint_path": str(checkpoint_path),
            "model_name": checkpoint_model_name,
            "image_size": image_size,
            "class_names": class_names,
            "error_type_definitions": list(ERROR_TYPES),
            "data_dir": str(data_dir),
            "metadata_path": str(metadata_path) if metadata_path else None,
            "metadata_available": bool(metadata_index),
            "missing_metadata_count": sum(
                1 for row in rows if not row.get("patient_id") and not row.get("magnification")
            ),
        }
    )

    examples_summary = write_examples(
        rows=rows,
        output_dir=output_dir,
        model=model,
        class_names=class_names,
        image_size=image_size,
        device=device,
        max_examples=args.max_examples,
        skip_gradcam=args.skip_gradcam,
    )
    summary.update(examples_summary)

    save_predictions_csv(rows, output_dir / "predictions.csv")
    save_summary_json(summary, output_dir / "summary.json")

    print(f"Model: {checkpoint_model_name}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Total images: {summary['total_images']}")
    print(f"Correct: {summary['correct_count']}")
    print(f"Errors: {summary['error_count']}")
    print(f"False positives: {summary['false_positive_count']}")
    print(f"False negatives: {summary['false_negative_count']}")
    print(f"Accuracy: {summary['accuracy']:.4f}")
    print(f"Average confidence correct: {summary['average_confidence_correct']}")
    print(f"Average confidence errors: {summary['average_confidence_errors']}")
    print(f"Predictions: {output_dir / 'predictions.csv'}")
    print(f"Summary: {output_dir / 'summary.json'}")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError, RuntimeError, OSError) as exc:
        raise SystemExit(f"Error: {exc}") from exc
