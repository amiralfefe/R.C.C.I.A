"""Error analysis helpers for model predictions on the test split."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader
from tqdm import tqdm

from cancer_cell_vision.data import build_dataset
from cancer_cell_vision.gradcam import (
    GradCAM,
    denormalize_image,
    get_gradcam_target_layer,
    image_to_tensor,
    overlay_cam,
)
from cancer_cell_vision.model import load_checkpoint
from cancer_cell_vision.utils import ensure_dir, get_device, read_json, save_json


PREDICTION_COLUMNS = [
    "image_path",
    "true_label",
    "predicted_label",
    "confidence",
    "prob_normal",
    "prob_leukemia_blast",
    "is_correct",
    "error_type",
]


@dataclass(frozen=True)
class ErrorAnalysisConfig:
    data_dir: Path
    checkpoint: Path
    output_dir: Path = Path("outputs/error_analysis")
    model_name: str | None = None
    batch_size: int = 16
    num_workers: int = 0
    max_examples: int = 12
    generate_gradcam: bool = True


def resolve_binary_labels(class_names: list[str]) -> tuple[str, str]:
    normal_label = "normal" if "normal" in class_names else class_names[0]
    if "leukemia_blast" in class_names:
        positive_label = "leukemia_blast"
    else:
        positive_label = next(label for label in class_names if label != normal_label)
    return normal_label, positive_label


def error_type_for(
    true_label: str,
    predicted_label: str,
    normal_label: str,
    positive_label: str,
) -> str:
    if true_label == predicted_label:
        return "correct"
    if true_label == normal_label and predicted_label == positive_label:
        return "false_positive"
    if true_label == positive_label and predicted_label == normal_label:
        return "false_negative"
    return "false_positive" if predicted_label == positive_label else "false_negative"


def probability_for(
    probabilities: list[float],
    class_names: list[str],
    class_name: str,
) -> float:
    try:
        return probabilities[class_names.index(class_name)]
    except ValueError:
        return 0.0


@torch.inference_mode()
def collect_predictions(
    model: torch.nn.Module,
    dataset,
    class_names: list[str],
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> list[dict[str, Any]]:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    normal_label, positive_label = resolve_binary_labels(class_names)
    rows: list[dict[str, Any]] = []
    sample_offset = 0

    model.eval()
    for inputs, labels in tqdm(loader, leave=False):
        inputs = inputs.to(device)
        logits = model(inputs)
        probabilities = torch.softmax(logits, dim=1).detach().cpu()

        for index, label in enumerate(labels.tolist()):
            sample_path = Path(dataset.samples[sample_offset + index][0])
            probability_values = [float(value) for value in probabilities[index].tolist()]
            predicted_index = int(probabilities[index].argmax().item())
            true_label = class_names[int(label)]
            predicted_label = class_names[predicted_index]
            confidence = float(probability_values[predicted_index])
            is_correct = true_label == predicted_label

            rows.append(
                {
                    "image_path": str(sample_path),
                    "true_label": true_label,
                    "predicted_label": predicted_label,
                    "confidence": confidence,
                    "prob_normal": probability_for(probability_values, class_names, normal_label),
                    "prob_leukemia_blast": probability_for(
                        probability_values,
                        class_names,
                        positive_label,
                    ),
                    "is_correct": is_correct,
                    "error_type": error_type_for(
                        true_label=true_label,
                        predicted_label=predicted_label,
                        normal_label=normal_label,
                        positive_label=positive_label,
                    ),
                }
            )
        sample_offset += len(labels)

    return rows


def mean_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def compact_prediction_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "image_path": row["image_path"],
        "true_label": row["true_label"],
        "predicted_label": row["predicted_label"],
        "confidence": float(row["confidence"]),
        "prob_normal": float(row["prob_normal"]),
        "prob_leukemia_blast": float(row["prob_leukemia_blast"]),
        "error_type": row["error_type"],
    }


def build_summary(rows: list[dict[str, Any]], max_items: int = 10) -> dict[str, Any]:
    total_images = len(rows)
    correct_rows = [row for row in rows if row["is_correct"]]
    error_rows = [row for row in rows if not row["is_correct"]]
    false_positive_rows = [row for row in rows if row["error_type"] == "false_positive"]
    false_negative_rows = [row for row in rows if row["error_type"] == "false_negative"]

    return {
        "total_images": total_images,
        "correct_count": len(correct_rows),
        "error_count": len(error_rows),
        "false_positive_count": len(false_positive_rows),
        "false_negative_count": len(false_negative_rows),
        "accuracy": len(correct_rows) / max(total_images, 1),
        "average_confidence_correct": mean_or_none(
            [float(row["confidence"]) for row in correct_rows]
        ),
        "average_confidence_errors": mean_or_none(
            [float(row["confidence"]) for row in error_rows]
        ),
        "most_confident_errors": [
            compact_prediction_row(row)
            for row in sorted(error_rows, key=lambda item: item["confidence"], reverse=True)[
                :max_items
            ]
        ],
        "least_confident_correct": [
            compact_prediction_row(row)
            for row in sorted(correct_rows, key=lambda item: item["confidence"])[:max_items]
        ],
        "class_distribution": {
            "true": dict(Counter(row["true_label"] for row in rows)),
            "predicted": dict(Counter(row["predicted_label"] for row in rows)),
        },
    }


def save_display_image(image: Image.Image, output_path: Path, max_size: int = 512) -> None:
    output = image.convert("RGB").copy()
    output.thumbnail((max_size, max_size))
    output.save(output_path)


def save_annotated_image(image: Image.Image, row: dict[str, Any], output_path: Path) -> None:
    preview = image.convert("RGB").copy()
    preview.thumbnail((512, 512))

    lines = [
        f"True: {row['true_label']}",
        f"Pred: {row['predicted_label']}",
        f"Confidence: {row['confidence']:.2%}",
        f"P(normal): {row['prob_normal']:.2%}",
        f"P(leukemia_blast): {row['prob_leukemia_blast']:.2%}",
    ]
    line_height = 18
    padding = 10
    text_height = padding * 2 + line_height * len(lines)
    canvas = Image.new("RGB", (preview.width, preview.height + text_height), "white")
    canvas.paste(preview, (0, 0))

    draw = ImageDraw.Draw(canvas)
    y = preview.height + padding
    for line in lines:
        draw.text((padding, y), line, fill=(20, 20, 20))
        y += line_height

    canvas.save(output_path)


def save_gradcam_image(
    image: Image.Image,
    row: dict[str, Any],
    model: torch.nn.Module,
    class_names: list[str],
    image_size: int,
    device: torch.device,
    output_path: Path,
) -> None:
    gradcam = None
    try:
        tensor = image_to_tensor(image, image_size=image_size, device=device)
        target_index = class_names.index(row["predicted_label"])
        gradcam = GradCAM(model, get_gradcam_target_layer(model))
        cam, _ = gradcam.generate(tensor, target_index=target_index)
        overlay = overlay_cam(denormalize_image(tensor), cam)
        output = Image.fromarray(np.uint8(overlay * 255))
        output.save(output_path)
    finally:
        if gradcam is not None:
            gradcam.remove_hooks()


def selected_examples(rows: list[dict[str, Any]], max_examples: int) -> list[tuple[str, dict[str, Any]]]:
    if max_examples <= 0:
        return []

    per_group = max(1, max_examples // 4)
    correct_rows = [row for row in rows if row["is_correct"]]
    groups = {
        "false_positive": sorted(
            [row for row in rows if row["error_type"] == "false_positive"],
            key=lambda item: item["confidence"],
            reverse=True,
        ),
        "false_negative": sorted(
            [row for row in rows if row["error_type"] == "false_negative"],
            key=lambda item: item["confidence"],
            reverse=True,
        ),
        "correct_high_confidence": sorted(
            correct_rows,
            key=lambda item: item["confidence"],
            reverse=True,
        ),
        "correct_low_confidence": sorted(correct_rows, key=lambda item: item["confidence"]),
    }

    selected: list[tuple[str, dict[str, Any]]] = []
    for group_name, group_rows in groups.items():
        for row in group_rows[:per_group]:
            if len(selected) >= max_examples:
                return selected
            selected.append((group_name, row))
    return selected


def export_examples(
    rows: list[dict[str, Any]],
    output_dir: Path,
    max_examples: int,
    generate_gradcam: bool,
    model: torch.nn.Module,
    class_names: list[str],
    image_size: int,
    device: torch.device,
) -> dict[str, list[str]]:
    examples_dir = ensure_dir(output_dir / "examples")
    counters: Counter[str] = Counter()
    generated_files: list[str] = []
    gradcam_warnings: list[str] = []

    for group_name, row in selected_examples(rows, max_examples=max_examples):
        counters[group_name] += 1
        stem = f"{group_name}_{counters[group_name]:03d}"
        image_path = Path(row["image_path"])
        try:
            image = Image.open(image_path).convert("RGB")
        except OSError as exc:
            gradcam_warnings.append(f"Could not open {image_path}: {exc}")
            continue

        raw_path = examples_dir / f"{stem}.png"
        annotated_path = examples_dir / f"{stem}_annotated.png"
        save_display_image(image, raw_path)
        save_annotated_image(image, row, annotated_path)
        generated_files.extend([str(raw_path), str(annotated_path)])

        if generate_gradcam and group_name in {"false_positive", "false_negative"}:
            gradcam_path = examples_dir / f"{stem}_gradcam.png"
            try:
                save_gradcam_image(
                    image=image,
                    row=row,
                    model=model,
                    class_names=class_names,
                    image_size=image_size,
                    device=device,
                    output_path=gradcam_path,
                )
                generated_files.append(str(gradcam_path))
            except Exception as exc:  # pragma: no cover - defensive output path.
                gradcam_warnings.append(f"Grad-CAM failed for {image_path}: {exc}")

    return {
        "generated_files": generated_files,
        "gradcam_warnings": gradcam_warnings,
    }


def run_error_analysis(config: ErrorAnalysisConfig) -> dict[str, Any]:
    output_dir = ensure_dir(config.output_dir)
    device = get_device()
    model, checkpoint = load_checkpoint(config.checkpoint, device=device)
    class_names = list(checkpoint["class_names"])
    checkpoint_model = str(checkpoint.get("model_name", "resnet18"))
    image_size = int(checkpoint.get("image_size", 224))

    if config.model_name is not None and config.model_name != checkpoint_model:
        raise ValueError(
            f"Requested model '{config.model_name}' but checkpoint stores '{checkpoint_model}'."
        )

    test_dataset = build_dataset(config.data_dir, split="test", image_size=image_size)
    if test_dataset.classes != class_names:
        raise ValueError(
            f"Checkpoint classes {class_names} do not match test classes {test_dataset.classes}."
        )

    rows = collect_predictions(
        model=model,
        dataset=test_dataset,
        class_names=class_names,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        device=device,
    )
    predictions_path = output_dir / "predictions.csv"
    pd.DataFrame(rows, columns=PREDICTION_COLUMNS).to_csv(predictions_path, index=False)

    summary = build_summary(rows, max_items=min(10, max(config.max_examples, 1)))
    summary["checkpoint_path"] = str(config.checkpoint)
    summary["model_name"] = checkpoint_model
    summary["image_size"] = image_size
    summary["predictions_path"] = str(predictions_path)

    exports = export_examples(
        rows=rows,
        output_dir=output_dir,
        max_examples=config.max_examples,
        generate_gradcam=config.generate_gradcam,
        model=model,
        class_names=class_names,
        image_size=image_size,
        device=device,
    )
    summary.update(exports)
    save_json(summary, output_dir / "summary.json")
    return summary


def load_error_analysis_artifacts(output_dir: Path) -> dict[str, Any]:
    summary_path = output_dir / "summary.json"
    examples_dir = output_dir / "examples"
    if not summary_path.exists():
        return {
            "available": False,
            "summary": None,
            "summary_path": summary_path,
            "examples_dir": examples_dir,
            "examples": [],
        }

    examples = sorted(examples_dir.glob("*.png")) if examples_dir.exists() else []
    return {
        "available": True,
        "summary": read_json(summary_path),
        "summary_path": summary_path,
        "examples_dir": examples_dir,
        "examples": examples,
    }
