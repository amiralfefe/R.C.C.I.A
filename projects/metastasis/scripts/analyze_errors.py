"""Analyze errors and metastatic decision thresholds for R.C.C.I.A Metastasis."""

from __future__ import annotations

import argparse
import shutil
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

from rccia_metastasis.data import build_dataset  # noqa: E402
from rccia_metastasis.error_analysis import (  # noqa: E402
    NEGATIVE_CLASS,
    POSITIVE_CLASS,
    compute_threshold_analysis,
    plot_threshold_analysis,
    prediction_row,
    save_json,
    save_rows_csv,
    summarize_predictions,
)
from rccia_metastasis.gradcam import (  # noqa: E402
    GradCAM,
    denormalize_image,
    get_gradcam_target_layer,
    image_to_tensor,
    overlay_cam,
)
from rccia_metastasis.model import SUPPORTED_MODEL_NAMES, load_checkpoint  # noqa: E402
from rccia_metastasis.utils import get_device  # noqa: E402


DEFAULT_THRESHOLDS = (0.30, 0.40, 0.50, 0.60, 0.70)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze Metastasis prediction errors.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model", choices=SUPPORTED_MODEL_NAMES, default=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/error_analysis"))
    parser.add_argument("--thresholds", nargs="+", type=float, default=list(DEFAULT_THRESHOLDS))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-examples", type=int, default=20)
    parser.add_argument("--skip-gradcam", action="store_true")
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else (Path.cwd() / path).resolve()


def fallback_checkpoint(preferred_path: Path) -> Path:
    if preferred_path.exists():
        return preferred_path

    candidates = [
        PROJECT_ROOT / "outputs" / "model_comparison" / "efficientnet_b0" / "best_model.pt",
        PROJECT_ROOT / "outputs" / "model_comparison" / "resnet18" / "best_model.pt",
        PROJECT_ROOT / "outputs" / "best_model.pt",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Checkpoint not found: {preferred_path}")


@torch.inference_mode()
def collect_prediction_rows(
    model: torch.nn.Module,
    loader: DataLoader,
    dataset_samples: list[tuple[str, int]],
    class_names: list[str],
    device: torch.device,
) -> list[dict[str, Any]]:
    if NEGATIVE_CLASS not in class_names or POSITIVE_CLASS not in class_names:
        raise ValueError(
            f"Metastasis error analysis expects {NEGATIVE_CLASS!r} and {POSITIVE_CLASS!r}; "
            f"got {class_names}."
        )

    negative_index = class_names.index(NEGATIVE_CLASS)
    positive_index = class_names.index(POSITIVE_CLASS)
    rows: list[dict[str, Any]] = []
    sample_offset = 0

    model.eval()
    for inputs, labels in tqdm(loader, leave=False):
        batch_size = labels.shape[0]
        inputs = inputs.to(device)
        logits = model(inputs)
        probabilities = torch.softmax(logits, dim=1).detach().cpu()
        batch_samples = dataset_samples[sample_offset : sample_offset + batch_size]

        for local_index, (image_path, true_index) in enumerate(batch_samples):
            rows.append(
                prediction_row(
                    image_path=image_path,
                    true_label=class_names[int(true_index)],
                    prob_non_metastatic=float(probabilities[local_index, negative_index].item()),
                    prob_metastatic=float(probabilities[local_index, positive_index].item()),
                    threshold=0.5,
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
    image_size: int,
    device: torch.device,
) -> None:
    image = Image.open(row["image_path"]).convert("RGB")
    tensor = image_to_tensor(image, image_size=image_size, device=device)
    target_index = 0 if row["predicted_label"] == POSITIVE_CLASS else 1
    target_layer = get_gradcam_target_layer(model)
    gradcam = GradCAM(model=model, target_layer=target_layer)
    try:
        cam, _ = gradcam.generate(tensor, target_index=target_index)
    finally:
        gradcam.remove_hooks()

    overlay = overlay_cam(denormalize_image(tensor), cam)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.uint8(overlay * 255)).save(output_path)


def example_groups(rows: list[dict[str, Any]]) -> list[tuple[str, list[dict[str, Any]]]]:
    error_rows = [row for row in rows if not row["is_correct"]]
    correct_rows = [row for row in rows if row["is_correct"]]
    return [
        ("false_positive", [row for row in error_rows if row["error_type"] == "false_positive"]),
        ("false_negative", [row for row in error_rows if row["error_type"] == "false_negative"]),
        (
            "most_confident_error",
            sorted(error_rows, key=lambda row: float(row["confidence"]), reverse=True),
        ),
        (
            "low_confidence_correct",
            sorted(correct_rows, key=lambda row: float(row["confidence"])),
        ),
    ]


def write_examples(
    rows: list[dict[str, Any]],
    output_dir: Path,
    model: torch.nn.Module,
    image_size: int,
    device: torch.device,
    max_examples: int,
    skip_gradcam: bool,
) -> dict[str, Any]:
    examples_dir = output_dir / "examples"
    if examples_dir.exists():
        shutil.rmtree(examples_dir)
    examples_dir.mkdir(parents=True, exist_ok=True)

    exported: list[dict[str, Any]] = []
    gradcam_errors: list[str] = []
    for prefix, selected_rows in example_groups(rows):
        for index, row in enumerate(selected_rows[:max_examples], start=1):
            image_path = examples_dir / f"{prefix}_{index:03d}.png"
            save_example_image(row, image_path)

            gradcam_path = examples_dir / f"{prefix}_{index:03d}_gradcam.png"
            gradcam_status = "skipped" if skip_gradcam else "created"
            if not skip_gradcam:
                try:
                    save_gradcam_image(
                        row=row,
                        output_path=gradcam_path,
                        model=model,
                        image_size=image_size,
                        device=device,
                    )
                except (RuntimeError, ValueError, LookupError, OSError) as exc:
                    gradcam_status = "failed"
                    gradcam_errors.append(f"{image_path.name}: {exc}")

            exported.append(
                {
                    "kind": prefix,
                    "image_path": str(image_path),
                    "gradcam_path": str(gradcam_path) if gradcam_path.exists() else None,
                    "gradcam_status": gradcam_status,
                    "true_label": row["true_label"],
                    "predicted_label": row["predicted_label"],
                    "prob_metastatic": row["prob_metastatic"],
                    "confidence": row["confidence"],
                    "error_type": row["error_type"],
                }
            )

    return {
        "examples_dir": str(examples_dir),
        "exported_examples": exported,
        "gradcam_generated": any(example.get("gradcam_path") for example in exported),
        "gradcam_errors": gradcam_errors,
    }


def main() -> None:
    args = parse_args()
    data_dir = resolve_path(args.data_dir)
    checkpoint_path = fallback_checkpoint(resolve_path(args.checkpoint))
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()
    model, checkpoint = load_checkpoint(checkpoint_path, device=device)
    checkpoint_model_name = checkpoint.get("model_name", "resnet18")
    if args.model is not None and args.model != checkpoint_model_name:
        raise ValueError(
            f"--model={args.model} does not match checkpoint model '{checkpoint_model_name}'."
        )

    class_names = list(checkpoint["class_names"])
    image_size = int(args.image_size or checkpoint.get("image_size", 96))
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
    rows = collect_prediction_rows(
        model=model,
        loader=loader,
        dataset_samples=test_dataset.samples,
        class_names=class_names,
        device=device,
    )
    threshold_rows = compute_threshold_analysis(rows, thresholds=args.thresholds)
    examples_summary = write_examples(
        rows=rows,
        output_dir=output_dir,
        model=model,
        image_size=image_size,
        device=device,
        max_examples=args.max_examples,
        skip_gradcam=args.skip_gradcam,
    )
    summary = summarize_predictions(
        rows,
        thresholds=args.thresholds,
        max_examples=args.max_examples,
    )
    summary.update(
        {
            "model": checkpoint_model_name,
            "checkpoint_path": str(checkpoint_path),
            "data_dir": str(data_dir),
            "image_size": image_size,
            "class_names": class_names,
            "thresholds": [float(threshold) for threshold in args.thresholds],
            **examples_summary,
        }
    )

    save_rows_csv(rows, output_dir / "error_cases.csv")
    save_json(summary, output_dir / "error_summary.json")
    save_rows_csv(threshold_rows, output_dir / "threshold_analysis.csv")
    plot_threshold_analysis(threshold_rows, output_dir / "threshold_analysis.png")

    print(f"model={summary['model']}")
    print(f"checkpoint={summary['checkpoint_path']}")
    print(f"total_images={summary['total_images']}")
    print(f"accuracy_at_0_50={summary['accuracy_at_0_50']:.4f}")
    print(f"false_positives={summary['false_positive_count']}")
    print(f"false_negatives={summary['false_negative_count']}")
    print(f"average_confidence_correct={summary['average_confidence_correct']}")
    print(f"average_confidence_errors={summary['average_confidence_errors']}")
    print(f"roc_auc={summary['roc_auc']}")
    print(f"pr_auc={summary['pr_auc']}")
    print(f"error_cases={output_dir / 'error_cases.csv'}")
    print(f"summary={output_dir / 'error_summary.json'}")
    print(f"thresholds={output_dir / 'threshold_analysis.csv'}")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError, RuntimeError, OSError) as exc:
        raise SystemExit(f"Error: {exc}") from exc
