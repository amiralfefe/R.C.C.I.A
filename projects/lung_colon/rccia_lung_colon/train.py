"""Train the image classification model."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import torch
from torch import nn
from torch.optim import AdamW
from tqdm import tqdm

from rccia_lung_colon.config import TrainConfig
from rccia_lung_colon.data import build_dataloaders, class_counts
from rccia_lung_colon.metrics import save_training_curves
from rccia_lung_colon.model import SUPPORTED_MODEL_NAMES, create_model, save_checkpoint
from rccia_lung_colon.utils import ensure_dir, get_device, save_json, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Lung Colon Vision.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--report-dir", type=Path, default=None)
    parser.add_argument("--model", choices=SUPPORTED_MODEL_NAMES, default="resnet18")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def run_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: AdamW | None = None,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    context = torch.enable_grad() if is_train else torch.inference_mode()
    with context:
        for inputs, labels in tqdm(loader, leave=False):
            inputs = inputs.to(device)
            labels = labels.to(device)

            if is_train:
                optimizer.zero_grad(set_to_none=True)

            logits = model(inputs)
            loss = criterion(logits, labels)

            if is_train:
                loss.backward()
                optimizer.step()

            predictions = logits.argmax(dim=1)
            batch_size = labels.size(0)
            total_loss += float(loss.item()) * batch_size
            total_correct += int((predictions == labels).sum().item())
            total_samples += batch_size

    return {
        "loss": total_loss / max(total_samples, 1),
        "accuracy": total_correct / max(total_samples, 1),
    }


def config_to_json(config: TrainConfig) -> dict:
    payload = asdict(config)
    payload["data_dir"] = str(config.data_dir)
    payload["output_dir"] = str(config.output_dir)
    payload["report_dir"] = str(config.report_dir)
    return payload


def main() -> None:
    args = parse_args()
    config = TrainConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        report_dir=args.report_dir or args.output_dir / "train",
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        image_size=args.image_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        pretrained=not args.no_pretrained,
        seed=args.seed,
    )

    set_seed(config.seed)
    ensure_dir(config.output_dir)
    ensure_dir(config.report_dir)

    device = get_device()
    loaders, class_names = build_dataloaders(
        data_dir=config.data_dir,
        image_size=config.image_size,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
    )

    model = create_model(
        num_classes=len(class_names),
        model_name=config.model_name,
        pretrained=config.pretrained,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    best_val_accuracy = -1.0
    history: list[dict[str, float | int]] = []

    save_json(
        {
            "config": config_to_json(config),
            "class_names": class_names,
            "class_counts": {
                split: class_counts(loader.dataset) for split, loader in loaders.items()
            },
        },
        config.report_dir / "training_setup.json",
    )

    for epoch in range(1, config.epochs + 1):
        print(f"Epoch {epoch}/{config.epochs}")
        train_metrics = run_epoch(model, loaders["train"], criterion, device, optimizer)
        val_metrics = run_epoch(model, loaders["val"], criterion, device)

        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
        }
        history.append(row)
        print(
            "train_loss={train_loss:.4f} train_acc={train_accuracy:.4f} "
            "val_loss={val_loss:.4f} val_acc={val_accuracy:.4f}".format(**row)
        )

        if val_metrics["accuracy"] > best_val_accuracy:
            best_val_accuracy = val_metrics["accuracy"]
            save_checkpoint(
                path=config.output_dir / "best_model.pt",
                model=model,
                class_names=class_names,
                image_size=config.image_size,
                model_name=config.model_name,
                metrics={
                    "val_loss": val_metrics["loss"],
                    "val_accuracy": val_metrics["accuracy"],
                    "epoch": float(epoch),
                },
            )

    save_json({"history": history}, config.report_dir / "training_history.json")
    save_training_curves(history, config.report_dir)
    print(f"Best validation accuracy: {best_val_accuracy:.4f}")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"Error: {exc}") from exc
