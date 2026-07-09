"""Train a Metastasis Vision classifier."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn

from .data import build_dataloaders
from .model import SUPPORTED_MODEL_NAMES, create_model, save_checkpoint
from .utils import get_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Metastasis classifier.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--model", choices=SUPPORTED_MODEL_NAMES, default="resnet18")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--no-pretrained", action="store_true")
    return parser.parse_args()


def run_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)

    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            logits = model(inputs)
            loss = criterion(logits, labels)
            if is_train:
                loss.backward()
                optimizer.step()

        running_loss += float(loss.item()) * labels.size(0)
        predictions = logits.argmax(dim=1)
        correct += int((predictions == labels).sum().item())
        total += int(labels.size(0))

    return {
        "loss": running_loss / total if total else 0.0,
        "accuracy": correct / total if total else 0.0,
    }


def plot_history(history: list[dict[str, float]], output_dir: Path) -> None:
    train_dir = output_dir / "train"
    train_dir.mkdir(parents=True, exist_ok=True)
    epochs = [int(row["epoch"]) for row in history]

    plt.figure(figsize=(6, 4))
    plt.plot(epochs, [row["train_loss"] for row in history], label="train")
    plt.plot(epochs, [row["val_loss"] for row in history], label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(train_dir / "training_loss.png")
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.plot(epochs, [row["train_accuracy"] for row in history], label="train")
    plt.plot(epochs, [row["val_accuracy"] for row in history], label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(train_dir / "training_accuracy.png")
    plt.close()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()
    loaders, class_names = build_dataloaders(
        data_dir=args.data_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    model = create_model(
        num_classes=len(class_names),
        model_name=args.model,
        pretrained=not args.no_pretrained,
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    history: list[dict[str, float]] = []
    best_val_accuracy = -1.0
    best_metrics: dict[str, float] = {}

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, loaders["train"], criterion, device, optimizer)
        val_metrics = run_epoch(model, loaders["val"], criterion, device)

        row = {
            "epoch": float(epoch),
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
        }
        history.append(row)
        print(
            f"epoch={epoch} "
            f"train_loss={row['train_loss']:.4f} train_acc={row['train_accuracy']:.4f} "
            f"val_loss={row['val_loss']:.4f} val_acc={row['val_accuracy']:.4f}"
        )

        if val_metrics["accuracy"] > best_val_accuracy:
            best_val_accuracy = val_metrics["accuracy"]
            best_metrics = {
                "best_val_accuracy": best_val_accuracy,
                "best_val_loss": val_metrics["loss"],
            }
            save_checkpoint(
                path=args.output_dir / "best_model.pt",
                model=model,
                class_names=class_names,
                image_size=args.image_size,
                model_name=args.model,
                metrics=best_metrics,
            )

    train_dir = args.output_dir / "train"
    train_dir.mkdir(parents=True, exist_ok=True)
    with (train_dir / "training_history.json").open("w", encoding="utf-8") as file:
        json.dump(history, file, indent=2)
        file.write("\n")
    plot_history(history, args.output_dir)
    print(f"best_val_accuracy={best_val_accuracy:.4f}")
    print(f"checkpoint={args.output_dir / 'best_model.pt'}")


if __name__ == "__main__":
    main()

