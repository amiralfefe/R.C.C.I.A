"""Model creation and prediction helpers."""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torchvision import models

from cancer_cell_vision.data import build_image_transform


def create_model(
    num_classes: int,
    model_name: str = "resnet18",
    pretrained: bool = True,
) -> nn.Module:
    if model_name != "resnet18":
        raise ValueError(f"Unsupported model '{model_name}'. V1 currently supports resnet18.")

    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def save_checkpoint(
    path: Path,
    model: nn.Module,
    class_names: list[str],
    image_size: int,
    model_name: str,
    metrics: dict[str, float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "class_names": class_names,
            "image_size": image_size,
            "model_name": model_name,
            "metrics": metrics,
        },
        path,
    )


def load_checkpoint(path: Path, device: torch.device) -> tuple[nn.Module, dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {path}. Train a model first with "
            "python -m cancer_cell_vision.train."
        )

    checkpoint = torch.load(path, map_location=device)
    required_keys = {"model_state", "class_names", "image_size"}
    missing_keys = required_keys.difference(checkpoint)
    if missing_keys:
        raise ValueError(f"Invalid checkpoint {path}. Missing keys: {sorted(missing_keys)}.")

    class_names = checkpoint["class_names"]
    model_name = checkpoint.get("model_name", "resnet18")
    model = create_model(num_classes=len(class_names), model_name=model_name, pretrained=False)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model, checkpoint


@torch.inference_mode()
def predict_image(
    image: Image.Image,
    model: nn.Module,
    class_names: list[str],
    image_size: int,
    device: torch.device,
) -> dict[str, float | str | list[float]]:
    transform = build_image_transform(image_size=image_size)
    tensor = transform(image.convert("RGB")).unsqueeze(0).to(device)
    logits = model(tensor)
    probabilities = torch.softmax(logits, dim=1).squeeze(0).detach().cpu()
    predicted_index = int(probabilities.argmax().item())

    return {
        "class_name": class_names[predicted_index],
        "confidence": float(probabilities[predicted_index].item()),
        "probabilities": [float(value) for value in probabilities],
    }
