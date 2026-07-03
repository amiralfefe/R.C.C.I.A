"""Run prediction on a single breast histopathology image."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from .model import load_checkpoint, predict_image
from .utils import get_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict one image with R.C.C.I.A Breast.")
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/best_model.pt"))
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--pretty", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.image.exists():
        raise FileNotFoundError(f"Image not found: {args.image}")

    try:
        image = Image.open(args.image).convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError(f"Invalid image file: {args.image}") from exc

    device = get_device()
    model, checkpoint = load_checkpoint(args.checkpoint, device=device)
    class_names = checkpoint["class_names"]
    image_size = int(checkpoint.get("image_size", 224))
    result = predict_image(
        image=image,
        model=model,
        class_names=class_names,
        image_size=image_size,
        device=device,
    )

    indent = 2 if args.pretty else None
    print(json.dumps(result, indent=indent))


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"Error: {exc}") from exc
