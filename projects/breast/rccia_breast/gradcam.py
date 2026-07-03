"""Grad-CAM visualization for convolutional classifiers."""

from __future__ import annotations

import cv2
import numpy as np
import torch
from PIL import Image
from torch import nn

from .config import IMAGENET_MEAN, IMAGENET_STD
from .data import build_image_transform


class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self.forward_handle = target_layer.register_forward_hook(self._save_activation)
        self.backward_handle = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, _module: nn.Module, _inputs: tuple, output: torch.Tensor) -> None:
        self.activations = output.detach()

    def _save_gradient(
        self,
        _module: nn.Module,
        _grad_input: tuple,
        grad_output: tuple[torch.Tensor],
    ) -> None:
        self.gradients = grad_output[0].detach()

    def remove_hooks(self) -> None:
        self.forward_handle.remove()
        self.backward_handle.remove()

    def generate(
        self,
        input_tensor: torch.Tensor,
        target_index: int | None = None,
    ) -> tuple[np.ndarray, int]:
        self.model.eval()
        self.model.zero_grad(set_to_none=True)

        logits = self.model(input_tensor)
        if target_index is None:
            target_index = int(logits.argmax(dim=1).item())

        score = logits[:, target_index].sum()
        score.backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations and gradients.")

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = torch.nn.functional.interpolate(
            cam,
            size=input_tensor.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        cam = cam.squeeze().detach().cpu().numpy()
        cam = cam - cam.min()
        max_value = cam.max()
        if max_value > 0:
            cam = cam / max_value
        return cam, target_index


def get_gradcam_target_layer(model: nn.Module) -> nn.Module:
    if hasattr(model, "layer4"):
        return model.layer4[-1].conv2

    if hasattr(model, "features") and isinstance(model.features, nn.Sequential):
        return model.features[-1]

    raise ValueError("Grad-CAM expects a CNN model with a ResNet layer4 or features block.")


def image_to_tensor(image: Image.Image, image_size: int, device: torch.device) -> torch.Tensor:
    transform = build_image_transform(image_size=image_size)
    return transform(image.convert("RGB")).unsqueeze(0).to(device)


def denormalize_image(tensor: torch.Tensor) -> np.ndarray:
    image = tensor.squeeze(0).detach().cpu().permute(1, 2, 0).numpy()
    mean = np.array(IMAGENET_MEAN)
    std = np.array(IMAGENET_STD)
    image = (image * std + mean).clip(0, 1)
    return image


def overlay_cam(image: np.ndarray, cam: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    overlay = ((1 - alpha) * image + alpha * heatmap).clip(0, 1)
    return overlay
