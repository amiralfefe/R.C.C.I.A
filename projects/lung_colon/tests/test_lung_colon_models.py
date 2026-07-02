from __future__ import annotations

import torch

from rccia_lung_colon.model import SUPPORTED_MODEL_NAMES, create_model


def test_supported_models_forward_with_five_classes() -> None:
    inputs = torch.randn(2, 3, 32, 32)
    for model_name in SUPPORTED_MODEL_NAMES:
        model = create_model(num_classes=5, model_name=model_name, pretrained=False)
        model.eval()
        with torch.inference_mode():
            outputs = model(inputs)

        assert outputs.shape == (2, 5)
