from __future__ import annotations

import torch

from cancer_cell_vision.model import SUPPORTED_MODEL_NAMES, create_model


def test_supported_models_forward_without_pretrained_weights() -> None:
    inputs = torch.randn(2, 3, 64, 64)

    for model_name in SUPPORTED_MODEL_NAMES:
        model = create_model(num_classes=2, model_name=model_name, pretrained=False)
        model.eval()
        with torch.inference_mode():
            outputs = model(inputs)

        assert outputs.shape == (2, 2)
