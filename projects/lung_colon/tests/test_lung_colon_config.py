from __future__ import annotations

from pathlib import Path

from rccia_lung_colon.config import TrainConfig


def test_train_config_defaults() -> None:
    config = TrainConfig(data_dir=Path("data/processed"))

    assert config.model_name == "resnet18"
    assert config.output_dir == Path("outputs")
    assert config.image_size == 224
