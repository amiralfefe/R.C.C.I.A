from pathlib import Path

from rccia_leukemia.config import TrainConfig


def test_train_config_defaults_are_portfolio_v1_friendly() -> None:
    config = TrainConfig(data_dir=Path("data/processed"))

    assert config.model_name == "resnet18"
    assert config.image_size == 224
    assert config.output_dir == Path("outputs")
    assert config.pretrained is True
