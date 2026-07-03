from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import torch
from PIL import Image

from rccia_lung_colon.binary import (
    BINARY_CLASS_MAPPING,
    BINARY_CLASSES,
    SOURCE_CLASSES_BY_BINARY,
    binary_label_for_source,
)


def load_prepare_binary_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "prepare_binary_dataset.py"
    spec = importlib.util.spec_from_file_location("prepare_binary_dataset", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_lung_colon_app_module():
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    spec = importlib.util.spec_from_file_location("lung_colon_app", app_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def create_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 16), (120, 90, 200)).save(path)


def test_binary_mapping_rules() -> None:
    assert BINARY_CLASSES == ("benign", "malignant")
    assert binary_label_for_source("colon_benign") == "benign"
    assert binary_label_for_source("lung_benign") == "benign"
    assert binary_label_for_source("colon_adenocarcinoma") == "malignant"
    assert binary_label_for_source("lung_adenocarcinoma") == "malignant"
    assert binary_label_for_source("lung_squamous_cell_carcinoma") == "malignant"
    assert SOURCE_CLASSES_BY_BINARY["benign"] == ("colon_benign", "lung_benign")


def test_prepare_binary_dataset_copies_processed_splits(tmp_path: Path) -> None:
    module = load_prepare_binary_module()
    input_dir = tmp_path / "processed"
    output_dir = tmp_path / "binary_processed"

    for split in module.SPLITS:
        for class_name in BINARY_CLASS_MAPPING:
            for index in range(2):
                create_image(input_dir / split / class_name / f"{class_name}_{index}.png")

    summary = module.prepare_binary_dataset(input_dir=input_dir, output_dir=output_dir)

    for split in module.SPLITS:
        assert summary[split] == {"benign": 4, "malignant": 6}
        assert len(list((output_dir / split / "benign").glob("*.png"))) == 4
        assert len(list((output_dir / split / "malignant").glob("*.png"))) == 6

    assert len(list((input_dir / "train" / "colon_benign").glob("*.png"))) == 2

    summary_path = output_dir / "binary_dataset_summary.json"
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert payload["splits"]["test"]["malignant"] == 6


def test_streamlit_binary_checkpoint_missing_is_safe(tmp_path: Path) -> None:
    app = load_lung_colon_app_module()
    missing_checkpoint = tmp_path / "missing_binary_model.pt"

    model, checkpoint, error = app.load_checkpoint_safely(
        missing_checkpoint,
        device=torch.device("cpu"),
    )

    assert model is None
    assert checkpoint is None
    assert "Checkpoint not found" in error
