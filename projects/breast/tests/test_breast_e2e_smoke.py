from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw


CLASSES = ("benign", "malignant")


def run_command(args: list[str], cwd: Path, timeout: int = 180) -> subprocess.CompletedProcess:
    result = subprocess.run(
        args,
        cwd=cwd,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    assert result.returncode == 0, (
        f"Command failed: {' '.join(args)}\n"
        f"STDOUT:\n{result.stdout}\n"
        f"STDERR:\n{result.stderr}"
    )
    return result


def create_synthetic_image(path: Path, class_index: int, image_index: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    base = 80 + class_index * 90
    image = Image.new("RGB", (64, 64), (base, 180 - class_index * 60, 140))
    draw = ImageDraw.Draw(image)
    offset = 6 + image_index
    draw.ellipse((offset, offset, 60 - offset, 60 - offset), outline=(255, 255, 255), width=2)
    draw.line((8, 14 + class_index * 20, 56, 14 + class_index * 20), fill=(20, 20, 20), width=2)
    image.save(path)


def create_raw_dataset(raw_dir: Path) -> None:
    for class_index, class_name in enumerate(CLASSES):
        for image_index in range(6):
            create_synthetic_image(
                raw_dir / class_name / f"{class_name}_{image_index}.png",
                class_index,
                image_index,
            )


def test_breast_first_end_to_end_run_with_synthetic_dataset(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[1]
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    output_dir = tmp_path / "outputs"
    eval_dir = output_dir / "eval"

    create_raw_dataset(raw_dir)

    run_command(
        [
            sys.executable,
            "scripts/split_image_folder.py",
            "--input",
            str(raw_dir),
            "--output",
            str(processed_dir),
            "--train-ratio",
            "0.5",
            "--val-ratio",
            "0.25",
            "--test-ratio",
            "0.25",
        ],
        cwd=project_root,
    )

    for class_name in CLASSES:
        assert len(list((processed_dir / "train" / class_name).glob("*.png"))) == 3
        assert len(list((processed_dir / "val" / class_name).glob("*.png"))) == 1
        assert len(list((processed_dir / "test" / class_name).glob("*.png"))) == 2

    run_command(
        [
            sys.executable,
            "-m",
            "rccia_breast.train",
            "--data-dir",
            str(processed_dir),
            "--epochs",
            "1",
            "--batch-size",
            "4",
            "--image-size",
            "32",
            "--output-dir",
            str(output_dir),
            "--no-pretrained",
            "--num-workers",
            "0",
        ],
        cwd=project_root,
    )

    checkpoint_path = output_dir / "best_model.pt"
    assert checkpoint_path.exists()
    assert (output_dir / "train" / "training_history.json").exists()
    assert (output_dir / "train" / "training_loss.png").exists()

    run_command(
        [
            sys.executable,
            "-m",
            "rccia_breast.evaluate",
            "--data-dir",
            str(processed_dir),
            "--checkpoint",
            str(checkpoint_path),
            "--output-dir",
            str(eval_dir),
            "--batch-size",
            "4",
            "--num-workers",
            "0",
        ],
        cwd=project_root,
    )

    assert (eval_dir / "test_classification_report.json").exists()
    assert (eval_dir / "test_classification_report.csv").exists()
    assert (eval_dir / "test_confusion_matrix.png").exists()

    image_path = next((processed_dir / "test").glob("*/*.png"))
    prediction = run_command(
        [
            sys.executable,
            "-m",
            "rccia_breast.predict",
            "--checkpoint",
            str(checkpoint_path),
            "--image",
            str(image_path),
            "--pretty",
        ],
        cwd=project_root,
    )
    payload = json.loads(prediction.stdout)

    assert payload["class_name"] in CLASSES
    assert 0 <= payload["confidence"] <= 1
    assert len(payload["probabilities"]) == 2


def test_breast_app_imports_without_checkpoint() -> None:
    project_root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("breast_streamlit_app", project_root / "app.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert hasattr(module, "render_app")
