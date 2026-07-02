from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw


CLASSES = (
    "colon_adenocarcinoma",
    "colon_benign",
    "lung_adenocarcinoma",
    "lung_benign",
    "lung_squamous_cell_carcinoma",
)


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
    base = 35 + class_index * 35
    image = Image.new("RGB", (64, 64), (base, 245 - base // 2, 120 + class_index * 20))
    draw = ImageDraw.Draw(image)
    offset = 6 + image_index * 2
    draw.rectangle((offset, offset, 58 - offset, 58 - offset), outline=(255, 255, 255), width=2)
    draw.line((8, 12 + class_index * 4, 56, 12 + class_index * 4), fill=(20, 20, 20), width=2)
    image.save(path)


def create_raw_dataset(raw_dir: Path) -> None:
    for class_index, class_name in enumerate(CLASSES):
        for image_index in range(4):
            create_synthetic_image(
                raw_dir / class_name / f"{class_name}_{image_index}.png",
                class_index,
                image_index,
            )


def test_lung_colon_first_end_to_end_run_with_synthetic_dataset(tmp_path: Path) -> None:
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
        assert len(list((processed_dir / "train" / class_name).glob("*.png"))) == 2
        assert len(list((processed_dir / "val" / class_name).glob("*.png"))) == 1
        assert len(list((processed_dir / "test" / class_name).glob("*.png"))) == 1

    run_command(
        [
            sys.executable,
            "-m",
            "rccia_lung_colon.train",
            "--data-dir",
            str(processed_dir),
            "--epochs",
            "1",
            "--batch-size",
            "5",
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
            "rccia_lung_colon.evaluate",
            "--data-dir",
            str(processed_dir),
            "--checkpoint",
            str(checkpoint_path),
            "--output-dir",
            str(eval_dir),
            "--batch-size",
            "5",
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
            "rccia_lung_colon.predict",
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
    assert len(payload["probabilities"]) == 5
