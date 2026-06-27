from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw


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


def create_synthetic_image(path: Path, class_name: str, index: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (64, 64), (245, 245, 245))
    draw = ImageDraw.Draw(image)

    if class_name == "normal":
        draw.ellipse((16, 16, 48, 48), fill=(80, 160, 220), outline=(30, 90, 150))
        draw.line((16, 32 + index, 48, 32 + index), fill=(255, 255, 255), width=2)
    else:
        draw.rectangle((14, 14, 50, 50), fill=(210, 70, 100), outline=(120, 20, 50))
        draw.line((32 + index, 14, 32 + index, 50), fill=(255, 230, 230), width=2)

    image.save(path)


def create_raw_dataset(raw_dir: Path) -> None:
    for class_name in ("normal", "cancer"):
        for index in range(4):
            create_synthetic_image(raw_dir / class_name / f"{class_name}_{index}.png", class_name, index)


def test_first_end_to_end_run_with_synthetic_dataset(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
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
        cwd=repo_root,
    )

    assert len(list((processed_dir / "train" / "normal").glob("*.png"))) == 2
    assert len(list((processed_dir / "val" / "normal").glob("*.png"))) == 1
    assert len(list((processed_dir / "test" / "normal").glob("*.png"))) == 1

    run_command(
        [
            sys.executable,
            "-m",
            "cancer_cell_vision.train",
            "--data-dir",
            str(processed_dir),
            "--epochs",
            "1",
            "--batch-size",
            "2",
            "--image-size",
            "32",
            "--output-dir",
            str(output_dir),
            "--no-pretrained",
            "--num-workers",
            "0",
        ],
        cwd=repo_root,
    )

    checkpoint_path = output_dir / "best_model.pt"
    assert checkpoint_path.exists()
    assert (output_dir / "train" / "training_history.json").exists()

    run_command(
        [
            sys.executable,
            "-m",
            "cancer_cell_vision.evaluate",
            "--data-dir",
            str(processed_dir),
            "--checkpoint",
            str(checkpoint_path),
            "--output-dir",
            str(eval_dir),
            "--batch-size",
            "2",
            "--num-workers",
            "0",
        ],
        cwd=repo_root,
    )

    assert (eval_dir / "test_classification_report.json").exists()
    assert (eval_dir / "test_confusion_matrix.png").exists()

    image_path = next((processed_dir / "test").glob("*/*.png"))
    prediction = run_command(
        [
            sys.executable,
            "-m",
            "cancer_cell_vision.predict",
            "--checkpoint",
            str(checkpoint_path),
            "--image",
            str(image_path),
            "--pretty",
        ],
        cwd=repo_root,
    )
    payload = json.loads(prediction.stdout)

    assert payload["class_name"] in {"normal", "cancer"}
    assert 0 <= payload["confidence"] <= 1
    assert len(payload["probabilities"]) == 2
