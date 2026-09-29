from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from rccia_common.dataset_paths import validate_dataset_paths

PROJECTS = Path(__file__).resolve().parents[2]


def load_script(project: str, filename: str):
    spec = importlib.util.spec_from_file_location(
        f"audit_{project}_{filename}", PROJECTS / project / "scripts" / filename
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("relationship", ["same", "ancestor", "descendant"])
def test_dataset_overlap_rejected_without_deleting(tmp_path, relationship):
    source = tmp_path / "dataset" / "raw"
    source.mkdir(parents=True)
    marker = source / "important.txt"
    marker.write_text("preserve")
    output = {"same": source, "ancestor": source.parent, "descendant": source / "out"}[relationship]
    with pytest.raises(ValueError, match="overlap"):
        validate_dataset_paths(source, output)
    assert marker.read_text() == "preserve"


@pytest.mark.parametrize("project", ["leukemia", "lung_colon", "breast", "metastasis"])
@pytest.mark.parametrize("failure", ["ancestor", "ratio", "nan", "empty"])
def test_split_rejects_before_overwrite(tmp_path, monkeypatch, project, failure):
    module = load_script(project, "split_image_folder.py")
    source = tmp_path / "source" / "raw"
    (source / "class_a").mkdir(parents=True)
    if failure != "empty":
        (source / "class_a" / "a.png").write_bytes(b"test fixture")
    output = source.parent if failure == "ancestor" else tmp_path / "processed"
    output.mkdir(exist_ok=True)
    sentinel = output / "keep.txt"
    sentinel.write_text("existing dataset")
    argv = ["split", "--input", str(source), "--output", str(output), "--overwrite"]
    if failure in {"ratio", "nan"}:
        argv += ["--val-ratio", "-1" if failure == "ratio" else "nan"]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(ValueError):
        module.main()
    assert sentinel.read_text() == "existing dataset"
    assert source.is_dir()


@pytest.mark.parametrize(
    ("project", "script", "source_flag"),
    [
        ("leukemia", "prepare_leukemia_dataset.py", "--source"),
        ("lung_colon", "prepare_lc25000_dataset.py", "--source"),
        ("breast", "prepare_breakhis_dataset.py", "--input"),
        ("metastasis", "prepare_pcam_dataset.py", "--input"),
        ("lung_colon", "prepare_binary_dataset.py", "--input"),
    ],
)
@pytest.mark.parametrize("relationship", ["same", "ancestor", "descendant"])
def test_prepare_overlap_rejected(tmp_path, monkeypatch, project, script, source_flag, relationship):
    source = tmp_path / "dataset" / "source"
    source.mkdir(parents=True)
    sentinel = source / "keep.txt"
    sentinel.write_text("preserve")
    output = {"same": source, "ancestor": source.parent, "descendant": source / "out"}[relationship]
    monkeypatch.setattr(
        sys, "argv", ["prepare", source_flag, str(source), "--output", str(output), "--overwrite"]
    )
    with pytest.raises(ValueError, match="overlap"):
        load_script(project, script).main()
    assert sentinel.read_text() == "preserve"


def test_hdf5_input_inside_output_rejected(tmp_path, monkeypatch):
    source = tmp_path / "dataset"
    source.mkdir()
    x, y = source / "x.h5", source / "y.h5"
    x.write_bytes(b"preserve")
    y.write_bytes(b"preserve")
    monkeypatch.setattr(sys, "argv", [
        "prepare", "--x-h5", str(x), "--y-h5", str(y),
        "--output", str(source), "--overwrite",
    ])
    with pytest.raises(ValueError, match="overlap"):
        load_script("metastasis", "prepare_pcam_dataset.py").main()
    assert x.read_bytes() == y.read_bytes() == b"preserve"


def test_patient_metadata_required_before_overwrite(tmp_path, monkeypatch):
    source, output = tmp_path / "raw", tmp_path / "processed"
    source.mkdir()
    output.mkdir()
    sentinel = output / "keep.txt"
    sentinel.write_text("preserve")
    monkeypatch.setattr(sys, "argv", [
        "split", "--input", str(source), "--output", str(output), "--patient-aware", "--overwrite"
    ])
    with pytest.raises(ValueError, match="metadata"):
        load_script("breast", "split_image_folder.py").main()
    assert sentinel.read_text() == "preserve"
