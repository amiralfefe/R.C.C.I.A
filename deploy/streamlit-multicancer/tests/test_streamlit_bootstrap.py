from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "app.py"
SPEC = importlib.util.spec_from_file_location("streamlit_multicancer_bootstrap", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
bootstrap = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = bootstrap
SPEC.loader.exec_module(bootstrap)


@pytest.fixture
def downloader_module():
    return bootstrap.load_downloader_module()


def _create_expected_files(repo_root: Path, module) -> None:
    for model_file in module.MODEL_FILES:
        checkpoint = repo_root / model_file.local_path
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(b"test-checkpoint")


def test_mapping_comes_from_existing_downloader(downloader_module) -> None:
    mapping = {
        model_file.remote_path: model_file.local_path.as_posix()
        for model_file in downloader_module.MODEL_FILES
    }

    assert mapping == {
        "leukemia/best_model.pt": "projects/leukemia/outputs/best_model.pt",
        "breast/efficientnet_b0/best_model.pt": (
            "projects/breast/outputs/model_comparison/efficientnet_b0/best_model.pt"
        ),
        "metastasis/efficientnet_b0/best_model.pt": (
            "projects/metastasis/outputs/model_comparison/efficientnet_b0/best_model.pt"
        ),
        "lung_colon/multiclass/efficientnet_b0/best_model.pt": (
            "projects/lung_colon/outputs/model_comparison/efficientnet_b0/best_model.pt"
        ),
        "lung_colon/binary/resnet18/best_model.pt": (
            "projects/lung_colon/outputs/binary_resnet18/best_model.pt"
        ),
    }


def test_existing_checkpoints_skip_downloader(
    tmp_path: Path,
    downloader_module,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _create_expected_files(tmp_path, downloader_module)

    def unexpected_download(**kwargs) -> None:
        raise AssertionError("Downloader must not run when every checkpoint exists.")

    monkeypatch.setattr(downloader_module, "download_models", unexpected_download)

    paths = bootstrap.ensure_checkpoints(
        tmp_path,
        downloader_module=downloader_module,
    )

    assert len(paths) == 5
    assert all(path.is_file() for path in paths)


def test_missing_checkpoints_call_downloader(
    tmp_path: Path,
    downloader_module,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[Path] = []

    def fake_download(*, repo_root: Path) -> None:
        calls.append(repo_root)
        _create_expected_files(repo_root, downloader_module)

    monkeypatch.setattr(downloader_module, "download_models", fake_download)

    paths = bootstrap.ensure_checkpoints(
        tmp_path,
        downloader_module=downloader_module,
    )

    assert calls == [tmp_path]
    assert len(paths) == 5
    assert all(path.is_file() for path in paths)


def test_downloader_failure_is_controlled_and_does_not_expose_token(
    tmp_path: Path,
    downloader_module,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    secret = "opaque-value-that-must-not-appear"
    monkeypatch.setenv("HF_TOKEN", secret)

    def failed_download(**kwargs) -> None:
        raise downloader_module.ModelDownloadError("private repository unavailable")

    monkeypatch.setattr(downloader_module, "download_models", failed_download)

    with pytest.raises(bootstrap.BootstrapError, match="Checkpoint download failed") as error:
        bootstrap.ensure_checkpoints(
            tmp_path,
            downloader_module=downloader_module,
        )

    captured = capsys.readouterr()
    assert secret not in str(error.value)
    assert secret not in captured.out
    assert secret not in captured.err


def test_missing_files_after_download_raise_clear_error(
    tmp_path: Path,
    downloader_module,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(downloader_module, "download_models", lambda **kwargs: None)

    with pytest.raises(bootstrap.BootstrapError, match="preparation incomplete") as error:
        bootstrap.ensure_checkpoints(
            tmp_path,
            downloader_module=downloader_module,
        )

    assert "projects/leukemia/outputs/best_model.pt" in str(error.value)


def test_hub_delegation_uses_real_project_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hub_path = tmp_path / "projects" / "multicancer" / "app.py"
    hub_path.parent.mkdir(parents=True)
    hub_path.write_text("# test hub", encoding="ascii")
    calls: list[tuple[str, str]] = []

    def fake_run_path(path: str, *, run_name: str) -> None:
        calls.append((path, run_name))

    monkeypatch.setattr(bootstrap.runpy, "run_path", fake_run_path)

    bootstrap.run_multicancer_hub(tmp_path)

    assert calls == [(str(hub_path), "__main__")]
