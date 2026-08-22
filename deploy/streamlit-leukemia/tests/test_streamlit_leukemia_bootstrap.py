from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "app.py"
SPEC = importlib.util.spec_from_file_location("streamlit_leukemia_bootstrap", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
bootstrap = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = bootstrap
SPEC.loader.exec_module(bootstrap)


@pytest.fixture
def downloader_module():
    return bootstrap.load_downloader_module()


def test_checkpoint_mapping_comes_from_shared_downloader(downloader_module) -> None:
    model_file = bootstrap.leukemia_model_file(
        downloader_module=downloader_module,
    )

    assert model_file.remote_path == "leukemia/best_model.pt"
    assert model_file.local_path.as_posix() == (
        "projects/leukemia/outputs/best_model.pt"
    )


def test_existing_checkpoint_skips_download(
    tmp_path: Path,
    downloader_module,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = bootstrap.expected_checkpoint_path(
        tmp_path,
        downloader_module=downloader_module,
    )
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"local-checkpoint")

    def unexpected_download(**kwargs) -> None:
        raise AssertionError("Downloader must not run when the checkpoint exists.")

    monkeypatch.setattr(downloader_module, "download_models", unexpected_download)

    result = bootstrap.ensure_checkpoint(
        tmp_path,
        downloader_module=downloader_module,
    )

    assert result == checkpoint


def test_missing_checkpoint_downloads_only_leukemia(
    tmp_path: Path,
    downloader_module,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    def fake_download(*, repo_root: Path, model_files) -> None:
        selected = tuple(model_files)
        calls.append((repo_root, selected))
        destination = repo_root / selected[0].local_path
        destination.parent.mkdir(parents=True)
        destination.write_bytes(b"downloaded-checkpoint")

    monkeypatch.setattr(downloader_module, "download_models", fake_download)

    checkpoint = bootstrap.ensure_checkpoint(
        tmp_path,
        downloader_module=downloader_module,
    )

    assert checkpoint.is_file()
    assert len(calls) == 1
    assert calls[0][0] == tmp_path
    assert len(calls[0][1]) == 1
    assert calls[0][1][0].remote_path == "leukemia/best_model.pt"


def test_downloader_failure_is_controlled_and_hides_token(
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
        bootstrap.ensure_checkpoint(
            tmp_path,
            downloader_module=downloader_module,
        )

    captured = capsys.readouterr()
    assert secret not in str(error.value)
    assert secret not in captured.out
    assert secret not in captured.err


def test_missing_checkpoint_after_download_is_explicit(
    tmp_path: Path,
    downloader_module,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        downloader_module,
        "download_models",
        lambda **kwargs: None,
    )

    with pytest.raises(bootstrap.BootstrapError, match="preparation incomplete") as error:
        bootstrap.ensure_checkpoint(
            tmp_path,
            downloader_module=downloader_module,
        )

    assert "projects/leukemia/outputs/best_model.pt" in str(error.value)


def test_app_delegation_uses_project_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app_path = tmp_path / "projects" / "leukemia" / "app.py"
    app_path.parent.mkdir(parents=True)
    app_path.write_text("# test app", encoding="ascii")
    calls: list[tuple[str, str]] = []

    def fake_run_path(path: str, *, run_name: str) -> None:
        calls.append((path, run_name))

    monkeypatch.setattr(bootstrap.runpy, "run_path", fake_run_path)
    monkeypatch.setattr(bootstrap.sys, "path", list(bootstrap.sys.path))

    bootstrap.run_leukemia_app(tmp_path)

    assert calls == [(str(app_path), "__main__")]
    assert str(app_path.parent) in bootstrap.sys.path
