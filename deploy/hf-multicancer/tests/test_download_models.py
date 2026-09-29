from __future__ import annotations

import importlib.util
import sys
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "download_models.py"
SPEC = importlib.util.spec_from_file_location("hf_multicancer_download_models", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
download_module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = download_module
SPEC.loader.exec_module(download_module)


def test_concurrent_installs_have_independent_staging_files(tmp_path, monkeypatch):
    cache = tmp_path / "cache.pt"
    cache.write_bytes(b"complete checkpoint")
    barrier = Barrier(2)
    original_copy = download_module.shutil.copyfile
    staged_paths = []

    def synchronized_copy(source, destination):
        staged_paths.append(destination)
        original_copy(source, destination)
        barrier.wait(timeout=10)

    monkeypatch.setattr(download_module.shutil, "copyfile", synchronized_copy)
    def install():
        return download_module.download_models(
            repo_root=tmp_path / "repo", repo_id="owner/models",
            token="test-only", downloader=lambda *_: cache,
            model_files=download_module.MODEL_FILES[:1],
        )
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda _: install(), range(2)))
    assert len(set(staged_paths)) == 2
    assert results[0][0].destination.read_bytes() == cache.read_bytes()
    assert not list((tmp_path / "repo").rglob("*.part"))


def test_empty_download_is_rejected(tmp_path):
    cache = tmp_path / "empty.pt"
    cache.touch()
    with pytest.raises(download_module.ModelDownloadError, match="unavailable"):
        download_module.download_models(
            repo_root=tmp_path / "repo", repo_id="owner/models", token="test-only",
            downloader=lambda *_: cache, model_files=download_module.MODEL_FILES[:1],
        )
    assert not (tmp_path / "repo").exists()


def test_checkpoint_mapping_matches_frozen_adapter_paths() -> None:
    mapping = {
        model_file.remote_path: model_file.local_path.as_posix()
        for model_file in download_module.MODEL_FILES
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


def test_downloads_all_files_and_skips_existing(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    calls: list[tuple[str, str, str | None]] = []

    def fake_downloader(repo_id: str, filename: str, token: str | None) -> Path:
        calls.append((repo_id, filename, token))
        cached_file = cache / filename
        cached_file.parent.mkdir(parents=True, exist_ok=True)
        cached_file.write_bytes(f"checkpoint:{filename}".encode())
        return cached_file

    first_results = download_module.download_models(
        repo_root=tmp_path / "repo",
        repo_id="owner/models",
        token="opaque-test-token",
        downloader=fake_downloader,
    )

    assert len(calls) == 5
    assert {result.status for result in first_results} == {"downloaded"}
    for result in first_results:
        assert result.destination.is_file()
        assert result.destination.stat().st_size > 0

    calls.clear()
    second_results = download_module.download_models(
        repo_root=tmp_path / "repo",
        repo_id="owner/models",
        token="opaque-test-token",
        downloader=fake_downloader,
    )

    assert calls == []
    assert {result.status for result in second_results} == {"skipped"}


def test_downloads_only_selected_model_files(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    calls: list[tuple[str, str, str | None]] = []

    def fake_downloader(repo_id: str, filename: str, token: str | None) -> Path:
        calls.append((repo_id, filename, token))
        cached_file = cache / filename
        cached_file.parent.mkdir(parents=True, exist_ok=True)
        cached_file.write_bytes(b"selected-checkpoint")
        return cached_file

    leukemia_model = next(
        model_file
        for model_file in download_module.MODEL_FILES
        if model_file.remote_path == "leukemia/best_model.pt"
    )
    results = download_module.download_models(
        repo_root=tmp_path / "repo",
        repo_id="owner/models",
        token="opaque-test-token",
        downloader=fake_downloader,
        model_files=(leukemia_model,),
    )

    assert calls == [
        ("owner/models", "leukemia/best_model.pt", "opaque-test-token")
    ]
    assert len(results) == 1
    assert results[0].model_file == leukemia_model
    assert results[0].status == "downloaded"
    assert results[0].destination.is_file()


def test_missing_model_repository_configuration_is_explicit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("HF_MODEL_REPO_ID", raising=False)

    with pytest.raises(download_module.ModelDownloadError, match="HF_MODEL_REPO_ID"):
        download_module.download_models(repo_root=tmp_path)


def test_private_repository_without_token_has_safe_guidance(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    class RepositoryNotFoundError(Exception):
        pass

    def denied_downloader(repo_id: str, filename: str, token: str | None) -> Path:
        raise RepositoryNotFoundError("private repository")

    with pytest.raises(download_module.ModelDownloadError, match="read-only"):
        download_module.download_models(
            repo_root=tmp_path,
            repo_id="owner/private-models",
            token=None,
            downloader=denied_downloader,
        )

    assert "HF_TOKEN" in capsys.readouterr().out


def test_missing_remote_file_is_precise_and_never_prints_token(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    secret = "opaque-value-that-must-not-appear"

    class EntryNotFoundError(Exception):
        pass

    def missing_downloader(repo_id: str, filename: str, token: str | None) -> Path:
        raise EntryNotFoundError("missing file")

    with pytest.raises(download_module.ModelDownloadError) as error:
        download_module.download_models(
            repo_root=tmp_path,
            repo_id="owner/models",
            token=secret,
            downloader=missing_downloader,
        )

    output = capsys.readouterr()
    assert "leukemia/best_model.pt" in str(error.value)
    assert secret not in output.out
    assert secret not in output.err
    assert secret not in str(error.value)
