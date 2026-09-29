"""Download the five frozen MultiCancer checkpoints without loading any model."""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from threading import Lock
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


@dataclass(frozen=True)
class ModelFile:
    project: str
    remote_path: str
    local_path: Path


@dataclass(frozen=True)
class DownloadResult:
    model_file: ModelFile
    status: str
    destination: Path


MODEL_FILES = (
    ModelFile(
        project="Leukemia",
        remote_path="leukemia/best_model.pt",
        local_path=Path("projects/leukemia/outputs/best_model.pt"),
    ),
    ModelFile(
        project="Breast",
        remote_path="breast/efficientnet_b0/best_model.pt",
        local_path=Path(
            "projects/breast/outputs/model_comparison/efficientnet_b0/best_model.pt"
        ),
    ),
    ModelFile(
        project="Metastasis",
        remote_path="metastasis/efficientnet_b0/best_model.pt",
        local_path=Path(
            "projects/metastasis/outputs/model_comparison/efficientnet_b0/best_model.pt"
        ),
    ),
    ModelFile(
        project="LungColon multiclass",
        remote_path="lung_colon/multiclass/efficientnet_b0/best_model.pt",
        local_path=Path(
            "projects/lung_colon/outputs/model_comparison/efficientnet_b0/best_model.pt"
        ),
    ),
    ModelFile(
        project="LungColon binary",
        remote_path="lung_colon/binary/resnet18/best_model.pt",
        local_path=Path("projects/lung_colon/outputs/binary_resnet18/best_model.pt"),
    ),
)


class ModelDownloadError(RuntimeError):
    """Raised when deployment checkpoint preparation cannot complete safely."""


HubDownloader = Callable[[str, str, str | None], Path]
INSTALL_LOCK = Lock()


def _download_from_hub(repo_id: str, filename: str, token: str | None) -> Path:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise ModelDownloadError(
            "The huggingface_hub runtime dependency is unavailable."
        ) from exc

    return Path(
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="model",
            token=token,
        )
    )


def _download_error_message(
    exc: Exception,
    *,
    repo_id: str,
    remote_path: str,
    token_present: bool,
) -> str:
    error_name = type(exc).__name__
    status_code = getattr(getattr(exc, "response", None), "status_code", None)

    if error_name == "EntryNotFoundError":
        return f"Checkpoint '{remote_path}' is absent from model repository '{repo_id}'."

    access_errors = {"RepositoryNotFoundError", "GatedRepoError"}
    if error_name in access_errors or status_code in {401, 403}:
        if not token_present:
            return (
                f"Model repository '{repo_id}' is unavailable. If it is private, configure "
                "HF_TOKEN as a read-only Hugging Face Space secret."
            )
        return (
            f"Access to model repository '{repo_id}' was denied. Verify that the configured "
            "HF_TOKEN is read-only and can access this repository."
        )

    if status_code == 404:
        return f"Model repository '{repo_id}' or checkpoint '{remote_path}' was not found."

    return (
        f"Unable to download checkpoint '{remote_path}' from '{repo_id}' "
        f"({error_name})."
    )


def _resolve_repo_root(repo_root: Path | str | None) -> Path:
    if repo_root is not None:
        return Path(repo_root).resolve()

    configured_root = os.getenv("RCCIA_REPO_ROOT", "").strip()
    if configured_root:
        return Path(configured_root).resolve()

    return Path(__file__).resolve().parents[2]


def download_models(
    *,
    repo_root: Path | str | None = None,
    repo_id: str | None = None,
    token: str | None = None,
    downloader: HubDownloader | None = None,
    model_files: Iterable[ModelFile] | None = None,
) -> tuple[DownloadResult, ...]:
    resolved_repo_id = (repo_id or os.getenv("HF_MODEL_REPO_ID", "")).strip()
    if not resolved_repo_id:
        raise ModelDownloadError(
            "HF_MODEL_REPO_ID is not configured. Set it to the Hugging Face model "
            "repository containing the five MultiCancer checkpoints."
        )

    resolved_token = token if token is not None else os.getenv("HF_TOKEN")
    resolved_token = resolved_token.strip() if resolved_token else None
    resolved_root = _resolve_repo_root(repo_root)
    fetch = downloader or _download_from_hub
    selected_model_files = MODEL_FILES if model_files is None else tuple(model_files)

    print(f"[models] Repository: {resolved_repo_id}")
    if resolved_token is None:
        print("[models] HF_TOKEN is not configured; only public repositories are accessible.")

    results: list[DownloadResult] = []
    for model_file in selected_model_files:
        destination = resolved_root / model_file.local_path
        if destination.is_file() and destination.stat().st_size > 0:
            print(f"[models] Skip existing {model_file.project}: {model_file.local_path}")
            results.append(DownloadResult(model_file, "skipped", destination))
            continue
        if destination.exists() and not destination.is_file():
            raise ModelDownloadError(
                f"Checkpoint destination is not a file: '{model_file.local_path}'."
            )

        print(f"[models] Download {model_file.project}: {model_file.remote_path}")
        try:
            cached_file = fetch(
                resolved_repo_id,
                model_file.remote_path,
                resolved_token,
            )
        except ModelDownloadError:
            raise
        except Exception as exc:
            message = _download_error_message(
                exc,
                repo_id=resolved_repo_id,
                remote_path=model_file.remote_path,
                token_present=resolved_token is not None,
            )
            raise ModelDownloadError(message) from exc

        if not cached_file.is_file() or cached_file.stat().st_size == 0:
            raise ModelDownloadError(
                f"Downloaded checkpoint is unavailable in the local Hub cache: "
                f"'{model_file.remote_path}'."
            )

        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        status = "downloaded"
        try:
            # Each session owns its staging file; replacement remains atomic.
            with tempfile.NamedTemporaryFile(
                dir=destination.parent, prefix=destination.name + ".", suffix=".part", delete=False
            ) as staged:
                temporary = Path(staged.name)
            shutil.copyfile(cached_file, temporary)
            with INSTALL_LOCK:
                if destination.is_file() and destination.stat().st_size > 0:
                    status = "skipped"
                else:
                    temporary.replace(destination)
        except OSError as exc:
            raise ModelDownloadError(
                f"Cannot install checkpoint at '{model_file.local_path}'."
            ) from exc
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

        results.append(DownloadResult(model_file, status, destination))

    downloaded = sum(result.status == "downloaded" for result in results)
    skipped = sum(result.status == "skipped" for result in results)
    print(f"[models] Ready: {downloaded} downloaded, {skipped} already present.")
    return tuple(results)


def main() -> int:
    try:
        download_models()
    except ModelDownloadError as exc:
        print(f"[models] ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
