"""Streamlit Community Cloud bootstrap for the frozen MultiCancer V1 hub."""

from __future__ import annotations

import importlib.util
import runpy
import sys
from pathlib import Path
from types import ModuleType


ROOT = Path(__file__).resolve().parents[2]
DOWNLOADER_PATH = ROOT / "deploy" / "hf-multicancer" / "download_models.py"
HUB_APP_PATH = ROOT / "projects" / "multicancer" / "app.py"
DOWNLOADER_MODULE_NAME = "rccia_multicancer_model_downloader"


class BootstrapError(RuntimeError):
    """Raised when disk checkpoint preparation cannot complete safely."""


def load_downloader_module(path: Path = DOWNLOADER_PATH) -> ModuleType:
    cached_module = sys.modules.get(DOWNLOADER_MODULE_NAME)
    if cached_module is not None:
        return cached_module

    if not path.is_file():
        raise BootstrapError(f"Model downloader not found: '{path}'.")

    spec = importlib.util.spec_from_file_location(DOWNLOADER_MODULE_NAME, path)
    if spec is None or spec.loader is None:
        raise BootstrapError(f"Cannot import model downloader: '{path}'.")

    module = importlib.util.module_from_spec(spec)
    sys.modules[DOWNLOADER_MODULE_NAME] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        sys.modules.pop(DOWNLOADER_MODULE_NAME, None)
        raise BootstrapError("Cannot initialize the model downloader.") from exc
    return module


def expected_checkpoint_paths(
    repo_root: Path = ROOT,
    *,
    downloader_module: ModuleType | None = None,
) -> tuple[Path, ...]:
    module = downloader_module or load_downloader_module()
    return tuple(repo_root / model_file.local_path for model_file in module.MODEL_FILES)


def missing_checkpoint_paths(
    repo_root: Path = ROOT,
    *,
    downloader_module: ModuleType | None = None,
) -> tuple[Path, ...]:
    return tuple(
        path
        for path in expected_checkpoint_paths(
            repo_root,
            downloader_module=downloader_module,
        )
        if not path.is_file() or path.stat().st_size <= 0
    )


def ensure_checkpoints(
    repo_root: Path = ROOT,
    *,
    downloader_module: ModuleType | None = None,
) -> tuple[Path, ...]:
    """Prepare missing files on disk without loading any PyTorch model."""
    module = downloader_module or load_downloader_module()
    expected = expected_checkpoint_paths(repo_root, downloader_module=module)
    missing_before = missing_checkpoint_paths(repo_root, downloader_module=module)
    if not missing_before:
        return expected

    try:
        module.download_models(repo_root=repo_root)
    except module.ModelDownloadError as exc:
        raise BootstrapError(f"Checkpoint download failed: {exc}") from exc
    except Exception as exc:
        raise BootstrapError(
            f"Checkpoint download failed ({type(exc).__name__})."
        ) from exc

    missing_after = missing_checkpoint_paths(repo_root, downloader_module=module)
    if missing_after:
        relative_paths = ", ".join(
            path.relative_to(repo_root).as_posix() for path in missing_after
        )
        raise BootstrapError(
            f"Checkpoint preparation incomplete. Missing files: {relative_paths}."
        )
    return expected


def run_multicancer_hub(repo_root: Path = ROOT) -> None:
    hub_path = repo_root / "projects" / "multicancer" / "app.py"
    if not hub_path.is_file():
        raise BootstrapError(f"MultiCancer hub not found: '{hub_path}'.")
    runpy.run_path(str(hub_path), run_name="__main__")


def render_bootstrap_error(message: str) -> None:
    import streamlit as st

    st.set_page_config(page_title="R.C.C.I.A MultiCancer", layout="wide")
    st.title("R.C.C.I.A MultiCancer")
    st.error("Le deploiement ne peut pas preparer ses checkpoints.")
    st.code(message)
    st.warning(
        "Demonstrateur educatif / portfolio uniquement. Aucun diagnostic medical."
    )


def main() -> None:
    try:
        ensure_checkpoints(ROOT)
        run_multicancer_hub(ROOT)
    except BootstrapError as exc:
        render_bootstrap_error(str(exc))


if __name__ == "__main__":
    main()
