"""Streamlit Community Cloud bootstrap for the R.C.C.I.A Leukemia demo."""

from __future__ import annotations

import importlib.util
import runpy
import sys
from pathlib import Path
from types import ModuleType


ROOT = Path(__file__).resolve().parents[2]
DOWNLOADER_PATH = ROOT / "deploy" / "hf-multicancer" / "download_models.py"
LEUKEMIA_APP_PATH = ROOT / "projects" / "leukemia" / "app.py"
LEUKEMIA_REMOTE_PATH = "leukemia/best_model.pt"
DOWNLOADER_MODULE_NAME = "rccia_multicancer_model_downloader"


class BootstrapError(RuntimeError):
    """Raised when the Leukemia checkpoint cannot be prepared safely."""


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


def leukemia_model_file(*, downloader_module: ModuleType | None = None):
    module = downloader_module or load_downloader_module()
    matches = tuple(
        model_file
        for model_file in module.MODEL_FILES
        if model_file.remote_path == LEUKEMIA_REMOTE_PATH
    )
    if len(matches) != 1:
        raise BootstrapError(
            f"Expected one downloader mapping for '{LEUKEMIA_REMOTE_PATH}'."
        )
    return matches[0]


def expected_checkpoint_path(
    repo_root: Path = ROOT,
    *,
    downloader_module: ModuleType | None = None,
) -> Path:
    model_file = leukemia_model_file(downloader_module=downloader_module)
    return repo_root / model_file.local_path


def checkpoint_is_ready(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def ensure_checkpoint(
    repo_root: Path = ROOT,
    *,
    downloader_module: ModuleType | None = None,
) -> Path:
    """Use the local checkpoint first, otherwise download only Leukemia."""
    module = downloader_module or load_downloader_module()
    model_file = leukemia_model_file(downloader_module=module)
    checkpoint = repo_root / model_file.local_path
    if checkpoint_is_ready(checkpoint):
        return checkpoint

    try:
        module.download_models(
            repo_root=repo_root,
            model_files=(model_file,),
        )
    except module.ModelDownloadError as exc:
        raise BootstrapError(f"Checkpoint download failed: {exc}") from exc
    except Exception as exc:
        raise BootstrapError(
            f"Checkpoint download failed ({type(exc).__name__})."
        ) from exc

    if not checkpoint_is_ready(checkpoint):
        relative_path = checkpoint.relative_to(repo_root).as_posix()
        raise BootstrapError(
            f"Checkpoint preparation incomplete. Missing file: {relative_path}."
        )
    return checkpoint


def run_leukemia_app(repo_root: Path = ROOT) -> None:
    app_path = repo_root / "projects" / "leukemia" / "app.py"
    if not app_path.is_file():
        raise BootstrapError(f"Leukemia app not found: '{app_path}'.")

    app_dir = str(app_path.parent)
    if app_dir not in sys.path:
        sys.path.insert(0, app_dir)
    runpy.run_path(str(app_path), run_name="__main__")


def render_bootstrap_error(message: str) -> None:
    import streamlit as st

    st.set_page_config(page_title="R.C.C.I.A Leukemia", layout="wide")
    st.title("R.C.C.I.A Leukemia")
    st.error("Le deploiement ne peut pas preparer son checkpoint.")
    st.code(message)
    st.warning(
        "Demonstrateur educatif / portfolio uniquement. Aucun diagnostic medical."
    )


def main() -> None:
    try:
        ensure_checkpoint(ROOT)
        run_leukemia_app(ROOT)
    except BootstrapError as exc:
        render_bootstrap_error(str(exc))


if __name__ == "__main__":
    main()
