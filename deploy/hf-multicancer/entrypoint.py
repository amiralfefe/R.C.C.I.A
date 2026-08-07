"""Prepare deployment checkpoints, then replace this process with Streamlit."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from download_models import ModelDownloadError, download_models


def main() -> int:
    repo_root = Path(os.environ.get("RCCIA_REPO_ROOT", "/home/user/app/source")).resolve()
    app_path = repo_root / "projects" / "multicancer" / "app.py"
    if not app_path.is_file():
        print(f"[startup] ERROR: MultiCancer app not found at '{app_path}'.", file=sys.stderr)
        return 1

    try:
        download_models(repo_root=repo_root)
    except ModelDownloadError as exc:
        print(f"[startup] ERROR: {exc}", file=sys.stderr)
        return 1

    os.chdir(repo_root)
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
        "--server.address=0.0.0.0",
        "--server.port=7860",
        "--server.headless=true",
    ]
    print("[startup] Checkpoints ready. Starting Streamlit on 0.0.0.0:7860.")
    os.execv(sys.executable, command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
