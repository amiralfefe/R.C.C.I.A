from __future__ import annotations

import runpy
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
LEUKEMIA_APP_DIR = REPO_ROOT / "projects" / "leukemia"

if str(LEUKEMIA_APP_DIR) not in sys.path:
    sys.path.insert(0, str(LEUKEMIA_APP_DIR))

runpy.run_path(str(LEUKEMIA_APP_DIR / "app.py"), run_name="__main__")
