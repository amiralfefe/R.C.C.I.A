from __future__ import annotations

import runpy
import sys
from pathlib import Path


CURRENT_DIR = Path(__file__).resolve().parent
LEUKEMIA_APP_DIR_CANDIDATES = [
    CURRENT_DIR / "projects" / "leukemia",
    CURRENT_DIR.parents[1] / "projects" / "leukemia",
]

LEUKEMIA_APP_DIR = next(
    (path for path in LEUKEMIA_APP_DIR_CANDIDATES if (path / "app.py").exists()),
    None,
)

if LEUKEMIA_APP_DIR is None:
    raise FileNotFoundError(
        "Could not find projects/leukemia/app.py. Preserve the Space structure "
        "documented in deploy/hf-leukemia/README.md."
    )

sys.path.insert(0, str(LEUKEMIA_APP_DIR))
runpy.run_path(str(LEUKEMIA_APP_DIR / "app.py"), run_name="__main__")
