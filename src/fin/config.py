"""Project-wide paths. Override locations with environment variables."""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("FIN_DATA_DIR", PROJECT_ROOT / "data"))
RUNS_DIR = Path(os.environ.get("FIN_RUNS_DIR", PROJECT_ROOT / "runs"))
