"""Calibration compare repeat-run helpers."""

from __future__ import annotations

import os
from pathlib import Path

DEFAULT_OUTPUT_ROOT = Path(
    os.environ.get("WALLA_OUTPUT_ROOT", "/research/projects/ecoai/yl2310/WALLA")
).expanduser()


def resolve_output_dir(path_str: str) -> Path:
    path = Path(path_str).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (DEFAULT_OUTPUT_ROOT / path).resolve()
