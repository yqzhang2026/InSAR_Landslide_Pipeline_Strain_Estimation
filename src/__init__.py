"""Landslide PySide6 desktop app package."""

import os

from src.paths import program_dir

_mpl_cache = program_dir() / ".mplconfig"
_mpl_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_cache))
