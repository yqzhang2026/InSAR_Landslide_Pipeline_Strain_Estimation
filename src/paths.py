from __future__ import annotations

import sys
from pathlib import Path


def program_dir() -> Path:
    """Directory next to the program (and the ``data/`` folder).

    Development: folder containing ``main.py``.
    Frozen ``.app``: folder that contains the ``.app`` bundle.
    Frozen exe: folder that contains the executable.
    """
    if getattr(sys, "frozen", False):
        exe = Path(sys.executable).resolve()
        if exe.parent.name == "MacOS" and exe.parent.parent.name == "Contents":
            return exe.parent.parent.parent.parent
        return exe.parent
    return Path(__file__).resolve().parents[1]


def resource_dir() -> Path:
    """Bundled resources (logos, etc.). Frozen apps use ``_MEIPASS``."""
    meipass = getattr(sys, "_MEIPASS", None)
    if getattr(sys, "frozen", False) and meipass:
        return Path(meipass)
    return program_dir()
