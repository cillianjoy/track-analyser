"""Test configuration for the track-analyser project."""

from __future__ import annotations

import sys
from pathlib import Path


def _ensure_src_on_path() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    src_dir = repo_root / "src"
    if src_dir.exists():
        src = str(src_dir)
        if src not in sys.path:
            sys.path.insert(0, src)


_ensure_src_on_path()
