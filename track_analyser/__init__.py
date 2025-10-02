"""Top-level package for track_analyser."""

from __future__ import annotations

from importlib.metadata import version, PackageNotFoundError

from .pipeline import analyse_track, TrackAnalysisResult

__all__ = ["analyse_track", "TrackAnalysisResult", "get_version"]


def get_version() -> str:
    """Return the installed package version.

    Falls back to ``"0.0.0"`` when the distribution metadata is missing,
    which is the case when the package is executed from a source checkout
    without being installed in editable mode.
    """

    try:
        return version("track-analyser")
    except PackageNotFoundError:  # pragma: no cover - defensive guard
        return "0.0.0"
