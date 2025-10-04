"""Backward compatibility shims for :mod:`track_analyser.harmony`."""

from __future__ import annotations

import warnings

from ..harmony import (
    HarmonyAnalysis,
    KeyEstimate,
    SpectralBalance,
    StereoImage,
    analyse_harmony,
)
from ..harmony import ChordHint, MidiSuggestion
from ..harmony import key_estimate as key_estimate  # public alias


def analyse_harmonic(*args, **kwargs) -> HarmonyAnalysis:  # pragma: no cover - shim
    warnings.warn(
        "track_analyser.analysis.harmonic is deprecated; import from"
        " track_analyser.harmony instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return analyse_harmony(*args, **kwargs)


__all__ = [
    "HarmonyAnalysis",
    "SpectralBalance",
    "StereoImage",
    "KeyEstimate",
    "ChordHint",
    "MidiSuggestion",
    "analyse_harmonic",
    "key_estimate",
]

