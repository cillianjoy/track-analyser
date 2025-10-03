from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from track_analyser.analysis.beats import BeatAnalysis
from track_analyser.analysis.harmonic import (
    HarmonicAnalysis,
    KeyEstimate,
    MidiSuggestion,
    SpectralBalance,
    StereoImage,
)
from track_analyser.analysis.loudness import LoudnessAnalysis
from track_analyser.analysis.structure import StructuralSegment, StructureAnalysis
from track_analyser.pipeline import TrackAnalysisResult
from track_analyser.rendering.outputs import render_all
from track_analyser.utils import AudioInput


def test_render_all_writes_summary_json(tmp_path):
    audio = AudioInput(
        samples=np.zeros(4_410, dtype=np.float32),
        sample_rate=4_410,
        path="dummy.wav",
    )
    beat = BeatAnalysis(bpm=120.0, beat_times=[0.0, 0.5], beat_frames=[0, 220], confidence=0.9)
    structure = StructureAnalysis(
        segments=[StructuralSegment(label="A", start=0.0, end=1.0, confidence=1.0)],
        novelty_curve=[0.1, 0.2],
    )
    loudness = LoudnessAnalysis(
        integrated_lufs=-14.0,
        short_term_lufs=[-14.0, -13.5],
        momentary_lufs=[-13.8, -13.2],
        loudness_range=1.2,
        true_peak_dbfs=-1.0,
        rms_dbfs=-12.0,
    )
    spectral_balance = SpectralBalance(low_band=0.3, mid_band=0.4, high_band=0.3)
    stereo_image = StereoImage(correlation=1.0, balance=0.0)
    key_estimate = KeyEstimate(key="C major", confidence=0.95)
    empty_notes = pd.DataFrame(columns=["start", "duration", "pitch", "velocity"])
    harmonic = HarmonicAnalysis(
        spectral_balance=spectral_balance,
        stereo_image=stereo_image,
        key_estimate=key_estimate,
        chord_hints=[],
        hook_suggestion=MidiSuggestion(name="hook", notes=empty_notes),
        bass_suggestion=MidiSuggestion(name="bass", notes=empty_notes),
    )

    result = TrackAnalysisResult(
        audio=audio,
        beat=beat,
        downbeat=None,
        structure=structure,
        loudness=loudness,
        harmonic=harmonic,
        stems=None,
    )

    render_all(result, tmp_path)

    summary_path = tmp_path / "summary.json"
    assert summary_path.exists(), "summary.json should be rendered"
    with summary_path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    assert data["structure"], "Structure entries should be serialised"
    assert "integrated_lufs" in data["loudness"], "Loudness dataclass should be serialised"
