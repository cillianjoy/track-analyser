from __future__ import annotations

import json
import numpy as np
import pandas as pd

from track_analyser.analysis.beats import BeatAnalysis
from track_analyser.harmony import (
    ChordChangePoint,
    ChordHint,
    HarmonyAnalysis,
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
    beat = BeatAnalysis(
        bpm=120.0, beat_times=[0.0, 0.5], beat_frames=[0, 220], confidence=0.9
    )
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
    primary_key = KeyEstimate(key="C major", confidence=0.95)
    secondary_key = KeyEstimate(key="G major", confidence=0.65)
    empty_notes = pd.DataFrame(
        columns=["start", "duration", "pitch", "velocity", "channel"]
    )
    harmonic = HarmonyAnalysis(
        spectral_balance=spectral_balance,
        stereo_image=stereo_image,
        primary_key=primary_key,
        secondary_key=secondary_key,
        chord_hints=[ChordHint(time=0.0, chord="Cmaj", confidence=0.9)],
        chord_change_points=[ChordChangePoint(time=0.5, strength=0.8)],
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
    assert "integrated_lufs" in data["loudness"], (
        "Loudness dataclass should be serialised"
    )
    harmonic_data = data["harmonic"]
    assert harmonic_data["key"] == "C major"
    assert harmonic_data["secondary_key"]["key"] == "G major"
    assert harmonic_data["chord_change_points"], "Change points should be exported"
