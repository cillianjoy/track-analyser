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
from track_analyser.features import (
    FeatureAnalysis,
    FeatureSeries,
    LongTermAverageSpectrum,
)
from track_analyser.pipeline import TrackAnalysisResult
from track_analyser.rendering.outputs import render_all
from track_analyser.report import ReportOutputs
from track_analyser.stereo import StereoAnalysis, StereoWidthBands
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
        segments=[
            StructuralSegment(
                label="A",
                category="intro",
                start=0.0,
                end=1.0,
                confidence=1.0,
                percussive_energy=0.0,
                harmonic_energy=0.0,
                percussive_ratio=0.0,
            )
        ],
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

    features = FeatureAnalysis(
        ltas=LongTermAverageSpectrum(
            frequencies=np.array([0.0, 1_000.0], dtype=np.float32),
            magnitude=np.array([1.0, 0.5], dtype=np.float32),
        ),
        spectral_centroid=FeatureSeries(values=np.array([100.0, 200.0])),
        spectral_rolloff=FeatureSeries(values=np.array([5_000.0, 6_000.0])),
    )
    stereo = StereoAnalysis(
        mid_rms=0.5,
        side_rms=0.0,
        correlation=1.0,
        width=StereoWidthBands(low=0.0, mid=0.0, high=0.0),
    )

    result = TrackAnalysisResult(
        audio=audio,
        beat=beat,
        downbeat=None,
        structure=structure,
        loudness=loudness,
        harmonic=harmonic,
        features=features,
        stereo=stereo,
        stems=None,
    )

    report_outputs = render_all(result, tmp_path)
    assert isinstance(report_outputs, ReportOutputs)

    report_path = tmp_path / "report.json"
    assert report_outputs.json == report_path
    assert report_path.exists(), "report.json should be rendered"
    with report_path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    assert data["structure"], "Structure entries should be serialised"
    assert "integrated_lufs" in data["loudness"], (
        "Loudness dataclass should be serialised"
    )
    harmonic_data = data["harmonic"]
    assert harmonic_data["key"] == "C major"
    assert harmonic_data["secondary_key"]["key"] == "G major"
    assert harmonic_data["chord_change_points"], "Change points should be exported"
    feature_data = data["features"]
    assert feature_data["spectral_centroid"]["mean"] == np.mean([100.0, 200.0])
    stereo_data = data["stereo"]
    assert stereo_data["mid_rms"] == 0.5

    assert "sections" in report_outputs.csv
    sections_path = report_outputs.csv["sections"]
    assert sections_path.exists(), "sections.csv should be rendered"
    sections = pd.read_csv(sections_path)
    expected_columns = {
        "label",
        "category",
        "start",
        "end",
        "confidence",
        "percussive_energy",
        "harmonic_energy",
        "percussive_ratio",
    }
    assert expected_columns.issubset(set(sections.columns))

    assert "beats" in report_outputs.csv
    beats_path = report_outputs.csv["beats"]
    beats = pd.read_csv(beats_path)
    assert set(beats.columns) == {"index", "time", "frame", "is_downbeat"}

    expected_plots = {
        "waveform_beats",
        "tempogram",
        "novelty",
        "ltas",
        "stereo_width",
    }
    assert expected_plots.issubset(set(report_outputs.plots))
    for plot in expected_plots:
        assert report_outputs.plots[plot].exists(), f"{plot} plot should exist"
