from __future__ import annotations

import numpy as np

from track_analyser.analysis.beats import BeatAnalysis
from track_analyser.analysis.structure import analyse_structure
from track_analyser.utils import AudioInput


def test_structure_detects_boundary_when_drums_mute():
    sample_rate = 22_050
    duration = 32.0
    t = np.linspace(0.0, duration, int(sample_rate * duration), endpoint=False)

    harmonic = 0.3 * np.sin(2 * np.pi * 110.0 * t)
    drum_times = np.arange(0.0, duration, 0.5)
    active = drum_times[(drum_times < 12.0) | (drum_times >= 20.0)]
    drum_track = np.zeros_like(t)
    hit_length = int(sample_rate * 0.05)
    envelope = np.linspace(1.0, 0.0, hit_length, dtype=np.float32)
    for time in active:
        start = int(time * sample_rate)
        end = min(len(drum_track), start + hit_length)
        span = end - start
        if span > 0:
            drum_track[start:end] += envelope[:span]

    samples = (harmonic + drum_track).astype(np.float32)
    audio = AudioInput(samples=samples, sample_rate=sample_rate)

    beat_times = np.arange(0.0, duration, 0.5)
    hop_length = 512
    beat_frames = (beat_times * sample_rate / hop_length).astype(int)
    beat = BeatAnalysis(
        bpm=120.0,
        beat_times=beat_times.astype(float).tolist(),
        beat_frames=beat_frames.astype(int).tolist(),
        confidence=1.0,
    )

    analysis = analyse_structure(audio, beat, seed=123)
    boundary_times = [segment.start for segment in analysis.segments[1:]]
    assert any(abs(boundary - 12.0) <= 0.5 for boundary in boundary_times)
