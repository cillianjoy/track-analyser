from __future__ import annotations

import numpy as np
import librosa

from track_analyser.analysis import beats
from track_analyser.harmony import analyse_harmony, key_estimate
from track_analyser.utils import AudioInput


def _synth_triad(midi_root: int, quality: str, sr: int, duration: float) -> np.ndarray:
    intervals = {"maj": [0, 4, 7], "min": [0, 3, 7]}
    t = np.linspace(0.0, duration, int(sr * duration), endpoint=False)
    chord = np.zeros_like(t)
    for interval in intervals[quality]:
        freq = librosa.midi_to_hz(midi_root + interval)
        chord += np.sin(2 * np.pi * freq * t)
    envelope = np.hanning(t.size)
    if np.max(np.abs(envelope)) > 0:
        chord *= envelope / np.max(envelope)
    return chord.astype(np.float32)


def test_harmony_pipeline_detects_key_and_changes() -> None:
    sr = 22_050
    duration = 1.0
    progression = [
        _synth_triad(60, "maj", sr, duration),  # C major
        _synth_triad(65, "maj", sr, duration),  # F major
        _synth_triad(67, "maj", sr, duration),  # G major
        _synth_triad(60, "maj", sr, duration),  # C major return
    ]
    audio_samples = np.concatenate(progression)
    audio_samples /= np.max(np.abs(audio_samples))
    audio_samples = audio_samples.astype(np.float32)

    key_result = key_estimate(audio_samples, sr)
    assert key_result.best.key == "C major"
    assert key_result.best.confidence > key_result.second_best.confidence
    assert key_result.second_best.key in {"G major", "F major"}

    audio = AudioInput(samples=audio_samples, sample_rate=sr)
    beat_times = np.arange(len(progression)) * duration
    beat_analysis = beats.build_beat_analysis(
        bpm=60.0,
        beat_times=beat_times,
        sr=sr,
    )

    harmony_result = analyse_harmony(audio, beat_analysis, None, seed=123)
    assert harmony_result.primary_key.key == "C major"
    assert harmony_result.primary_key.confidence > harmony_result.secondary_key.confidence
    assert harmony_result.secondary_key.key in {"G major", "F major"}

    change_times = np.array([point.time for point in harmony_result.chord_change_points])
    assert change_times.size > 0
    expected = np.array([1.0, 2.0, 3.0])
    matches = 0
    for boundary in expected:
        if np.any(np.abs(change_times - boundary) <= 0.25):
            matches += 1
    accuracy = matches / expected.size
    assert accuracy >= 0.7
    assert all(0.0 <= point.strength <= 1.0 for point in harmony_result.chord_change_points)
