"""Integration tests for tempo utilities."""

from __future__ import annotations

import numpy as np

from track_analyser.tempo import beat_grid, estimate_bpm


def _noisy_click_track(
    bpm: float = 120.0,
    bars: int = 64,
    sr: int = 48000,
    noise_level: float = 0.02,
) -> tuple[np.ndarray, int, np.ndarray]:
    beats_per_bar = 4
    total_beats = bars * beats_per_bar
    beat_period = 60.0 / bpm
    duration = total_beats * beat_period
    length = int(duration * sr)

    click = np.zeros(length, dtype=np.float32)
    beat_samples = (np.arange(total_beats) * beat_period * sr).astype(int)
    click_length = int(0.01 * sr)
    decay = np.exp(-np.linspace(0.0, 6.0, click_length))

    for idx in beat_samples:
        end = min(length, idx + click_length)
        click[idx:end] += decay[: end - idx]

    rng = np.random.default_rng(1234)
    noise = rng.normal(scale=noise_level, size=length)
    signal = click + noise.astype(np.float32)

    beat_times = beat_samples / sr
    return signal.astype(np.float32), sr, beat_times


def test_estimate_bpm_for_noisy_click_track() -> None:
    y, sr, _ = _noisy_click_track()
    bpm = estimate_bpm(y, sr)
    assert abs(bpm - 120.0) <= 0.1


def test_beat_grid_alignment_remains_under_five_milliseconds() -> None:
    y, sr, expected_times = _noisy_click_track()
    grid = beat_grid(y, sr)

    assert grid.shape[0] >= expected_times.size

    actual_times = grid["time"].to_numpy()[: expected_times.size]
    misalignment = np.abs(actual_times - expected_times[: actual_times.size])
    assert float(np.max(misalignment)) <= 0.005
