from __future__ import annotations

import numpy as np
import pytest

from track_analyser.features import (
    analyse_features,
    compute_ltas,
    spectral_centroid_series,
    spectral_rolloff_series,
)
from track_analyser.utils import AudioInput


def test_compute_ltas_identifies_dominant_frequency():
    sample_rate = 22_050
    duration = 1.0
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    tone = np.sin(2 * np.pi * 440.0 * t).astype(np.float32)

    ltas = compute_ltas(tone, sample_rate)
    peak_frequency = float(ltas.frequencies[np.argmax(ltas.magnitude)])

    assert peak_frequency == pytest.approx(440.0, abs=5.0)


def test_spectral_centroid_matches_expected_for_sine():
    sample_rate = 22_050
    t = np.linspace(0, 1.0, sample_rate, endpoint=False)
    tone = np.sin(2 * np.pi * 1_000.0 * t).astype(np.float32)

    centroid_series = spectral_centroid_series(tone, sample_rate)

    assert centroid_series.mean == pytest.approx(1_000.0, abs=20.0)


def test_spectral_rolloff_increases_with_broadband_signal():
    rng = np.random.default_rng(1337)
    sample_rate = 22_050
    noise = rng.normal(size=sample_rate).astype(np.float32)

    rolloff_series = spectral_rolloff_series(noise, sample_rate)

    assert np.all(rolloff_series.values > 5_000.0)


def test_analyse_features_returns_consistent_structures():
    sample_rate = 22_050
    t = np.linspace(0, 1.0, sample_rate, endpoint=False)
    tone = np.sin(2 * np.pi * 440.0 * t).astype(np.float32)
    audio = AudioInput(samples=tone, sample_rate=sample_rate)

    analysis = analyse_features(audio)

    assert analysis.ltas.frequencies.shape == analysis.ltas.magnitude.shape
    assert analysis.spectral_centroid.values.ndim == 1
    assert analysis.spectral_rolloff.values.ndim == 1
