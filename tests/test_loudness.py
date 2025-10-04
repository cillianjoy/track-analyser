"""Regression tests for loudness analysis utilities."""

from __future__ import annotations

import math

import numpy as np
import pytest

from track_analyser.analysis.loudness import (
    analyse_loudness,
    measure_loudness,
    true_peak_dbtp,
)
from track_analyser.utils import AudioInput

pyloudnorm = pytest.importorskip("pyloudnorm")


def _generate_minus_18_dbfs_sine(
    sample_rate: int = 44_100,
    duration: float = 1.0,
    frequency: float = 1000.0,
) -> np.ndarray:
    """Return a sine tone with a nominal loudness of −18 dBFS."""

    time = np.linspace(0.0, duration, int(sample_rate * duration), endpoint=False)
    rms_amplitude = 10 ** (-18.0 / 20.0)
    peak_amplitude = rms_amplitude * math.sqrt(2.0)
    return (peak_amplitude * np.sin(2.0 * np.pi * frequency * time)).astype(np.float32)


def test_measure_loudness_matches_expected_values() -> None:
    """The helper should return LUFS close to the expected programme level."""

    sample_rate = 48_000
    samples = _generate_minus_18_dbfs_sine(sample_rate=sample_rate)

    integrated, short_term, momentary, lra = measure_loudness(samples, sample_rate)

    assert integrated == pytest.approx(-18.0, abs=0.3)
    assert short_term  # pyloudnorm returns multiple frames
    assert momentary


def test_true_peak_dbtp_polyphase_oversampling() -> None:
    """True peak estimation should align with the theoretical peak of the tone."""

    sample_rate = 44_100
    samples = _generate_minus_18_dbfs_sine(sample_rate=sample_rate)
    expected = 20.0 * math.log10(float(np.max(np.abs(samples))))

    true_peak = true_peak_dbtp(samples, sample_rate, oversample=8)

    assert true_peak == pytest.approx(expected, abs=0.2)


def test_analyse_loudness_uses_helpers() -> None:
    """The public API should expose the helper results."""

    sample_rate = 48_000
    samples = _generate_minus_18_dbfs_sine(sample_rate=sample_rate)
    audio = AudioInput(samples=samples, sample_rate=sample_rate)

    result = analyse_loudness(audio, seed=0)
    expected_integrated, expected_short, expected_momentary, expected_lra = measure_loudness(
        samples, sample_rate
    )
    expected_true_peak = true_peak_dbtp(samples, sample_rate)

    assert result.integrated_lufs == pytest.approx(expected_integrated, abs=1e-6)
    assert result.short_term_lufs == expected_short
    assert result.momentary_lufs == expected_momentary
    assert result.loudness_range == pytest.approx(expected_lra, abs=1e-6)
    assert result.true_peak_dbfs == pytest.approx(expected_true_peak, abs=1e-6)
