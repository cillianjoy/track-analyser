"""Regression tests for loudness analysis utilities."""

from __future__ import annotations

import numpy as np

from track_analyser.analysis.loudness import analyse_loudness
from track_analyser.utils import AudioInput


def test_analyse_loudness_returns_with_pyloudnorm_installed() -> None:
    """``analyse_loudness`` should run without raising when pyloudnorm is available."""

    sample_rate = 44_100
    duration = 0.5
    time = np.linspace(0.0, duration, int(sample_rate * duration), endpoint=False)
    samples = (0.1 * np.sin(2.0 * np.pi * 440.0 * time)).astype(np.float32)
    audio = AudioInput(samples=samples, sample_rate=sample_rate)

    result = analyse_loudness(audio, seed=0)

    assert result.integrated_lufs is not None
    assert result.short_term_lufs
    assert result.momentary_lufs
