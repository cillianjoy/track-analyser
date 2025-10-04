from __future__ import annotations

import numpy as np
import pytest

from track_analyser.stereo import (
    analyse_stereo,
    frequency_dependent_width,
    mid_side_rms,
    mono_compatibility_correlation,
)
from track_analyser.utils import AudioInput


def test_mono_audio_yields_zero_side_and_full_correlation():
    sample_rate = 22_050
    t = np.linspace(0, 1.0, sample_rate, endpoint=False)
    mono = np.sin(2 * np.pi * 440.0 * t).astype(np.float32)
    audio = AudioInput(samples=mono, sample_rate=sample_rate)

    analysis = analyse_stereo(audio)

    assert analysis.side_rms == pytest.approx(0.0, abs=1e-6)
    assert analysis.correlation == pytest.approx(1.0, abs=1e-6)
    assert analysis.width.low == pytest.approx(0.0, abs=1e-6)
    assert analysis.width.mid == pytest.approx(0.0, abs=1e-6)
    assert analysis.width.high == pytest.approx(0.0, abs=1e-6)


def test_mid_side_rms_for_imbalanced_stereo_signal():
    sample_rate = 22_050
    t = np.linspace(0, 1.0, sample_rate, endpoint=False)
    left = np.sin(2 * np.pi * 440.0 * t).astype(np.float32)
    right = 0.5 * left
    stereo = np.vstack([left, right])

    mid_rms_value, side_rms_value = mid_side_rms(stereo)

    assert mid_rms_value > side_rms_value > 0.0


def test_frequency_dependent_width_increases_with_phase_difference():
    sample_rate = 22_050
    t = np.linspace(0, 1.0, sample_rate, endpoint=False)
    left = np.sin(2 * np.pi * 440.0 * t).astype(np.float32)
    right = np.sin(2 * np.pi * 440.0 * t + np.pi / 2).astype(np.float32)
    stereo = np.vstack([left, right])

    width = frequency_dependent_width(stereo, sample_rate)

    assert width.low >= 0.0
    assert width.mid >= 0.0
    assert width.high >= 0.0
    assert max(width.low, width.mid, width.high) > 0.0


def test_mono_compatibility_handles_constant_channels():
    left = np.ones(10, dtype=np.float32)
    right = np.ones(10, dtype=np.float32)
    stereo = np.vstack([left, right])

    corr = mono_compatibility_correlation(stereo)

    assert corr == pytest.approx(1.0)
