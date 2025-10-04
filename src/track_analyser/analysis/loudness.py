"""Loudness and dynamics analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import librosa
from scipy import signal

from ..utils import AudioInput, seed_everything

try:  # pragma: no cover - optional dependency guard
    import pyloudnorm
except ImportError:  # pragma: no cover - fallback implementation
    pyloudnorm = None  # type: ignore[assignment]


@dataclass(slots=True)
class LoudnessAnalysis:
    integrated_lufs: float
    short_term_lufs: List[float]
    momentary_lufs: List[float]
    loudness_range: float
    true_peak_dbfs: float
    rms_dbfs: float


def _windowed_loudness(
    samples: np.ndarray, sample_rate: int, meter_block_size: float
) -> np.ndarray:
    """Return LUFS estimates computed over sliding windows."""

    frame_length = max(1024, int(round(sample_rate * meter_block_size)))
    if frame_length % 2:
        frame_length += 1
    hop_length = max(1, frame_length // 2)
    rms = librosa.feature.rms(
        y=samples, frame_length=frame_length, hop_length=hop_length
    )[0]
    return librosa.amplitude_to_db(rms + 1e-9, ref=1.0)


def measure_loudness(
    samples: np.ndarray,
    sample_rate: int,
    meter_block_size: float = 0.400,
) -> Tuple[float, List[float], List[float], float]:
    """Measure LUFS and loudness range metrics for ``samples``."""

    samples = np.asarray(samples, dtype=np.float32)
    if samples.ndim != 1:
        raise ValueError("measure_loudness expects mono audio samples")

    short_term = _windowed_loudness(samples, sample_rate, meter_block_size=3.0)
    momentary = _windowed_loudness(samples, sample_rate, meter_block_size=meter_block_size)

    if pyloudnorm is not None:
        meter = pyloudnorm.Meter(sample_rate, block_size=meter_block_size)
        integrated = float(meter.integrated_loudness(samples))
        loudness_range_fn = getattr(meter, "loudness_range", None)
        if callable(loudness_range_fn):
            lra = float(loudness_range_fn(samples))
        else:  # pragma: no cover - pyloudnorm<0.1.2 compatibility
            lra = float(
                np.percentile(momentary, 95) - np.percentile(momentary, 5)
            )
    else:  # pragma: no cover - fallback path
        integrated = float(np.mean(momentary))
        lra = float(np.percentile(momentary, 95) - np.percentile(momentary, 5))

    return (
        integrated,
        np.asarray(short_term, dtype=float).tolist(),
        np.asarray(momentary, dtype=float).tolist(),
        lra,
    )


def true_peak_dbtp(samples: np.ndarray, sample_rate: int, *, oversample: int = 8) -> float:
    """Estimate dB true peak using polyphase oversampling."""

    if oversample < 1:
        raise ValueError("oversample must be >= 1")

    samples = np.asarray(samples, dtype=np.float32)
    if samples.ndim != 1:
        raise ValueError("true_peak_dbtp expects mono audio samples")

    if oversample == 1:
        upsampled = samples
    else:
        upsampled = signal.resample_poly(samples, oversample, 1)

    peak = float(np.max(np.abs(upsampled)))
    return float(20.0 * np.log10(peak + 1e-12))


def analyse_loudness(
    audio: AudioInput | str,
    *,
    seed: int,
    meter_block_size: float = 0.400,
) -> LoudnessAnalysis:
    """Compute LUFS, loudness range and crest factor information."""

    if not isinstance(audio, AudioInput):
        raise TypeError("analyse_loudness expects an AudioInput instance")
    seed_everything(seed)

    samples = audio.samples.astype(np.float32)

    integrated, short_term, momentary, loudness_range = measure_loudness(
        samples, audio.sample_rate, meter_block_size
    )
    true_peak_dbfs = true_peak_dbtp(samples, audio.sample_rate)
    rms_val = float(np.sqrt(np.mean(samples**2)))
    rms_dbfs = float(20.0 * np.log10(rms_val + 1e-12))

    return LoudnessAnalysis(
        integrated_lufs=integrated,
        short_term_lufs=short_term,
        momentary_lufs=momentary,
        loudness_range=loudness_range,
        true_peak_dbfs=true_peak_dbfs,
        rms_dbfs=rms_dbfs,
    )
