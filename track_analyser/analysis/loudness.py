"""Loudness and dynamics analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import librosa

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

    if pyloudnorm is not None:
        meter = pyloudnorm.Meter(audio.sample_rate, block_size=meter_block_size)
        integrated = float(meter.integrated_loudness(samples))
        short_term = meter.short_term_loudness(samples)
        momentary = meter.momentary_loudness(samples)
        lra = float(meter.loudness_range(samples))
    else:  # pragma: no cover - fallback path
        frame_length = max(1024, int(audio.sample_rate * meter_block_size))
        if frame_length % 2:
            frame_length += 1
        hop_length = frame_length // 2
        rms = librosa.feature.rms(
            y=samples, frame_length=frame_length, hop_length=hop_length
        )[0]
        rms_db = librosa.amplitude_to_db(rms + 1e-9, ref=np.max)
        integrated = float(np.mean(rms_db))
        short_term = rms_db.tolist()
        momentary = rms_db.tolist()
        lra = float(np.percentile(rms_db, 95) - np.percentile(rms_db, 5))

    true_peak = float(np.max(np.abs(samples)))
    true_peak_dbfs = float(20.0 * np.log10(true_peak + 1e-9))
    rms_val = float(np.sqrt(np.mean(samples**2)))
    rms_dbfs = float(20.0 * np.log10(rms_val + 1e-9))

    return LoudnessAnalysis(
        integrated_lufs=integrated,
        short_term_lufs=np.asarray(short_term, dtype=float).tolist(),
        momentary_lufs=np.asarray(momentary, dtype=float).tolist(),
        loudness_range=lra,
        true_peak_dbfs=true_peak_dbfs,
        rms_dbfs=rms_dbfs,
    )
