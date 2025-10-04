"""Stereo image analysis helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

try:  # pragma: no cover - optional dependency guard
    import librosa
except ImportError as exc:  # pragma: no cover - library is required for the package
    raise RuntimeError("librosa is required for track_analyser") from exc

from .utils import AudioInput

_EPS = 1e-12


@dataclass(slots=True)
class StereoWidthBands:
    """Frequency dependent stereo width estimates."""

    low: float
    mid: float
    high: float

    def as_dict(self) -> dict[str, float]:
        return {"low": self.low, "mid": self.mid, "high": self.high}


@dataclass(slots=True)
class StereoAnalysis:
    """Aggregate container for stereo image metrics."""

    mid_rms: float
    side_rms: float
    correlation: float
    width: StereoWidthBands


def _ensure_stereo_array(audio: AudioInput) -> np.ndarray:
    if audio.stereo_samples is None:
        mono = np.asarray(audio.samples, dtype=np.float32)
        if mono.ndim == 1:
            return np.vstack([mono, mono])
        return mono[:2]

    stereo = np.asarray(audio.stereo_samples, dtype=np.float32)
    if stereo.ndim == 1:
        return np.vstack([stereo, stereo])
    if stereo.shape[0] == 2:
        return stereo
    if stereo.shape[1] == 2:
        return np.transpose(stereo)
    return stereo[:2]


def mid_side_rms(stereo: np.ndarray) -> tuple[float, float]:
    left, right = np.asarray(stereo, dtype=np.float32)
    mid = 0.5 * (left + right)
    side = 0.5 * (left - right)
    if mid.size == 0:
        return 0.0, 0.0
    mid_rms = float(np.sqrt(np.mean(np.square(mid))))
    side_rms = float(np.sqrt(np.mean(np.square(side))))
    return mid_rms, side_rms


def mono_compatibility_correlation(stereo: np.ndarray) -> float:
    left, right = np.asarray(stereo, dtype=np.float32)
    if left.size == 0 or right.size == 0:
        return 1.0
    left = left - np.mean(left)
    right = right - np.mean(right)
    denom = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denom <= _EPS:
        return 1.0
    corr = float(np.dot(left, right) / denom)
    return float(np.clip(corr, -1.0, 1.0))


def frequency_dependent_width(
    stereo: np.ndarray,
    sample_rate: int,
    *,
    bands: Sequence[tuple[str, float, float]] | None = None,
    n_fft: int = 2_048,
    hop_length: int = 512,
) -> StereoWidthBands:
    left, right = np.asarray(stereo, dtype=np.float32)
    stft_left = librosa.stft(left, n_fft=n_fft, hop_length=hop_length, window="hann")
    stft_right = librosa.stft(right, n_fft=n_fft, hop_length=hop_length, window="hann")
    mid = 0.5 * (stft_left + stft_right)
    side = 0.5 * (stft_left - stft_right)

    freqs = librosa.fft_frequencies(sr=sample_rate, n_fft=n_fft)
    nyquist = sample_rate / 2.0
    if bands is None:
        bands = (
            ("low", 0.0, min(200.0, nyquist)),
            ("mid", 200.0, min(2_000.0, nyquist)),
            ("high", 2_000.0, nyquist),
        )

    mid_energy = np.abs(mid) ** 2
    side_energy = np.abs(side) ** 2
    width_map: dict[str, float] = {"low": 0.0, "mid": 0.0, "high": 0.0}
    for name, low, high in bands:
        mask = (freqs >= low) & (freqs <= high)
        if not np.any(mask):
            width_map[name] = 0.0
            continue
        mid_band_energy = float(np.mean(mid_energy[mask]))
        side_band_energy = float(np.mean(side_energy[mask]))
        if mid_band_energy <= _EPS:
            width_map[name] = 0.0
        else:
            width_map[name] = float(np.sqrt(side_band_energy / mid_band_energy))

    return StereoWidthBands(
        low=width_map.get("low", 0.0),
        mid=width_map.get("mid", 0.0),
        high=width_map.get("high", 0.0),
    )


def analyse_stereo(
    audio: AudioInput,
    *,
    n_fft: int = 2_048,
    hop_length: int = 512,
    bands: Sequence[tuple[str, float, float]] | None = None,
) -> StereoAnalysis:
    stereo = _ensure_stereo_array(audio)
    mid_rms_value, side_rms_value = mid_side_rms(stereo)
    correlation = mono_compatibility_correlation(stereo)
    width = frequency_dependent_width(
        stereo,
        audio.sample_rate,
        bands=bands,
        n_fft=n_fft,
        hop_length=hop_length,
    )
    return StereoAnalysis(
        mid_rms=mid_rms_value,
        side_rms=side_rms_value,
        correlation=correlation,
        width=width,
    )
