"""Spectral feature computations for the analysis pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

try:  # pragma: no cover - optional dependency guard
    import librosa
except ImportError as exc:  # pragma: no cover - library is required for the package
    raise RuntimeError("librosa is required for track_analyser") from exc

from .utils import AudioInput


@dataclass(slots=True)
class LongTermAverageSpectrum:
    """Represents the long-term average spectrum (LTAS) of a signal."""

    frequencies: np.ndarray
    magnitude: np.ndarray

    def as_dict(self) -> dict[str, Sequence[float]]:
        """Serialise the LTAS into JSON friendly containers."""

        return {
            "frequencies": self.frequencies.tolist(),
            "magnitude": self.magnitude.tolist(),
        }


@dataclass(slots=True)
class FeatureSeries:
    """Container for frame-wise spectral features."""

    values: np.ndarray

    @property
    def mean(self) -> float:
        if self.values.size == 0:
            return 0.0
        return float(np.mean(self.values))

    @property
    def median(self) -> float:
        if self.values.size == 0:
            return 0.0
        return float(np.median(self.values))

    @property
    def as_list(self) -> list[float]:
        return self.values.tolist()


@dataclass(slots=True)
class FeatureAnalysis:
    """Aggregates the spectral feature outputs."""

    ltas: LongTermAverageSpectrum
    spectral_centroid: FeatureSeries
    spectral_rolloff: FeatureSeries


def compute_ltas(
    samples: np.ndarray,
    sample_rate: int,
    *,
    n_fft: int = 2_048,
    hop_length: int = 512,
    window: str = "hann",
) -> LongTermAverageSpectrum:
    """Compute the long-term average spectrum for ``samples``."""

    mono = np.asarray(samples, dtype=np.float32)
    if mono.ndim > 1:
        mono = np.mean(mono, axis=0)
    stft = librosa.stft(mono, n_fft=n_fft, hop_length=hop_length, window=window)
    magnitude = np.mean(np.abs(stft), axis=1)
    frequencies = librosa.fft_frequencies(sr=sample_rate, n_fft=n_fft)
    return LongTermAverageSpectrum(frequencies=frequencies, magnitude=magnitude)


def spectral_centroid_series(
    samples: np.ndarray,
    sample_rate: int,
    *,
    n_fft: int = 2_048,
    hop_length: int = 512,
) -> FeatureSeries:
    """Return the spectral centroid trajectory for ``samples``."""

    mono = np.asarray(samples, dtype=np.float32)
    if mono.ndim > 1:
        mono = np.mean(mono, axis=0)
    centroid = librosa.feature.spectral_centroid(
        y=mono, sr=sample_rate, n_fft=n_fft, hop_length=hop_length
    )
    return FeatureSeries(values=np.squeeze(centroid, axis=0))


def spectral_rolloff_series(
    samples: np.ndarray,
    sample_rate: int,
    *,
    roll_percent: float = 0.85,
    n_fft: int = 2_048,
    hop_length: int = 512,
) -> FeatureSeries:
    """Return the spectral roll-off trajectory for ``samples``."""

    mono = np.asarray(samples, dtype=np.float32)
    if mono.ndim > 1:
        mono = np.mean(mono, axis=0)
    rolloff = librosa.feature.spectral_rolloff(
        y=mono,
        sr=sample_rate,
        roll_percent=roll_percent,
        n_fft=n_fft,
        hop_length=hop_length,
    )
    return FeatureSeries(values=np.squeeze(rolloff, axis=0))


def analyse_features(
    audio: AudioInput,
    *,
    n_fft: int = 2_048,
    hop_length: int = 512,
    roll_percent: float = 0.85,
) -> FeatureAnalysis:
    """Derive spectral summary features for ``audio``."""

    ltas = compute_ltas(audio.samples, audio.sample_rate, n_fft=n_fft, hop_length=hop_length)
    centroid = spectral_centroid_series(
        audio.samples,
        audio.sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
    )
    rolloff = spectral_rolloff_series(
        audio.samples,
        audio.sample_rate,
        roll_percent=roll_percent,
        n_fft=n_fft,
        hop_length=hop_length,
    )
    return FeatureAnalysis(ltas=ltas, spectral_centroid=centroid, spectral_rolloff=rolloff)
