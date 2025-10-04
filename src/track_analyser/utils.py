"""Utility helpers shared across analysis modules."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np

try:  # pragma: no cover - optional dependency guard
    import librosa
except ImportError as exc:  # pragma: no cover - library is required for the package
    raise RuntimeError("librosa is required for track_analyser") from exc

try:  # pragma: no cover - optional dependency guard
    import resampy
except ImportError:  # pragma: no cover - resampy is optional, librosa can resample
    resampy = None  # type: ignore[assignment]

from .io import load_audio

DEFAULT_SR = 44_100
DEFAULT_SEED = 13_370


@dataclass(slots=True)
class AudioInput:
    """Representation of audio data with optional stereo information."""

    samples: np.ndarray
    sample_rate: int
    path: Optional[str] = None
    stereo_samples: Optional[np.ndarray] = None

    @property
    def duration(self) -> float:
        return float(len(self.samples)) / float(self.sample_rate)


def deterministic_rng(seed: int = DEFAULT_SEED) -> np.random.Generator:
    """Return a :class:`numpy.random.Generator` seeded deterministically."""

    return np.random.default_rng(seed)


def seed_everything(seed: int = DEFAULT_SEED) -> None:
    """Seed global RNGs for deterministic behaviour."""

    np.random.seed(seed)
    random.seed(seed)


def _resample(samples: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return samples
    if samples.ndim == 1:
        if resampy is not None:
            return resampy.resample(samples, orig_sr, target_sr)
        return librosa.resample(samples, orig_sr=orig_sr, target_sr=target_sr)
    channels = []
    for channel in np.atleast_2d(samples):
        if resampy is not None:
            channels.append(resampy.resample(channel, orig_sr, target_sr))
        else:
            channels.append(
                librosa.resample(channel, orig_sr=orig_sr, target_sr=target_sr)
            )
    return np.asarray(channels)


def coerce_audio(
    source: str
    | Path
    | Sequence[float]
    | np.ndarray
    | AudioInput
    | tuple[Iterable[float], int],
    *,
    target_sr: int = DEFAULT_SR,
    mono: bool = True,
) -> AudioInput:
    """Normalise ``source`` into an :class:`AudioInput` instance."""

    if isinstance(source, AudioInput):
        samples = np.asarray(source.samples, dtype=np.float32)
        if source.sample_rate != target_sr:
            samples = _resample(samples, source.sample_rate, target_sr)
        stereo = None
        if source.stereo_samples is not None:
            stereo = np.asarray(source.stereo_samples, dtype=np.float32)
            if source.sample_rate != target_sr:
                stereo = _resample(stereo, source.sample_rate, target_sr)
        return AudioInput(
            samples=samples,
            sample_rate=target_sr,
            path=source.path,
            stereo_samples=stereo,
        )

    if isinstance(source, (str, Path)):
        path = str(source)
        samples, sr, _meta = load_audio(path, mono=False)
        stereo: Optional[np.ndarray]
        if samples.ndim > 1:
            stereo = np.asarray(samples, dtype=np.float32)
            mono_samples = np.mean(stereo, axis=0)
        else:
            stereo = None
            mono_samples = np.asarray(samples, dtype=np.float32)
        mono_samples = _resample(mono_samples, sr, target_sr)
        if stereo is not None:
            stereo = _resample(stereo, sr, target_sr)
            if mono:
                mono_samples = np.mean(stereo, axis=0)
        return AudioInput(
            samples=mono_samples,
            sample_rate=target_sr,
            path=path,
            stereo_samples=stereo,
        )

    if isinstance(source, np.ndarray):
        samples = np.asarray(source, dtype=np.float32)
        stereo = None
        if samples.ndim > 1:
            stereo = samples
            if mono:
                samples = np.mean(samples, axis=0)
        return AudioInput(samples=samples, sample_rate=target_sr, stereo_samples=stereo)

    if isinstance(source, tuple) and len(source) == 2:
        data, sr = source
        samples = np.asarray(list(data), dtype=np.float32)
        stereo = None
        if samples.ndim > 1:
            stereo = samples
            if mono:
                samples = np.mean(samples, axis=0)
        samples = _resample(samples, int(sr), target_sr)
        if stereo is not None:
            stereo = _resample(stereo, int(sr), target_sr)
        return AudioInput(samples=samples, sample_rate=target_sr, stereo_samples=stereo)

    raise TypeError(f"Unsupported audio source type: {type(source)!r}")
