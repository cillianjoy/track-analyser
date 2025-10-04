"""Tempo estimation utilities built on onset envelope autocorrelation."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
import librosa
from librosa import util

DEFAULT_HOP_LENGTH = 512
BEATS_PER_BAR = 4


def _onset_envelope(y: np.ndarray, sr: int, hop_length: int) -> np.ndarray:
    """Return the onset strength envelope for ``y``."""

    envelope = librosa.onset.onset_strength(
        y=y, sr=sr, hop_length=hop_length, aggregate=np.mean
    )
    if envelope.size == 0:
        return np.zeros(1, dtype=float)
    return envelope


def estimate_bpm(
    y: np.ndarray,
    sr: int,
    bpm_min: float = 90.0,
    bpm_max: float = 135.0,
    *,
    hop_length: int = DEFAULT_HOP_LENGTH,
) -> float:
    """Estimate tempo using autocorrelation of the onset strength envelope."""

    onset_env = _onset_envelope(y, sr, hop_length)
    autocorr = librosa.autocorrelate(onset_env)
    if autocorr.size <= 1:
        return float(bpm_min)

    autocorr = autocorr[1:]  # discard zero-lag peak
    lags = np.arange(1, autocorr.size + 1, dtype=float)
    tempi = 60.0 * sr / (lags * hop_length)

    mask = (tempi >= bpm_min) & (tempi <= bpm_max)
    if not np.any(mask):
        mask = tempi > 0

    masked_autocorr = util.normalize(autocorr[mask])
    masked_lags = lags[mask]
    peak_index = int(np.argmax(masked_autocorr))

    refined_lag = masked_lags[peak_index]
    if 0 < peak_index < masked_autocorr.size - 1:
        left = masked_autocorr[peak_index - 1]
        center = masked_autocorr[peak_index]
        right = masked_autocorr[peak_index + 1]
        denominator = left - 2 * center + right
        if abs(denominator) > 1e-9:
            shift = 0.5 * (left - right) / denominator
            refined_lag = float(masked_lags[peak_index] + shift)

    refined_lag = max(refined_lag, 1.0)
    bpm = float(60.0 * sr / (refined_lag * hop_length))

    regression = _fit_onset_regression(onset_env, sr, hop_length, 60.0 / bpm)
    if regression is not None:
        _, slope = regression
        if slope > 0:
            refined_bpm = 60.0 / slope
            if bpm_min <= refined_bpm <= bpm_max:
                bpm = float(refined_bpm)

    return bpm


def _initial_beat_time(
    onset_env: np.ndarray, sr: int, hop_length: int
) -> Tuple[float, int]:
    onset_frames = librosa.onset.onset_detect(
        onset_envelope=onset_env,
        sr=sr,
        hop_length=hop_length,
        backtrack=True,
        units="frames",
    )
    if onset_frames.size == 0:
        return 0.0, 0

    first_frame = int(onset_frames[0])
    start_time = librosa.frames_to_time(first_frame, sr=sr, hop_length=hop_length)
    return float(start_time), first_frame


def _fit_onset_regression(
    onset_env: np.ndarray, sr: int, hop_length: int, beat_period: float
) -> Tuple[float, float] | None:
    onset_times = np.asarray(
        librosa.onset.onset_detect(
            onset_envelope=onset_env,
            sr=sr,
            hop_length=hop_length,
            backtrack=True,
            units="time",
        ),
        dtype=float,
    )
    if onset_times.size < 4 or beat_period <= 0:
        return None

    indices = np.round(onset_times / beat_period).astype(int)
    mask = indices >= 0
    if not np.any(mask):
        return None

    unique: dict[int, float] = {}
    for idx, time in zip(indices[mask], onset_times[mask]):
        unique.setdefault(int(idx), float(time))

    if len(unique) < 4:
        return None

    sorted_indices = np.array(sorted(unique))
    times = np.array([unique[i] for i in sorted_indices])
    A = np.vstack([np.ones_like(sorted_indices), sorted_indices]).T
    intercept, slope = np.linalg.lstsq(A, times, rcond=None)[0]
    return float(intercept), float(slope)


def beat_grid(
    y: np.ndarray,
    sr: int,
    *,
    hop_length: int = DEFAULT_HOP_LENGTH,
    beats_per_bar: int = BEATS_PER_BAR,
) -> pd.DataFrame:
    """Return a beat grid annotated with bar positions."""

    onset_env = _onset_envelope(y, sr, hop_length)
    bpm = estimate_bpm(y, sr, hop_length=hop_length)
    beat_period = 60.0 / bpm

    regression = _fit_onset_regression(onset_env, sr, hop_length, beat_period)
    if regression is not None:
        start_time = max(regression[0], 0.0)
    else:
        start_time, _ = _initial_beat_time(onset_env, sr, hop_length)
    if start_time < 0.0:
        start_time = 0.0

    duration = len(y) / float(sr)
    if start_time > duration:
        start_time = 0.0

    total_beats = max(1, int(np.floor((duration - start_time) / beat_period)) + 1)
    times = start_time + np.arange(total_beats, dtype=float) * beat_period
    times = times[times <= duration + 1e-3]

    frames = librosa.time_to_frames(times, sr=sr, hop_length=hop_length)
    beat_index = np.arange(times.size)
    bars = beat_index // beats_per_bar + 1
    beats = beat_index % beats_per_bar + 1

    grid = pd.DataFrame(
        {
            "time": times,
            "frame": frames.astype(int),
            "bar": bars.astype(int),
            "beat": beats.astype(int),
            "is_downbeat": beats == 1,
        }
    )

    return grid


__all__ = ["estimate_bpm", "beat_grid"]
