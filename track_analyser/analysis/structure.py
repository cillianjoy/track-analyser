"""Structural segmentation utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import scipy.ndimage
import librosa

from ..utils import AudioInput, seed_everything
from .beats import BeatAnalysis


@dataclass(slots=True)
class StructuralSegment:
    label: str
    start: float
    end: float
    confidence: float


@dataclass(slots=True)
class StructureAnalysis:
    segments: List[StructuralSegment]
    novelty_curve: List[float]


def analyse_structure(
    audio: AudioInput | str,
    beat_result: BeatAnalysis,
    *,
    seed: int,
    frame_length: int = 2048,
    hop_length: int = 512,
) -> StructureAnalysis:
    """Detect structural boundaries using a novelty curve heuristic."""

    if not isinstance(audio, AudioInput):
        raise TypeError("analyse_structure expects an AudioInput instance")
    seed_everything(seed)

    S = np.abs(
        librosa.stft(
            audio.samples, n_fft=frame_length, hop_length=hop_length, window="hann"
        )
    )
    mel = librosa.feature.melspectrogram(S=S, sr=audio.sample_rate)
    log_mel = librosa.power_to_db(mel + 1e-9)
    novelty = np.mean(np.abs(np.diff(log_mel, axis=1)), axis=0)
    novelty = scipy.ndimage.gaussian_filter1d(novelty, sigma=2.0)
    peaks = librosa.util.peak_pick(
        novelty,
        pre_max=8,
        post_max=8,
        pre_avg=32,
        post_avg=32,
        delta=np.std(novelty) * 0.5,
        wait=16,
    )

    boundaries = np.sort(np.unique(np.concatenate(([0], peaks, [len(novelty) - 1]))))
    times = librosa.frames_to_time(
        boundaries, sr=audio.sample_rate, hop_length=hop_length
    )
    if beat_result.beat_times:
        beat_times = np.asarray(beat_result.beat_times)
        snapped = []
        for t in times:
            idx = int(np.argmin(np.abs(beat_times - t)))
            snapped.append(float(beat_times[idx]))
        times = np.maximum.accumulate(np.asarray(snapped))

    labels = _label_segments(len(boundaries) - 1)
    segments: List[StructuralSegment] = []
    for idx, start_idx in enumerate(boundaries[:-1]):
        end_idx = boundaries[idx + 1]
        window = novelty[start_idx:end_idx]
        seg_novelty = float(np.mean(window)) if window.size else 0.0
        segments.append(
            StructuralSegment(
                label=labels[idx],
                start=float(times[idx]),
                end=float(times[idx + 1]),
                confidence=float(
                    np.clip(seg_novelty / (np.max(novelty) + 1e-9), 0.0, 1.0)
                ),
            )
        )

    return StructureAnalysis(segments=segments, novelty_curve=novelty.tolist())


def _label_segments(count: int) -> List[str]:
    alphabet = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    labels = []
    for idx in range(count):
        labels.append(alphabet[idx % len(alphabet)])
    return labels
