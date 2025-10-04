"""Beat and downbeat estimation routines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

try:  # pragma: no cover - optional dependency
    import madmom
    from madmom.features.downbeats import (
        DBNDownBeatTrackingProcessor,
        RNNDownBeatProcessor,
    )
except ImportError:  # pragma: no cover - optional dependency not installed
    madmom = None  # type: ignore[assignment]

import librosa

from ..utils import AudioInput, seed_everything


@dataclass(slots=True)
class BeatAnalysis:
    """Summary of the beat grid."""

    bpm: float
    beat_times: List[float]
    beat_frames: List[int]
    confidence: float


@dataclass(slots=True)
class DownbeatAnalysis:
    """Downbeat estimates optionally powered by madmom."""

    downbeat_times: List[float]
    beat_positions: List[int]
    source: str


def _compute_confidence(beat_times: np.ndarray) -> float:
    if len(beat_times) < 2:
        return 0.0
    intervals = np.diff(beat_times)
    if np.allclose(intervals, intervals[0]):
        return 1.0
    return float(
        np.clip(1.0 - np.std(intervals) / (np.mean(intervals) + 1e-9), 0.0, 1.0)
    )


def analyse_beats(
    audio: AudioInput | str,
    *,
    hop_length: int = 512,
    seed: int,
) -> Tuple[BeatAnalysis, Optional[DownbeatAnalysis]]:
    """Estimate the beat grid and optional downbeats from ``audio``."""

    seed_everything(seed)
    if not isinstance(audio, AudioInput):
        raise TypeError("analyse_beats expects an AudioInput instance")

    onset_env = librosa.onset.onset_strength(y=audio.samples, sr=audio.sample_rate)
    tempo, beat_frames = librosa.beat.beat_track(
        onset_envelope=onset_env,
        sr=audio.sample_rate,
        hop_length=hop_length,
        tightness=100.0,
    )
    beat_times = librosa.frames_to_time(
        beat_frames, sr=audio.sample_rate, hop_length=hop_length
    )
    confidence = _compute_confidence(beat_times)
    beat_result = BeatAnalysis(
        bpm=float(tempo),
        beat_times=beat_times.tolist(),
        beat_frames=beat_frames.astype(int).tolist(),
        confidence=confidence,
    )

    downbeat_result = _analyse_downbeats(audio, beat_result, hop_length, seed)
    return beat_result, downbeat_result


def _analyse_downbeats(
    audio: AudioInput,
    beat_result: BeatAnalysis,
    hop_length: int,
    seed: int,
) -> Optional[DownbeatAnalysis]:
    seed_everything(seed)
    if madmom is None:  # pragma: no cover - optional path
        return _fallback_downbeats(beat_result)

    try:
        proc = RNNDownBeatProcessor()
        act = proc(audio.path or np.ascontiguousarray(audio.samples))
        tracker = DBNDownBeatTrackingProcessor(beats_per_bar=[3, 4], fps=proc.fps)
        tracked = tracker(act)
        downbeat_times = tracked[:, 0].tolist()
        beat_positions = tracked[:, 1].astype(int).tolist()
        source = "madmom"
        if not downbeat_times:
            return _fallback_downbeats(beat_result)
        return DownbeatAnalysis(
            downbeat_times=downbeat_times, beat_positions=beat_positions, source=source
        )
    except Exception:  # pragma: no cover - madmom failure fallback
        return _fallback_downbeats(beat_result)


def _fallback_downbeats(beat_result: BeatAnalysis) -> DownbeatAnalysis:
    beat_positions = []
    downbeat_times = []
    for idx, beat_time in enumerate(beat_result.beat_times):
        if idx % 4 == 0:
            downbeat_times.append(float(beat_time))
            beat_positions.append(1)
        else:
            beat_positions.append((idx % 4) + 1)
    return DownbeatAnalysis(
        downbeat_times=downbeat_times, beat_positions=beat_positions, source="heuristic"
    )
