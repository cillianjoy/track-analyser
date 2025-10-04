"""Harmony analysis primitives shared across the pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import librosa

from .utils import AudioInput, deterministic_rng, seed_everything
from .analysis.beats import BeatAnalysis, DownbeatAnalysis

MAJOR_PROFILE = np.array(
    [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
)
MINOR_PROFILE = np.array(
    [6.33, 2.68, 3.52, 5.38, 2.6, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
)
PITCH_CLASS_NAMES = [
    "C",
    "C#",
    "D",
    "Eb",
    "E",
    "F",
    "F#",
    "G",
    "Ab",
    "A",
    "Bb",
    "B",
]


@dataclass(slots=True)
class SpectralBalance:
    low_band: float
    mid_band: float
    high_band: float


@dataclass(slots=True)
class StereoImage:
    correlation: float
    balance: float


@dataclass(slots=True)
class KeyEstimate:
    key: str
    confidence: float


@dataclass(slots=True)
class KeyEstimation:
    best: KeyEstimate
    second_best: KeyEstimate


@dataclass(slots=True)
class ChordHint:
    time: float
    chord: str
    confidence: float


@dataclass(slots=True)
class ChordChangePoint:
    time: float
    strength: float


@dataclass(slots=True)
class MidiSuggestion:
    name: str
    notes: pd.DataFrame


@dataclass(slots=True)
class HarmonyAnalysis:
    spectral_balance: SpectralBalance
    stereo_image: StereoImage
    primary_key: KeyEstimate
    secondary_key: KeyEstimate
    chord_hints: List[ChordHint]
    chord_change_points: List[ChordChangePoint]
    hook_suggestion: MidiSuggestion
    bass_suggestion: MidiSuggestion

    @property
    def key_estimate(self) -> KeyEstimate:
        """Backward compatible accessor for the best key estimate."""

        return self.primary_key


def key_estimate(y: np.ndarray, sr: int) -> KeyEstimation:
    """Return the best and second best key hypotheses for ``y``.

    The estimator evaluates both CQT and STFT chroma projections against the
    Krumhansl–Schmuckler key profiles. Scores from the two projections are
    combined before the best ranking is derived.
    """

    chroma_cqt = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)

    combined_scores, keys = _score_keys([chroma_cqt, chroma_stft])
    if not combined_scores.size:
        fallback = KeyEstimate(key="C major", confidence=0.0)
        return KeyEstimation(best=fallback, second_best=fallback)

    # Convert to probabilities and pick the two strongest hypotheses.
    positive_scores = np.maximum(combined_scores, 0.0)
    score_sum = float(np.sum(positive_scores)) or 1.0
    confidences = positive_scores / score_sum

    best_idx = int(np.argmax(confidences))
    best = KeyEstimate(key=keys[best_idx], confidence=float(confidences[best_idx]))

    confidences[best_idx] = -np.inf
    second_idx = int(np.argmax(confidences))
    second = KeyEstimate(
        key=keys[second_idx], confidence=float(max(confidences[second_idx], 0.0))
    )

    return KeyEstimation(best=best, second_best=second)


def analyse_harmony(
    audio: AudioInput | str,
    beat_result: BeatAnalysis,
    downbeat_result: Optional[DownbeatAnalysis],
    *,
    seed: int,
) -> HarmonyAnalysis:
    if not isinstance(audio, AudioInput):
        raise TypeError("analyse_harmony expects an AudioInput instance")

    seed_everything(seed)
    rng = deterministic_rng(seed)

    spectral_balance = _spectral_balance(audio)
    stereo_image = _stereo_image(audio)

    chroma_cqt = librosa.feature.chroma_cqt(y=audio.samples, sr=audio.sample_rate)
    chroma_stft = librosa.feature.chroma_stft(y=audio.samples, sr=audio.sample_rate)

    key_result = _estimate_keys_from_chroma(chroma_cqt, chroma_stft)

    chord_hints = _estimate_chords(chroma_cqt, beat_result, rng)
    change_points = _detect_chord_changes(chroma_cqt, beat_result, chord_hints)

    start_offset = (
        downbeat_result.downbeat_times[0]
        if downbeat_result and downbeat_result.downbeat_times
        else (beat_result.beat_times[0] if beat_result.beat_times else 0.0)
    )

    hook = _generate_midi(
        chroma_cqt,
        beat_result,
        key_result.best,
        rng,
        name="hook",
        start_offset=start_offset,
    )
    bass = _generate_midi(
        chroma_cqt,
        beat_result,
        key_result.best,
        rng,
        name="bass",
        octave=-1,
        start_offset=start_offset,
    )

    return HarmonyAnalysis(
        spectral_balance=spectral_balance,
        stereo_image=stereo_image,
        primary_key=key_result.best,
        secondary_key=key_result.second_best,
        chord_hints=chord_hints,
        chord_change_points=change_points,
        hook_suggestion=hook,
        bass_suggestion=bass,
    )


def _score_keys(chroma_matrices: Sequence[np.ndarray]) -> Tuple[np.ndarray, List[str]]:
    if not chroma_matrices:
        return np.array([]), []

    profiles = {
        "major": MAJOR_PROFILE / np.linalg.norm(MAJOR_PROFILE),
        "minor": MINOR_PROFILE / np.linalg.norm(MINOR_PROFILE),
    }

    aggregated = np.zeros(24, dtype=float)
    keys: List[str] = []
    if not keys:
        for idx, pitch in enumerate(PITCH_CLASS_NAMES):
            keys.append(f"{pitch} major")
        for idx, pitch in enumerate(PITCH_CLASS_NAMES):
            keys.append(f"{pitch} minor")

    for chroma in chroma_matrices:
        if chroma.size == 0:
            continue
        chroma_mean = np.mean(chroma, axis=1)
        norm = np.linalg.norm(chroma_mean)
        if norm <= 0:
            continue
        chroma_norm = chroma_mean / norm
        major_scores = _correlate_chroma(chroma_norm, profiles["major"])
        minor_scores = _correlate_chroma(chroma_norm, profiles["minor"])
        aggregated[:12] += major_scores
        aggregated[12:] += minor_scores

    return aggregated, keys


def _estimate_keys_from_chroma(
    chroma_cqt: np.ndarray, chroma_stft: np.ndarray
) -> KeyEstimation:
    scores, keys = _score_keys([chroma_cqt, chroma_stft])
    if not scores.size:
        fallback = KeyEstimate(key="C major", confidence=0.0)
        return KeyEstimation(best=fallback, second_best=fallback)

    scores = np.maximum(scores, 0.0)
    total = float(np.sum(scores)) or 1.0
    confidences = scores / total

    best_idx = int(np.argmax(confidences))
    best = KeyEstimate(key=keys[best_idx], confidence=float(confidences[best_idx]))
    confidences[best_idx] = -np.inf
    second_idx = int(np.argmax(confidences))
    second = KeyEstimate(
        key=keys[second_idx], confidence=float(max(confidences[second_idx], 0.0))
    )
    return KeyEstimation(best=best, second_best=second)


def _correlate_chroma(chroma: np.ndarray, template: np.ndarray) -> np.ndarray:
    return np.array([
        float(np.dot(chroma, np.roll(template, shift))) for shift in range(12)
    ])


def _spectral_balance(audio: AudioInput) -> SpectralBalance:
    spec = np.abs(librosa.stft(audio.samples, n_fft=4096, hop_length=1024))
    freqs = librosa.fft_frequencies(sr=audio.sample_rate, n_fft=4096)
    total_power = np.sum(spec)
    if total_power <= 0:
        return SpectralBalance(0.0, 0.0, 0.0)

    low_mask = freqs < 200
    mid_mask = (freqs >= 200) & (freqs < 2000)
    high_mask = freqs >= 2000

    low = float(np.sum(spec[low_mask]) / total_power)
    mid = float(np.sum(spec[mid_mask]) / total_power)
    high = float(np.sum(spec[high_mask]) / total_power)
    return SpectralBalance(low_band=low, mid_band=mid, high_band=high)


def _stereo_image(audio: AudioInput) -> StereoImage:
    samples = (
        audio.stereo_samples if audio.stereo_samples is not None else audio.samples
    )
    if samples.ndim == 1:
        return StereoImage(correlation=1.0, balance=0.0)

    left = samples[0]
    right = samples[1]
    corr = float(np.corrcoef(left, right)[0, 1]) if left.size and right.size else 0.0
    balance = float(np.mean(np.abs(left)) - np.mean(np.abs(right)))
    return StereoImage(correlation=corr, balance=balance)


def _estimate_chords(
    chroma: np.ndarray,
    beat_result: BeatAnalysis,
    rng: np.random.Generator,
) -> List[ChordHint]:
    beat_frames = beat_result.beat_frames
    if not beat_frames:
        return []
    hints: List[ChordHint] = []
    chord_templates = _build_chord_templates()
    for idx, frame in enumerate(beat_frames):
        window = chroma[:, max(0, frame - 2) : frame + 2]
        if window.size == 0:
            continue
        chroma_mean = np.mean(window, axis=1)
        norm = np.linalg.norm(chroma_mean)
        if norm <= 0:
            continue
        chroma_norm = chroma_mean / norm
        chords = list(chord_templates.items())
        base_scores = np.array(
            [float(np.dot(template, chroma_norm)) for _, template in chords]
        )
        noise = rng.normal(0.0, 1e-6, size=base_scores.shape)
        idx_best = int(np.argmax(base_scores + noise))
        chord, score = chords[idx_best]
        score_values = base_scores + 1e-9
        confidence = float(base_scores[idx_best] / float(np.max(score_values)))
        hints.append(
            ChordHint(
                time=float(beat_result.beat_times[idx]),
                chord=chord,
                confidence=confidence,
            )
        )
    return hints


def _build_chord_templates() -> Dict[str, np.ndarray]:
    intervals = {
        "maj": [0, 4, 7],
        "min": [0, 3, 7],
        "dim": [0, 3, 6],
        "sus2": [0, 2, 7],
        "sus4": [0, 5, 7],
    }
    templates: Dict[str, np.ndarray] = {}
    base = np.zeros(12)
    for root_idx, pitch in enumerate(PITCH_CLASS_NAMES):
        for quality, ints in intervals.items():
            template = base.copy()
            for interval in ints:
                template[(root_idx + interval) % 12] = 1.0
            norm = np.linalg.norm(template)
            if norm > 0:
                template = template / norm
            templates[f"{pitch}{quality}"] = template
    return templates


def _detect_chord_changes(
    chroma: np.ndarray, beat_result: BeatAnalysis, chord_hints: Sequence[ChordHint]
) -> List[ChordChangePoint]:
    beat_frames = beat_result.beat_frames
    if len(beat_frames) < 2:
        return []

    profiles: List[np.ndarray] = []
    times: List[float] = []
    for idx, frame in enumerate(beat_frames):
        window = chroma[:, max(0, frame - 2) : frame + 2]
        if window.size == 0:
            continue
        chroma_mean = np.mean(window, axis=1)
        norm = np.linalg.norm(chroma_mean)
        if norm <= 0:
            continue
        profiles.append(chroma_mean / norm)
        times.append(float(beat_result.beat_times[idx]))

    if len(profiles) < 2:
        return []

    changes: List[ChordChangePoint] = []
    strengths = []
    for prev, curr, time in zip(profiles, profiles[1:], times[1:]):
        similarity = float(np.clip(np.dot(prev, curr), -1.0, 1.0))
        strength = float(np.clip(1.0 - similarity, 0.0, 1.0))
        strengths.append(strength)
        changes.append(ChordChangePoint(time=time, strength=strength))

    if not strengths:
        return []

    change_map: Dict[float, float] = {}
    if strengths:
        strengths_arr = np.asarray(strengths)
        keep = max(1, int(np.ceil(len(strengths_arr) * 0.9)))
        if keep >= len(strengths_arr):
            threshold = float(np.min(strengths_arr))
        else:
            cutoff_index = len(strengths_arr) - keep
            partitioned = np.partition(strengths_arr, cutoff_index)
            threshold = float(partitioned[cutoff_index])
        threshold = float(max(threshold, 0.15))
        for change in changes:
            if change.strength >= threshold:
                change_map[change.time] = max(change_map.get(change.time, 0.0), change.strength)
        if changes:
            first_change = changes[0]
            change_map[first_change.time] = max(
                change_map.get(first_change.time, 0.0), first_change.strength
            )

    if len(chord_hints) >= 2:
        templates = _build_chord_templates()
        for prev_hint, curr_hint in zip(chord_hints, chord_hints[1:]):
            if curr_hint.chord == prev_hint.chord:
                continue
            prev_template = templates.get(prev_hint.chord)
            curr_template = templates.get(curr_hint.chord)
            if prev_template is None or curr_template is None:
                similarity = 0.0
            else:
                similarity = float(np.clip(np.dot(prev_template, curr_template), -1.0, 1.0))
            strength = float(np.clip(1.0 - similarity, 0.0, 1.0))
            change_map[curr_hint.time] = max(change_map.get(curr_hint.time, 0.0), strength)

    if not change_map:
        return []

    max_strength = max(change_map.values()) or 1.0
    return [
        ChordChangePoint(time=float(time), strength=float(value / max_strength))
        for time, value in sorted(change_map.items())
    ]


def _generate_midi(
    chroma: np.ndarray,
    beat_result: BeatAnalysis,
    key_estimate: KeyEstimate,
    rng: np.random.Generator,
    *,
    name: str,
    octave: int = 0,
    start_offset: float = 0.0,
) -> MidiSuggestion:
    scale = _scale_for_key(key_estimate.key)
    beats = [max(0.0, beat - start_offset) for beat in beat_result.beat_times[:8]]
    if not beats:
        beats = [0.0, 0.5, 1.0, 1.5]
    notes = []
    duration = np.median(np.diff(beats)) if len(beats) > 1 else 0.5
    for idx, beat_time in enumerate(beats):
        pitch_class = int(scale[int(rng.integers(0, len(scale)))])
        velocity = int(np.clip(96 + rng.integers(-12, 12), 20, 127))
        pitch = 60 + pitch_class + octave * 12
        notes.append(
            {
                "start": float(beat_time),
                "duration": float(duration),
                "pitch": int(pitch),
                "velocity": int(velocity),
                "channel": 0,
            }
        )
    df = pd.DataFrame(
        notes, columns=["start", "duration", "pitch", "velocity", "channel"]
    )
    return MidiSuggestion(name=name, notes=df)


def _scale_for_key(key: str) -> List[int]:
    key_root, _, mode = key.partition(" ")
    root_idx = PITCH_CLASS_NAMES.index(key_root)
    if mode.strip().lower().startswith("major"):
        pattern = [0, 2, 4, 5, 7, 9, 11]
    else:
        pattern = [0, 2, 3, 5, 7, 8, 10]
    return [(root_idx + interval) % 12 for interval in pattern]


__all__ = [
    "HarmonyAnalysis",
    "ChordChangePoint",
    "ChordHint",
    "KeyEstimation",
    "KeyEstimate",
    "MidiSuggestion",
    "SpectralBalance",
    "StereoImage",
    "analyse_harmony",
    "key_estimate",
]

