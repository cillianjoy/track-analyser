"""Harmonic, spectral and MIDI hook analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import librosa

from ..utils import AudioInput, deterministic_rng, seed_everything
from .beats import BeatAnalysis, DownbeatAnalysis

MAJOR_PROFILE = np.array(
    [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
)
MINOR_PROFILE = np.array(
    [6.33, 2.68, 3.52, 5.38, 2.6, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
)
PITCH_CLASS_NAMES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]


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
class ChordHint:
    time: float
    chord: str
    confidence: float


@dataclass(slots=True)
class MidiSuggestion:
    name: str
    notes: pd.DataFrame


@dataclass(slots=True)
class HarmonicAnalysis:
    spectral_balance: SpectralBalance
    stereo_image: StereoImage
    key_estimate: KeyEstimate
    chord_hints: List[ChordHint]
    hook_suggestion: MidiSuggestion
    bass_suggestion: MidiSuggestion


def analyse_harmonic(
    audio: AudioInput | str,
    beat_result: BeatAnalysis,
    downbeat_result: Optional[DownbeatAnalysis],
    *,
    seed: int,
) -> HarmonicAnalysis:
    if not isinstance(audio, AudioInput):
        raise TypeError("analyse_harmonic expects an AudioInput instance")

    seed_everything(seed)
    rng = deterministic_rng(seed)

    spectral_balance = _spectral_balance(audio)
    stereo_image = _stereo_image(audio)
    key_estimate, chroma = _estimate_key(audio)
    chord_hints = _estimate_chords(chroma, beat_result, rng)
    start_offset = (
        downbeat_result.downbeat_times[0]
        if downbeat_result and downbeat_result.downbeat_times
        else (beat_result.beat_times[0] if beat_result.beat_times else 0.0)
    )
    hook = _generate_midi(
        chroma,
        beat_result,
        key_estimate,
        rng,
        name="hook",
        start_offset=start_offset,
    )
    bass = _generate_midi(
        chroma,
        beat_result,
        key_estimate,
        rng,
        name="bass",
        octave=-1,
        start_offset=start_offset,
    )

    return HarmonicAnalysis(
        spectral_balance=spectral_balance,
        stereo_image=stereo_image,
        key_estimate=key_estimate,
        chord_hints=chord_hints,
        hook_suggestion=hook,
        bass_suggestion=bass,
    )


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


def _estimate_key(audio: AudioInput) -> tuple[KeyEstimate, np.ndarray]:
    chroma = librosa.feature.chroma_cqt(y=audio.samples, sr=audio.sample_rate)
    chroma_mean = np.mean(chroma, axis=1)

    def correlate(template: np.ndarray) -> np.ndarray:
        return np.array([np.dot(np.roll(template, i), chroma_mean) for i in range(12)])

    major_scores = correlate(MAJOR_PROFILE)
    minor_scores = correlate(MINOR_PROFILE)

    major_idx = int(np.argmax(major_scores))
    minor_idx = int(np.argmax(minor_scores))

    if major_scores[major_idx] >= minor_scores[minor_idx]:
        key = f"{PITCH_CLASS_NAMES[major_idx]} major"
        confidence = float(major_scores[major_idx] / (np.sum(major_scores) + 1e-9))
    else:
        key = f"{PITCH_CLASS_NAMES[minor_idx]} minor"
        confidence = float(minor_scores[minor_idx] / (np.sum(minor_scores) + 1e-9))

    return KeyEstimate(key=key, confidence=confidence), chroma


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
        chords = list(chord_templates.items())
        base_scores = np.array(
            [float(np.dot(template, chroma_mean)) for _, template in chords]
        )
        noise = rng.normal(0.0, 1e-6, size=base_scores.shape)
        idx_best = int(np.argmax(base_scores + noise))
        chord, score = chords[idx_best]
        score_values = base_scores + 1e-9
        hints.append(
            ChordHint(
                time=float(beat_result.beat_times[idx]),
                chord=chord,
                confidence=float(base_scores[idx_best] / float(np.max(score_values))),
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
            templates[f"{pitch}{quality}"] = template
    return templates


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
