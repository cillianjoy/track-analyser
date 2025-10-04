"""Structural segmentation utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np
import scipy.ndimage
import librosa

from ..utils import AudioInput, seed_everything
from .beats import BeatAnalysis


@dataclass(slots=True)
class StructuralSegment:
    label: str
    category: str
    start: float
    end: float
    confidence: float
    percussive_energy: float
    harmonic_energy: float
    percussive_ratio: float


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

    stft = librosa.stft(
        audio.samples, n_fft=frame_length, hop_length=hop_length, window="hann"
    )
    magnitude = np.abs(stft)
    harmonic, percussive = librosa.decompose.hpss(magnitude)
    mel = librosa.feature.melspectrogram(
        y=audio.samples,
        sr=audio.sample_rate,
        n_fft=frame_length,
        hop_length=hop_length,
        power=2.0,
    )

    novelty, energy_novelty = _combined_novelty_curve(
        mel,
        percussive,
        harmonic,
        hop_length=hop_length,
        sample_rate=audio.sample_rate,
    )

    if novelty.size == 0:
        duration = float(audio.duration)
        fallback_segment = StructuralSegment(
            label="A",
            category="intro",
            start=0.0,
            end=duration,
            confidence=0.0,
            percussive_energy=float(np.sum(percussive)),
            harmonic_energy=float(np.sum(harmonic)),
            percussive_ratio=0.0,
        )
        return StructureAnalysis(
            segments=[fallback_segment],
            novelty_curve=novelty.tolist(),
        )

    frames_per_second = audio.sample_rate / float(hop_length)
    min_spacing_seconds = 8.0
    min_spacing_frames = max(1, int(round(min_spacing_seconds * frames_per_second)))
    peaks = librosa.util.peak_pick(
        novelty,
        pre_max=8,
        post_max=8,
        pre_avg=32,
        post_avg=32,
        delta=np.std(novelty) * 0.4,
        wait=min_spacing_frames,
    )

    peaks = _refine_boundaries(peaks, energy_novelty, int(round(frames_per_second * 3.0)))
    peaks = _enforce_min_frame_spacing(peaks, novelty, min_spacing_frames)
    total_frames = len(novelty)
    boundaries = np.concatenate(([0], peaks, [total_frames - 1]))
    boundaries = np.asarray(np.unique(boundaries), dtype=int)
    times = librosa.frames_to_time(
        boundaries, sr=audio.sample_rate, hop_length=hop_length
    )
    if beat_result.beat_times:
        beat_times = np.asarray(beat_result.beat_times)
        snapped = []
        for t in times:
            idx = int(np.argmin(np.abs(beat_times - t)))
            snapped.append(float(beat_times[idx]))
        snapped = np.maximum.accumulate(np.asarray(snapped))
        spacing_mask = _enforce_min_time_spacing(
            snapped,
            boundaries,
            novelty,
            min_spacing_seconds,
        )
        times = snapped[spacing_mask]
        boundaries = boundaries[spacing_mask]
    else:
        spacing_mask = _enforce_min_time_spacing(
            times,
            boundaries,
            novelty,
            min_spacing_seconds,
        )
        times = times[spacing_mask]
        boundaries = boundaries[spacing_mask]

    labels = _label_segments(len(boundaries) - 1)
    segment_percussive: List[float] = []
    segment_harmonic: List[float] = []
    segment_ratio: List[float] = []
    segments: List[StructuralSegment] = []
    for idx, start_idx in enumerate(boundaries[:-1]):
        end_idx = boundaries[idx + 1]
        window = novelty[start_idx:end_idx]
        seg_novelty = float(np.mean(window)) if window.size else 0.0
        start_time = float(times[idx])
        end_time = float(times[idx + 1])
        perc_energy = float(np.sum(percussive[:, start_idx:end_idx]))
        harm_energy = float(np.sum(harmonic[:, start_idx:end_idx]))
        ratio = float(
            perc_energy / (perc_energy + harm_energy + 1e-9)
        )
        segment_percussive.append(perc_energy)
        segment_harmonic.append(harm_energy)
        segment_ratio.append(ratio)
        segments.append(
            StructuralSegment(
                label=labels[idx],
                category="",
                start=start_time,
                end=end_time,
                confidence=float(
                    np.clip(seg_novelty / (np.max(novelty) + 1e-9), 0.0, 1.0)
                ),
                percussive_energy=perc_energy,
                harmonic_energy=harm_energy,
                percussive_ratio=ratio,
            )
        )

    categories = _classify_segments(segment_ratio, segment_percussive, segment_harmonic)
    for segment, category in zip(segments, categories, strict=True):
        segment.category = category

    return StructureAnalysis(segments=segments, novelty_curve=novelty.tolist())


def _label_segments(count: int) -> List[str]:
    alphabet = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    labels = []
    for idx in range(count):
        labels.append(alphabet[idx % len(alphabet)])
    return labels


def _combined_novelty_curve(
    mel_spectrogram: np.ndarray,
    percussive: np.ndarray,
    harmonic: np.ndarray,
    *,
    hop_length: int,
    sample_rate: int,
    context_seconds: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    mel = np.asarray(mel_spectrogram, dtype=float)
    if mel.size == 0:
        return np.zeros(0, dtype=float)

    log_mel = librosa.power_to_db(mel + 1e-9)
    spectral_flux = librosa.onset.onset_strength(S=mel, sr=sample_rate, hop_length=hop_length)
    spectral_flux = np.asarray(spectral_flux, dtype=float)

    frames = log_mel.shape[1]
    mfcc = librosa.feature.mfcc(S=log_mel, sr=sample_rate, n_mfcc=13)
    mfcc = scipy.ndimage.gaussian_filter1d(mfcc, sigma=1.0, axis=1)
    context = max(2, int(round(context_seconds * sample_rate / float(hop_length))))
    self_similarity = np.zeros(frames, dtype=float)
    for frame in range(context, frames - context):
        left = mfcc[:, frame - context : frame]
        right = mfcc[:, frame : frame + context]
        left_mean = np.mean(left, axis=1)
        right_mean = np.mean(right, axis=1)
        left_norm = left_mean / (np.linalg.norm(left_mean) + 1e-9)
        right_norm = right_mean / (np.linalg.norm(right_mean) + 1e-9)
        self_similarity[frame] = 1.0 - float(np.dot(left_norm, right_norm))

    perc_curve = np.sum(percussive, axis=0) if percussive.size else np.zeros(frames)
    harm_curve = np.sum(harmonic, axis=0) if harmonic.size else np.zeros(frames)
    ratio_curve = perc_curve / (perc_curve + harm_curve + 1e-9)
    ratio_sigma = max(1.0, 0.5 * sample_rate / float(hop_length))
    ratio_curve = scipy.ndimage.gaussian_filter1d(ratio_curve, sigma=ratio_sigma)
    energy_novelty = np.abs(np.diff(ratio_curve, prepend=ratio_curve[0]))

    spectral_flux = _normalise_curve(spectral_flux)
    self_similarity = _normalise_curve(self_similarity)
    energy_novelty = _normalise_curve(energy_novelty)
    combined = 0.5 * spectral_flux + 0.3 * self_similarity + 0.2 * energy_novelty
    smoothed = scipy.ndimage.gaussian_filter1d(combined, sigma=1.5)
    return smoothed, energy_novelty


def _normalise_curve(curve: np.ndarray) -> np.ndarray:
    if curve.size == 0:
        return curve
    min_val = float(np.min(curve))
    max_val = float(np.max(curve))
    if max_val - min_val < 1e-9:
        return np.zeros_like(curve)
    return (curve - min_val) / (max_val - min_val)


def _enforce_min_frame_spacing(
    peaks: np.ndarray,
    novelty: np.ndarray,
    min_spacing: int,
) -> np.ndarray:
    if peaks.size == 0:
        return peaks
    selected: List[int] = []
    for idx in np.sort(peaks):
        if not selected:
            selected.append(int(idx))
            continue
        if idx - selected[-1] < min_spacing:
            if novelty[idx] > novelty[selected[-1]]:
                selected[-1] = int(idx)
        else:
            selected.append(int(idx))
    return np.asarray(selected, dtype=int)


def _enforce_min_time_spacing(
    times: Sequence[float],
    frames: Sequence[int],
    novelty: np.ndarray,
    min_spacing_seconds: float,
) -> np.ndarray:
    times = np.asarray(times, dtype=float)
    frames = np.asarray(frames, dtype=int)
    if times.size == 0:
        return np.zeros(0, dtype=bool)
    if times.size <= 2:
        return np.ones(times.shape, dtype=bool)

    kept_indices: List[int] = [0]
    for idx in range(1, len(times) - 1):
        previous_idx = kept_indices[-1]
        if times[idx] - times[previous_idx] < min_spacing_seconds:
            if previous_idx == 0:
                continue
            prev_frame = frames[previous_idx]
            current_frame = frames[idx]
            if novelty[current_frame] > novelty[prev_frame]:
                kept_indices[-1] = idx
        else:
            kept_indices.append(idx)

    kept_indices.append(len(times) - 1)
    mask = np.zeros(times.shape, dtype=bool)
    mask[kept_indices] = True
    return mask


def _refine_boundaries(
    peaks: np.ndarray,
    energy_novelty: np.ndarray,
    search_radius: int,
) -> np.ndarray:
    if peaks.size == 0:
        return peaks
    refined: List[int] = []
    total = energy_novelty.shape[0]
    radius = max(1, search_radius)
    for idx in peaks:
        start = max(0, int(idx) - radius)
        end = min(total, int(idx) + radius + 1)
        window = energy_novelty[start:end]
        if window.size == 0:
            refined.append(int(idx))
            continue
        offset = int(np.argmax(window))
        refined.append(start + offset)
    return np.asarray(refined, dtype=int)


def _classify_segments(
    percussive_ratios: Sequence[float],
    percussive_energy: Sequence[float],
    harmonic_energy: Sequence[float],
) -> List[str]:
    ratios = np.asarray(percussive_ratios, dtype=float)
    perc_energy = np.asarray(percussive_energy, dtype=float)
    harm_energy = np.asarray(harmonic_energy, dtype=float)
    total_energy = perc_energy + harm_energy
    if total_energy.size == 0:
        return []
    median_energy = float(np.median(total_energy)) if total_energy.size else 0.0
    categories: List[str] = []
    for idx, (ratio, energy) in enumerate(zip(ratios, total_energy, strict=True)):
        if idx == 0:
            categories.append("intro")
            continue
        if idx == len(ratios) - 1:
            categories.append("outro")
            continue

        if energy < 0.5 * median_energy and ratio < 0.35:
            categories.append("breakdown")
        elif ratio > 0.65 and energy >= 0.75 * median_energy:
            categories.append("drop")
        elif ratio > 0.45:
            categories.append("groove")
        elif ratio < 0.35:
            categories.append("breakdown")
        else:
            categories.append("bridge")
    return categories
