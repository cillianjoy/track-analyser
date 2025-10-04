"""Utilities for persisting analysis artefacts as a structured report."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import librosa

from .pipeline import TrackAnalysisResult


@dataclass(slots=True)
class ReportRequest:
    """Configuration describing which artefacts should be generated."""

    include_json: bool = True
    include_csv: bool = True
    include_plots: bool = True
    json_path: Path | None = None
    csv_dir: Path | None = None
    plots_dir: Path | None = None


@dataclass(slots=True)
class ReportOutputs:
    """Paths to the artefacts produced when generating a report."""

    json: Path | None
    csv: Dict[str, Path]
    plots: Dict[str, Path]


def generate_report(
    result: TrackAnalysisResult,
    output_dir: Path,
    request: ReportRequest | None = None,
) -> ReportOutputs:
    """Persist a structured analysis report to ``output_dir``.

    Parameters
    ----------
    result:
        The aggregate analysis container returned by :func:`analyse_track`.
    output_dir:
        Base directory that will be used for any artefacts that do not have an
        explicit override in ``request``.
    request:
        Optional :class:`ReportRequest` detailing which artefacts should be
        written and where they should live. When omitted all artefacts are
        generated inside ``output_dir``.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    request = request or ReportRequest()

    json_path: Path | None = None
    if request.include_json:
        json_path = request.json_path or output_dir / "report.json"
        json_path.parent.mkdir(parents=True, exist_ok=True)
        _write_report_json(result, json_path)

    csv_outputs: Dict[str, Path] = {}
    if request.include_csv:
        csv_dir = request.csv_dir or output_dir
        csv_dir.mkdir(parents=True, exist_ok=True)
        csv_outputs = _write_csv_tables(result, csv_dir)

    plot_outputs: Dict[str, Path] = {}
    if request.include_plots:
        plots_dir = request.plots_dir or output_dir
        plots_dir.mkdir(parents=True, exist_ok=True)
        plot_outputs = _write_plots(result, plots_dir)

    return ReportOutputs(json=json_path, csv=csv_outputs, plots=plot_outputs)


def _write_report_json(result: TrackAnalysisResult, path: Path) -> None:
    """Serialise ``result`` into a JSON summary."""

    summary = {
        "audio": {
            "path": result.audio.path,
            "sample_rate": result.audio.sample_rate,
            "duration": result.audio.duration,
        },
        "beat": {
            "bpm": result.beat.bpm,
            "confidence": result.beat.confidence,
            "count": len(result.beat.beat_times),
        },
        "downbeat": {
            "source": result.downbeat.source if result.downbeat else None,
            "count": len(result.downbeat.downbeat_times)
            if result.downbeat
            else 0,
        },
        "structure": [
            {
                "label": seg.label,
                "category": seg.category,
                "start": seg.start,
                "end": seg.end,
                "confidence": seg.confidence,
            }
            for seg in result.structure.segments
        ],
        "loudness": {
            "integrated_lufs": result.loudness.integrated_lufs,
            "loudness_range": result.loudness.loudness_range,
            "true_peak_dbfs": result.loudness.true_peak_dbfs,
            "rms_dbfs": result.loudness.rms_dbfs,
        },
        "harmonic": {
            "key": result.harmonic.primary_key.key,
            "key_confidence": result.harmonic.primary_key.confidence,
            "secondary_key": {
                "key": result.harmonic.secondary_key.key,
                "confidence": result.harmonic.secondary_key.confidence,
            },
            "chord_change_points": [
                {
                    "time": point.time,
                    "strength": point.strength,
                }
                for point in result.harmonic.chord_change_points
            ],
        },
        "features": {
            "ltas": result.features.ltas.as_dict(),
            "spectral_centroid": {
                "mean": result.features.spectral_centroid.mean,
                "median": result.features.spectral_centroid.median,
            },
            "spectral_rolloff": {
                "mean": result.features.spectral_rolloff.mean,
                "median": result.features.spectral_rolloff.median,
            },
        },
        "stereo": {
            "mid_rms": result.stereo.mid_rms,
            "side_rms": result.stereo.side_rms,
            "correlation": result.stereo.correlation,
            "width": result.stereo.width.as_dict(),
        },
    }

    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def _write_csv_tables(result: TrackAnalysisResult, output_dir: Path) -> Dict[str, Path]:
    """Write beats and structural sections CSV tables."""

    beat_times = np.asarray(result.beat.beat_times, dtype=float)
    beat_frames = np.asarray(result.beat.beat_frames, dtype=int)
    downbeat_times: np.ndarray = (
        np.asarray(result.downbeat.downbeat_times, dtype=float)
        if result.downbeat
        else np.zeros(0, dtype=float)
    )
    beats = pd.DataFrame(
        {
            "index": np.arange(1, beat_times.size + 1, dtype=int),
            "time": beat_times,
            "frame": beat_frames,
            "is_downbeat": _flag_downbeats(beat_times, downbeat_times),
        }
    )
    beats_path = output_dir / "beats.csv"
    beats.to_csv(beats_path, index=False)

    sections = pd.DataFrame(
        [
            {
                "label": seg.label,
                "category": seg.category,
                "start": seg.start,
                "end": seg.end,
                "confidence": seg.confidence,
                "percussive_energy": seg.percussive_energy,
                "harmonic_energy": seg.harmonic_energy,
                "percussive_ratio": seg.percussive_ratio,
            }
            for seg in result.structure.segments
        ]
    )
    sections_path = output_dir / "sections.csv"
    sections.to_csv(sections_path, index=False)

    return {
        "beats": beats_path,
        "sections": sections_path,
    }


def _flag_downbeats(beat_times: np.ndarray, downbeat_times: np.ndarray) -> np.ndarray:
    if beat_times.size == 0:
        return np.zeros(0, dtype=bool)
    if downbeat_times.size == 0:
        return np.zeros_like(beat_times, dtype=bool)
    flags = np.zeros_like(beat_times, dtype=bool)
    for idx, time in enumerate(beat_times):
        if np.any(np.isclose(time, downbeat_times, atol=1e-2)):
            flags[idx] = True
    return flags


def _write_plots(result: TrackAnalysisResult, output_dir: Path) -> Dict[str, Path]:
    """Render PNG visualisations for key analysis artefacts."""

    plots: Dict[str, Path] = {}
    plots["waveform_beats"] = _plot_waveform_with_beats(result, output_dir)
    plots["tempogram"] = _plot_tempogram(result, output_dir)
    plots["novelty"] = _plot_novelty_with_boundaries(result, output_dir)
    plots["ltas"] = _plot_ltas(result, output_dir)
    plots["stereo_width"] = _plot_stereo_width(result, output_dir)
    return plots


def _plot_waveform_with_beats(
    result: TrackAnalysisResult, output_dir: Path
) -> Path:
    samples = np.asarray(result.audio.samples, dtype=float)
    if samples.ndim > 1:
        samples = np.mean(samples, axis=0)
    times = _time_axis(samples.size, result.audio.sample_rate)
    plt.figure(figsize=(10, 4))
    if samples.size:
        plt.plot(times, samples, linewidth=0.8, color="#1f77b4")
    else:
        plt.text(0.5, 0.5, "No audio samples", ha="center", va="center")
    for beat_time in result.beat.beat_times:
        plt.axvline(beat_time, color="#ff7f0e", alpha=0.3, linewidth=0.8)
    plt.title("Waveform with beats")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    path = output_dir / "waveform_beats.png"
    plt.savefig(path)
    plt.close()
    return path


def _plot_tempogram(result: TrackAnalysisResult, output_dir: Path) -> Path:
    samples = np.asarray(result.audio.samples, dtype=float)
    if samples.ndim > 1:
        samples = np.mean(samples, axis=0)
    hop_length = 512
    if samples.size:
        tempogram = librosa.feature.tempogram(
            y=samples, sr=result.audio.sample_rate, hop_length=hop_length
        )
    else:
        tempogram = np.zeros((1, 1), dtype=float)
    tempogram = np.asarray(tempogram, dtype=float)
    if tempogram.size == 0 or tempogram.shape[1] == 0:
        tempogram = np.zeros((1, 1), dtype=float)
    bpm = librosa.tempo_frequencies(
        tempogram.shape[0], sr=result.audio.sample_rate, hop_length=hop_length
    )
    bpm = np.asarray(bpm, dtype=float)
    if bpm.size == 0:
        bpm = np.array([0.0], dtype=float)
    bpm = np.nan_to_num(bpm, nan=0.0, posinf=0.0, neginf=0.0)
    if not np.any(np.isfinite(bpm)):
        bpm = np.array([0.0], dtype=float)
    times = np.arange(tempogram.shape[1], dtype=float) * hop_length / result.audio.sample_rate
    if times.size == 0:
        times = np.array([0.0], dtype=float)
    x_max = times[-1] if times.size > 1 else times[0] + 1e-6
    y_max = bpm[-1] if bpm.size > 1 else bpm[0] + 1e-6
    plt.figure(figsize=(10, 4))
    extent = [times[0], x_max, bpm[0], y_max]
    plt.imshow(
        tempogram,
        aspect="auto",
        origin="lower",
        extent=extent,
        cmap="magma",
    )
    plt.colorbar(label="Tempogram strength")
    plt.xlabel("Time (s)")
    plt.ylabel("Tempo (BPM)")
    plt.title("Tempogram")
    plt.tight_layout()
    path = output_dir / "tempogram.png"
    plt.savefig(path)
    plt.close()
    return path


def _plot_novelty_with_boundaries(
    result: TrackAnalysisResult, output_dir: Path
) -> Path:
    novelty = np.asarray(result.structure.novelty_curve, dtype=float)
    plt.figure(figsize=(10, 4))
    if novelty.size:
        times = np.linspace(0.0, result.audio.duration, num=novelty.size)
        plt.plot(times, novelty, color="#2ca02c")
        for segment in result.structure.segments:
            plt.axvline(segment.start, color="#d62728", alpha=0.3, linewidth=0.8)
    else:
        plt.text(0.5, 0.5, "No novelty data", ha="center", va="center")
    plt.title("Novelty vs structural boundaries")
    plt.xlabel("Time (s)")
    plt.ylabel("Novelty")
    plt.tight_layout()
    path = output_dir / "novelty_boundaries.png"
    plt.savefig(path)
    plt.close()
    return path


def _plot_ltas(result: TrackAnalysisResult, output_dir: Path) -> Path:
    frequencies = np.asarray(result.features.ltas.frequencies, dtype=float)
    magnitude = np.asarray(result.features.ltas.magnitude, dtype=float)
    plt.figure(figsize=(10, 4))
    if frequencies.size and magnitude.size:
        plt.semilogx(frequencies, magnitude, color="#9467bd")
    else:
        plt.text(0.5, 0.5, "No LTAS data", ha="center", va="center")
    plt.title("Long-term average spectrum")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.tight_layout()
    path = output_dir / "ltas.png"
    plt.savefig(path)
    plt.close()
    return path


def _plot_stereo_width(result: TrackAnalysisResult, output_dir: Path) -> Path:
    width = result.stereo.width
    labels = ["Low", "Mid", "High"]
    values = [width.low, width.mid, width.high]
    plt.figure(figsize=(6, 4))
    plt.bar(labels, values, color="#8c564b")
    plt.ylim(0.0, max(values + [1.0]))
    plt.title("Mid/Side width by band")
    plt.ylabel("Width")
    plt.tight_layout()
    path = output_dir / "stereo_width.png"
    plt.savefig(path)
    plt.close()
    return path


def _time_axis(sample_count: int, sample_rate: int) -> np.ndarray:
    if sample_count == 0:
        return np.zeros(0, dtype=float)
    duration = sample_count / float(sample_rate)
    return np.linspace(0.0, duration, num=sample_count)

