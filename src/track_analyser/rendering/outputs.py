"""Rendering helpers for persisting analysis artefacts."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from ..pipeline import TrackAnalysisResult

try:  # pragma: no cover - optional dependency
    import mido
except ImportError:  # pragma: no cover - fallback minimal MIDI writer
    mido = None  # type: ignore[assignment]


def render_all(result: TrackAnalysisResult, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(result, output_dir / "summary.json")
    _write_csv_tables(result, output_dir)
    _write_plots(result, output_dir)
    _write_html_report(result, output_dir / "report.html")
    _write_midi(result.harmonic.hook_suggestion, output_dir / "hook.mid")
    _write_midi(result.harmonic.bass_suggestion, output_dir / "bass.mid")


def _write_json(result: TrackAnalysisResult, path: Path) -> None:
    summary = {
        "audio": {
            "path": result.audio.path,
            "sample_rate": result.audio.sample_rate,
            "duration": result.audio.duration,
        },
        "beat": {
            "bpm": result.beat.bpm,
            "confidence": result.beat.confidence,
        },
        "downbeat": {
            "source": result.downbeat.source if result.downbeat else None,
        },
        "structure": [asdict(segment) for segment in result.structure.segments],
        "loudness": asdict(result.loudness),
        "harmonic": {
            "key": result.harmonic.primary_key.key,
            "key_confidence": result.harmonic.primary_key.confidence,
            "secondary_key": asdict(result.harmonic.secondary_key),
            "chord_change_points": [
                asdict(point) for point in result.harmonic.chord_change_points
            ],
        },
    }
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def _write_csv_tables(result: TrackAnalysisResult, output_dir: Path) -> None:
    segments = pd.DataFrame(
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
    segments.to_csv(output_dir / "sections.csv", index=False)
    segments[["label", "start", "end", "confidence"]].to_csv(
        output_dir / "segments.csv", index=False
    )

    chords = pd.DataFrame(
        [
            {
                "time": hint.time,
                "chord": hint.chord,
                "confidence": hint.confidence,
            }
            for hint in result.harmonic.chord_hints
        ]
    )
    chords.to_csv(output_dir / "chords.csv", index=False)

    changes = pd.DataFrame(
        [
            {
                "time": point.time,
                "strength": point.strength,
            }
            for point in result.harmonic.chord_change_points
        ],
        columns=["time", "strength"],
    )
    changes.to_csv(output_dir / "chord_changes.csv", index=False)

    loudness = pd.DataFrame(
        {
            "short_term_lufs": result.loudness.short_term_lufs,
            "momentary_lufs": result.loudness.momentary_lufs,
        }
    )
    loudness.to_csv(output_dir / "loudness.csv", index=False)


def _write_plots(result: TrackAnalysisResult, output_dir: Path) -> None:
    plt.figure(figsize=(10, 6))
    plt.plot(result.loudness.short_term_lufs, label="Short-term LUFS")
    plt.plot(result.loudness.momentary_lufs, label="Momentary LUFS", alpha=0.7)
    plt.title("Loudness over time")
    plt.xlabel("Frame")
    plt.ylabel("LUFS")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "loudness.png")
    plt.close()

    plt.figure(figsize=(10, 4))
    plt.bar(
        ["Low", "Mid", "High"],
        [
            result.harmonic.spectral_balance.low_band,
            result.harmonic.spectral_balance.mid_band,
            result.harmonic.spectral_balance.high_band,
        ],
    )
    plt.title("Spectral balance")
    plt.tight_layout()
    plt.savefig(output_dir / "spectral_balance.png")
    plt.close()


def _write_html_report(result: TrackAnalysisResult, path: Path) -> None:
    rows = "".join(
        f"<tr><td>{seg.label}</td><td>{seg.start:.2f}</td><td>{seg.end:.2f}</td><td>{seg.confidence:.2f}</td></tr>"
        for seg in result.structure.segments
    )
    html = f"""
    <html>
    <head>
        <meta charset='utf-8'/>
        <title>Track Analysis Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 2rem; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #ccc; padding: 0.5rem; text-align: left; }}
        </style>
    </head>
    <body>
        <h1>Track Analysis Report</h1>
        <p><strong>Audio:</strong> {result.audio.path or "In-memory"} ({result.audio.duration:.2f}s)</p>
        <p><strong>BPM:</strong> {result.beat.bpm:.2f} (confidence {result.beat.confidence:.2f})</p>
        <p><strong>Key:</strong> {result.harmonic.primary_key.key} (confidence {result.harmonic.primary_key.confidence:.2f})</p>
        <p><strong>Second choice:</strong> {result.harmonic.secondary_key.key} (confidence {result.harmonic.secondary_key.confidence:.2f})</p>
        <h2>Structure</h2>
        <table>
            <tr><th>Label</th><th>Start</th><th>End</th><th>Confidence</th></tr>
            {rows}
        </table>
    </body>
    </html>
    """
    path.write_text(html, encoding="utf-8")


def _write_midi(suggestion: Optional, path: Path) -> None:
    if suggestion is None:
        return
    if suggestion.notes.empty:
        return
    if mido is None:  # pragma: no cover - fallback simple MIDI writer
        _write_basic_midi(suggestion, path)
        return

    mid = mido.MidiFile(type=1)
    track = mido.MidiTrack()
    mid.tracks.append(track)
    tempo = mido.bpm2tempo(120)
    track.append(mido.MetaMessage("set_tempo", tempo=tempo))
    ticks_per_beat = mid.ticks_per_beat
    events = _note_events(suggestion.notes)
    last_tick = 0
    for time_sec, message in events:
        tick = int(round(time_sec * ticks_per_beat))
        delta = tick - last_tick
        message.time = max(0, delta)
        track.append(message)
        last_tick = tick
    mid.save(path)


def _write_basic_midi(suggestion, path: Path) -> None:
    """Write a very small type-0 MIDI file without external deps."""

    header = (
        b"MThd"
        + (6).to_bytes(4, "big")
        + (0).to_bytes(2, "big")
        + (1).to_bytes(2, "big")
        + (480).to_bytes(2, "big")
    )
    events = _note_events(suggestion.notes)
    bytes_events: List[bytes] = []
    last_tick = 0
    for time_sec, message in events:
        tick = int(round(time_sec * 480))
        delta = tick - last_tick
        last_tick = tick
        note_on = getattr(message, "type", "note_on") == "note_on"
        pitch = getattr(message, "note", 60)
        velocity = getattr(message, "velocity", 100)
        bytes_events.append(_midi_event(delta, note_on, pitch, velocity))
    track_data = b"".join(bytes_events) + b"\x00\xff\x2f\x00"
    track_chunk = b"MTrk" + len(track_data).to_bytes(4, "big") + track_data
    path.write_bytes(header + track_chunk)


def _note_events(notes: pd.DataFrame) -> List[Tuple[float, object]]:
    events: List[Tuple[float, object]] = []
    for _, row in notes.iterrows():
        start = float(row["start"])
        end = start + float(row["duration"])
        pitch = int(row["pitch"])
        velocity = int(row["velocity"])
        events.append(
            (
                start,
                mido.Message("note_on", note=pitch, velocity=velocity, time=0)
                if mido is not None
                else _BasicMessage(True, pitch, velocity),
            )
        )
        events.append(
            (
                end,
                mido.Message("note_off", note=pitch, velocity=0, time=0)
                if mido is not None
                else _BasicMessage(False, pitch, 0),
            )
        )
    events.sort(key=lambda item: item[0])
    return events


def _midi_event(delta: int, note_on: bool, pitch: int, velocity: int) -> bytes:
    status = 0x90 if note_on else 0x80
    return _var_len(delta) + bytes([status, pitch, velocity])


def _var_len(value: int) -> bytes:
    buffer = value & 0x7F
    bytes_out = []
    while (value >> 7) != 0:
        value >>= 7
        buffer <<= 8
        buffer |= (value & 0x7F) | 0x80
    while True:
        bytes_out.append(buffer & 0xFF)
        if buffer & 0x80:
            buffer >>= 8
        else:
            break
    return bytes(bytes_out[::-1])


@dataclass
class _BasicMessage:
    note_on: bool
    note: int
    velocity: int
