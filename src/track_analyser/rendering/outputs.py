"""Rendering helpers for persisting analysis artefacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd

from .. import report as report_module
from ..pipeline import TrackAnalysisResult

try:  # pragma: no cover - optional dependency
    import mido
except ImportError:  # pragma: no cover - fallback minimal MIDI writer
    mido = None  # type: ignore[assignment]


def render_all(
    result: TrackAnalysisResult,
    output_dir: Path,
    *,
    report_request: report_module.ReportRequest | None = None,
) -> report_module.ReportOutputs:
    output_dir.mkdir(parents=True, exist_ok=True)
    report_outputs = report_module.generate_report(result, output_dir, report_request)
    _write_html_report(result, output_dir / "report.html")
    _write_midi(result.harmonic.hook_suggestion, output_dir / "hook.mid")
    _write_midi(result.harmonic.bass_suggestion, output_dir / "bass.mid")
    return report_outputs


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
        <h2>Spectral features</h2>
        <p><strong>Mean spectral centroid:</strong> {result.features.spectral_centroid.mean:.2f} Hz</p>
        <p><strong>Mean spectral roll-off:</strong> {result.features.spectral_rolloff.mean:.2f} Hz</p>
        <h2>Stereo image</h2>
        <p><strong>Mid RMS:</strong> {result.stereo.mid_rms:.4f}</p>
        <p><strong>Side RMS:</strong> {result.stereo.side_rms:.4f}</p>
        <p><strong>Correlation:</strong> {result.stereo.correlation:.2f}</p>
        <table>
            <tr><th>Band</th><th>Width</th></tr>
            <tr><td>Low</td><td>{result.stereo.width.low:.3f}</td></tr>
            <tr><td>Mid</td><td>{result.stereo.width.mid:.3f}</td></tr>
            <tr><td>High</td><td>{result.stereo.width.high:.3f}</td></tr>
        </table>
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
