"""CLI regression tests."""

from __future__ import annotations

import math
import wave
from pathlib import Path

import numpy as np
from click.testing import CliRunner

from track_analyser.cli import cli


def test_analyze_help_lists_new_flags() -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["analyze", "--help"])
    assert result.exit_code == 0, result.output
    help_text = result.output
    for flag in ("--out", "--plots", "--json", "--csv"):
        assert flag in help_text
    assert "analyze" in help_text


def test_analyze_generates_requested_outputs(tmp_path) -> None:
    audio_path = tmp_path / "tone.wav"
    _write_test_tone(audio_path)
    output_dir = tmp_path / "report"
    plots_dir = tmp_path / "plots"
    csv_dir = tmp_path / "tables"
    json_path = tmp_path / "custom" / "custom_report.json"

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "analyze",
            str(audio_path),
            "--out",
            str(output_dir),
            "--plots",
            str(plots_dir),
            "--json",
            str(json_path),
            "--csv",
            str(csv_dir),
        ],
    )
    assert result.exit_code == 0, result.output
    assert json_path.exists(), "Custom JSON path should exist"
    assert (csv_dir / "beats.csv").exists(), "beats.csv should be created"
    assert (csv_dir / "sections.csv").exists(), "sections.csv should be created"
    expected_plots = {
        "waveform_beats.png",
        "tempogram.png",
        "novelty_boundaries.png",
        "ltas.png",
        "stereo_width.png",
    }
    for plot_name in expected_plots:
        assert (plots_dir / plot_name).exists(), f"Plot {plot_name} should exist"


def _write_test_tone(path: Path, *, sr: int = 22_050, duration: float = 0.5) -> None:
    sample_count = int(sr * duration)
    times = np.linspace(0.0, duration, num=sample_count, endpoint=False)
    waveform = 0.25 * np.sin(2.0 * math.pi * 220.0 * times)
    pcm = np.clip(waveform, -1.0, 1.0)
    int_samples = (pcm * 32767).astype(np.int16)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sr)
        handle.writeframes(int_samples.tobytes())
