"""Generate a tiny click track for smoke-testing the analyser CLI."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf

DEFAULT_OUTPUT = Path("examples/tiny_click_120.wav")
SAMPLE_RATE = 44_100
BPM = 120
BEATS_PER_BAR = 4
CLICK_DURATION_SECONDS = 0.03
ACCENT_FREQUENCY = 1500.0
REGULAR_FREQUENCY = 1000.0


def _synth_click(
    frequency: float, amplitude: float, sample_rate: int, duration: float
) -> np.ndarray:
    """Return a short, exponentially decaying sine click."""
    sample_count = int(duration * sample_rate)
    times = np.linspace(0.0, duration, sample_count, endpoint=False)
    envelope = np.exp(-times * 50.0)
    waveform = amplitude * np.sin(2 * np.pi * frequency * times) * envelope
    return waveform.astype(np.float32)


def make_click_track(path: Path) -> Path:
    """Create a one-bar, four-beat click track at 120 BPM."""
    seconds_per_beat = 60.0 / BPM
    click = _synth_click(REGULAR_FREQUENCY, 0.6, SAMPLE_RATE, CLICK_DURATION_SECONDS)
    accent = _synth_click(ACCENT_FREQUENCY, 0.9, SAMPLE_RATE, CLICK_DURATION_SECONDS)

    click_length = click.shape[0]
    bar_samples = int(np.ceil(BEATS_PER_BAR * seconds_per_beat * SAMPLE_RATE))
    total_samples = bar_samples + click_length
    audio = np.zeros(total_samples, dtype=np.float32)

    for beat in range(BEATS_PER_BAR):
        start = int(round(beat * seconds_per_beat * SAMPLE_RATE))
        end = start + click_length
        waveform = accent if beat == 0 else click
        audio[start:end] += waveform[: total_samples - start]

    audio = np.clip(audio, -1.0, 1.0)

    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, audio, SAMPLE_RATE)
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "output",
        nargs="?",
        default=str(DEFAULT_OUTPUT),
        help="Destination path for the generated WAV (default: %(default)s)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    written_path = make_click_track(output_path)
    print(f"Wrote click track to {written_path}")


if __name__ == "__main__":
    main()
