# track-analyser

Analyse full audio tracks and export structured artefacts.

## Installation

```bash
pip install -r requirements.txt
pip install -e .
```

## Command line usage

```bash
track-analyser analyse path/to/audio.wav --output reports/song --use-stems
```

The command performs deterministic beat, structure, loudness and harmonic analysis. It exports JSON summaries, CSV tables, PNG plots, HTML reports and MIDI hook/bass suggestions.

Use `--no-stems` to disable the optional demucs integration when torch is unavailable.

## Python API

```python
from track_analyser import analyse_track

result = analyse_track("path/to/audio.wav")
print(result.beat.bpm, result.harmonic.key_estimate)
```

The returned dataclass `TrackAnalysisResult` bundles per-module analysis outputs, making it straightforward to persist or run further downstream tasks.
