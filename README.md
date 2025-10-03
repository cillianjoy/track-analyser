# track-analyser

Analyse full audio tracks and export structured artefacts.

## Installation

### 1. Create or activate the Conda environment

```bash
conda create -n audio python=3.11
conda activate audio
```

### 2. Install the package

Install the core analyser with the pinned, reproducible dependency set:

```bash
pip install .
```

Optional extras provide integrations that require heavier dependencies:

```bash
# Beat-tracking refinement with madmom
pip install .[madmom]

# Torch-based models (e.g. GPU accelerated inference)
pip install .[torch]

# Demucs stem separation support (requires torch extras on most systems)
pip install .[demucs]
```

When using Conda on Windows, make sure the `audio` environment is active before running `pip install` so that binaries such as `numpy` and `librosa` resolve correctly.

## Command line usage

Once installed, the CLI is available as `track-analyser`:

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
