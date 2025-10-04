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

**Windows/Conda note:** install `Cython>=0.29` before running `pip install .[madmom]`. The upstream `madmom` sdist imports Cython during its build, so pre-installing it avoids failures while the extra resolves dependencies.

When using Conda on Windows, make sure the `audio` environment is active before running `pip install` so that binaries such as `numpy` and `librosa` resolve correctly.

## Command line usage

Once installed, the CLI is available as `track-analyser`:

```bash
track-analyser analyze path/to/audio.wav --out reports/song
```

The command performs deterministic beat, structure, loudness and harmonic analysis. It exports `report.json`, `beats.csv`, `sections.csv`, optional PNG visualisations (waveform with beats, tempogram, novelty curve, LTAS and stereo width), HTML reports and MIDI hook/bass suggestions.

Pass `--plots`, `--json` or `--csv` to control the generated artefacts. Provide a custom path to move them elsewhere or pass `skip`/`none` to disable that output entirely. When omitted, the artefacts are written inside the `--out` directory.

Refer to the [CLI flag reference](RUNBOOK.md#cli-flag-reference) for a complete description of the available options and operational guidance.

## Python API

```python
from track_analyser import analyse_track

result = analyse_track("path/to/audio.wav")
print(result.beat.bpm, result.harmonic.key_estimate)
```

The returned dataclass `TrackAnalysisResult` bundles per-module analysis outputs, making it straightforward to persist or run further downstream tasks.

## Testing

Install the project in your environment (editable installs are fine) along with the test dependencies:

```bash
pip install -e .
pip install pytest
```

From the repository root, execute the automated suite with:

```bash
pytest -q
```

The current regression tests focus on three critical paths:

- `tests/test_loudness.py` checks the integrated-loudness regression to guard against changes in analysis math or default parameters.
- `tests/test_rendering_outputs.py` verifies the report renderer to ensure the structured artefacts remain stable.
- `tests/test_cli.py::test_analyze_generates_requested_outputs` exercises the CLI end-to-end to confirm the requested files are created.

When you add features, please extend or supplement these tests so we maintain coverage of the most important behaviours.

For a quick manual smoke test, you can also run the CLI end-to-end. First generate
the example click track (this writes to the git-ignored `examples/` directory):

```bash
python scripts/make_tiny_click.py
```

Then analyse the generated file:

```bash
track-analyser analyze examples/tiny_click_120.wav --out reports/smoke
```

Because `examples/` is ignored by git, the rendered WAV stays local to your machine. The full [smoke test runbook](RUNBOOK.md#smoke-test-end-to-end-cli-run) captures the checklist and post-run validation steps.

Additional operational procedures—such as dependency refreshes and packaging validation—are documented in the [maintenance workflows](RUNBOOK.md#maintenance-workflows) section of the runbook.
