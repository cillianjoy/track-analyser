# RUNBOOK

Operational procedures and checklists for `track-analyser`.

## Table of contents
- [CLI flag reference](#cli-flag-reference)
- [Smoke test: end-to-end CLI run](#smoke-test-end-to-end-cli-run)
- [Maintenance workflows](#maintenance-workflows)

## CLI flag reference

### `track-analyser analyse`

| Argument / flag | Type | Default | Description |
| --- | --- | --- | --- |
| `audio_path` | Path to an audio file | _required_ | Input track to analyse. Must exist and point to a readable audio file. |
| `--output` / `--output-dir` | Directory path | _required_ | Destination folder for generated artefacts (JSON, CSV, plots, HTML, MIDI). Created automatically if missing. |
| `--use-stems / --no-stems` | Boolean flag | `--no-stems` | Toggle Demucs-assisted stem separation. Enable when the Demucs (and Torch) extras are installed. Disable to skip the extra dependency and speed up runs. |
| `--seed` | Integer | `42` | Deterministic random seed shared with the analysis pipeline. Override to explore non-deterministic paths or to reproduce a reported issue exactly. |

**Operational notes**
- The CLI reports rich progress output and a success summary containing BPM and key estimates.
- When Demucs is unavailable, keep `--no-stems` to avoid errors. The command runs without stems by default.
- The `--seed` option is surfaced primarily for reproducing debugging scenarios; normal operations should keep the default.

## Smoke test: end-to-end CLI run

Use this checklist after modifying the analysis pipeline or rendering code, or before publishing a release.

1. (If not already installed) install dependencies and the project in editable mode:
   ```bash
   pip install -r requirements.txt
   pip install -e .
   ```
2. Generate the sample click track (writes to the git-ignored `examples/` directory):
   ```bash
   python scripts/make_tiny_click.py
   ```
3. Run the CLI against the generated file:
   ```bash
   track-analyser analyse examples/tiny_click_120.wav --output reports/smoke
   ```
4. Inspect the console output for the success summary, then spot-check the artefacts written to `reports/smoke` (especially the JSON and plots) to confirm the pipeline executed end-to-end.

## Maintenance workflows

### Refresh dependencies
1. Review `pyproject.toml` and `requirements.txt` for pinned versions that need updates.
2. Update the pins and run the full test suite locally: `pytest -q`.
3. Regenerate the CLI smoke test artefacts (see [Smoke test](#smoke-test-end-to-end-cli-run)).
4. Update the changelog or release notes to highlight dependency changes.

### Validate packaging metadata
1. Ensure `pyproject.toml` metadata (version, dependencies, entry points) is up-to-date.
2. Build the distribution artifacts:
   ```bash
   python -m build
   ```
3. Inspect the generated wheel and sdist in `dist/`.
4. Optionally perform a local install test: `pip install dist/track_analyser-<version>-py3-none-any.whl` inside a clean virtual environment.

### Scheduled reports clean-up
1. Remove historical analysis artefacts from `reports/` if disk usage grows large.
2. Archive important reports before deletion.
3. Ensure the `reports/` directory remains git-ignored to prevent accidental commits of generated content.
