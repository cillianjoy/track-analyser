# RUNBOOK

Operational procedures and checklists for `track-analyser`.

## Table of contents
- [CLI flag reference](#cli-flag-reference)
- [Smoke test: end-to-end CLI run](#smoke-test-end-to-end-cli-run)
- [Maintenance workflows](#maintenance-workflows)

## CLI flag reference

### `track-analyser analyze`

| Argument / flag | Type | Default | Description |
| --- | --- | --- | --- |
| `audio_path` | Path to an audio file | _required_ | Input track to analyse. Must exist and point to a readable audio file. |
| `--out` | Directory path | _required_ | Destination folder for generated artefacts (HTML report, MIDI files and any artefacts not explicitly redirected). Created automatically if missing. |
| `--plots` | Directory path or `skip` | _optional_ | Generate PNG plots (waveform + beats, tempogram, novelty, LTAS, stereo width). Provide a directory to store them or `skip`/`none` to disable. Defaults to the `--out` directory when omitted. |
| `--json` | File path or `skip` | _optional_ | Generate `report.json`. Provide a file path to relocate it or `skip`/`none` to disable. Defaults to `report.json` inside `--out`. |
| `--csv` | Directory path or `skip` | _optional_ | Generate `beats.csv` and `sections.csv`. Provide a directory to relocate them or `skip`/`none` to disable. Defaults to the `--out` directory when omitted. |

**Operational notes**
- The CLI reports rich progress output and a success summary containing BPM and key estimates plus the final artefact locations.
- Redirecting or disabling JSON/CSV/plot outputs is useful when integrating with automated pipelines that expect files in dedicated directories.
- Artefacts not covered by the dedicated flags (HTML and MIDI) always remain inside `--out`.

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
   track-analyser analyze examples/tiny_click_120.wav --out reports/smoke
   ```
4. Inspect the console output for the success summary, then spot-check the artefacts written to `reports/smoke` (especially `report.json`, `beats.csv`, `sections.csv` and the plots) to confirm the pipeline executed end-to-end.

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
