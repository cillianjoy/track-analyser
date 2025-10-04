# AGENTS.md

## Setup commands
- Install deps: `pip install -r requirements.txt`
- Install package: `pip install -e .`
- Run tests: `pytest -q`
- Before the CLI smoke test, generate the click track: `python scripts/make_tiny_click.py`
- Run CLI smoke test: `track-analyser analyze examples/tiny_click_120.wav --out reports/smoke`

## Code style
- Python 3.11
- Use Ruff for lint and format: `ruff check --fix . && ruff format .`

## Expectations for PRs
- Add or update tests for new features.
- Keep functions pure where practical. Avoid global state.
- Document new CLI flags in [README.md](README.md) and keep the [RUNBOOK CLI flag reference](RUNBOOK.md#cli-flag-reference) current.
- Update operational procedures in the [RUNBOOK](RUNBOOK.md) when workflows change (e.g. smoke tests, maintenance tasks).
