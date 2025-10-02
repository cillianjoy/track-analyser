# AGENTS.md

## Setup commands
- Install deps: `pip install -r requirements.txt`
- Install package: `pip install -e .`
- Run tests: `pytest -q`
- Run CLI smoke test: `track-analyser analyse examples/tiny_click_120.wav --out reports/smoke --plots --json --csv`

## Code style
- Python 3.11
- Use Ruff for lint and format: `ruff check --fix . && ruff format .`

## Expectations for PRs
- Add or update tests for new features.
- Keep functions pure where practical. Avoid global state.
- Document new CLI flags in README and RUNBOOK.
