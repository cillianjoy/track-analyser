"""CLI regression tests."""

from __future__ import annotations

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
