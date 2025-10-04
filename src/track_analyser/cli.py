"""Command line interface for :mod:`track_analyser`."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple

import click
from rich.console import Console
from rich.progress import Progress

from .pipeline import analyse_track
from . import report as report_module
from .rendering import outputs as outputs_module


SKIP_VALUES = {"skip", "none", "false", "off"}


@click.group()
def cli() -> None:
    """Track analyser command line utilities."""


@cli.command("analyze")
@click.argument(
    "audio_path", type=click.Path(exists=True, dir_okay=False, path_type=Path)
)
@click.option(
    "--out",
    "output_dir",
    type=click.Path(file_okay=False, path_type=Path),
    required=True,
    help="Destination for generated artefacts (HTML, MIDI, tables, plots).",
)
@click.option(
    "--plots",
    "plots_option",
    type=str,
    default=None,
    help="Generate plot PNGs. Provide a directory path or 'skip' to disable.",
)
@click.option(
    "--json",
    "json_option",
    type=str,
    default=None,
    help="Generate report.json. Provide a file path or 'skip' to disable.",
)
@click.option(
    "--csv",
    "csv_option",
    type=str,
    default=None,
    help="Generate CSV tables. Provide a directory path or 'skip' to disable.",
)
def analyze_command(
    audio_path: Path,
    output_dir: Path,
    plots_option: str | None,
    json_option: str | None,
    csv_option: str | None,
) -> None:
    """Analyse ``audio_path`` and render artefacts to disk."""

    console = Console()
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        with Progress(transient=True) as progress:
            task = progress.add_task("Analysing", total=0)

            stages_seen = 0

            def _advance(_: str) -> None:
                nonlocal stages_seen
                stages_seen += 1
                progress.update(task, total=stages_seen)
                progress.advance(task)

            result = analyse_track(
                audio_path,
                progress_callback=_advance,
            )
        report_request = _build_report_request(
            output_dir,
            plots_option=plots_option,
            json_option=json_option,
            csv_option=csv_option,
        )
        report_outputs = outputs_module.render_all(
            result,
            output_dir,
            report_request=report_request,
        )
        _advance("render")
        console.print(
            f"[green]Analysis completed[/green] -> {output_dir}\n"
            f"BPM: {result.beat.bpm:.2f}, Key: {result.harmonic.key_estimate.key}\n"
            f"JSON: {_format_json_destination(report_outputs.json)}\n"
            f"CSV: {_format_collection(report_outputs.csv.values())}\n"
            f"Plots: {_format_collection(report_outputs.plots.values())}"
        )
    except Exception as exc:  # pragma: no cover - CLI guard
        console.print(f"[red]Error:[/red] {exc}")
        raise SystemExit(1) from exc


def _build_report_request(
    output_dir: Path,
    *,
    plots_option: str | None,
    json_option: str | None,
    csv_option: str | None,
) -> report_module.ReportRequest:
    include_plots, plots_path = _parse_option(plots_option)
    include_json, json_path = _parse_option(json_option)
    include_csv, csv_path = _parse_option(csv_option)
    return report_module.ReportRequest(
        include_plots=include_plots,
        include_json=include_json,
        include_csv=include_csv,
        plots_dir=_resolve_path(output_dir, plots_path) if plots_path else None,
        json_path=_resolve_path(output_dir, json_path) if json_path else None,
        csv_dir=_resolve_path(output_dir, csv_path) if csv_path else None,
    )


def _parse_option(value: str | None) -> Tuple[bool, Path | None]:
    if value is None:
        return True, None
    lowered = value.lower()
    if lowered in SKIP_VALUES:
        return False, None
    return True, Path(value)


def _resolve_path(output_dir: Path, path: Path) -> Path:
    if path.is_absolute():
        return path
    return (output_dir / path).resolve()


def _format_json_destination(path: Path | None) -> str:
    if path is None:
        return "skipped"
    return str(path)


def _format_collection(paths: Iterable[Path]) -> str:
    realised = list(paths)
    if not realised:
        return "skipped"
    parents = {p.parent for p in realised}
    if len(parents) == 1:
        return str(parents.pop())
    return ", ".join(str(p) for p in realised)


def main() -> None:
    cli()


if __name__ == "__main__":  # pragma: no cover
    main()
