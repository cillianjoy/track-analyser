"""Command line interface for :mod:`track_analyser`."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable

import click
from rich.console import Console
from rich.progress import Progress

from .pipeline import analyse_track

JSON_FILENAME = "summary.json"
CSV_FILENAMES = ("segments.csv", "chords.csv", "loudness.csv")
PLOT_FILENAMES = ("loudness.png", "spectral_balance.png")


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
    "plots_dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Optional directory to relocate generated plot images.",
)
@click.option(
    "--json",
    "json_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Optional path for the summary JSON file.",
)
@click.option(
    "--csv",
    "csv_dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Optional directory to relocate CSV tables.",
)
def analyze_command(
    audio_path: Path,
    output_dir: Path,
    plots_dir: Path | None,
    json_path: Path | None,
    csv_dir: Path | None,
) -> None:
    """Analyse ``audio_path`` and render artefacts to disk."""

    console = Console()
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        stages = 6  # audio, beats, structure, loudness, harmonic, render
        with Progress(transient=True) as progress:
            task = progress.add_task("Analysing", total=stages)

            def _advance(_: str) -> None:
                progress.advance(task)

            result = analyse_track(
                audio_path,
                output_dir=output_dir,
                progress_callback=_advance,
            )
        destinations = _relocate_outputs(
            output_dir, plots_dir=plots_dir, json_path=json_path, csv_dir=csv_dir
        )
        console.print(
            f"[green]Analysis completed[/green] -> {output_dir}\n"
            f"BPM: {result.beat.bpm:.2f}, Key: {result.harmonic.key_estimate.key}\n"
            f"JSON: {destinations['json']}\n"
            f"CSV: {destinations['csv']}\n"
            f"Plots: {destinations['plots']}"
        )
    except Exception as exc:  # pragma: no cover - CLI guard
        console.print(f"[red]Error:[/red] {exc}")
        raise SystemExit(1) from exc


def _relocate_outputs(
    output_dir: Path,
    *,
    plots_dir: Path | None,
    json_path: Path | None,
    csv_dir: Path | None,
) -> Dict[str, Path]:
    """Return the final locations for rendered artefacts, moving files if needed."""

    destinations: Dict[str, Path] = {}

    json_source = output_dir / JSON_FILENAME
    if json_path and json_source.exists():
        _move_file(json_source, json_path)
        destinations["json"] = json_path
    else:
        destinations["json"] = json_source

    csv_destination_dir = csv_dir or output_dir
    if csv_dir:
        _move_named_files(output_dir, csv_destination_dir, CSV_FILENAMES)
    destinations["csv"] = csv_destination_dir

    plots_destination_dir = plots_dir or output_dir
    if plots_dir:
        _move_named_files(output_dir, plots_destination_dir, PLOT_FILENAMES)
    destinations["plots"] = plots_destination_dir

    return destinations


def _move_named_files(
    base_dir: Path, destination_dir: Path, filenames: Iterable[str]
) -> None:
    """Move a collection of files from ``base_dir`` into ``destination_dir``."""

    destination_dir.mkdir(parents=True, exist_ok=True)
    for name in filenames:
        source = base_dir / name
        if not source.exists():
            continue
        destination = destination_dir / name
        _move_file(source, destination)


def _move_file(source: Path, destination: Path) -> None:
    """Move ``source`` to ``destination``, overwriting if necessary."""

    try:
        if source.resolve() == destination.resolve(strict=False):
            return
    except FileNotFoundError:
        # ``destination`` may not exist yet; fall back to simple path comparison.
        if source == destination:
            return
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.is_dir():
            raise IsADirectoryError(destination)
        destination.unlink()
    source.replace(destination)


def main() -> None:
    cli()


if __name__ == "__main__":  # pragma: no cover
    main()
