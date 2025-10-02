"""Command line interface for track_analyser."""

from __future__ import annotations

from pathlib import Path

import click
from rich.console import Console
from rich.progress import Progress

from .pipeline import analyse_track
from .utils import DEFAULT_SEED


@click.group()
def cli() -> None:
    """Entry point for the track analyser CLI."""


@cli.command("analyse")
@click.argument(
    "audio_path", type=click.Path(exists=True, dir_okay=False, path_type=Path)
)
@click.option(
    "--output",
    "output_dir",
    type=click.Path(file_okay=False, path_type=Path),
    required=True,
)
@click.option(
    "--use-stems/--no-stems",
    default=False,
    help="Attempt stem assisted analysis via demucs when available",
)
@click.option(
    "--seed", default=DEFAULT_SEED, show_default=True, help="Deterministic random seed"
)
def analyse_command(
    audio_path: Path, output_dir: Path, use_stems: bool, seed: int
) -> None:
    """Analyse a full audio track and export structured artefacts."""

    console = Console()
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        stages = (
            5 + int(use_stems) + 1
        )  # audio, beats, structure, loudness, harmonic, [stems], render
        with Progress(transient=True) as progress:
            task = progress.add_task("Analysing", total=stages)

            def _advance(_: str) -> None:
                progress.advance(task)

            result = analyse_track(
                audio_path,
                output_dir=output_dir,
                use_stems=use_stems,
                seed=seed,
                progress_callback=_advance,
            )
        console.print(
            f"[green]Analysis completed[/green] -> {output_dir}\n"
            f"BPM: {result.beat.bpm:.2f}, Key: {result.harmonic.key_estimate.key}"
        )
    except Exception as exc:  # pragma: no cover - CLI guard
        console.print(f"[red]Error:[/red] {exc}")
        raise SystemExit(1) from exc


def main() -> None:
    cli()


if __name__ == "__main__":  # pragma: no cover
    main()
