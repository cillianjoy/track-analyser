"""High level orchestration for audio track analysis."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from .analysis import beats, harmonic, loudness, structure, stems
from .utils import AudioInput, coerce_audio, DEFAULT_SEED


@dataclass
class TrackAnalysisResult:
    """Container aggregating all per-module analysis artefacts."""

    audio: AudioInput
    beat: beats.BeatAnalysis
    downbeat: Optional[beats.DownbeatAnalysis]
    structure: structure.StructureAnalysis
    loudness: loudness.LoudnessAnalysis
    harmonic: harmonic.HarmonicAnalysis
    stems: Optional[stems.StemBundle] = None


def analyse_track(
    source: str | AudioInput,
    *,
    output_dir: Optional[str | Path] = None,
    use_stems: bool = False,
    seed: int = DEFAULT_SEED,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> TrackAnalysisResult:
    """Run the deterministic analysis pipeline on ``source``.

    Parameters
    ----------
    source:
        File path to the audio file or a preloaded :class:`~track_analyser.utils.AudioInput`.
    output_dir:
        Optional destination folder for persisted artefacts. When omitted the
        caller can render the outputs manually using :mod:`track_analyser.rendering.outputs`.
    use_stems:
        Whether to attempt stem separation for enhanced harmonic analysis. The
        demucs / torch dependency is optional and automatically guarded.
    seed:
        Random seed used to initialise deterministic components in the
        underlying analysis modules.
    """

    audio = source if isinstance(source, AudioInput) else coerce_audio(source)
    if progress_callback:
        progress_callback("audio")

    beat_result, downbeat_result = beats.analyse_beats(audio, seed=seed)
    if progress_callback:
        progress_callback("beats")

    structure_result = structure.analyse_structure(audio, beat_result, seed=seed)
    if progress_callback:
        progress_callback("structure")

    loudness_result = loudness.analyse_loudness(audio, seed=seed)
    if progress_callback:
        progress_callback("loudness")

    harmonic_result = harmonic.analyse_harmonic(
        audio, beat_result, downbeat_result, seed=seed
    )
    if progress_callback:
        progress_callback("harmonic")

    stem_result: Optional[stems.StemBundle] = None
    if use_stems:
        stem_result = stems.separate_stems(audio.path, output_dir, seed=seed)
        if progress_callback:
            progress_callback("stems")

    result = TrackAnalysisResult(
        audio=audio,
        beat=beat_result,
        downbeat=downbeat_result,
        structure=structure_result,
        loudness=loudness_result,
        harmonic=harmonic_result,
        stems=stem_result,
    )

    if output_dir is not None:
        from .rendering import outputs  # local import to avoid circular dependency

        outputs.render_all(result, Path(output_dir))
        if progress_callback:
            progress_callback("render")

    return result
