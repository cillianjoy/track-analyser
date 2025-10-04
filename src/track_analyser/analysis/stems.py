"""Optional stem separation using demucs when available."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

from ..utils import DEFAULT_SEED

try:  # pragma: no cover - optional dependency guard
    import torch
    from demucs.apply import apply_model
    from demucs.pretrained import get_model
    from demucs.separate import load_track
except ImportError:  # pragma: no cover - optional dependency not installed
    torch = None  # type: ignore[assignment]


@dataclass(slots=True)
class StemBundle:
    stems: Dict[str, Path]
    model_name: str


def separate_stems(
    audio_path: Optional[str],
    output_dir: Optional[str | Path],
    *,
    seed: int = DEFAULT_SEED,
) -> Optional[StemBundle]:
    """Return separated stems if the demucs stack is available."""

    if audio_path is None or torch is None:  # pragma: no cover - optional path
        return None

    out_dir = Path(output_dir) if output_dir is not None else Path.cwd() / "stems"
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        model = get_model(name="htdemucs")
        model.eval()
        torch.manual_seed(seed)
        if torch.cuda.is_available():  # pragma: no cover - optional GPU path
            torch.cuda.manual_seed_all(seed)
        waveform, sample_rate = load_track(audio_path)
        with torch.no_grad():
            stems = apply_model(model, waveform[None])[0]
        stem_names = ["drums", "bass", "other", "vocals"]
        stem_paths: Dict[str, Path] = {}
        try:
            import soundfile as sf  # type: ignore
        except ImportError:
            return None
        for idx, name in enumerate(stem_names):
            path = out_dir / f"{Path(audio_path).stem}_{name}.wav"
            sf.write(path, stems[idx].cpu().T, sample_rate)
            stem_paths[name] = path
        return StemBundle(stems=stem_paths, model_name="htdemucs")
    except Exception:  # pragma: no cover - demucs failure fallback
        return None
