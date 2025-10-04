"""Audio input/output helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np

try:  # pragma: no cover - optional dependency guard
    import soundfile as sf
except ImportError:  # pragma: no cover - ``audioread`` fallback handles decoding
    sf = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency guard
    import audioread
except ImportError as exc:  # pragma: no cover - decoding requires audioread
    raise RuntimeError("audioread is required for audio decoding") from exc

try:  # pragma: no cover - optional dependency guard
    import resampy
except ImportError:  # pragma: no cover - resampling will fall back to librosa
    resampy = None  # type: ignore[assignment]

import librosa


def _buf_to_stereo(buffer: np.ndarray, channels: int) -> np.ndarray:
    if channels <= 0:
        raise RuntimeError("Invalid channel count from decoder")
    if buffer.size % channels:
        raise RuntimeError("Decoded frame buffer is not divisible by channel count")
    if channels > 1:
        return buffer.reshape((-1, channels)).T
    return buffer.reshape((1, -1))


def _resample(samples: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return samples
    if samples.ndim == 1:
        if resampy is not None:
            return resampy.resample(samples, orig_sr, target_sr)
        return librosa.resample(samples, orig_sr=orig_sr, target_sr=target_sr)
    channels = []
    for channel in np.atleast_2d(samples):
        if resampy is not None:
            channels.append(resampy.resample(channel, orig_sr, target_sr))
        else:
            channels.append(
                librosa.resample(channel, orig_sr=orig_sr, target_sr=target_sr)
            )
    return np.asarray(channels)


def load_audio(
    path: str | Path,
    target_sr: Optional[int] = None,
    mono: bool = True,
) -> Tuple[np.ndarray, int, dict[str, object]]:
    """Load ``path`` into memory and return samples, sample rate, and metadata.

    The loader prefers :mod:`soundfile` for decode fidelity and falls back to
    :mod:`audioread` when the backend does not support the provided format.
    """

    file_path = str(path)
    data: Optional[np.ndarray] = None
    sr: Optional[int] = None
    meta: dict[str, object] = {}

    if sf is not None:
        try:
            with sf.SoundFile(file_path) as handle:
                sr = int(handle.samplerate)
                channels = int(handle.channels)
                frames = int(len(handle))
                raw = handle.read(always_2d=True, dtype="float32")
                data = np.asarray(raw.T)
                meta = {
                    "channels": channels,
                    "duration": frames / float(sr) if sr else 0.0,
                    "file_type": handle.format,
                    "subtype": handle.subtype,
                }
        except RuntimeError:
            data = None
            sr = None
            meta = {}

    if data is None or sr is None:
        try:
            with audioread.audio_open(file_path) as handle:
                sr = int(handle.samplerate)
                channels = int(handle.channels)
                duration = float(handle.duration) if handle.duration else None
                buffers = []
                for chunk in handle:
                    frame = librosa.util.buf_to_float(chunk, dtype=np.float32)
                    if not len(frame):
                        continue
                    buffers.append(frame)
                if buffers:
                    stacked = np.concatenate(buffers)
                    data = _buf_to_stereo(stacked, channels)
                else:
                    data = np.zeros((channels, 0), dtype=np.float32)
                meta = {
                    "channels": channels,
                    "duration": duration
                    if duration is not None
                    else data.shape[-1] / float(sr),
                    "file_type": Path(file_path).suffix.lstrip(".").upper() or "UNKNOWN",
                }
        except audioread.NoBackendError as exc:
            raise RuntimeError(f"Could not decode audio file: {file_path}") from exc

    if data is None or sr is None:
        raise RuntimeError(f"Failed to load audio file: {file_path}")

    if data.ndim == 1:
        data = data[np.newaxis, :]

    original_channels = int(data.shape[0])

    if target_sr is not None and sr != target_sr:
        data = _resample(data, sr, target_sr)
        sr = target_sr

    if mono and data.shape[0] > 1:
        data = np.mean(data, axis=0, keepdims=True)

    meta["channels"] = original_channels
    meta["duration"] = data.shape[-1] / float(sr)
    meta["file_type"] = meta.get("file_type") or Path(file_path).suffix.lstrip(".").upper() or "UNKNOWN"

    if mono:
        return data.squeeze(axis=0), sr, meta
    return data, sr, meta
