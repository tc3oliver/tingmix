"""Audio helpers: file -> float32 mono 16 kHz, and PCM int16 <-> float32."""

from __future__ import annotations

import os

import numpy as np

SAMPLE_RATE = 16000


def load_audio(src: str | os.PathLike | np.ndarray, sr: int | None = None) -> np.ndarray:
    """Float32 mono 16 kHz samples from a file path or an array (`sr` required for arrays). Needs the `large` extra (embed2asr)."""
    try:
        from embed2asr.audio import load_audio as _load
    except ImportError as e:
        raise ImportError("loading audio needs embed2asr (pip install 'tingmix[large]')") from e
    return _load(src, sr)


def pcm16_to_float(data: bytes) -> np.ndarray:
    return np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0


def float_to_pcm16(x: np.ndarray) -> bytes:
    return (np.clip(x, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
