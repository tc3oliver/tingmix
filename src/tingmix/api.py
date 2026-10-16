"""Python entry points: `load_model()` and the offline `transcribe()` convenience."""

from __future__ import annotations

from .core.audio import load_audio
from .core.registry import load_model

__all__ = ["load_model", "transcribe"]


def transcribe(audio, model=None, sr: int | None = None) -> str:
    """Offline transcription of a file path or a float array (`sr` required). `model`: loaded model, model name, or None for Large."""
    if model is None or isinstance(model, str):
        model = load_model(model or "large")
    return model.transcribe_batch([load_audio(audio, sr)])[0]
