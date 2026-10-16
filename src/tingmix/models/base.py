"""Model interface shared by TingMix-Large and TingMix-Small."""

from __future__ import annotations

from typing import Protocol

import numpy as np


class ASRModel(Protocol):
    name: str
    sample_rate: int
    max_window_s: float  # longest audio one transcribe_batch item should cover
    streaming_native: bool  # False: offline model, streaming is buffered re-decoding (Large); True: stateful streaming (Small)

    def transcribe_batch(self, wavs: list[np.ndarray]) -> list[str]:
        """float32 mono `sample_rate` waveforms -> texts."""
        ...


class Stream(Protocol):
    """One utterance of a natively streaming model: audio is consumed once and the model keeps its own state."""

    def accept(self, pcm: np.ndarray) -> str:
        """Feed float32 samples; returns the current (revisable) hypothesis of this utterance."""
        ...

    def finalize(self) -> str:
        """No more audio: the final text of this utterance (never revised afterwards)."""
        ...


class StreamingASRModel(ASRModel, Protocol):
    """Models with streaming_native = True (TingMix-Small) additionally open per-utterance streams; StreamingSession can
    then feed audio incrementally instead of re-decoding the whole open segment."""

    def new_stream(self) -> Stream: ...
