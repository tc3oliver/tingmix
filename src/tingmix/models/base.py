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
