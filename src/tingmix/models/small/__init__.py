"""TingMix-Small (CPU): placeholder. Intended to be a natively streaming model with recurrent/cached state, so audio is
consumed once instead of being re-decoded. Not implemented; it only fixes the interface."""

from __future__ import annotations

import numpy as np


class SmallModel:
    name = "tingmix-small"
    sample_rate = 16000
    max_window_s = float("inf")
    streaming_native = True

    def __init__(self, *args, **kwargs):
        raise NotImplementedError("TingMix-Small is not implemented yet; use model='large'")

    def transcribe_batch(self, wavs: list[np.ndarray]) -> list[str]:
        raise NotImplementedError
