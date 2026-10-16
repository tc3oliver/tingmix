"""Frame-based voice activity detection. A VAD has `frame_ms` and `update(frame) -> bool` (speech state after smoothing),
so a neural VAD (e.g. Silero) can replace the energy detector without touching the session."""

from __future__ import annotations

import numpy as np


class EnergyVAD:
    """RMS threshold with hysteresis: speech starts after `start_frames` loud frames in a row and ends after `hang_frames` quiet ones.
    ponytail: fixed threshold, no noise-floor tracking; use a neural VAD or add an adaptive floor for noisy rooms."""

    def __init__(self, threshold_db: float = -50.0, frame_ms: int = 20, start_frames: int = 2, hang_frames: int = 10):
        self.threshold_db, self.frame_ms = threshold_db, frame_ms
        self.start_frames, self.hang_frames = start_frames, hang_frames
        self.speech = False
        self._run = 0  # consecutive frames that disagree with the current state

    def update(self, frame: np.ndarray) -> bool:
        loud = 20 * np.log10(float(np.sqrt(np.mean(np.square(frame, dtype=np.float64)))) + 1e-10) > self.threshold_db
        self._run = self._run + 1 if loud != self.speech else 0
        if self._run >= (self.start_frames if loud else self.hang_frames):
            self.speech, self._run = loud, 0
        return self.speech
