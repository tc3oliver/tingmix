"""Frame-based voice activity detection. A VAD has `frame_ms` and `update(frame) -> bool` (speech state after smoothing),
so a neural VAD (e.g. Silero) can replace the energy detector without touching the session."""

from __future__ import annotations

import numpy as np


class EnergyVAD:
    """RMS threshold with hysteresis: speech starts after `start_frames` loud frames in a row and ends after `hang_frames` quiet ones.
    Loud = above `threshold_db` AND `margin_db` above a tracked noise floor (falls at once to quieter frames, rises at most
    `rise_db` per frame, 10x faster outside speech), so steady room noise (fan, hum, hiss) does not open segments. margin_db=0 -> fixed threshold only.
    ponytail: energy only; a neural VAD (e.g. Silero) handles non-stationary noise (music, keyboard) better."""

    def __init__(
        self,
        threshold_db: float = -50.0,
        frame_ms: int = 20,
        start_frames: int = 2,
        hang_frames: int = 10,
        margin_db: float = 10.0,
        rise_db: float = 0.05,
    ):
        self.threshold_db, self.frame_ms = threshold_db, frame_ms
        self.start_frames, self.hang_frames = start_frames, hang_frames
        self.margin_db, self.rise_db = margin_db, rise_db
        self.floor = None
        self.speech = False
        self._run = 0  # consecutive frames that disagree with the current state

    def update(self, frame: np.ndarray) -> bool:
        db = 20 * np.log10(float(np.sqrt(np.mean(np.square(frame, dtype=np.float64)))) + 1e-10)
        if self.floor is None:
            self.floor = self.threshold_db - self.margin_db  # start as the fixed threshold
        rise = self.rise_db * (1 if self.speech else 10)
        self.floor = db if db < self.floor else self.floor + min(db - self.floor, rise)
        loud = db > self.threshold_db and (self.margin_db <= 0 or db > self.floor + self.margin_db)
        self._run = self._run + 1 if loud != self.speech else 0
        if self._run >= (self.start_frames if loud else self.hang_frames):
            self.speech, self._run = loud, 0
        return self.speech
