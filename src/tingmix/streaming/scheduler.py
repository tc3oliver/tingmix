"""Decides when to re-decode the open segment: every `hop_ms` of new audio, stretched when decoding is slow so that
compute stays below `max_load` x real time (audio that arrived meanwhile is simply covered by the next decode)."""

from __future__ import annotations


class Scheduler:
    def __init__(self, sample_rate: int, hop_ms: float = 480, max_load: float = 0.5):
        self.sample_rate, self.hop_s, self.max_load = sample_rate, hop_ms / 1000, max_load
        self.reset()

    def reset(self) -> None:
        self._last = 0  # segment length (samples) at the last decode
        self._cost_s = 0.0

    def due(self, segment_samples: int) -> bool:
        hop = max(self.hop_s, self._cost_s / self.max_load)
        return segment_samples - self._last >= hop * self.sample_rate

    def mark(self, segment_samples: int, cost_s: float) -> None:
        self._last, self._cost_s = segment_samples, cost_s
