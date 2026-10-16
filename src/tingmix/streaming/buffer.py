"""Ring buffer of samples addressed by absolute sample index (index 0 = first sample ever appended)."""

from __future__ import annotations

import numpy as np


class RingBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self._b = np.zeros(capacity, dtype=np.float32)
        self.end = 0  # absolute index one past the newest sample

    @property
    def start(self) -> int:
        """Absolute index of the oldest sample still held."""
        return max(0, self.end - self.capacity)

    def append(self, x: np.ndarray) -> None:
        n = len(x)
        x = x[-self.capacity :]
        self._b[np.arange(self.end + n - len(x), self.end + n) % self.capacity] = x
        self.end += n

    def get(self, a: int, b: int) -> np.ndarray:
        if a < self.start or b > self.end or a > b:
            raise IndexError(f"[{a}, {b}) outside buffered range [{self.start}, {self.end})")
        return self._b[np.arange(a, b) % self.capacity]
