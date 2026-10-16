"""Stable-prefix detection by local agreement: the prefix shared by the last `k` successive partial hypotheses."""

from __future__ import annotations

from collections import deque


class StablePrefix:
    def __init__(self, k: int = 2):
        self._hyps: deque[str] = deque(maxlen=k)

    def reset(self) -> None:
        self._hyps.clear()

    def update(self, text: str) -> int:
        """Add a hypothesis; returns the length of the prefix of `text` that all of the last k hypotheses share (0 until k seen)."""
        self._hyps.append(text)
        if len(self._hyps) < self._hyps.maxlen:
            return 0
        n = 0
        for chars in zip(*self._hyps):
            if len(set(chars)) > 1:
                break
            n += 1
        return n
