"""Endpointing: a segment opens on the first speech frame and closes after `silence_ms` of non-speech or at `max_segment_s`."""

from __future__ import annotations


class Endpointer:
    def __init__(self, silence_ms: float = 600, max_segment_s: float = 25.0):
        self.silence_s, self.max_segment_s = silence_ms / 1000, max_segment_s
        self.active = False
        self.length_s = 0.0
        self.trailing_s = 0.0  # non-speech at the end of the segment; kept after it closes

    def update(self, speech: bool, dur_s: float) -> str | None:
        """Feed one frame. Returns "silence" or "max_length" when the segment ended with this frame, else None."""
        if not self.active:
            if not speech:
                return None
            self.active, self.length_s, self.trailing_s = True, 0.0, 0.0
        self.length_s += dur_s
        self.trailing_s = 0.0 if speech else self.trailing_s + dur_s
        reason = "silence" if self.trailing_s >= self.silence_s else "max_length" if self.length_s >= self.max_segment_s else None
        if reason:
            self.active = False
        return reason
