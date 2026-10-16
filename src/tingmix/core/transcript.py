"""Revision manager: numbers the partials of each open segment and commits finals, which can never change afterwards."""

from __future__ import annotations

from .events import FinalEvent, PartialEvent


class Transcript:
    def __init__(self):
        self.finals: list[FinalEvent] = []
        self._open: dict[int, PartialEvent] = {}
        self._closed: set[int] = set()

    def current(self, segment_id: int) -> PartialEvent | None:
        """Latest partial of an open segment."""
        return self._open.get(segment_id)

    def partial(self, segment_id: int, text: str, start: float, end: float, stable_prefix_len: int = 0) -> PartialEvent:
        if segment_id in self._closed:
            raise ValueError(f"segment {segment_id} is final; it cannot be revised")
        prev = self._open.get(segment_id)
        ev = PartialEvent(segment_id, prev.revision + 1 if prev else 0, text, start, end, stable_prefix_len)
        self._open[segment_id] = ev
        return ev

    def final(self, segment_id: int, text: str, start: float, end: float) -> FinalEvent:
        if segment_id in self._closed:
            raise ValueError(f"segment {segment_id} is already final")
        prev = self._open.pop(segment_id, None)
        self._closed.add(segment_id)
        ev = FinalEvent(segment_id, prev.revision + 1 if prev else 0, text, start, end)
        self.finals.append(ev)
        return ev

    @property
    def text(self) -> str:
        parts = [f.text for f in self.finals if f.text]
        return " ".join(parts) if " " in "".join(parts) else "".join(parts)  # CJK-only output joins without spaces
