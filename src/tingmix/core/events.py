"""Streaming events and their JSON form. A partial may be revised by a later partial of the same segment; a final never changes."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import ClassVar


@dataclass(frozen=True)
class Event:
    segment_id: int
    revision: int  # 0, 1, ... per segment; the final carries the last revision + 1
    text: str
    start: float  # seconds from the start of the stream
    end: float
    type: ClassVar[str]

    def to_dict(self) -> dict:
        return {"type": self.type, **asdict(self)}


@dataclass(frozen=True)
class PartialEvent(Event):
    stable_prefix_len: int = 0  # text[:stable_prefix_len] agreed across the last partials and is unlikely to change
    type: ClassVar[str] = "partial"


@dataclass(frozen=True)
class FinalEvent(Event):
    type: ClassVar[str] = "final"


def event_from_dict(d: dict) -> Event:
    cls = {"partial": PartialEvent, "final": FinalEvent}[d["type"]]
    return cls(**{f.name: d[f.name] for f in fields(cls) if f.name in d})
