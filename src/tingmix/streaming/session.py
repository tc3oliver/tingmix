"""Streaming session: audio chunks in, partial/final events out.

For models with `streaming_native == False` (TingMix-Large) this is BUFFERED streaming, not native streaming: while a
segment is open, the offline model is re-run on all audio of that segment every `hop_ms` (window growing up to the
model's limit), so compute per update grows with segment length. At an endpoint (silence or max length) the segment is
decoded one last time, emitted as a final, and its audio is dropped. Partials may change between updates; finals never do.
A forced cut at max length can split a word."""

from __future__ import annotations

import time

import numpy as np

from ..core.events import Event
from ..core.transcript import Transcript
from .buffer import RingBuffer
from .endpoint import Endpointer
from .scheduler import Scheduler
from .stability import StablePrefix
from .vad import EnergyVAD


class StreamingSession:
    def __init__(
        self,
        model,
        hop_ms: float = 480,
        silence_ms: float = 600,
        max_segment_s: float = 25.0,
        preroll_ms: float = 800,
        tail_pad_ms: float = 0,
        stability_k: int = 2,
        vad=None,
    ):
        self.model, self.sr = model, model.sample_rate
        self.vad = vad or EnergyVAD(threshold_db=-50.0)
        self.preroll, self.tail_pad = int(preroll_ms * self.sr / 1000), int(tail_pad_ms * self.sr / 1000)
        # the open segment (plus preroll) must fit the model window
        self.endpointer = Endpointer(silence_ms, min(max_segment_s, model.max_window_s - preroll_ms / 1000 - 0.1))
        self.scheduler = Scheduler(self.sr, hop_ms)
        self.stability = StablePrefix(stability_k)
        self.transcript = Transcript()
        self._frame = int(self.vad.frame_ms * self.sr / 1000)
        self._buf = RingBuffer(int((min(model.max_window_s, max_segment_s) + 30) * self.sr))
        self._rem = np.zeros(0, dtype=np.float32)  # samples of an incomplete VAD frame
        self._pos = 0  # samples processed (whole frames)
        self._prev_end = 0  # end of the last closed segment
        self._seg_start = 0
        self._seg_id = -1
        self._last: tuple[int, int, str] | None = None  # (start, end, text) of the latest decode

    def feed(self, pcm: np.ndarray) -> list[Event]:
        """Add float32 mono samples at the model's sample rate; returns the events they trigger."""
        pcm = np.asarray(pcm, dtype=np.float32)
        self._buf.append(pcm)
        x = np.concatenate([self._rem, pcm])
        events: list[Event] = []
        for i in range(len(x) // self._frame):
            speech = self.vad.update(x[i * self._frame : (i + 1) * self._frame])
            self._pos += self._frame
            was_active = self.endpointer.active
            reason = self.endpointer.update(speech, self._frame / self.sr)
            if not was_active and (self.endpointer.active or reason):
                self._open(self._pos - self._frame)
            if reason:
                events += self._close(reason)
        self._rem = x[len(x) // self._frame * self._frame :]
        if self.endpointer.active and self.scheduler.due(self._pos - self._seg_start):
            events += self._partial()
        return events

    def flush(self) -> list[Event]:
        """End of stream: finalize the open segment, if any."""
        if not self.endpointer.active:
            return []
        self.endpointer.active = False
        return self._close("flush")

    def _open(self, frame_start: int) -> None:
        self._seg_start = max(frame_start - self.preroll, self._prev_end, self._buf.start)
        self._seg_id += 1
        self.scheduler.reset()
        self.stability.reset()
        if getattr(self.model, "streaming_native", False):  # stateful model: one stream per segment, audio fed once
            self._stream, self._fed = self.model.new_stream(), self._seg_start

    def _decode(self, a: int, b: int) -> str:
        t0 = time.perf_counter()
        if getattr(self.model, "streaming_native", False):
            if b > self._fed:  # only the new audio is processed; the model keeps its state
                text, self._fed = self._stream.accept(self._buf.get(self._fed, b)).strip(), b
            else:
                text = self._stream.finalize().strip()
        else:
            text = self.model.transcribe_batch([self._buf.get(a, b)])[0].strip() if b - a >= self.sr // 10 else ""
        self.scheduler.mark(b - a, time.perf_counter() - t0)
        self._last = (a, b, text)
        return text

    def _partial(self) -> list[Event]:
        a, b = self._seg_start, self._pos
        text = self._decode(a, b)
        stable = self.stability.update(text)
        prev = self.transcript.current(self._seg_id)
        if not text or (prev and (prev.text, prev.stable_prefix_len) == (text, stable)):
            return []
        return [self.transcript.partial(self._seg_id, text, a / self.sr, b / self.sr, stable)]

    def _close(self, reason: str) -> list[Event]:
        a, b = self._seg_start, self._pos
        if reason != "max_length":
            b = max(a, b - max(0, int(self.endpointer.trailing_s * self.sr) - self.tail_pad))  # keep a little trailing silence
        self._prev_end = b
        if self._last and self._last[:2] == (a, b):
            text = self._last[2]
        else:
            text = self._decode(a, b)
        self._last = None
        if not text and not self.transcript.current(self._seg_id):
            return []  # false trigger, nothing was ever shown
        return [self.transcript.final(self._seg_id, text, a / self.sr, b / self.sr)]
