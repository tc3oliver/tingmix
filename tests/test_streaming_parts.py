import numpy as np
import pytest
from conftest import SR, silence, tone

from tingmix.core.events import FinalEvent, PartialEvent, event_from_dict
from tingmix.core.transcript import Transcript
from tingmix.streaming.buffer import RingBuffer
from tingmix.streaming.endpoint import Endpointer
from tingmix.streaming.scheduler import Scheduler
from tingmix.streaming.stability import StablePrefix
from tingmix.streaming.vad import EnergyVAD


def test_ring_buffer_absolute_indices():
    b = RingBuffer(10)
    b.append(np.arange(7, dtype=np.float32))
    b.append(np.arange(7, 15, dtype=np.float32))
    assert (b.start, b.end) == (5, 15)
    assert b.get(5, 15).tolist() == list(range(5, 15))
    assert b.get(12, 14).tolist() == [12, 13]
    with pytest.raises(IndexError):
        b.get(4, 6)
    b.append(np.arange(15, 40, dtype=np.float32))  # larger than capacity keeps the newest samples
    assert b.get(30, 40).tolist() == list(range(30, 40))


def test_vad_hysteresis():
    v = EnergyVAD(start_frames=2, hang_frames=3)
    loud, quiet = tone(0.02), silence(0.02)
    assert [v.update(f) for f in [quiet, loud, loud, quiet, quiet, quiet, quiet]] == [False, False, True, True, True, False, False]
    assert not EnergyVAD(start_frames=1).update(silence(0.02))


def test_endpointer_silence_and_max_length():
    e = Endpointer(silence_ms=100, max_segment_s=0.5)
    assert e.update(False, 0.02) is None and not e.active
    assert [e.update(True, 0.02) for _ in range(3)] == [None] * 3 and e.active
    assert [e.update(False, 0.02) for _ in range(5)] == [None] * 4 + ["silence"]
    assert not e.active and e.trailing_s == pytest.approx(0.1)
    reasons = [e.update(True, 0.1) for _ in range(5)]
    assert reasons == [None] * 4 + ["max_length"]


def test_stable_prefix():
    s = StablePrefix(k=2)
    assert s.update("hello wor") == 0
    assert s.update("hello world") == 9
    assert s.update("hallo world!") == 1
    s.reset()
    assert s.update("x") == 0


def test_scheduler_hop_and_backoff():
    s = Scheduler(SR, hop_ms=500, max_load=0.5)
    assert not s.due(SR // 4) and s.due(SR // 2)
    s.mark(SR // 2, cost_s=0.6)  # slow decode: hop stretches to 1.2 s
    assert not s.due(SR // 2 + SR) and s.due(SR // 2 + int(1.2 * SR))


def test_transcript_finals_are_immutable():
    t = Transcript()
    p0 = t.partial(0, "he", 0, 1)
    p1 = t.partial(0, "hello", 0, 2, stable_prefix_len=2)
    f = t.final(0, "hello", 0, 2)
    assert (p0.revision, p1.revision, f.revision) == (0, 1, 2)
    with pytest.raises(ValueError):
        t.partial(0, "changed", 0, 3)
    with pytest.raises(ValueError):
        t.final(0, "changed", 0, 3)
    with pytest.raises(AttributeError):
        f.text = "x"
    assert t.finals == [f] and t.text == "hello"


def test_event_json_round_trip():
    for ev in (PartialEvent(1, 2, "你好", 0.0, 1.5, 1), FinalEvent(1, 3, "你好", 0.0, 1.5)):
        assert event_from_dict(ev.to_dict()) == ev
    assert PartialEvent(0, 0, "x", 0, 1).to_dict()["type"] == "partial"
