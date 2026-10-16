from conftest import FakeModel, chunks, silence, tone

from tingmix.streaming.session import StreamingSession


def run(session, audio):
    events = []
    for c in chunks(audio):
        events += session.feed(c)
    return events + session.flush()


def test_partials_then_final_per_segment(model):
    events = run(StreamingSession(model), __import__("numpy").concatenate([tone(2), silence(1.5), tone(1), silence(1.5)]))
    by_seg = {}
    for e in events:
        by_seg.setdefault(e.segment_id, []).append(e)
    assert sorted(by_seg) == [0, 1]
    for evs in by_seg.values():
        assert [e.type for e in evs[:-1]] == ["partial"] * (len(evs) - 1) and evs[-1].type == "final"
        assert [e.revision for e in evs] == list(range(len(evs)))
        assert len(evs) > 2  # several partials before the final
    f0 = by_seg[0][-1]
    assert 1.8 < f0.end - f0.start < 2.8 and f0.text.startswith("a")
    # events of a segment never appear after its final
    seen_final = set()
    for e in events:
        assert e.segment_id not in seen_final
        if e.type == "final":
            seen_final.add(e.segment_id)


def test_flush_finalizes_open_segment_and_silence_makes_nothing(model):
    s = StreamingSession(model)
    assert run(s, silence(2)) == [] and model.calls == []
    events = run(s, tone(1))
    assert events[-1].type == "final"


def test_max_segment_forces_cut_below_model_window(model):
    import numpy as np

    s = StreamingSession(model, max_segment_s=3.0)
    events = run(s, tone(10))
    finals = [e for e in events if e.type == "final"]
    assert len(finals) >= 3
    assert max(model.calls) <= 3.5 * 16000 and max(model.calls) < model.max_window_s * 16000
    assert all(a.end <= b.start + 1e-6 for a, b in zip(finals, finals[1:]))  # no overlap between segments
    assert np.isclose(finals[-1].end, 10, atol=0.05)


def test_stable_prefix_reported(model):
    events = run(StreamingSession(model, stability_k=2), tone(3))
    partials = [e for e in events if e.type == "partial"]
    assert partials[0].stable_prefix_len == 0 and partials[-1].stable_prefix_len > 0
    assert FakeModel.streaming_native is False
