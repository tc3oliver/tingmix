"""TingMix-Small: chunked streaming must reproduce the offline pass exactly (random weights, CPU)."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tingmix.models.small.model import LogMel, SmallNet, Stream  # noqa: E402


def test_logmel_frames():
    assert LogMel.frames(511) == 0 and LogMel.frames(512) == 1 and LogMel.frames(672) == 2


@pytest.mark.parametrize("chunk", [160, 777, 4000])
def test_stream_equals_offline(chunk):
    torch.manual_seed(0)
    net = SmallNet(vocab_size=50, d=32, hidden=48, layers=2, dropout=0.0).eval()
    wav = np.random.default_rng(1).standard_normal(16000 * 3).astype(np.float32) * 0.1
    with torch.inference_mode():
        lp, lens = net(torch.from_numpy(wav)[None], torch.tensor([len(wav)]))
    offline = lp[0, : int(lens[0])]
    s = Stream(net, lambda ids: "")
    for i in range(0, len(wav), chunk):
        s.accept(wav[i : i + chunk])
    streamed = torch.cat(s.logp)
    assert streamed.shape == offline.shape
    assert torch.allclose(streamed, offline, atol=1e-4)  # same math; only float summation order differs


def test_session_native_path():
    from tingmix.streaming.session import StreamingSession

    class Fake:
        sample_rate, max_window_s, streaming_native = 16000, float("inf"), True

        def new_stream(self):
            class S:
                n = 0

                def accept(self, pcm):
                    self.n += len(pcm)
                    return "a" * (self.n // 8000)

                def finalize(self):
                    return "a" * (self.n // 8000)

            return S()

    sess = StreamingSession(Fake())
    rng = np.random.default_rng(0)
    speech = (rng.standard_normal(16000 * 2) * 0.3).astype(np.float32)
    ev = []
    for x in (np.zeros(8000, np.float32), speech, np.zeros(24000, np.float32)):
        for i in range(0, len(x), 1600):
            ev += sess.feed(x[i : i + 1600])
    ev += sess.flush()
    kinds = [e.type for e in ev]
    assert kinds[-1] == "final" and "partial" in kinds and ev[-1].text.startswith("a")
