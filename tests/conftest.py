import numpy as np
import pytest

SR = 16000


class FakeModel:
    """Text = one 'a' per 0.1 s of audio, so partials grow and agree on their prefix."""

    name, sample_rate, max_window_s, streaming_native = "fake", SR, 29.5, False

    def __init__(self):
        self.calls = []

    def transcribe_batch(self, wavs):
        self.calls += [len(w) for w in wavs]
        return ["a" * (len(w) // 1600) for w in wavs]


def tone(seconds: float) -> np.ndarray:
    t = np.arange(int(seconds * SR)) / SR
    return (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def silence(seconds: float) -> np.ndarray:
    return np.zeros(int(seconds * SR), dtype=np.float32)


def chunks(x, ms=100):
    n = SR * ms // 1000
    return [x[i : i + n] for i in range(0, len(x), n)]


@pytest.fixture
def model():
    return FakeModel()
