"""Encoder parity on the real EmbeddingGemma 2 weights (slow: needs the model download; GPU optional)."""

import os
import sys

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def setup():
    from tingmix.models.large import encoder as E

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tower = E.load_tower().to(dev)
    fe = E.load_feature_extractor()
    rng = np.random.default_rng(0)
    wavs = [rng.standard_normal(16000 * s).astype(np.float32) * 0.1 for s in (3, 7, 5)]
    inp = fe(wavs, sampling_rate=16000, return_tensors="pt")
    mel, mask = inp["input_features"].to(dev, torch.bfloat16), inp["input_features_mask"].to(dev)
    with torch.no_grad():
        ref = tower(mel, mask, return_dict=True)
    return E, tower, mel, mask, ref


def rel(a, b):
    return ((a - b).norm() / b.norm()).item()


def test_frozen_matches_forward(setup):
    E, tower, mel, mask, ref = setup
    x, lens = E.encode(tower, mel, mask, "frozen")
    assert torch.equal(lens.cpu(), ref.attention_mask.sum(1).cpu())
    v = ref.attention_mask
    assert rel(x[v], ref.last_hidden_state.float()[v]) < 0.03  # ROCm forward is not bit-deterministic (~1% run-to-run on noise input)


def test_split_at_10_matches_eg2_cache(setup):
    E, tower, mel, mask, ref = setup
    sys.path.insert(0, os.environ.get("EG2_CACHE_DIR", ""))  # dir holding the eg2_cache.py research helper; skip if absent
    eg2 = pytest.importorskip("eg2_cache")
    with torch.no_grad(), torch.autocast(mel.device.type, torch.bfloat16, enabled=mel.device.type == "cuda"):
        h, lens = E.subsample(tower, mel, mask)
        h10 = E.run_layers(tower, h, lens, 0, 10, proj=False)
        a = E.run_layers(tower, h10, lens, 10)
        b = eg2.top_layers(tower, h10, lens)
    v = ref.attention_mask
    assert rel(a.float()[v], b.float()[v]) < 1e-3
    assert rel(a.float()[v], ref.last_hidden_state.float()[v]) < 0.03  # bf16 rounding on noise input


def test_lora_and_partial_exact_at_init(setup):
    E, tower, mel, mask, ref = setup
    import copy

    v = ref.attention_mask
    x0, _ = E.encode(tower, mel, mask, "frozen")
    t = copy.deepcopy(tower)
    p = E.apply_lora(t, r=8, alpha=16)
    assert len(p) == 12 * 8 * 2
    x1, _ = E.encode(t, mel, mask, "lora")
    assert x1.requires_grad
    x1.sum().backward()

    assert p[1].grad.abs().sum() > 0 and p[-1].grad.abs().sum() > 0  # first and last layer B receive gradient
    assert rel(x1.detach()[v], x0[v]) < 0.03  # B=0: equal up to run-to-run noise
    t2 = copy.deepcopy(tower)
    E.set_partial(t2, 10)
    x2, _ = E.encode(t2, mel, mask, "partial")
    assert rel(x2.detach()[v], x0[v]) < 0.03
