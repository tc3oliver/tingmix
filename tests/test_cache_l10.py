"""Shard atomicity / resume logic of the layer-10 cache with a fake encoder (CPU, no model)."""

import json

import numpy as np
import pytest
import torch

from training.large import cache_l10 as C


def fake_load(path):
    return np.zeros(int(float(path) * 16000), dtype=np.float32)  # "audio" = its duration in seconds


def fake_encode(wavs):
    return [torch.full((len(w) // 640, C.DIM), len(w) / 16000, dtype=torch.bfloat16) for w in wavs]


@pytest.fixture
def manifest(tmp_path):
    secs = [1.0 + (i % 7) for i in range(25)] + [0.1]  # last row is too short -> skipped
    p = tmp_path / "m.jsonl"
    p.write_text("".join(json.dumps({"audio": str(s), "text": f"t{i}", "lang": "en", "seconds": s}) + "\n" for i, s in enumerate(secs)))
    return p


def go(manifest, out, **kw):
    return C.run(manifest, out, fake_encode, fake_load, {"encoder": "fake"}, shard_rows=10, batch_size=4, threads=2, **kw)


def test_roundtrip(manifest, tmp_path):
    out = tmp_path / "c"
    st = go(manifest, out)
    assert st["built_shards"] == 3 and st["skipped_rows"] == 1
    rows = C.load_l10_index(out)
    assert [r["id"] for r in rows] == list(range(25))
    for r in rows:
        x = C.read_l10(r)
        assert x.shape == (r["frames"], C.DIM) and x.dtype == torch.bfloat16 and float(x[0, 0]) == r["seconds"]
    assert not list(out.glob("*.tmp"))


def test_resume_skips_complete_redoes_corrupt_and_partial(manifest, tmp_path):
    out = tmp_path / "c"
    go(manifest, out)
    ref = {k: (C.shard_base(out, k).with_suffix(".bin")).read_bytes() for k in range(3)}
    with open(C.shard_base(out, 1).with_suffix(".bin"), "r+b") as f:  # corrupt shard 1
        f.write(b"\xff\xff")
    C.shard_base(out, 2).with_suffix(".sha256").unlink()  # partial shard 2 (no commit marker)
    assert not C.shard_ok(C.shard_base(out, 1)) and not C.shard_ok(C.shard_base(out, 2))
    st = go(manifest, out)
    assert (st["skipped_shards"], st["built_shards"]) == (1, 2)
    assert all(C.shard_base(out, k).with_suffix(".bin").read_bytes() == ref[k] for k in range(3))


def test_workers_partition_shards(manifest, tmp_path):
    out = tmp_path / "c"
    a, b = go(manifest, out, num_workers=2, worker_id=0), go(manifest, out, num_workers=2, worker_id=1)
    assert (a["built_shards"], b["built_shards"]) == (2, 1)
    assert len(C.load_l10_index(out)) == 25


def test_meta_mismatch_refused(manifest, tmp_path):
    out = tmp_path / "c"
    go(manifest, out, max_shards=1)
    with pytest.raises(SystemExit):
        C.run(manifest, out, fake_encode, fake_load, {"encoder": "other"}, shard_rows=10, batch_size=4)
    with pytest.raises(SystemExit):
        C.run(manifest, out, fake_encode, fake_load, {"encoder": "fake"}, shard_rows=10, batch_size=8)


def test_merge_index(manifest, tmp_path):
    out = tmp_path / "c"
    go(manifest, out)
    C.shard_base(out, 1).with_suffix(".bin").write_bytes(b"x")  # corrupt -> excluded from merged index
    assert C.merge_index(out) == 15
    rows = [json.loads(line) for line in (out / "index.jsonl").open()]
    assert all(
        r["bin"].startswith("/") and r["dim"] == C.DIM and {"off", "frames", "audio", "text", "lang", "seconds"} <= r.keys() for r in rows
    )
