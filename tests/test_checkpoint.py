"""Checkpoint atomicity, fallback to the previous generation, and the run lock (CPU, no model)."""

import json

import torch

from training.large import checkpoint as C


def test_save_keeps_previous_and_falls_back(tmp_path):
    C.save_resume({"step": 1}, tmp_path)
    C.save_resume({"step": 2}, tmp_path)
    assert torch.load(tmp_path / "resume.prev.pt")["step"] == 1
    (tmp_path / "resume.pt").write_bytes(b"truncated")  # a crash mid-write can only ever leave *.tmp, but be defensive
    obj, p = C.load_resume(tmp_path)
    assert obj["step"] == 1 and p.name == "resume.prev.pt"


def test_interrupted_write_leaves_last_valid(tmp_path, monkeypatch):
    C.save_resume({"step": 5}, tmp_path)

    def boom(obj, f):
        f.write(b"partial")
        raise KeyboardInterrupt

    monkeypatch.setattr(torch, "save", boom)
    try:
        C.save_atomic({"step": 6}, tmp_path / "resume.pt")
    except KeyboardInterrupt:
        pass
    monkeypatch.undo()
    obj, p = C.load_resume(tmp_path)
    assert obj["step"] == 5 and p.name == "resume.pt"


def test_lock_is_exclusive(tmp_path):
    a = C.lock_run(tmp_path)
    assert a is not None and C.lock_run(tmp_path) is None
    a.close()
    assert C.lock_run(tmp_path) is not None


def test_status(tmp_path):
    C.set_status(tmp_path, "FAILED", "oom", permanent=True)
    s = json.loads((tmp_path / "status.json").read_text())
    assert s["state"] == "FAILED" and s["permanent"]
