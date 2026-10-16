"""Atomic checkpoints with one previous generation, plus the run lock and status file used by the job manager.

<run>/resume.pt       latest complete checkpoint (written to resume.pt.tmp, fsync, os.replace)
<run>/resume.prev.pt  the one before it (a crash while writing never loses both)
<run>/status.json     {"state": RUNNING|DONE|FAILED|PAUSED, "reason", "permanent", "time"}"""

from __future__ import annotations

import fcntl
import json
import os
import time
from pathlib import Path

import torch


def save_atomic(obj, path: Path):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        torch.save(obj, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    dfd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(dfd)
    finally:
        os.close(dfd)


def save_resume(obj, run: Path):
    cur, prev = Path(run) / "resume.pt", Path(run) / "resume.prev.pt"
    if cur.exists():
        os.replace(cur, prev)
    save_atomic(obj, cur)


def load_resume(run: Path):
    """Newest loadable checkpoint (resume.pt, else resume.prev.pt) or None. A corrupt file is reported, not deleted."""
    for name in ("resume.pt", "resume.prev.pt"):
        p = Path(run) / name
        if not p.exists():
            continue
        try:
            return torch.load(p, map_location="cpu", weights_only=False), p
        except Exception as e:  # truncated / corrupt
            print(f"[resume] {p} unreadable ({type(e).__name__}: {e}); trying the previous one", flush=True)
    return None, None


def set_status(run: Path, state: str, reason: str = "", permanent: bool = False, **extra):
    p = Path(run) / "status.json"
    tmp = p.with_name("status.json.tmp")
    tmp.write_text(json.dumps({"state": state, "reason": reason, "permanent": permanent, "time": time.strftime("%F %T"), **extra}))
    os.replace(tmp, p)


def lock_run(run: Path):
    """Exclusive lock for the run dir; returns the open file (keep it alive) or None if another trainer holds it."""
    f = open(Path(run) / ".lock", "w")
    try:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        f.close()
        return None
    f.write(str(os.getpid()))
    f.flush()
    return f
