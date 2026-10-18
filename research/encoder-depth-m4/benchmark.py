"""Phase 3: synchronize each actual MPS early-exit forward; never benchmark feature-cache reads."""

from __future__ import annotations

import argparse
import json
import statistics
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from common import early, environment, executed_params, load_adapted, mel_input, require_mps, sync, write_json


def percentile(xs, q):
    a = sorted(xs)
    pos = (len(a) - 1) * q
    lo, hi = int(pos), min(int(pos) + 1, len(a) - 1)
    return a[lo] * (1 - (pos - lo)) + a[hi] * (pos - lo)


def measure(fn, n, warmup, gpu):
    for _ in range(warmup):
        fn()
        if gpu:
            sync()
    samples = []
    for _ in range(n):
        if gpu:
            sync()
        t = time.perf_counter()
        fn()
        if gpu:
            sync()
        samples.append((time.perf_counter() - t) * 1000)
    return {
        "p50_ms": percentile(samples, 0.50),
        "p95_ms": percentile(samples, 0.95),
        "min_ms": min(samples),
        "mean_ms": statistics.mean(samples),
        "n": len(samples),
        "warmup": warmup,
    }


def waveform(duration, input_path=None):
    if input_path:
        from embed2asr.audio import load_audio

        w = load_audio(input_path)
        if len(w) < duration * 16000:
            raise ValueError(f"Audio file must contain >= {duration} seconds of speech")
        return w[: duration * 16000].astype(np.float32), "speech"
    t = np.arange(duration * 16000, dtype=np.float32) / 16000
    x = 0.02 * np.random.default_rng(17).standard_normal(len(t)).astype(np.float32)
    return (x + 0.05 * np.sin(2 * np.pi * 220 * t)).astype(np.float32), "synthetic"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--audio", help="file with >=25s speech; otherwise synthetic fixed-length workload")
    p.add_argument("--parity", type=Path, default=Path("research/encoder-depth-m4/runs/parity.json"))
    p.add_argument("--revision", help="optional HF commit; must match parity source")
    p.add_argument("--iterations", type=int, default=50)
    p.add_argument("--warmup", type=int, default=6)
    p.add_argument("--out", type=Path, default=Path("research/encoder-depth-m4/runs/latency.json"))
    a = p.parse_args()
    res = {
        "phase": "latency",
        "status": "BLOCKED",
        "environment": environment(),
        "device": "mps",
        "precision": "fp32",
        "batch_size": 1,
        "feature_cache": False,
        "durations_s": [5, 15, 25],
        "measures": {},
    }
    try:
        require_mps()
        parity = json.loads(a.parity.read_text(encoding="utf-8"))
        if parity.get("status") != "PASS":
            raise RuntimeError("Phase 0 MPS parity must pass before benchmarking")
        if a.iterations < 50:
            raise ValueError("At least 50 iterations required")
        model, origin = load_adapted(a.revision or parity["origin"]["hf_revision"])
        if parity["origin"]["hf_revision"] != origin["hf_revision"] or parity["origin"]["delta_sha256"] != origin["delta_sha256"]:
            raise RuntimeError("Benchmark release differs from parity release")
        res["origin"] = origin
        tower, fe = model.asr.encoder.tower, model.asr.encoder.feature_extractor
        res["loaded_encoder_params"] = sum(p.numel() for p in tower.parameters())
        res["memory_note"] = "No structural pruning; all 12 layers remain resident. Peak unified memory not measured."
        res["mps_allocated_bytes_start"] = torch.mps.current_allocated_memory()
        for secs in res["durations_s"]:
            wav, kind = waveform(secs, a.audio)

            def inp():
                return fe([wav], sampling_rate=16000, return_tensors="pt")

            prep = measure(inp, a.iterations, a.warmup, gpu=False)
            mel, mask = mel_input(wav, fe)
            for depth in (6, 9, 12):

                def f():
                    return early(tower, mel, mask, depth)

                with torch.inference_mode():
                    r = measure(f, a.iterations, a.warmup, gpu=True)
                    y, lens = f()
                    sync()
                key = f"E{depth}/{secs}s"
                r.update(
                    rtf=r["p50_ms"] / (secs * 1000),
                    executed_parameters=executed_params(tower, depth),
                    output_shape=list(y.shape),
                    output_frames=int(lens.item()),
                    includes_projection=(depth == 12),
                    audio_kind=kind,
                    scope="mel_to_hidden_includes_subsample_and_active_layers",
                )
                res["measures"][key] = r
            e2e = measure(lambda: model.transcribe_batch([wav]), a.iterations, a.warmup, gpu=True)
            e2e["scope"] = "file_waveform_to_official_reader_greedy_decode"
            res["measures"][f"e2e_full/{secs}s"] = e2e
            prep["scope"] = "audio_waveform_to_feature_extractor_cpu"
            res["measures"][f"preprocess/{secs}s"] = prep
        res["mps_allocated_bytes_end"] = torch.mps.current_allocated_memory()
        res["status"] = "PASS"
    except Exception as exc:
        res["error"] = f"{type(exc).__name__}: {exc}"
        res["traceback"] = traceback.format_exc()
    write_json(a.out, res)
    print(f"{res['status']}: {a.out}")
    if res["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
