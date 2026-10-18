"""Phase 2: synchronized MPS latency of real early exits. Every timed call starts from the input mel; no feature cache is read.

Conditions are interleaved round-robin inside each iteration (rotating start) so thermal / clock drift hits all of them equally."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from common import E, early, environment, executed_params, load_adapted, mel_input, require_mps, sync, write_json


def percentile(xs, q):
    a = sorted(xs)
    pos = (len(a) - 1) * q
    lo, hi = int(pos), min(int(pos) + 1, len(a) - 1)
    return a[lo] * (1 - (pos - lo)) + a[hi] * (pos - lo)


def summarize(samples, warmup):
    return {
        "p50_ms": percentile(samples, 0.50),
        "p95_ms": percentile(samples, 0.95),
        "min_ms": min(samples),
        "mean_ms": statistics.mean(samples),
        "std_ms": statistics.pstdev(samples),
        "n": len(samples),
        "warmup": warmup,
    }


def measure_interleaved(fns, n, warmup, gpu):
    """fns: {name: callable}. Returns {name: [ms,...]} with n samples each; the order rotates every iteration."""
    names = list(fns)
    for _ in range(warmup):
        for k in names:
            fns[k]()
            if gpu.get(k):
                sync()
    out = {k: [] for k in names}
    for i in range(n):
        for k in names[i % len(names) :] + names[: i % len(names)]:
            if gpu.get(k):
                sync()
            t = time.perf_counter()
            fns[k]()
            if gpu.get(k):
                sync()
            out[k].append((time.perf_counter() - t) * 1000)
    return out


def load_speech(paths, seconds):
    """Real speech: concatenate the given files (16 kHz mono) and cut to exactly `seconds`."""
    from embed2asr.audio import load_audio

    need, parts = seconds * 16000, []
    while sum(map(len, parts)) < need:
        for p in paths:
            parts.append(load_audio(p))
            if sum(map(len, parts)) >= need:
                break
    return np.concatenate(parts)[:need].astype(np.float32)


def thermal():
    try:
        return subprocess.check_output(["pmset", "-g", "therm"], text=True, timeout=5).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--audio", nargs="+", required=True, help="real speech files (16 kHz-convertible); concatenated and cut to 5/15/25 s")
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
        "audio_files": [str(Path(x).name) for x in a.audio],
        "thermal_before": thermal(),
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
        res["pruned"] = False
        res["memory_note"] = (
            "Early exit skips compute only: all 12 layers + LoRA stay resident (loaded_encoder_params). "
            "Allocator numbers are torch.mps current/driver allocated after the run, not a true peak."
        )
        res["mps_allocated_bytes_start"] = torch.mps.current_allocated_memory()
        for secs in res["durations_s"]:
            wav = load_speech(a.audio, secs)
            mel, mask = mel_input(wav, fe)

            def sub():
                h, lens = E.subsample(tower, mel, mask)
                return h, lens

            with torch.inference_mode():
                feats12, lens12 = early(tower, mel, mask, 12)
            asr = model.asr

            def reader_greedy():
                with torch.inference_mode():
                    return asr.decode(asr.reader(feats12.float(), lens12), lens12, asr.tok)

            fns = {
                "reader_greedy": reader_greedy,
                "preprocess_cpu": lambda: fe([wav], sampling_rate=16000, return_tensors="pt"),
                "E0_subsample": sub,
                "E6": lambda: early(tower, mel, mask, 6),
                "E9": lambda: early(tower, mel, mask, 9),
                "E12_noproj": lambda: E.run_layers(tower, *E.subsample(tower, mel, mask), 0, 12, proj=False),
                "E12": lambda: early(tower, mel, mask, 12),
                "e2e_full": lambda: model.transcribe_batch([wav]),
            }
            gpu = {k: k != "preprocess_cpu" for k in fns}
            with torch.inference_mode():
                samples = measure_interleaved(fns, a.iterations, a.warmup, gpu)
                shapes = {}
                for d in (6, 9, 12):
                    y, lens = early(tower, mel, mask, d)
                    sync()
                    shapes[f"E{d}"] = (list(y.shape), int(lens.item()))
                yn = E.run_layers(tower, *E.subsample(tower, mel, mask), 0, 12, proj=False)
                shapes["E12_noproj"] = (list(yn.shape), shapes["E12"][1])
            res["mps_driver_allocated_bytes"] = torch.mps.driver_allocated_memory()
            for k, xs in samples.items():
                r = summarize(xs, a.warmup)
                r["rtf_p50"] = r["p50_ms"] / (secs * 1000)
                if k in ("E6", "E9", "E12", "E12_noproj"):
                    depth = int(k[1:].split("_")[0])
                    r.update(
                        executed_parameters=executed_params(tower, depth, proj=(k == "E12")),
                        output_shape=shapes[k][0],
                        output_frames=shapes[k][1],
                        includes_projection=(k == "E12"),
                        scope="mel_to_hidden_includes_subsample_and_active_layers",
                    )
                res["measures"][f"{k}/{secs}s"] = r
            for k in ("E6", "E9", "E12_noproj"):
                m, base = res["measures"][f"{k}/{secs}s"], res["measures"][f"E12/{secs}s"]
                m["p50_reduction_vs_E12"] = 1 - m["p50_ms"] / base["p50_ms"]
            m, base = res["measures"][f"E9/{secs}s"], res["measures"][f"E12_noproj/{secs}s"]
            res["measures"][f"E9/{secs}s"]["p50_reduction_vs_E12_noproj"] = 1 - m["p50_ms"] / base["p50_ms"]
            m = res["measures"][f"E6/{secs}s"]
            m["p50_reduction_vs_E12_noproj"] = 1 - m["p50_ms"] / base["p50_ms"]
            res["measures"][f"e2e_full/{secs}s"]["scope"] = "waveform_to_reader_greedy_decode_full_12_layer"
            res["measures"][f"preprocess_cpu/{secs}s"]["scope"] = "waveform_to_feature_extractor_cpu"
        res["mps_allocated_bytes_end"] = torch.mps.current_allocated_memory()
        res["thermal_after"] = thermal()
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
