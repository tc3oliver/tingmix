#!/usr/bin/env python3
"""Real-time check of a natively streaming model on CPU: feed audio in fixed chunks, measure compute vs audio time.

  python -m evaluation.cpu_rtf --model DIR_OR_REPO --manifest dev.jsonl [--threads 4] [--chunk-ms 100] [--n 50]

Reports RTF (compute seconds / audio seconds; < 1 = real time), per-chunk latency p50/p95/max, and the streamed text
accuracy vs reference (CER / MER by `lang`). Uses `Stream.accept` exactly as StreamingSession does."""

from __future__ import annotations

import argparse
import json
import time


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--manifest", nargs="+", required=True)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--chunk-ms", type=int, default=100)
    p.add_argument("--n", type=int, default=50, help="utterances per manifest (evenly spaced)")
    p.add_argument("--out")
    p.add_argument("--lm", help="character LM for final texts (Small: partials greedy, finals beam search)")
    p.add_argument("--lm-alpha", type=float, default=0.35)
    p.add_argument("--lm-beta", type=float, default=1.5)
    a = p.parse_args(argv)

    import numpy as np
    import torch
    from embed2asr.audio import load_audio
    from embed2asr.metrics import score

    torch.set_num_threads(a.threads)
    from tingmix.core.registry import load_model

    model = load_model(a.model, device="cpu", **({"lm": a.lm, "lm_alpha": a.lm_alpha, "lm_beta": a.lm_beta} if a.lm else {}))
    model.transcribe_batch([np.zeros(16000, np.float32)])  # warm-up
    chunk = 16000 * a.chunk_ms // 1000
    lat, comp, audio, rows = [], 0.0, 0.0, []
    for path in a.manifest:
        ms = [json.loads(x) for x in open(path, encoding="utf-8")]
        for r in ms[:: max(1, len(ms) // a.n)][: a.n]:
            wav = load_audio(r["audio"])
            s = model.new_stream()
            text = ""
            for i in range(0, len(wav), chunk):
                t0 = time.perf_counter()
                text = s.accept(wav[i : i + chunk])
                dt = time.perf_counter() - t0
                lat.append(dt * 1000)
                comp += dt
            t0 = time.perf_counter()
            text = s.finalize()  # final decode (LM beam search when configured) counts as compute too
            comp += time.perf_counter() - t0
            audio += len(wav) / 16000
            rows.append({"ref": r["text"], "hyp": text, "lang": r.get("lang")})
    q = np.percentile(lat, [50, 95, 100])
    res = {
        "threads": a.threads,
        "chunk_ms": a.chunk_ms,
        "utts": len(rows),
        "audio_s": round(audio, 1),
        "rtf": round(comp / audio, 4),
        "chunk_ms_p50": round(q[0], 2),
        "chunk_ms_p95": round(q[1], 2),
        "chunk_ms_max": round(q[2], 2),
        "accuracy": {k: round(v["value"], 4) for k, v in score(rows).items()},
    }
    print(json.dumps(res, ensure_ascii=False))
    if a.out:
        open(a.out, "w").write(json.dumps({**res, "rows": rows}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
