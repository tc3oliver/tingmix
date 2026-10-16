"""Streaming smoke / latency measurement: feeds clips (and one concatenated stream) through StreamingSession as if live.

  python evaluation/streaming.py MANIFEST.jsonl [MANIFEST2.jsonl ...] --model-path RELEASE_DIR_OR_REPO_OR_PT --out RESULT.json [--n 50] [--map FROM=TO]

MANIFEST: JSON lines with an `audio` path and `text` (the reference). `--map` rewrites a path prefix. Per manifest, `--n` clips (evenly
spaced) are streamed one by one; accuracy is the edit-distance rate of the streaming final text against the reference and against
offline transcription (CER over characters, MER over CJK characters + latin words, punctuation and case ignored).
The first manifest also provides the concatenated stream.
Audio is fed in 100 ms chunks without sleeping; latencies are on the audio clock plus the measured compute:
  first_partial_s  audio time of the first partial minus segment start, plus the compute of that update
  final_s          audio time of the final minus segment end (the endpointing wait), plus the compute of that update
"""

from __future__ import annotations

import argparse
import itertools
import json
import re
import time

import numpy as np

from tingmix.core.audio import SAMPLE_RATE, load_audio
from tingmix.core.registry import load_model
from tingmix.streaming.session import StreamingSession
from tingmix.streaming.vad import EnergyVAD

CHUNK = SAMPLE_RATE // 10


class Timed:
    """Wraps a model and records the wall time of every transcribe_batch call."""

    def __init__(self, model):
        self.model, self.costs, self.offline = model, [], {}
        self.sample_rate, self.max_window_s = model.sample_rate, model.max_window_s

    def transcribe_batch(self, wavs):
        t0 = time.perf_counter()
        out = self.model.transcribe_batch(wavs)
        self.costs.append(time.perf_counter() - t0)
        return out


def run_stream(model: Timed, wav: np.ndarray, **kw) -> dict:
    session, events, decodes = StreamingSession(model, **kw), [], []
    model.costs.clear()
    for i in range(0, len(wav), CHUNK):
        out = session.feed(wav[i : i + CHUNK])
        fed = min(i + CHUNK, len(wav)) / SAMPLE_RATE
        compute = sum(model.costs)
        decodes += model.costs
        model.costs.clear()
        events += [(ev, fed, compute) for ev in out]
    out = session.flush()
    events += [(ev, len(wav) / SAMPLE_RATE, sum(model.costs)) for ev in out]
    decodes += model.costs
    seen, ok = {}, True  # per segment: revisions count up from 0, a final comes last and only once
    for ev, _, _ in events:
        ok &= seen.get(ev.segment_id, (-1, False))[0] == ev.revision - 1 and not seen.get(ev.segment_id, (0, False))[1]
        seen[ev.segment_id] = (ev.revision, ev.type == "final")
    ok &= all(done for _, done in seen.values())
    first, final = {}, []
    for ev, t, c in events:
        if ev.type == "partial":
            first.setdefault(ev.segment_id, round(t - ev.start + c, 3))
        else:
            final.append(round(t - ev.end + c, 3))
    d = np.array(decodes) * 1000
    return {
        "text": session.transcript.text,
        "segments": [{"start": f.start, "end": f.end, "text": f.text} for f in session.transcript.finals],
        "sequence_ok": bool(ok),
        "partials": sum(ev.type == "partial" for ev, _, _ in events),
        "decode_all_ms": [round(float(v), 1) for v in d],
        "decode_ms": {
            "n": len(d),
            **{k: round(float(v), 1) for k, v in zip(("mean", "p50", "p95", "max"), (d.mean(), *np.percentile(d, [50, 95]), d.max()))},
        },
        "first_partial_s": list(first.values()),
        "final_s": final,
    }


def edit_distance(a, b) -> int:
    row = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        prev, row[0] = row[0], i
        for j, y in enumerate(b, 1):
            prev, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, prev + (x != y))
    return row[-1]


def chars(t: str) -> list[str]:
    return list(re.sub(r"[\W_]+", "", t.lower()))


def units(t: str) -> list[str]:
    return re.findall(r"[\u4e00-\u9fff]|[a-z0-9']+", t.lower())


def rate(pairs, split) -> dict:
    """pairs: (reference, hypothesis) -> corpus error rate."""
    err = sum(edit_distance(split(r), split(h)) for r, h in pairs)
    n = sum(len(split(r)) for r, _ in pairs)
    return {"err": err, "n": n, "rate": round(err / max(n, 1), 4)}


def accuracy(rows: list[dict]) -> dict:
    out = {}
    for name, a, b in (("stream_vs_ref", "ref", "text"), ("offline_vs_ref", "ref", "offline"), ("stream_vs_offline", "offline", "text")):
        pairs = [(r[a], r[b]) for r in rows]
        out[name] = {"cer": rate(pairs, chars), "mer": rate(pairs, units)}
    return out


def pick(rows, n, lo, hi, prefix_from, prefix_to):
    ok = [r for r in rows if lo <= r.get("seconds", lo) <= hi]
    step = max(len(ok) // n, 1)
    return [
        (prefix_to + r["audio"][len(prefix_from) :] if r["audio"].startswith(prefix_from) else r["audio"], r["text"]) for r in ok[::step]
    ]


def stats(x) -> dict:
    x = np.array(x)
    if not len(x):
        return {"n": 0}
    return {
        "n": len(x),
        "mean": round(float(x.mean()), 3),
        "p50": round(float(np.percentile(x, 50)), 3),
        "p95": round(float(np.percentile(x, 95)), 3),
        "max": round(float(x.max()), 3),
    }


def evaluate(model: Timed, args, **kw) -> dict:
    """Stream the manifests' clips (and one concatenated stream) with StreamingSession(**kw); `threshold_db` sets the VAD."""
    if "threshold_db" in kw:
        kw["vad"] = EnergyVAD(kw.pop("threshold_db"))
    src, dst = args.map.split("=", 1) if args.map else ("", "")
    tail = np.zeros(int(args.tail_s * SAMPLE_RATE), np.float32)

    def offline_text(path, w):  # the offline reference does not depend on the session settings
        if path not in model.offline:
            model.offline[path] = model.transcribe_batch([w])[0]
        return model.offline[path]

    results = {"manifests": {}, "concat": None}
    decode_ms, first, final = [], [], []
    for mi, mf in enumerate(args.manifest):
        rows = [json.loads(line) for line in open(mf, encoding="utf-8")]
        clips = pick(rows, args.n + (40 if mi == 0 else 0), args.min_s, args.max_s, src, dst)
        singles = []
        for path, ref in clips[: args.n]:
            w = load_audio(path)
            res = run_stream(model, np.concatenate([w, tail]), **kw)
            decode_ms += res.pop("decode_all_ms")
            first += res["first_partial_s"]
            final += res["final_s"]
            singles.append({"seconds": round(len(w) / SAMPLE_RATE, 2), "ref": ref, "offline": offline_text(path, w), **res})
        results["manifests"][mf.rsplit("/", 2)[-2] if mf.endswith("index.jsonl") else mf.rsplit("/", 1)[-1]] = {
            "accuracy": accuracy(singles),
            "same_as_offline": sum(s["text"] == s["offline"] for s in singles) / len(singles),
            "clips": singles,
        }
        if mi == 0 and args.concat_s:
            parts, total = [], 0.0
            for path, ref in clips[args.n :]:
                parts.append((path, load_audio(path), ref))
                total += len(parts[-1][1]) / SAMPLE_RATE + args.gap_s
                if total >= args.concat_s:
                    break
            gap = np.zeros(int(args.gap_s * SAMPLE_RATE), np.float32)
            stream = np.concatenate([x for _, w, _ in parts for x in (w, gap)])
            offline = [offline_text(path, w) for path, w, _ in parts]
            res = run_stream(model, stream, **kw)
            res.pop("decode_all_ms")
            results["concat"] = {
                "seconds": round(len(stream) / SAMPLE_RATE, 1),
                "clips": len(parts),
                "segments_found": len(res["segments"]),
                "offline_texts": offline,
                "refs": [r for _, _, r in parts],
                **res,
            }

    results["sequence_ok"] = all(c["sequence_ok"] for m in results["manifests"].values() for c in m["clips"]) and (
        not results["concat"] or results["concat"]["sequence_ok"]
    )
    results["latency"] = {"decode_ms": stats(decode_ms), "first_partial_s": stats(first), "final_s": stats(final)}
    return results


def summary(r: dict) -> dict:
    out = {
        k: {n: m["accuracy"][n]["mer"]["rate"] for n in ("stream_vs_ref", "stream_vs_offline")} | {"same": m["same_as_offline"]}
        for k, m in r["manifests"].items()
    }
    c = r["concat"]
    return {"sequence_ok": r["sequence_ok"], "mer": out, "concat": c and f"{c['segments_found']}/{c['clips']}", "latency": r["latency"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest", nargs="+")
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=50, help="single-clip streams per manifest")
    ap.add_argument("--concat-s", type=float, default=60.0, help="length of the concatenated stream (0 = skip)")
    ap.add_argument("--gap-s", type=float, default=1.5, help="silence between concatenated clips")
    ap.add_argument("--tail-s", type=float, default=1.0, help="silence appended to single clips")
    ap.add_argument("--map", help="FROM=TO path prefix rewrite")
    ap.add_argument("--min-s", type=float, default=1.0)
    ap.add_argument("--max-s", type=float, default=20.0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--lm", help="character n-gram LM file -> CTC prefix beam search for every decode")
    ap.add_argument("--lm-alpha", type=float, default=0.25)
    ap.add_argument("--lm-beta", type=float, default=1.5)
    ap.add_argument("--session", default="{}", help="JSON of StreamingSession kwargs (+ threshold_db), e.g. '{\"preroll_ms\": 500}'")
    ap.add_argument("--grid", help="JSON {kwarg: [values]}: run every combination, write only summaries (use hop_ms=1e6 to skip partials)")
    args = ap.parse_args()

    model = Timed(
        load_model(
            args.model_path, device=args.device, **({"lm": args.lm, "lm_alpha": args.lm_alpha, "lm_beta": args.lm_beta} if args.lm else {})
        )
    )
    model.transcribe_batch([np.zeros(SAMPLE_RATE, np.float32) + 0.01])  # warm-up
    base = json.loads(args.session)
    if args.grid:
        grid = json.loads(args.grid)
        runs = []
        for values in itertools.product(*grid.values()):
            kw = base | dict(zip(grid, values))
            runs.append({"session": kw, **summary(evaluate(model, args, **kw))})
            print(json.dumps(runs[-1], ensure_ascii=False), flush=True)
            with open(args.out, "w", encoding="utf-8") as f:
                json.dump(runs, f, ensure_ascii=False, indent=1)
        return
    results = evaluate(model, args, **base)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    print(json.dumps(summary(results), ensure_ascii=False))


if __name__ == "__main__":
    main()
