#!/usr/bin/env python3
"""N-best rescoring of CTC + char-LM beam search with an LLM (OpenAI-compatible /v1/completions with prompt_logprobs, e.g. vLLM).
DEV sets only.

  nbest: python -m evaluation.nbest_llm nbest --logits DIR --lm LM --alpha 0.35 --beta 1.5 --n 8 --out nbest.json
  score: python -m evaluation.nbest_llm score --nbest nbest.json --url http://localhost:8822/v1 --model NAME --out scored.json
  tune:  python -m evaluation.nbest_llm tune --scored scored.json

final score = beam score + lam * LLM log P(text | prompt) + mu * #units; `tune` grids (lam, mu) on even utterances and reports
odd ones (held out), plus the N-best oracle."""

from __future__ import annotations

import argparse
import collections
import json
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import numpy as np

from tingmix.models.large.attention import ZeroLM, beam_nbest

PROMPT = "以下是一段語音的逐字稿（台灣華語、英文或中英夾雜）：\n"
_G = {}


def _init(logits, lm_path):
    from embed2asr.lm import CharLM, lm_units
    from embed2asr.tokenizer import Tok

    d = Path(logits)
    itos = json.loads((d / "tok.json").read_text())["itos"]
    _G["tok"] = Tok(itos, (d / "tok.sp").read_bytes() or None)
    _G["units"], _G["lm"], _G["V"] = lm_units(itos), CharLM.load(lm_path) if lm_path else ZeroLM(), len(itos)



def _job(job):
    v, i, alpha, beta, beam, n = job
    lp = np.full((len(v), _G["V"]), -1e30, dtype=np.float32)
    np.put_along_axis(lp, i.astype(np.int64), v.astype(np.float32), axis=1)
    return beam_nbest(lp, _G["tok"], _G["lm"], alpha, beta, beam, n, _G["units"])


def nbest(a):
    d, out = Path(a.logits), {}
    with ProcessPoolExecutor(a.workers, initializer=_init, initargs=(a.logits, a.lm)) as ex:
        for f in sorted(d.glob("*.npz")):
            z, m = np.load(f), json.loads((d / f"{f.stem}.json").read_text())
            jobs = [(z[f"v{k}"], z[f"i{k}"], a.alpha, a.beta, a.beam, a.n) for k in range(len(m["rows"]))]
            hyps = list(ex.map(_job, jobs, chunksize=8))
            out[f.stem] = {"metric": m["metric"], "rows": [r | {"nbest": h} for r, h in zip(m["rows"], hyps)]}
            print(f.stem, len(hyps), flush=True)
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False))


def score(a):
    data = json.loads(Path(a.nbest).read_text())
    base = len(_lp([PROMPT], a)[0])  # prompt tokens are not scored

    def one(texts):
        return [sum(x for x in lp[base:]) for lp in _lp([PROMPT + t for t in texts], a)]

    jobs = [(name, k) for name, s in data.items() for k in range(len(s["rows"]))]
    with ThreadPoolExecutor(a.concurrency) as ex:
        res = list(ex.map(lambda j: one([h["text"] or "。" for h in data[j[0]]["rows"][j[1]]["nbest"]]), jobs))
    for (name, k), llm in zip(jobs, res):
        for h, v in zip(data[name]["rows"][k]["nbest"], llm):
            h["llm"] = v
    Path(a.out).write_text(json.dumps(data, ensure_ascii=False))


def _lp(prompts, a):
    import urllib.request

    body = json.dumps({"model": a.model, "prompt": prompts, "max_tokens": 1, "prompt_logprobs": 0, "temperature": 0}).encode()
    req = urllib.request.Request(a.url + "/completions", body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        d = json.load(r)
    return [[next(iter(x.values()))["logprob"] if x else 0.0 for x in c["prompt_logprobs"]] for c in sorted(d["choices"], key=lambda c: c["index"])]


def _rate(data, pick, part):
    from embed2asr.metrics import corpus

    out = {}
    for name, s in data.items():
        rows = [r for k, r in enumerate(s["rows"]) if k % 2 == part]
        pairs = [(r["text"], pick(r["nbest"], r["text"])) for r in rows]
        if s["metric"] == "mixmer":
            pairs = [p for p, r in zip(pairs, rows) if r["lang"] in ("mix", "zh-en")]
            out[name] = corpus(pairs, "MER")["value"]
        else:
            out[name] = corpus(pairs, s["metric"].upper())["value"]
    return out


def tune(a):
    from embed2asr.metrics import corpus

    data = json.loads(Path(a.scored).read_text())
    w = {"cv": 0.30, "ascend": 0.20, "ntuml": 0.20, "parl": 0.15, "en": 0.15}
    tot = lambda r: sum(w[k] * r[k] for k in w if k in r)  # noqa: E731

    def oracle(nb, ref, m="CER"):
        return min(nb, key=lambda h: corpus([(ref, h["text"])], m)["errors"])["text"]

    for part in (0, 1):
        print("part", part, "beam", {k: round(v, 4) for k, v in _rate(data, lambda nb, ref: nb[0]["text"], part).items()})
    print("oracle(held-out)", {k: round(v, 4) for k, v in _rate(data, lambda nb, ref: oracle(nb, ref), 1).items()})
    best = None
    for lam in a.lam:
        for mu in a.mu:
            pick = lambda nb, ref, lam=lam, mu=mu: max(nb, key=lambda h: h["beam"] + lam * h[a.field] + mu * h["units"])["text"]  # noqa: E731
            r = _rate(data, pick, 0)
            if best is None or tot(r) < tot(best[2]):
                best = (lam, mu, r)
    lam, mu, r = best
    pick = lambda nb, ref: max(nb, key=lambda h: h["beam"] + lam * h[a.field] + mu * h["units"])["text"]  # noqa: E731
    print("best lam", lam, "mu", mu, "tune", {k: round(v, 4) for k, v in r.items()})
    print("held-out", {k: round(v, 4) for k, v in _rate(data, pick, 1).items()})


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    s = p.add_subparsers(dest="cmd", required=True)
    n = s.add_parser("nbest")
    n.add_argument("--logits", required=True)
    n.add_argument("--lm", help="char LM file (omit: no LM)")
    n.add_argument("--alpha", type=float, default=0.35)
    n.add_argument("--beta", type=float, default=1.5)
    n.add_argument("--beam", type=int, default=16)
    n.add_argument("--n", type=int, default=8)
    n.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 4))
    n.add_argument("--out", required=True)
    c = s.add_parser("score")
    c.add_argument("--nbest", required=True)
    c.add_argument("--url", default="http://localhost:8822/v1")
    c.add_argument("--model", default="qwen3.8-flash-next")
    c.add_argument("--concurrency", type=int, default=4)
    c.add_argument("--out", required=True)
    t = s.add_parser("tune")
    t.add_argument("--scored", required=True)
    t.add_argument("--field", default="llm", help="per-hypothesis score to combine (llm | att)")
    t.add_argument("--lam", nargs="+", type=float, default=[0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0])
    t.add_argument("--mu", nargs="+", type=float, default=[0.0, 0.5, 1.0, 2.0])
    a = p.parse_args(argv)
    {"nbest": nbest, "score": score, "tune": tune}[a.cmd](a)


if __name__ == "__main__":
    main()
