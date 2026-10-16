#!/usr/bin/env python3
"""Build a character n-gram LM and tune its decoding weights on DEV sets only.

  build: python -m evaluation.lm_tune build --text A.jsonl B.jsonl --exclude DEV_OR_TEST.jsonl ... --out lm.charlm.json.gz
  dump:  python -m evaluation.lm_tune dump --model PATH.pt --set NAME=MANIFEST:METRIC[:nd] ... --out logits_dir
  grid:  python -m evaluation.lm_tune grid --logits logits_dir --lm lm.charlm.json.gz --alpha 0.2 0.3 --beta 1 1.5 --out grid.json

`build` reads the "text" field of manifests (one utterance per line) and drops every sentence that also occurs in an
--exclude manifest (dev and test sets), so the LM never saw an evaluation sentence. `dump` stores per-utterance CTC
log-probs (top-k per frame, rest -inf; the beam search only expands the top 8) from the deployed end-to-end path.
`grid` decodes every (alpha, beta) in parallel on CPU and reports corpus CER / WER / MER per set; greedy is the reference."""

from __future__ import annotations

import argparse
import itertools
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np


def read_rows(spec):
    name, rest = spec.split("=", 1)
    nd = rest.endswith(":nd")
    path, metric = rest.removesuffix(":nd").rsplit(":", 1)
    rows = [json.loads(x) for x in open(path, encoding="utf-8")]
    if nd:
        rows = [r for r in rows if not any(c.isdigit() for c in r["text"])]
    return name, metric.lower(), rows


def build(a):
    from embed2asr.lm import CharLM, lm_text
    from embed2asr.normalize import norm

    excl = {norm(json.loads(x)["text"]).replace(" ", "") for p in a.exclude for x in open(p, encoding="utf-8")}
    sents, dropped = [], 0
    for p in a.text:
        for x in open(p, encoding="utf-8"):
            t = json.loads(x)["text"]
            if norm(t).replace(" ", "") in excl:
                dropped += 1
                continue
            s = lm_text(t)
            if s:
                sents.append(s)
    lm = CharLM(sents, a.order)
    sha = lm.save(a.out)
    meta = {
        "sentences": len(sents),
        "dropped_eval_overlap": dropped,
        "order": a.order,
        "sha256": sha,
        "sources": [Path(p).name for p in a.text],
    }
    Path(str(a.out) + ".meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta))


def dump(a):
    import torch
    from embed2asr.audio import load_audio

    from tingmix.core.registry import load_model

    asr = load_model(a.model, device=a.device).asr
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for spec in a.set:
        name, metric, rows = read_rows(spec)
        lps, meta = [], []
        with torch.inference_mode():
            for r in rows:
                feats, lens = asr.encoder([load_audio(r["audio"])])
                lp = asr.reader(feats.to(asr.device).float(), lens).float().log_softmax(-1)[0, : int(lens[0])].cpu()
                v, i = lp.topk(a.topk, -1)
                lps.append((v.half().numpy(), i.short().numpy()))
                meta.append({"text": r["text"], "lang": r.get("lang")})
        np.savez(out / f"{name}.npz", **{f"v{k}": v for k, (v, _) in enumerate(lps)}, **{f"i{k}": i for k, (_, i) in enumerate(lps)})
        (out / f"{name}.json").write_text(json.dumps({"metric": metric, "rows": meta}, ensure_ascii=False))
        print(f"{name}: {len(rows)} utterances", flush=True)
    sp = asr.tok.sp_bytes or b""
    (out / "tok.json").write_text(json.dumps({"itos": asr.tok.itos}, ensure_ascii=False))
    (out / "tok.sp").write_bytes(sp)


_G = {}


def _init(logits, lm_path):
    from embed2asr.lm import CharLM, lm_units
    from embed2asr.tokenizer import Tok

    d = Path(logits)
    itos = json.loads((d / "tok.json").read_text())["itos"]
    _G["tok"] = Tok(itos, (d / "tok.sp").read_bytes() or None)
    _G["units"] = lm_units(itos)
    _G["lm"] = CharLM.load(lm_path) if lm_path else None
    _G["V"] = len(itos)


def _decode(job):
    from embed2asr.ctc import collapse_ctc
    from embed2asr.lm import beam_decode

    v, i, alpha, beta, beam = job
    if alpha is None:  # greedy reference
        return _G["tok"].decode(collapse_ctc(i[:, 0].astype(np.int64).tolist()))
    lp = np.full((len(v), _G["V"]), -1e30, dtype=np.float32)
    np.put_along_axis(lp, i.astype(np.int64), v.astype(np.float32), axis=1)
    return beam_decode(lp, _G["tok"], _G["lm"], alpha, beta, beam, units=_G["units"])


def grid(a):
    from embed2asr.metrics import corpus

    d = Path(a.logits)
    sets = {}
    for f in sorted(d.glob("*.npz")):
        z, m = np.load(f), json.loads((d / f"{f.stem}.json").read_text())
        n = len(m["rows"])
        sets[f.stem] = (m, [(z[f"v{k}"], z[f"i{k}"]) for k in range(n)])
    combos = [(None, None)] + list(itertools.product(a.alpha, a.beta))
    results = []
    with ProcessPoolExecutor(a.workers, initializer=_init, initargs=(a.logits, a.lm)) as ex:
        for alpha, beta in combos:
            rec = {"alpha": alpha, "beta": beta}
            for name, (m, utts) in sets.items():
                hyps = list(ex.map(_decode, [(v, i, alpha, beta, a.beam) for v, i in utts], chunksize=16))
                pairs = [(r["text"], h) for r, h in zip(m["rows"], hyps)]
                if m["metric"] == "mixmer":
                    rec[name] = corpus([p for p, r in zip(pairs, m["rows"]) if r["lang"] in ("mix", "zh-en")], "MER")["value"]
                    rec[name + "_all"] = corpus(pairs, "MER")["value"]
                else:
                    rec[name] = corpus(pairs, m["metric"].upper())["value"]
            results.append(rec)
            print(json.dumps({k: round(v, 5) if isinstance(v, float) else v for k, v in rec.items()}), flush=True)
    Path(a.out).write_text(json.dumps(results, indent=1))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    s = p.add_subparsers(dest="cmd", required=True)
    b = s.add_parser("build")
    b.add_argument("--text", nargs="+", required=True)
    b.add_argument("--exclude", nargs="*", default=[])
    b.add_argument("--order", type=int, default=5)
    b.add_argument("--out", required=True)
    d = s.add_parser("dump")
    d.add_argument("--model", required=True)
    d.add_argument("--set", nargs="+", required=True)
    d.add_argument("--out", required=True)
    d.add_argument("--topk", type=int, default=16)
    d.add_argument("--device")
    g = s.add_parser("grid")
    g.add_argument("--logits", required=True)
    g.add_argument("--lm")
    g.add_argument("--alpha", nargs="+", type=float, default=[0.15, 0.25, 0.35, 0.5])
    g.add_argument("--beta", nargs="+", type=float, default=[0.5, 1.0, 1.5, 2.0, 3.0])
    g.add_argument("--beam", type=int, default=16)
    g.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 4))
    g.add_argument("--out", required=True)
    a = p.parse_args(argv)
    {"build": build, "dump": dump, "grid": grid}[a.cmd](a)


if __name__ == "__main__":
    main()
