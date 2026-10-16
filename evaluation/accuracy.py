#!/usr/bin/env python3
"""Corpus CER / WER / MER of one TingMix model, end to end from audio (the deployed inference path).

  python -m evaluation.accuracy --model large|PATH.pt|RELEASE_DIR --set NAME=MANIFEST:METRIC[:nd] ... --out DIR

METRIC: cer | wer | mer | mixmer (MER on code-switched rows, `lang` in mix/zh-en, plus MER on all rows).
nd: drop reference rows that contain digits (the labels spell numbers as the audio says them only in digit-free rows).
Writes DIR/<NAME>.jsonl (ref, hyp, lang, audio) and DIR/summary.json. One model per call: never mix checkpoints."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--set", nargs="+", required=True)
    p.add_argument("--map-prefix", nargs="*", default=[], help="OLD=NEW audio path rewrites")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--batch-files", type=int, default=16, help="files handed to the model per call (encoded one file at a time)")
    p.add_argument("--limit", type=int, default=0, help="first N rows per set (smoke tests only)")
    p.add_argument(
        "--by-lang", action="store_true", help="also score per `lang` (zh-TW CER, en-US WER, mix MER) like the Embed2ASR benchmark"
    )
    p.add_argument("--device")
    p.add_argument("--lm", help="character n-gram LM file (Embed2ASR format) -> prefix beam search")
    p.add_argument("--lm-alpha", type=float, default=0.25)
    p.add_argument("--lm-beta", type=float, default=1.5)
    p.add_argument("--beam", type=int, default=16)
    a = p.parse_args(argv)

    from embed2asr.audio import load_audio
    from embed2asr.metrics import corpus, score

    from tingmix.core.registry import load_model

    a.out.mkdir(parents=True, exist_ok=True)
    lm = dict(lm=a.lm, lm_alpha=a.lm_alpha, lm_beta=a.lm_beta, beam=a.beam) if a.lm else {}
    model = load_model(a.model, device=a.device, **lm)
    pm = [tuple(x.split("=", 1)) for x in a.map_prefix]
    summary = {"model": a.model, "lm": lm or None, "time": time.strftime("%F %T"), "sets": {}}
    for spec in a.set:
        name, rest = spec.split("=", 1)
        nd = rest.endswith(":nd")
        path, metric = rest.removesuffix(":nd").rsplit(":", 1)
        rows = [json.loads(x) for x in open(path, encoding="utf-8")]
        if nd:
            rows = [r for r in rows if not any(c.isdigit() for c in r["text"])]
        rows = rows[: a.limit] if a.limit else rows
        for r in rows:
            for old, new in pm:
                if r["audio"].startswith(old):
                    r["audio"] = new + r["audio"][len(old) :]
        t0, audio_s, hyps = time.time(), 0.0, []
        for i in range(0, len(rows), a.batch_files):
            wavs = [load_audio(r["audio"]) for r in rows[i : i + a.batch_files]]
            audio_s += sum(len(w) for w in wavs) / 16000
            hyps += model.transcribe_batch(wavs)
        dt = time.time() - t0
        with open(a.out / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for r, h in zip(rows, hyps):
                f.write(json.dumps({"ref": r["text"], "hyp": h, "lang": r.get("lang"), "audio": r["audio"]}, ensure_ascii=False) + "\n")
        pairs = [(r["text"], h) for r, h in zip(rows, hyps)]
        m = metric.lower()
        if m == "mixmer":
            res = corpus([(r["text"], h) for r, h in zip(rows, hyps) if r.get("lang") in ("mix", "zh-en")], "MER")
            res["all"] = corpus(pairs, "MER")
        else:
            res = corpus(pairs, m.upper())
        if a.by_lang:
            res["by_lang"] = score({"ref": r["text"], "hyp": h, "lang": r.get("lang")} for r, h in zip(rows, hyps))
        res.update(manifest=path, nodigit=nd, audio_h=round(audio_s / 3600, 3), rtf=round(dt / max(audio_s, 1e-9), 4))
        summary["sets"][name] = res
        print(json.dumps({name: res}, ensure_ascii=False), flush=True)
    (a.out / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
