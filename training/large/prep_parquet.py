#!/usr/bin/env python3
"""Hugging Face parquet ASR split (audio bytes + transcription) -> wav files + TingMix manifest.

  python -m training.large.prep_parquet --parquet 'DIR/train-*.parquet' --wav-dir OUT/wav --out OUT/train.jsonl
         [--exclude-text other.jsonl ...] [--group-regex '^\\d+-|\\.wav$']

Text: numbers spelled out (embed2asr.numnorm), then embed2asr.normalize.norm; lang from the text (CJK + Latin = mix).
Keeps 0.4-29.5 s clips. Rows whose normalized text appears in an --exclude-text manifest are dropped (dev/test).
`group` (e.g. lecture / video id) is derived from the file name with --group-regex and stored as speaker_id."""

from __future__ import annotations

import argparse
import collections
import glob
import io
import json
import re
from pathlib import Path

CJK = r"⺀-鿿豈-﫿㐀-䶿"


def lang_of(t: str) -> str:
    zh, en = bool(re.search(f"[{CJK}]", t)), bool(re.search("[a-z]", t))
    return "mix" if zh and en else "zh-TW" if zh else "en-US"


def main(argv=None):
    import pyarrow.parquet as pq
    import soundfile as sf
    from embed2asr.normalize import norm
    from embed2asr.numnorm import numnorm

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--parquet", required=True, help="glob")
    p.add_argument("--wav-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--dataset", default="parquet")
    p.add_argument("--exclude-text", nargs="*", default=[])
    p.add_argument("--group-regex", default=r"^\d+-|\.wav$")
    a = p.parse_args(argv)

    excl = {norm(json.loads(x)["text"]).replace(" ", "") for f in a.exclude_text for x in open(f, encoding="utf-8")}
    a.wav_dir.mkdir(parents=True, exist_ok=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    drop, seen, recs = collections.Counter(), set(), []
    for fp in sorted(glob.glob(a.parquet)):
        for b in pq.ParquetFile(fp).iter_batches(batch_size=256, columns=["audio", "transcription", "file"]):
            for r in b.to_pylist():
                raw = r["transcription"] or ""
                t = raw
                t, _ = numnorm(t, "zh" if re.search(f"[{CJK}]", t) else "en")
                text = norm(t)
                if not text:
                    drop["empty"] += 1
                    continue
                if text.replace(" ", "") in excl:
                    drop["eval_text"] += 1
                    continue
                base = Path(r["file"]).name
                if not re.fullmatch(r"[A-Za-z0-9_.\-]+\.wav", base) or base in seen:
                    drop["bad_or_dup_name"] += 1
                    continue
                seen.add(base)
                info = sf.info(io.BytesIO(r["audio"]["bytes"]))
                sec = info.frames / info.samplerate
                if not 0.4 <= sec <= 29.5 or info.samplerate != 16000:
                    drop["duration_or_rate"] += 1
                    continue
                out = a.wav_dir / base
                if not out.is_file():
                    out.write_bytes(r["audio"]["bytes"])
                recs.append(
                    {
                        "audio": str(out.resolve()),
                        "text": text,
                        "lang": lang_of(text),
                        "seconds": round(sec, 3),
                        "frames": int(sec * 25),
                        "speaker_id": re.sub(a.group_regex, "", base),
                        "dataset": a.dataset,
                    }
                )
    with open(a.out, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in recs)
    c = collections.Counter(r["lang"] for r in recs)
    print(
        json.dumps(
            {
                "rows": len(recs),
                "hours": round(sum(r["seconds"] for r in recs) / 3600, 2),
                "lang": dict(c),
                "groups": len({r["speaker_id"] for r in recs}),
                "dropped": dict(drop),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
