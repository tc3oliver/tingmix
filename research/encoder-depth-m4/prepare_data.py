"""Phase 2 data: build train/dev/test manifests from the *public* corpora (FLEURS en_us, ASCEND, Common Voice zh-TW).

Official splits are kept (train / dev / test); a seeded subsample is drawn from each. References are normalised with the
Embed2ASR pipeline (numnorm -> norm), rows the reader's tokenizer cannot represent are dropped, and train/dev/test are
audited for speaker, audio-byte and normalised-text overlap (text overlap is removed from dev/test, never from train).
Everything lands in the gitignored runs/ directory.

ASCEND transcripts are Simplified Chinese; the TingMix vocabulary is Traditional, so they are converted with OpenCC s2t
(character level). That makes absolute ASCEND error rates pessimistic; the depth comparison uses the same references for
every layer."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import random
import re
import tarfile
from pathlib import Path

DIGIT = re.compile(r"\d")
LANG = {"CV": "zh-TW", "FLEURS": "en-US", "ASCEND": "mix"}
# (train, dev, test) utterances kept per set; dev is only used to choose the shared probe learning rate.
SIZES = {"CV": (2000, 200, 500), "FLEURS": (1000, 200, 400), "ASCEND": (3000, 300, 600)}
MIN_S, MAX_S = 1.0, 29.5


def reference(text, kind):
    """Pipeline-identical reference text; None if it still holds digits (the pipeline's own digit filter)."""
    from embed2asr import numnorm
    from embed2asr.normalize import norm

    if kind == "zh":
        text = numnorm.zh(text)[0]
    elif kind == "en":
        text = numnorm.en(text)[0]
    out = norm(text)
    return None if not out or DIGIT.search(out) else out


def representable(tok, text):
    """True if tokenising loses nothing (no out-of-vocabulary character silently dropped)."""
    return tok.decode(tok.encode(text)) == text


def overlap_report(train, other):
    """Row indices of `other` that collide with train on normalised text; plus speaker / audio-hash intersections."""
    texts = {r["text"] for r in train}
    drop = {i for i, r in enumerate(other) if r["text"] in texts}
    spk_t = {r["speaker"] for r in train if r.get("speaker")}
    spk_o = {r["speaker"] for r in other if r.get("speaker")}
    sha_t = {r["sha256"] for r in train}
    return drop, sorted(spk_t & spk_o), sum(r["sha256"] in sha_t for r in other)


def rows_fleurs(root, split):
    base = root / "fleurs/data/en_us"
    rows = [
        {"member": rec[1], "raw": rec[2], "kind": "en", "speaker": None, "meta": {"gender": rec[6]}}
        for rec in csv.reader((base / f"{split}.tsv").open(encoding="utf-8"), delimiter="\t", quoting=csv.QUOTE_NONE)
    ]
    return rows, base / "audio" / f"{split}.tar.gz"


def rows_cv(root, split):
    base = root / "common_voice_22_0"
    rows = []
    for rec in csv.DictReader((base / f"transcript/zh-TW/{split}.tsv").open(encoding="utf-8"), delimiter="\t", quoting=csv.QUOTE_NONE):
        rows.append(
            {"member": f"zh-TW_{split}_0/{rec['path']}", "raw": rec["sentence"], "kind": "zh", "speaker": rec["client_id"], "meta": {}}
        )
    return rows, base / f"audio/zh-TW/{split}/zh-TW_{split}_0.tar"


def rows_ascend(root, split):
    import pyarrow.parquet as pq
    from opencc import OpenCC

    cc = OpenCC("s2t")
    name = {"train": "train-*", "dev": "validation-*", "test": "test-*"}[split]
    rows = []
    for f in sorted((root / "ASCEND/main").glob(f"{name}.parquet")):
        for r in pq.read_table(
            f, columns=["id", "audio", "transcription", "duration", "language", "original_speaker_id", "session_id"]
        ).to_pylist():
            if r["duration"] < MIN_S or r["duration"] > MAX_S:
                continue
            rows.append(
                {
                    "bytes": r["audio"]["bytes"],
                    "member": f"{f.stem}_{r['id']}.wav",
                    "raw": cc.convert(r["transcription"]),
                    "kind": "mix",
                    "speaker": f"ascend{r['original_speaker_id']}",
                    "meta": {"language": r["language"], "session": r["session_id"]},
                }
            )
    return rows, None


LOADERS = {"CV": rows_cv, "FLEURS": rows_fleurs, "ASCEND": rows_ascend}


def read_members(archive, names):
    """One sequential pass (random access into a .tar.gz re-inflates from the start every time)."""
    got = {}
    with tarfile.open(archive) as tf:
        for m in tf:
            if m.isfile() and Path(m.name).name in names:
                got[Path(m.name).name] = tf.extractfile(m).read()
                if len(got) == len(names):
                    break
    return got


def build(name, split, root, out, tok, rng, want):
    import soundfile as sf

    rows, archive = LOADERS[name](root, split)
    rng.shuffle(rows)
    stats = {"candidates": len(rows), "no_text_or_digits": 0, "not_representable": 0, "bad_duration": 0}
    cand = []
    for r in rows:
        if len(cand) >= int(want * 1.3) + 5:  # headroom for the duration filter
            break
        text = reference(r["raw"], r["kind"])
        if text is None:
            stats["no_text_or_digits"] += 1
        elif not representable(tok, text):
            stats["not_representable"] += 1
        else:
            cand.append((r, text))
    if archive:
        blobs = read_members(archive, {Path(r["member"]).name for r, _ in cand})
    audio_dir = out / "audio" / name / split
    audio_dir.mkdir(parents=True, exist_ok=True)
    kept = []
    for r, text in cand:
        if len(kept) >= want:
            break
        data = r.get("bytes") or blobs[Path(r["member"]).name]
        info = sf.info(io.BytesIO(data))
        seconds = info.frames / info.samplerate
        if not MIN_S <= seconds <= MAX_S:
            stats["bad_duration"] += 1
            continue
        path = audio_dir / Path(r["member"]).name
        path.write_bytes(data)
        kept.append(
            {
                "audio": str(path.resolve()),
                "text": text,
                "lang": LANG[name],
                "set": name,
                "split": split,
                "seconds": seconds,
                "speaker": r["speaker"],
                "sha256": hashlib.sha256(data).hexdigest(),
                **r["meta"],
            }
        )
    stats["kept"] = len(kept)
    return kept, stats


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=Path("research/encoder-depth-m4/runs/data"))
    p.add_argument("--out", type=Path, default=Path("research/encoder-depth-m4/runs/manifests"))
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--scale", type=float, default=1.0, help="multiply SIZES (e.g. 0.02 for a smoke run)")
    a = p.parse_args()
    from common import read_tokenizer, resolve_release

    release, origin = resolve_release()
    tok = read_tokenizer(release)
    report = {"seed": a.seed, "release": origin["hf_revision"], "sets": {}}
    a.out.mkdir(parents=True, exist_ok=True)
    for name, sizes in SIZES.items():
        built = {}
        for split, n in zip(("train", "dev", "test"), sizes):
            want = max(1, int(n * a.scale))
            # Over-draw dev/test so that text-overlap removal below still leaves ~`want` rows.
            built[split], st = build(
                name, split, a.data, a.out, tok, random.Random(f"{a.seed}-{name}-{split}"), want * (1 if split == "train" else 2)
            )
            report["sets"][f"{name}/{split}"] = st
        for split in ("dev", "test"):
            drop, speakers, same_audio = overlap_report(built["train"], built[split])
            built[split] = [r for i, r in enumerate(built[split]) if i not in drop][
                : max(1, int(SIZES[name][1 if split == "dev" else 2] * a.scale))
            ]
            report["sets"][f"{name}/{split}"].update(
                text_overlap_removed=len(drop), speaker_overlap=speakers, identical_audio_with_train=same_audio, kept=len(built[split])
            )
        dev_test = overlap_report(built["dev"], built["test"])
        report["sets"][f"{name}/dev_vs_test"] = {
            "text_overlap": len(dev_test[0]),
            "speaker_overlap": dev_test[1],
            "identical_audio": dev_test[2],
        }
        for split, rows in built.items():
            with (a.out / f"{name}-{split}.jsonl").open("w", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            st = report["sets"][f"{name}/{split}"]
            st["utterances"], st["hours"] = len(rows), sum(r["seconds"] for r in rows) / 3600
    (a.out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                k: {x: y for x, y in v.items() if x in ("utterances", "hours", "text_overlap_removed", "speaker_overlap")}
                for k, v in report["sets"].items()
            },
            indent=1,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
