"""Dataset audit + filtered manifests for TingMix-Large training.

    python -m training.large.audit --spec spec.json

spec.json (all paths private, kept outside the repo):
  {"audio_map": ["<prefix in rows>", "<prefix on this host>"], "out_dir": "...", "sample": 200,
   "train": {name: index.jsonl}, "dev": {name: index.jsonl}, "test": {name: index.jsonl}}
Row fields used: audio, text, lang, seconds, frames, speaker_id. Test sets are only read for overlap checks.
Writes <out_dir>/<audit_name>.json (summary + data_hash), train_<set>.jsonl and dev_<set>.jsonl (filtered).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
from concurrent.futures import ThreadPoolExecutor

from embed2asr.audio import load_audio
from embed2asr.normalize import norm

_UNITS = re.compile(r"[⺀-鿿豈-﫿㐀-䶿]|[^\s⺀-鿿豈-﫿㐀-䶿]+")
SHORT = 4  # <= SHORT units (CJK chars + latin words) counts as a short phrase ("yeah", "好")
FIELDS = ("audio", "text", "lang", "seconds", "frames", "speaker_id")


def sha256_file(p: str) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def units(text: str) -> int:
    return len(_UNITS.findall(text))


def real_speaker(s: str | None) -> bool:
    return bool(s) and s != "unk" and not s.endswith("-unk")


def load_rows(path: str, name: str, amap: tuple[str, str]) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            rows.append(
                {
                    "audio": r["audio"].replace(amap[0], amap[1], 1),
                    "text": r["text"],
                    "lang": r.get("lang"),
                    "seconds": r["seconds"],
                    "frames": r["frames"],
                    "speaker_id": r.get("speaker_id", ""),
                    "set": name,
                }
            )
    return rows


def decode_check(rows: list[dict], n: int, seed: int = 0) -> dict:
    pick = random.Random(seed).sample(rows, min(n, len(rows)))

    def one(r):
        try:
            return abs(len(load_audio(r["audio"])) / 16000 - r["seconds"]) <= 0.1
        except Exception:
            return None

    with ThreadPoolExecutor(16) as ex:
        res = list(ex.map(one, pick))
    return {"checked": len(pick), "decode_fail": res.count(None), "duration_mismatch": res.count(False)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--name", default="v1mix_audit")
    a = ap.parse_args()
    spec = json.load(open(a.spec, encoding="utf-8"))
    amap, out = tuple(spec["audio_map"]), spec["out_dir"]
    os.makedirs(out, exist_ok=True)

    data, sha = {}, {}
    for split in ("train", "dev", "test"):
        for name, p in spec[split].items():
            data[split, name] = load_rows(p, name, amap)
            sha[f"{split}/{name}"] = sha256_file(p)
    test_rows = [r for (s, _), rs in data.items() if s == "test" for r in rs]
    test_paths = {r["audio"] for r in test_rows}
    test_text = {norm(r["text"]) for r in test_rows}
    test_spk = {r["speaker_id"] for r in test_rows if real_speaker(r["speaker_id"])}
    dev_paths = {r["audio"] for (s, _), rs in data.items() if s == "dev" for r in rs}
    dev_spk = {r["speaker_id"] for (s, _), rs in data.items() if s == "dev" for r in rs if real_speaker(r["speaker_id"])}

    report, seen = {"sets": {}, "input_sha256": sha}, set()
    for (split, name), rows in data.items():
        miss = [os.path.exists(r["audio"]) for r in rows].count(False)
        s = {
            "split": split,
            "rows": len(rows),
            "hours": round(sum(r["seconds"] for r in rows) / 3600, 3),
            "missing_audio": miss,
            "dup_rows_in_set": len(rows) - len({r["audio"] for r in rows}),
        }
        if split != "test":
            s["path_overlap_test"] = sum(r["audio"] in test_paths for r in rows)
            s["sample_decode"] = decode_check([r for r in rows if os.path.exists(r["audio"])], spec.get("sample", 200))
            spk = {r["speaker_id"] for r in rows if real_speaker(r["speaker_id"])}
            s["speakers"], s["speakers_in_test"] = len(spk), len(spk & test_spk)
            tx = [r for r in rows if norm(r["text"]) in test_text]
            s["text_overlap_test"] = {"rows": len(tx), "short": sum(units(r["text"]) <= SHORT for r in tx)}
        if split == "train":
            s["path_overlap_dev"] = sum(r["audio"] in dev_paths for r in rows)
            s["speakers_in_dev"] = len(spk & dev_spk)
            s["dup_rows_vs_earlier_train_sets"] = sum(r["audio"] in seen for r in rows)
            keep = []
            for r in rows:
                if (
                    not os.path.exists(r["audio"])
                    or r["audio"] in test_paths
                    or r["audio"] in dev_paths
                    or r["audio"] in seen
                    or (norm(r["text"]) in test_text and units(r["text"]) > SHORT)
                ):
                    continue
                seen.add(r["audio"])
                keep.append(r)
        elif split == "dev":
            keep = [r for r in rows if os.path.exists(r["audio"]) and r["audio"] not in test_paths]
        else:
            keep = None
        if keep is not None:
            s["kept_rows"], s["kept_hours"] = len(keep), round(sum(r["seconds"] for r in keep) / 3600, 3)
            with open(f"{out}/{split}_{name}.jsonl", "w", encoding="utf-8") as f:
                for r in keep:
                    f.write(json.dumps({**{k: r[k] for k in FIELDS}, "set": name}, ensure_ascii=False) + "\n")
            s["manifest_sha256"] = sha256_file(f"{out}/{split}_{name}.jsonl")
        report["sets"][f"{split}/{name}"] = s
        print(split, name, s, flush=True)

    tr = [s for k, s in report["sets"].items() if k.startswith("train/")]
    report["totals"] = {
        "train_rows": sum(s["kept_rows"] for s in tr),
        "train_hours": round(sum(s["kept_hours"] for s in tr), 2),
        "train_missing": sum(s["missing_audio"] for s in tr),
        "dev_missing": sum(s["missing_audio"] for k, s in report["sets"].items() if k.startswith("dev/")),
        "train_path_overlap_test": sum(s["path_overlap_test"] for s in tr),
        "train_path_overlap_dev": sum(s["path_overlap_dev"] for s in tr),
    }
    report["data_hash"] = hashlib.sha256(
        json.dumps(sorted((k, v["manifest_sha256"]) for k, v in report["sets"].items() if "manifest_sha256" in v)).encode()
    ).hexdigest()
    json.dump(report, open(f"{out}/{a.name}.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(json.dumps(report["totals"]), report["data_hash"])
    bad = (
        report["totals"]["train_path_overlap_test"]
        + report["totals"]["train_path_overlap_dev"]
        + sum(s["path_overlap_test"] for k, s in report["sets"].items() if k.startswith("dev/"))
    )
    raise SystemExit(1 if bad else 0)


if __name__ == "__main__":
    main()
