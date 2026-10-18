"""Phase 1: save FP32 intermediate states from the same full forward for base and ASR-adapted towers."""
from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

import numpy as np
import torch

from common import (FEATURES, environment, load_adapted, load_base, mel_input,
                    no_overlap, read_manifest, sample_key, stages, write_json)


def specs(values):
    out = {}
    for spec in values:
        name, path = spec.split("=", 1)
        if name in out:
            raise ValueError(f"Duplicate dataset {name}")
        out[name] = Path(path)
    return out


def extract_set(tower, fe, rows, variant, split, name, root):
    from embed2asr.audio import load_audio

    folder = root / variant / split / name
    folder.mkdir(parents=True, exist_ok=True)
    index = []
    for r in rows:
        key = sample_key(r)
        path = folder / f"{key}.npz"
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing feature: {path}")
        wav = load_audio(r["audio"])
        if len(wav) / 16000 > 29.5:
            raise ValueError(f"Clip longer than the official 29.5 s window: {r['audio']}")
        mel, mask = mel_input(wav, fe)
        h, lengths = stages(tower, mel, mask)
        n = int(lengths.item())
        if n <= 0:
            raise RuntimeError(f"No valid frames in {r['audio']}")
        tensors = {k.replace("-", "_"): v[0, :n].detach().float().cpu().numpy() for k, v in h.items()}
        np.savez_compressed(path, **tensors)
        index.append({"id": key, "feature": str(path.resolve()), "text": r["text"], "lang": r.get("lang"),
                      "audio": str(Path(r["audio"]).resolve()), "seconds": len(wav) / 16000, "frames": n})
    with (folder / "index.jsonl").open("x", encoding="utf-8") as f:
        for row in index:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return {"samples": len(index), "index": str((folder / "index.jsonl").resolve())}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train-set", action="append", default=[], metavar="NAME=MANIFEST")
    p.add_argument("--dev-set", action="append", default=[], metavar="NAME=MANIFEST")
    p.add_argument("--revision", help="released TingMix-Large HF commit")
    p.add_argument("--limit", type=int, default=0, help="smoke subset; disallow release-level conclusions")
    p.add_argument("--out", type=Path, default=Path("research/encoder-depth-m4/runs/features"))
    a = p.parse_args()
    result = {"phase": "extract", "status": "BLOCKED", "environment": environment(),
              "sets": {}, "limit": a.limit, "variants": ("base", "adapted")}
    try:
        tr, dv = specs(a.train_set), specs(a.dev_set)
        if not tr and not dv:
            raise ValueError("At least one train/dev manifest is required")
        train = {k: read_manifest(v) for k, v in tr.items()}
        dev = {k: read_manifest(v) for k, v in dv.items()}
        if train and dev:
            result["overlap"] = no_overlap([r for rs in train.values() for r in rs],
                                            [r for rs in dev.values() for r in rs])
        else:
            result["overlap"] = {"exact_duplicates": "NOT_CHECKED", "near_duplicate_audit": "NOT_VERIFIED"}
        model, origin = load_adapted(a.revision)
        result["origin"] = origin
        fe = model.asr.encoder.feature_extractor
        towers = {"adapted": model.asr.encoder.tower, "base": load_base()}
        for variant, tower in towers.items():
            for split, groups in (("train", train), ("dev", dev)):
                for name, rows in groups.items():
                    rs = rows[:a.limit] if a.limit else rows
                    result["sets"][f"{variant}/{split}/{name}"] = extract_set(tower, fe, rs, variant, split, name, a.out)
        result["status"] = "PASS"
        result["layers"] = list(FEATURES)
        result["storage_dtype"] = "float32"
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()
    write_json(a.out / "extract_results.json", result)
    print(f"{result['status']}: {a.out / 'extract_results.json'}")
    if result["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
