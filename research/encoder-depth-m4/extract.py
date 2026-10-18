"""Phase 1: save intermediate encoder states for base and ASR-adapted towers from one shared mel per utterance.

Per (variant, set, split): one float16 array per layer, rows = valid frames of all utterances concatenated, plus index.json
(offsets, ids, texts). float16 is a storage format only (round-trip error is measured and recorded; the probe computes in fp32)."""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

import numpy as np
from common import FEATURES, environment, load_adapted, load_base, mel_input, no_overlap, read_manifest, stages, write_json

SPLITS = ("train", "dev", "test")
NAMES = ("CV", "FLEURS", "ASCEND")


def layer_file(layer):
    return layer.replace("-", "_") + ".npy"


def extract_set(towers, fe, rows, name, split, root):
    """Both towers see the identical mel, so base/adapted differ only by the LoRA delta."""
    from embed2asr.audio import load_audio

    buf = {v: {k: [] for k in FEATURES} for v in towers}
    err = {v: {k: [0.0, 0.0] for k in FEATURES} for v in towers}
    index = []
    for r in rows:
        wav = load_audio(r["audio"])
        if len(wav) / 16000 > 29.5:
            raise ValueError(f"Clip longer than the official 29.5 s window: {r['audio']}")
        mel, mask = mel_input(wav, fe)
        frames = None
        for variant, tower in towers.items():
            h, lengths = stages(tower, mel, mask)
            n = int(lengths.item())
            if n <= 0 or (frames is not None and n != frames):
                raise RuntimeError(f"Invalid or inconsistent frame count {n} for {r['audio']}")
            frames = n
            for k, v in h.items():
                x = v[0, :n].float().cpu().numpy()
                if not np.isfinite(x).all() or np.abs(x).max() > 6e4:
                    raise RuntimeError(f"{variant}/{k} not representable in float16 for {r['audio']}")
                x16 = x.astype(np.float16)
                err[variant][k][0] += float(((x - x16.astype(np.float32)) ** 2).sum())
                err[variant][k][1] += float((x**2).sum())
                buf[variant][k].append(x16)
        index.append(
            {
                "id": r["sha256"][:24],
                "text": r["text"],
                "lang": r["lang"],
                "set": name,
                "seconds": r["seconds"],
                "frames": frames,
                "speaker": r.get("speaker"),
                "language": r.get("language"),
                "sha256": r["sha256"],
            }
        )
    offsets = np.cumsum([0] + [i["frames"] for i in index]).tolist()
    for variant in towers:
        folder = root / variant / f"{name}-{split}"
        folder.mkdir(parents=True, exist_ok=False)
        for k in FEATURES:
            np.save(folder / layer_file(k), np.concatenate(buf[variant][k]))
        (folder / "index.json").write_text(json.dumps({"offsets": offsets, "rows": index}, ensure_ascii=False), encoding="utf-8")
    return {
        "utterances": len(index),
        "frames": offsets[-1],
        "fp16_roundtrip_rel_l2": {v: {k: (e[0] / max(e[1], 1e-30)) ** 0.5 for k, e in err[v].items()} for v in towers},
    }


def pooled(root, variant, name, split, layer="L6"):
    x = np.load(root / variant / f"{name}-{split}" / layer_file(layer), mmap_mode="r")
    off = json.loads((root / variant / f"{name}-{split}" / "index.json").read_text())["offsets"]
    return np.stack([np.asarray(x[a:b], dtype=np.float32).mean(0) for a, b in zip(off[:-1], off[1:])])


def acoustic_neighbour_audit(root, names, threshold=0.98):
    """Mean-pooled (globally centred) L6 states: cosine of every dev/test utterance to its nearest train utterance.
    A re-encoded or cropped copy of a train clip would sit near 1.0; genuinely different utterances do not."""
    out = {}
    for name in names:
        train = pooled(root, "base", name, "train")
        mu = train.mean(0)
        unit = lambda a: (a - mu) / np.linalg.norm(a - mu, axis=1, keepdims=True)  # noqa: E731
        t = unit(train)
        # reference scale: nearest *other* train utterance
        s = t @ t.T
        np.fill_diagonal(s, -1)
        out[f"{name}/train_vs_train"] = {"nn_cos_p50": float(np.median(s.max(1))), "nn_cos_max": float(s.max())}
        for split in ("dev", "test"):
            sim = unit(pooled(root, "base", name, split)) @ t.T
            nn = sim.max(1)
            out[f"{name}/{split}_vs_train"] = {
                "nn_cos_p50": float(np.median(nn)),
                "nn_cos_p99": float(np.percentile(nn, 99)),
                "nn_cos_max": float(nn.max()),
                f"n_above_{threshold}": int((nn > threshold).sum()),
            }
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifests", type=Path, default=Path("research/encoder-depth-m4/runs/manifests"))
    p.add_argument("--revision", help="released TingMix-Large HF commit")
    p.add_argument("--parity", type=Path, default=Path("research/encoder-depth-m4/runs/parity.json"))
    p.add_argument("--limit", type=int, default=0, help="smoke subset per set/split; results are exploratory only")
    p.add_argument("--out", type=Path, default=Path("research/encoder-depth-m4/runs/features"))
    a = p.parse_args()
    result = {
        "phase": "extract",
        "status": "BLOCKED",
        "environment": environment(),
        "sets": {},
        "limit": a.limit,
        "variants": ["base", "adapted"],
    }
    try:
        data = {(n, s): read_manifest(a.manifests / f"{n}-{s}.jsonl") for n in NAMES for s in SPLITS}
        for n in NAMES:  # exact audio-byte / path leakage between splits is a hard error
            no_overlap(data[n, "train"], data[n, "dev"] + data[n, "test"])
            no_overlap(data[n, "dev"], data[n, "test"])
        result["overlap"] = {"exact_duplicates": 0, "acoustic_audit": "see nn_audit"}
        parity = json.loads(a.parity.read_text(encoding="utf-8"))
        if parity.get("status") != "PASS":
            raise RuntimeError("MPS parity did not pass; feature extraction is blocked")
        model, origin = load_adapted(a.revision or parity["origin"]["hf_revision"])
        if origin["hf_revision"] != parity["origin"]["hf_revision"] or origin["delta_sha256"] != parity["origin"]["delta_sha256"]:
            raise RuntimeError("Feature extraction release differs from MPS parity release")
        result["origin"] = origin
        towers = {"base": load_base(), "adapted": model.asr.encoder.tower}
        fe = model.asr.encoder.feature_extractor
        for (name, split), rows in data.items():
            rows = rows[: a.limit] if a.limit else rows
            result["sets"][f"{name}/{split}"] = extract_set(towers, fe, rows, name, split, a.out)
            print(f"{name}/{split}: {len(rows)} utterances", flush=True)
        result["nn_audit"] = acoustic_neighbour_audit(a.out, NAMES)
        result["status"] = "PASS"
        result["layers"] = list(FEATURES)
        result["storage_dtype"] = "float16 (probe computes in fp32)"
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()
    write_json(a.out / "extract_results.json", result)
    print(f"{result['status']}: {a.out / 'extract_results.json'}")
    if result["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
