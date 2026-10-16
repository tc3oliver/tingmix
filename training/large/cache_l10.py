"""Resumable, sharded cache of layer-10 encoder states (1024-d bf16, input of layer 11) for partial fine-tuning.

    python -m training.large.cache_l10 --manifest train_x.jsonl --out cache/l10/train_x [--num-workers N --worker-id I]

Shard k = manifest rows [k*shard_rows, (k+1)*shard_rows), so --num-workers/--worker-id (shard k -> worker k % N) is
deterministic. A shard is shard_XXXXX.bin + .jsonl + .sha256, each written to *.tmp and os.replace'd; the .sha256 file is
the commit marker and is verified on resume (missing/corrupt shards are redone). meta.json pins everything that changes
the numbers; resuming into a dir with different meta is refused. Encoder is not padding-invariant: batches are
length-sorted inside a shard with a fixed batch size, both recorded in meta.

Reader: rows = load_l10_index(dir); read_l10(rows[i]) -> bf16 [frames, 1024].
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

CODE_VERSION = 1
DIM = 1024
MIN_SECONDS, MAX_SECONDS = 0.4, 29.5


def sha256_file(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def _atomic_write(path: Path, data: str) -> None:
    tmp = Path(f"{path}.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def shard_base(out: Path, k: int) -> Path:
    return out / f"shard_{k:05d}"


def shard_ok(base: Path) -> bool:
    """Complete = sha file present and both payload files hash to it."""
    try:
        want = Path(f"{base}.sha256").read_text().split()
        return len(want) == 2 and sha256_file(f"{base}.bin") == want[0] and sha256_file(f"{base}.jsonl") == want[1]
    except OSError:
        return False


def build_shard(base: Path, rows: list[dict], encode, load, batch_size: int, threads: int) -> int:
    """Encode `rows` (dicts with id/audio/seconds/...) into base.bin/.jsonl/.sha256. Returns rows skipped (bad audio/length)."""
    order = sorted(rows, key=lambda r: r["seconds"])
    index, skipped, off = [], 0, 0
    hb, tmp_bin = hashlib.sha256(), Path(f"{base}.bin.tmp")
    with ThreadPoolExecutor(threads) as ex, open(tmp_bin, "wb") as fb:
        futs = [ex.submit(load, r["audio"]) for r in order]
        for i in range(0, len(order), batch_size):
            batch, wavs = [], []
            for r, fu in zip(order[i : i + batch_size], futs[i : i + batch_size]):
                try:
                    w = fu.result()
                except Exception as e:  # undecodable file: skip, counted
                    print(f"skip {r['audio']}: {e}", flush=True)
                    skipped += 1
                    continue
                if not MIN_SECONDS <= len(w) / 16000 <= MAX_SECONDS:
                    skipped += 1
                    continue
                batch.append(r)
                wavs.append(w)
            futs[i : i + batch_size] = [None] * len(futs[i : i + batch_size])
            if not batch:
                continue
            for r, h in zip(batch, encode(wavs)):
                if h.shape[0] == 0 or not torch.isfinite(h.float()).all():
                    skipped += 1
                    continue
                b = h.contiguous().view(torch.int16).numpy().tobytes()
                fb.write(b)
                hb.update(b)
                index.append(
                    {
                        "id": r["id"],
                        "off": off,
                        "frames": h.shape[0],
                        "dim": DIM,
                        "text": r["text"],
                        "lang": r.get("lang"),
                        "seconds": r["seconds"],
                        "audio": r["audio"],
                        "speaker_id": r.get("speaker_id", ""),
                        "set": r.get("set"),
                    }
                )
                off += len(b)
        fb.flush()
        os.fsync(fb.fileno())
    index.sort(key=lambda x: x["id"])
    text = "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in index)
    _atomic_write(Path(f"{base}.jsonl"), text)
    os.replace(tmp_bin, f"{base}.bin")
    _atomic_write(Path(f"{base}.sha256"), f"{hb.hexdigest()} {hashlib.sha256(text.encode()).hexdigest()}\n")
    return skipped


def check_meta(out: Path, meta: dict) -> None:
    p = out / "meta.json"
    if p.exists():
        old = json.loads(p.read_text())
        if old != meta:
            raise SystemExit(f"refusing to resume: {p} differs from this run: {sorted(k for k in meta if old.get(k) != meta[k])}")
    else:
        _atomic_write(p, json.dumps(meta, indent=1, sort_keys=True))


def run(
    manifest: Path,
    out: Path,
    encode,
    load,
    meta_extra: dict,
    shard_rows=2000,
    batch_size=16,
    num_workers=1,
    worker_id=0,
    threads=16,
    max_shards: int | None = None,
) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in manifest.open(encoding="utf-8")]
    for i, r in enumerate(rows):
        r["id"] = i
    meta = {
        "feature": "layer10_hidden",
        "dim": DIM,
        "dtype": "bf16",
        "fps": 25,
        "batch_size": batch_size,
        "shard_rows": shard_rows,
        "batching": "length-sorted within shard",
        "manifest_sha256": sha256_file(manifest),
        "manifest_rows": len(rows),
        "code_version": CODE_VERSION,
        **meta_extra,
    }
    check_meta(out, meta)
    n_shards = -(-len(rows) // shard_rows)
    stats = {"skipped_shards": 0, "built_shards": 0, "skipped_rows": 0}
    for k in range(worker_id, n_shards, num_workers):
        if max_shards is not None and stats["built_shards"] >= max_shards:
            break
        base = shard_base(out, k)
        if shard_ok(base):
            stats["skipped_shards"] += 1
            print(f"shard {k}/{n_shards} ok, skip", flush=True)
            continue
        stats["skipped_rows"] += build_shard(base, rows[k * shard_rows : (k + 1) * shard_rows], encode, load, batch_size, threads)
        stats["built_shards"] += 1
        print(f"shard {k}/{n_shards} built", flush=True)
    return stats


def load_l10_index(cache_dir, verify: bool = False) -> list[dict]:
    """All rows of all complete shards, sorted by manifest row id; row['bin'] = absolute payload path.
    verify=True re-hashes every shard (slow); otherwise the .sha256 commit marker is trusted."""
    d, rows = Path(cache_dir).resolve(), []
    for j in sorted(d.glob("shard_*.jsonl")):
        if not j.with_suffix(".sha256").exists() or (verify and not shard_ok(j.with_suffix(""))):
            continue
        for line in j.open(encoding="utf-8"):
            r = json.loads(line)
            r["bin"] = str(j.with_suffix(".bin"))
            rows.append(r)
    return rows


_mm: dict[str, np.memmap] = {}


def read_l10(row: dict) -> torch.Tensor:
    mm = _mm.get(row["bin"])
    if mm is None:
        mm = _mm[row["bin"]] = np.memmap(row["bin"], dtype=np.int16, mode="r")
    a = mm[row["off"] // 2 : row["off"] // 2 + row["frames"] * row["dim"]]
    return torch.from_numpy(np.array(a)).view(torch.bfloat16).view(row["frames"], row["dim"])


def merge_index(cache_dir) -> int:
    """Write DIR/index.jsonl (rows of sha-verified shards, with absolute 'bin') for trainers that take one jsonl per pool."""
    rows = load_l10_index(cache_dir, verify=True)
    _atomic_write(Path(cache_dir) / "index.jsonl", "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    return len(rows)


def check_frames(l10_rows: list[dict], final_rows: list[dict], n: int = 20) -> int:
    """Compare frames with a final-feature cache (rows keyed by audio path after mapping); returns rows compared."""
    final = {r["audio"]: r["frames"] for r in final_rows}
    done = 0
    for r in l10_rows:
        if r["audio"] in final:
            assert final[r["audio"]] == r["frames"], (r["audio"], final[r["audio"]], r["frames"])
            done += 1
            if done == n:
                break
    return done


def make_encoder(device: str = "cuda"):
    """encode(list of 16 kHz float32 waveforms) -> list of bf16 CPU [frames, 1024] layer-10 states (pure bf16, no autocast)."""
    from embed2asr.audio import load_audio

    from tingmix.models.large import encoder as E

    tower = E.load_tower().to(device)
    fe = E.load_feature_extractor()

    @torch.inference_mode()
    def encode(wavs):
        inp = fe(wavs, sampling_rate=16000, return_tensors="pt")
        h, lens = E.subsample(tower, inp["input_features"].to(device, torch.bfloat16), inp["input_features_mask"].to(device))
        h = E.run_layers(tower, h, lens, 0, 10, proj=False)
        return [h[i, :n].cpu() for i, n in enumerate(lens.tolist())]

    return encode, load_audio


def main() -> None:
    from tingmix.models.large.encoder import ENCODER_ID, ENCODER_REVISION

    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--merge-index", type=Path, metavar="DIR", help="write DIR/index.jsonl from verified shards and exit")
    ap.add_argument("--shard-rows", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--num-workers", type=int, default=1)
    ap.add_argument("--worker-id", type=int, default=0)
    ap.add_argument("--max-shards", type=int)
    a = ap.parse_args()
    if a.merge_index:
        print(f"{merge_index(a.merge_index)} rows -> {a.merge_index}/index.jsonl")
        return
    if not (a.manifest and a.out):
        ap.error("--manifest and --out are required")
    encode, load = make_encoder()
    stats = run(
        a.manifest,
        a.out,
        encode,
        load,
        {"encoder": ENCODER_ID, "revision": ENCODER_REVISION},
        a.shard_rows,
        a.batch_size,
        a.num_workers,
        a.worker_id,
        max_shards=a.max_shards,
    )
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
