"""Training data for TingMix-Large: manifest rows, rotate sampler, batches, audio -> log-mel collate.

Manifest row (jsonl): audio (path), text (normalized), lang, seconds, frames (25 fps encoder frames). Rows that point into
a feature cache (path/off) are fine: only audio/text/lang/seconds/frames are used. Paths can be rewritten with
--map-prefix OLD=NEW (e.g. a container mount point).

Data order is a pure function of (seed, epoch, pools): resume needs only (epoch, batch index)."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch


def load_rows(path, prefix_map=(), nodigit=False, feat_root=None):
    """feat_root: resolve relative feature-cache paths (row "path"/"bin" + "off") against this dir; None = keep no cache ref."""
    rows = []
    for line in Path(path).open(encoding="utf-8"):
        r = json.loads(line)
        a = r["audio"]
        for old, new in prefix_map:
            if a.startswith(old):
                a = new + a[len(old) :]
                break
        row = {
            "audio": a,
            "text": r["text"].replace("臺", "台"),
            "lang": r.get("lang"),
            "seconds": float(r["seconds"]),
            "frames": int(r["frames"]),
        }
        if feat_root is not None and ("path" in r or "bin" in r):
            b = Path(r.get("bin") or r["path"])
            row.update(bin=str(b if b.is_absolute() else (Path(feat_root) / b).resolve()), off=int(r["off"]), dim=int(r.get("dim", 1536)))
        rows.append(row)
    if nodigit:  # labels with digits: audio says the number in words, label does not (as in the Embed2ASR-v1 recipe)
        rows = [r for r in rows if not any(c.isdigit() for c in r["text"])]
    return rows


class RotateSampler:
    """Per epoch take up to HOURS of the next unused rows of each pool's shuffled order; reshuffle after a full pass.
    Port of the Embed2ASR-v1 rotate sampler (same seeding): epoch e's rows are a pure function of (seed, pools, e)."""

    def __init__(self, pools, seed):
        self.pools, self.seed = pools, seed  # [(rows, hours_per_epoch)]
        self._orders, self._state = {}, {0: [(0, 0)] * len(pools)}

    def _order(self, k, n):
        if (k, n) not in self._orders:
            o = list(range(len(self.pools[k][0])))
            random.Random(self.seed * 1009 + k * 101 + n).shuffle(o)
            self._orders[k, n] = o
        return self._orders[k, n]

    def _take(self, k, n, pos):
        rows, want = self.pools[k]
        take, seen, h = [], set(), 0.0
        while h < want and len(take) < len(rows):
            o = self._order(k, n)
            if pos >= len(o):
                n, pos = n + 1, 0
                continue
            i = o[pos]
            pos += 1
            if i in seen:
                continue
            take.append(i)
            seen.add(i)
            h += rows[i]["seconds"] / 3600
        return take, n, pos

    def epoch(self, ep):
        e = max(x for x in self._state if x < ep)
        while e < ep:
            takes = [self._take(k, n, pos) for k, (n, pos) in enumerate(self._state[e])]
            e += 1
            self._state[e] = [(n, pos) for _, n, pos in takes]
        return [self.pools[k][0][i] for k, (take, _, _) in enumerate(takes) for i in take]


def make_batches(rows, max_seconds, max_bs, rng):
    """Shuffle, sort windows of ~50 batches by length, fill batches up to max_seconds of PADDED audio (longest x count),
    shuffle batch order. Bounds encoder activation memory independent of clip length."""
    idx = list(range(len(rows)))
    rng.shuffle(idx)
    win = max_bs * 50
    batches = []
    for c in range(0, len(idx), win):
        chunk = sorted(idx[c : c + win], key=lambda j: rows[j]["seconds"])
        b = []
        for j in chunk:
            if b and (len(b) + 1) * rows[j]["seconds"] > max_seconds or len(b) == max_bs:
                batches.append(b)
                b = []
            b.append(j)
        if b:
            batches.append(b)
    rng.shuffle(batches)
    return batches


def decode(path):
    """float32 mono 16 kHz (soundfile for wav/flac, ffmpeg otherwise; same as Embed2ASR inference)."""
    from embed2asr.audio import load_audio

    return load_audio(path)


def read_feat(r):
    """Cached encoder states of one row: bf16 [frames, dim] (Embed2ASR cache layout: raw bf16 rows at byte offset)."""
    mm = np.memmap(r["bin"], dtype=np.int16, mode="r", offset=r["off"], shape=(r["frames"], r["dim"]))
    return torch.from_numpy(np.array(mm)).view(torch.bfloat16)


class BatchSet(torch.utils.data.Dataset):
    """Item k = collated batch k: (inputs, mask, targets, target_lens, texts, langs).
    features="audio": inputs = log-mel [B,F,128] bf16, mask [B,F] (mel frames);
    features="cache": inputs = cached states [B,T,dim] bf16 zero-padded, mask = lengths [B]."""

    def __init__(self, rows, batches, fe, encode, features="audio"):
        self.rows, self.batches, self.fe, self.encode, self.features = rows, batches, fe, encode, features

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, k):
        rs = [self.rows[j] for j in self.batches[k]]
        tg = [self.encode(r["text"]) for r in rs]
        if self.features == "wave":  # raw padded waveforms + sample lengths (models with their own front end)
            ws = []
            for r in rs:
                try:
                    ws.append(torch.from_numpy(decode(r["audio"])))
                except (OSError, ValueError) as e:
                    print(f"[data] cannot decode {r['audio']}: {e}; replaced by 0.5 s of zeros", flush=True)
                    ws.append(torch.zeros(8000))
            x, m = torch.nn.utils.rnn.pad_sequence(ws, batch_first=True), torch.tensor([len(w) for w in ws])
        elif self.features == "cache":
            xs = [read_feat(r) for r in rs]
            x, m = torch.nn.utils.rnn.pad_sequence(xs, batch_first=True), torch.tensor([len(t) for t in xs])
        else:
            wavs = []
            for r in rs:
                try:
                    wavs.append(decode(r["audio"]))
                except (OSError, ValueError) as e:  # unreadable file: keep the batch shape;
                    print(f"[data] cannot decode {r['audio']}: {e}; replaced by 0.5 s of zeros", flush=True)
                    wavs.append(np.zeros(8000, np.float32))  # rare (audited); short labels may still train on it
            inp = self.fe(wavs, sampling_rate=16000, return_tensors="pt")
            x, m = inp["input_features"].to(torch.bfloat16), inp["input_features_mask"]
        return (
            x,
            m,
            torch.tensor([t for x in tg for t in x], dtype=torch.long),
            torch.tensor([len(x) for x in tg], dtype=torch.long),
            [r["text"] for r in rs],
            [r["lang"] for r in rs],
        )


def loader(rows, batches, fe, encode, workers=12, start=0, features="audio"):
    ds = BatchSet(rows, batches[start:], fe, encode, features)
    return torch.utils.data.DataLoader(
        ds, batch_size=None, shuffle=False, num_workers=workers, prefetch_factor=4 if workers else None, persistent_workers=False
    )


def time_mask(x, lens, frac, gen):
    """Zero ~frac of valid frames in spans of 4 (Embed2ASR-v1 augmentation on reader input)."""
    if frac <= 0:
        return x
    B, T, _ = x.shape
    n = (lens.float() * frac / 4).long()
    keep = torch.ones(B, T)
    for b in range(B):
        if n[b] > 0:
            s = torch.randint(0, max(1, int(lens[b]) - 4), (int(n[b]),), generator=gen)
            for t in s.tolist():
                keep[b, t : t + 4] = 0
    return x * keep.to(x.device)[..., None]
