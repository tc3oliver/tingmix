"""Phase 2: fair layer-wise frozen-feature Linear CTC probes. No dev/test training and no CPU fallback."""
from __future__ import annotations

import argparse
import json
import random
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from common import FEATURES, environment, no_overlap, require_mps, resolve_release, write_json
from embed2asr.ctc import ctc_min_len
from embed2asr.metrics import corpus
from embed2asr.pipeline import greedy
from embed2asr.tokenizer import Tok


def index_rows(folder, split, groups):
    all_rows = []
    for name in groups:
        index = folder / split / name / "index.jsonl"
        if not index.is_file():
            raise FileNotFoundError(index)
        for line in index.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            r["set"] = name
            all_rows.append(r)
    return all_rows


def read_tokenizer(release_path):
    cfg = json.loads((release_path / "config.json").read_text(encoding="utf-8"))
    units = json.loads((release_path / cfg["tokenizer"]["units"]).read_text(encoding="utf-8"))
    sp_name = cfg["tokenizer"].get("sentencepiece")
    return Tok(units, (release_path / sp_name).read_bytes() if sp_name else None)


def load_features(row, layer, device):
    with np.load(row["feature"], allow_pickle=False) as f:
        x = torch.from_numpy(f[layer.replace("-", "_")].astype(np.float32)).to(device)
    if x.ndim != 2 or x.shape[1] != (1536 if layer == "L12-Proj" else 1024):
        raise RuntimeError(f"Feature dimension mismatch: {layer} at {row['feature']}")
    return x


def eligible(rows, tok):
    kept, rejected = [], []
    for r in rows:
        y = tok.encode(r["text"])
        if not y or ctc_min_len(y) > int(r["frames"]):
            rejected.append({"id": r["id"], "reason": "empty_target_or_ctc_too_short"})
            continue
        kept.append({**r, "ids": y})
    return kept, rejected


class LinearCTC(nn.Module):
    """Identical layer norm (zero parameters) and one linear output layer for every 1024-d state."""
    def __init__(self, dim, vocab):
        super().__init__()
        self.linear = nn.Linear(dim, vocab)

    def forward(self, x):
        return self.linear(F.layer_norm(x, (x.shape[-1],)))


def train_one(layer, train_rows, dev_rows, tok, epochs, lr, seed, out, device, dev_specs):
    torch.manual_seed(seed)
    dim = 1536 if layer == "L12-Proj" else 1024
    head = LinearCTC(dim, len(tok.itos)).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=0.01)
    loss_fn = nn.CTCLoss(blank=0, reduction="mean", zero_infinity=False)
    history = []
    for epoch in range(epochs):
        order = list(range(len(train_rows)))
        random.Random(seed + epoch).shuffle(order)
        head.train()
        losses = []
        for index in order:
            r = train_rows[index]
            x = load_features(r, layer, device)
            target = torch.tensor(r["ids"], dtype=torch.long, device=device)
            if ctc_min_len(r["ids"]) > x.shape[0]:
                raise AssertionError("Invalid CTC frame alignment")
            logits = head(x)
            logp = logits.log_softmax(-1)[:, None, :]
            inp_len = torch.tensor([len(x)], device=device, dtype=torch.long)
            tgt_len = torch.tensor([len(target)], device=device, dtype=torch.long)
            loss = loss_fn(logp, target, inp_len, tgt_len)
            if not torch.isfinite(loss).item():
                raise RuntimeError(f"Nonfinite CTC loss on {r['id']}")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        history.append({"epoch": epoch + 1, "train_ctc": float(np.mean(losses))})
    head.eval()
    sets = {}
    with torch.inference_mode():
        for name, metric in dev_specs.items():
            pairs = []
            for r in dev_rows:
                if r["set"] != name:
                    continue
                x = load_features(r, layer, device)
                logits = head(x[None, ...])
                # greedy uses [B,T,V] with lengths on CPU
                pred = greedy(logits, torch.tensor([len(x)]), tok)[0]
                pairs.append((r["text"], pred))
            sets[name] = corpus(pairs, metric)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        raise FileExistsError(out)
    torch.save({"state_dict": {k: v.cpu() for k, v in head.state_dict().items()},
                "layer": layer, "seed": seed, "dim": dim, "epochs": epochs,
                "vocab_size": len(tok.itos)}, out)
    return {"parameter_count": sum(p.numel() for p in head.parameters()), "history": history,
            "dev": sets, "checkpoint": str(out.resolve()), "device": str(device)}


def verify_conditions(feature_root, train_sets, dev_sets):
    for split, groups in (("train", train_sets), ("dev", dev_sets)):
        base = index_rows(feature_root / "base", split, groups)
        adapted = index_rows(feature_root / "adapted", split, groups)
        def key(rs):
            return [(r["set"], r["id"], r["text"], r.get("lang"), r["frames"]) for r in rs]
        if key(base) != key(adapted):
            raise RuntimeError(f"{split} base/adapted manifests differ")
    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--features", type=Path, default=Path("research/encoder-depth-m4/runs/features"))
    p.add_argument("--train-sets", nargs="+", required=True, metavar="SET")
    p.add_argument("--dev-set", action="append", required=True, metavar="SET=CER|WER|MER")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--lr", type=float, default=0.0003)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--out", type=Path, default=Path("research/encoder-depth-m4/runs/probes"))
    a = p.parse_args()
    result = {"phase": "linear_probe", "status": "BLOCKED", "environment": environment(),
              "seed": a.seed, "epochs": a.epochs, "lr": a.lr,
              "scope": "frozen_cached_features_linear_CTC_no_reader", "results": {}}
    try:
        device = require_mps()
        if a.epochs <= 0:
            raise ValueError("epochs must be positive")
        dev_specs = {}
        for x in a.dev_set:
            k, v = x.split("=", 1)
            metric = v.upper()
            if metric not in ("CER", "WER", "MER"):
                raise ValueError(x)
            dev_specs[k] = metric
        verify_conditions(a.features, a.train_sets, list(dev_specs))
        release, origin = resolve_release()
        result["origin"] = origin
        tok = read_tokenizer(release)
        raw_train = index_rows(a.features / "adapted", "train", a.train_sets)
        raw_dev = index_rows(a.features / "adapted", "dev", list(dev_specs))
        # Exact cross-split checks are repeated on cached indexes; near-duplicates need separate audit.
        result["leakage"] = no_overlap(raw_train, raw_dev)
        train, rej_train = eligible(raw_train, tok)
        dev, rej_dev = eligible(raw_dev, tok)
        if not train or not dev:
            raise RuntimeError("No valid train/dev samples; dev must never substitute for train")
        result["counts"] = {"train": len(train), "dev": len(dev),
                             "invalid_train": rej_train, "invalid_dev": rej_dev}
        for variant in ("base", "adapted"):
            rows_tr = index_rows(a.features / variant, "train", a.train_sets)
            rows_dv = index_rows(a.features / variant, "dev", list(dev_specs))
            keep_t = {r["id"] for r in train}
            keep_d = {r["id"] for r in dev}
            rows_tr = [r for r in rows_tr if r["id"] in keep_t]
            rows_dv = [r for r in rows_dv if r["id"] in keep_d]
            # Guarantee identical targets and ordering, independent of variant and layer.
            for rows, control in ((rows_tr, train), (rows_dv, dev)):
                if [(r["id"], r["text"]) for r in rows] != [(r["id"], r["text"]) for r in control]:
                    raise RuntimeError("Probe sample alignment differs across variants")
                for r, c in zip(rows, control):
                    r["ids"] = c["ids"]
            for layer in FEATURES:
                key = f"{variant}/{layer}"
                checkpoint = a.out / variant / f"{layer.replace('-', '_')}-seed{a.seed}.pt"
                result["results"][key] = train_one(layer, rows_tr, rows_dv, tok, a.epochs, a.lr,
                                                   a.seed, checkpoint, device, dev_specs)
                write_json(a.out / "results.json", result)
        result["status"] = "PASS"
        result["known_limits"] = ["single seed", "no near-duplicate acoustic audit",
                                 "linear probes not equivalent to 22.2M GRU reader",
                                 "L12-Proj has a different parameter budget"]
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()
    write_json(a.out / "results.json", result)
    print(f"{result['status']}: {a.out / 'results.json'}")
    if result["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
