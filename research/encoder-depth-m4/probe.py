"""Phase 3: fair layer-wise frozen-feature Linear CTC probes (MPS, no CPU fallback).

Every condition (variant x layer) sees the same utterances in the same order, the same tokenizer / targets, the same seed,
AdamW + warmup/cosine schedule, batch size and number of updates. Only the input features differ. The learning rate is chosen
once on *dev CTC loss averaged over all conditions* (never on test, never per layer); test is scored once per condition.
L12-Proj is 1536-d, so its head is a different size and is labelled as such."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from common import FEATURES, environment, read_tokenizer, require_mps, resolve_release, write_json
from embed2asr.ctc import ctc_min_len
from embed2asr.metrics import COUNTS
from embed2asr.pipeline import greedy

NAMES = ("CV", "FLEURS", "ASCEND")
METRIC = {"CV": "CER", "FLEURS": "WER", "ASCEND": "MER"}
VARIANTS = ("base", "adapted")


def dim_of(layer):
    return 1536 if layer == "L12-Proj" else 1024


def load_split(root, variant, split, tok):
    """-> list of utterance dicts (shared across variants/layers) with token ids, in a fixed order."""
    rows = []
    for name in NAMES:
        folder = root / variant / f"{name}-{split}"
        idx = json.loads((folder / "index.json").read_text(encoding="utf-8"))
        for i, r in enumerate(idx["rows"]):
            rows.append({**r, "set": name, "folder": folder, "start": idx["offsets"][i], "ids": tok.encode(r["text"])})
    return rows


def eligible(rows):
    """CTC needs frames >= len(target) + repeats; identical filter for every condition."""
    kept = [r for r in rows if r["ids"] and ctc_min_len(r["ids"]) <= r["frames"]]
    return kept, len(rows) - len(kept)


def check_same_conditions(a, b):
    key = lambda rs: [(r["set"], r["id"], r["text"], r["frames"]) for r in rs]  # noqa: E731
    if key(a) != key(b):
        raise RuntimeError("base / adapted utterance index differs: features come from different runs")


class Feats:
    """All frames of one (variant, layer, split) as a single fp16 device tensor + per-utterance (start, length)."""

    def __init__(self, rows, variant, layer, device):
        chunks, self.span, pos = [], [], 0
        cache = {}
        for r in rows:
            key = r["folder"]
            if key not in cache:
                cache[key] = np.load(key / (layer.replace("-", "_") + ".npy"), mmap_mode="r")
            chunks.append(torch.from_numpy(np.ascontiguousarray(cache[key][r["start"] : r["start"] + r["frames"]])))
            self.span.append((pos, r["frames"]))
            pos += r["frames"]
        self.x = torch.cat(chunks).to(device)
        if self.x.shape[1] != dim_of(layer):
            raise RuntimeError(f"{layer}: dim {self.x.shape[1]}")

    def batch(self, idx):
        parts = [self.x[self.span[i][0] : self.span[i][0] + self.span[i][1]] for i in idx]
        lens = torch.tensor([p.shape[0] for p in parts], device=self.x.device)
        return nn.utils.rnn.pad_sequence(parts, batch_first=True).float(), lens


class LinearCTC(nn.Module):
    """Parameter-free LayerNorm + one linear layer, identical recipe for every state."""

    def __init__(self, dim, vocab):
        super().__init__()
        self.linear = nn.Linear(dim, vocab)

    def forward(self, x):
        return self.linear(F.layer_norm(x, (x.shape[-1],)))


def ctc(head, x, lens, targets):
    logp = head(x).log_softmax(-1).transpose(0, 1)
    flat = torch.tensor([t for ids in targets for t in ids], dtype=torch.long, device=x.device)
    tl = torch.tensor([len(ids) for ids in targets], dtype=torch.long, device=x.device)
    return F.ctc_loss(logp, flat, lens, tl, blank=0, reduction="mean", zero_infinity=False)


def lr_at(step, total, base):
    warm = max(1, int(0.05 * total))
    return base * (step + 1) / warm if step < warm else base * 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, total - warm)))


@torch.inference_mode()
def evaluate(head, feats, rows, tok, bs, device):
    """-> mean CTC loss and greedy hypotheses (same order as rows)."""
    hyps, losses, weights = [None] * len(rows), 0.0, 0
    order = sorted(range(len(rows)), key=lambda i: rows[i]["frames"])
    for b in range(0, len(order), bs):
        idx = order[b : b + bs]
        x, lens = feats.batch(idx)
        losses += ctc(head, x, lens, [rows[i]["ids"] for i in idx]).item() * len(idx)
        weights += len(idx)
        for i, h in zip(idx, greedy(head(x), lens.cpu(), tok)):
            hyps[i] = h
    return losses / weights, hyps


def score(rows, hyps):
    """Per-utterance (errors, reference units) under each set's metric -> arrays aligned with rows."""
    return np.array([COUNTS[METRIC[r["set"]]](r["text"], h) for r, h in zip(rows, hyps)], dtype=np.int64)


def train_one(variant, layer, train, dev, tok, args, seed, lr, device, test=None):
    ftr = Feats(train, variant, layer, device)
    fdv = Feats(dev, variant, layer, device)
    torch.manual_seed(seed)
    head = LinearCTC(dim_of(layer), len(tok.itos)).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=0.01)
    steps_per_epoch = math.ceil(len(train) / args.batch_size)
    total = steps_per_epoch * args.epochs
    step, history, t0 = 0, [], time.time()
    for epoch in range(args.epochs):
        order = list(range(len(train)))
        random.Random(seed * 1000 + epoch).shuffle(order)
        head.train()
        run = 0.0
        for b in range(0, len(order), args.batch_size):
            idx = order[b : b + args.batch_size]
            x, lens = ftr.batch(idx)
            for g in opt.param_groups:
                g["lr"] = lr_at(step, total, lr)
            loss = ctc(head, x, lens, [train[i]["ids"] for i in idx])
            if not torch.isfinite(loss).item():
                raise RuntimeError(f"Nonfinite CTC loss {variant}/{layer} step {step}")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            run += loss.item()
            step += 1
        if (epoch + 1) % args.eval_every == 0 or epoch + 1 == args.epochs:
            head.eval()
            history.append(
                {
                    "epoch": epoch + 1,
                    "train_ctc": run / steps_per_epoch,
                    "dev_ctc": evaluate(head, fdv, dev, tok, args.batch_size, device)[0],
                }
            )
    head.eval()
    out = {
        "variant": variant,
        "layer": layer,
        "seed": seed,
        "lr": lr,
        "updates": step,
        "head_parameters": sum(p.numel() for p in head.parameters()),
        "history": history,
        "dev_ctc": history[-1]["dev_ctc"],
        "seconds": time.time() - t0,
    }
    scores = None
    if test is not None:
        fte = Feats(test, variant, layer, device)
        loss, hyps = evaluate(head, fte, test, tok, args.batch_size, device)
        out["test_ctc"] = loss
        scores = score(test, hyps)
        out["test"] = summarise(test, scores)
    del ftr, fdv
    return out, scores


def summarise(rows, scores):
    def agg(mask, metric):
        e, n = scores[mask, 0].sum(), scores[mask, 1].sum()
        return {"metric": metric, "value": float(e / max(n, 1)), "errors": int(e), "ref_units": int(n), "utts": int(mask.sum())}

    out = {}
    for name in NAMES:
        m = np.array([r["set"] == name for r in rows])
        out[name] = agg(m, METRIC[name])
        if name == "ASCEND":
            for lang in ("mixed", "zh", "en"):
                sub = m & np.array([r.get("language") == lang for r in rows])
                if sub.any():
                    out[f"ASCEND-{lang}"] = agg(sub, "MER")
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--features", type=Path, default=Path("research/encoder-depth-m4/runs/features"))
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--eval-every", type=int, default=5)
    p.add_argument("--seeds", type=int, nargs="+", default=[1337, 1338, 1339])
    p.add_argument("--lr", type=float, help="skip the dev pilot and use this learning rate")
    p.add_argument("--pilot-lrs", type=float, nargs="+", default=[1e-3, 3e-3, 1e-2])
    p.add_argument("--pilot-epochs", type=int, default=6)
    p.add_argument("--layers", nargs="+", default=list(FEATURES))
    p.add_argument("--out", type=Path, default=Path("research/encoder-depth-m4/runs/probes"))
    a = p.parse_args()
    result = {
        "phase": "linear_probe",
        "status": "BLOCKED",
        "environment": environment(),
        "args": {k: str(v) for k, v in vars(a).items()},
        "scope": "frozen_cached_features_linear_CTC_no_reader",
        "pilot": [],
        "results": [],
    }
    try:
        device = require_mps()
        ext = json.loads((a.features / "extract_results.json").read_text(encoding="utf-8"))
        if ext.get("status") != "PASS" or ext.get("limit"):
            raise RuntimeError("features are missing, incomplete or a --limit smoke run")
        release, origin = resolve_release()
        if origin["hf_revision"] != ext["origin"]["hf_revision"]:
            raise RuntimeError("probe release differs from extraction release")
        result["origin"] = origin
        tok = read_tokenizer(release)
        splits = {}
        for split in ("train", "dev", "test"):
            rows = {v: load_split(a.features, v, split, tok) for v in VARIANTS}
            check_same_conditions(rows["base"], rows["adapted"])
            kept = {v: eligible(rows[v])[0] for v in VARIANTS}
            splits[split] = kept
            result.setdefault("counts", {})[split] = {
                "total": len(rows["base"]),
                "eligible": len(kept["base"]),
                "ctc_infeasible_or_empty": len(rows["base"]) - len(kept["base"]),
            }
        out_errors = {}
        lr = a.lr
        if lr is None:
            args_p = argparse.Namespace(**{**vars(a), "epochs": a.pilot_epochs})
            for cand in a.pilot_lrs:
                dev_losses = []
                for variant in VARIANTS:
                    for layer in a.layers:
                        r, _ = train_one(
                            variant, layer, splits["train"][variant], splits["dev"][variant], tok, args_p, a.seeds[0], cand, device
                        )
                        dev_losses.append(r["dev_ctc"])
                result["pilot"].append({"lr": cand, "mean_dev_ctc": float(np.mean(dev_losses)), "dev_ctc": dev_losses})
                print(f"pilot lr={cand}: mean dev CTC {np.mean(dev_losses):.4f}", flush=True)
                write_json(a.out / "results.json", result)
            lr = min(result["pilot"], key=lambda r: r["mean_dev_ctc"])["lr"]
        result["selected_lr"] = lr
        for seed in a.seeds:
            for variant in VARIANTS:
                for layer in a.layers:
                    r, sc = train_one(
                        variant,
                        layer,
                        splits["train"][variant],
                        splits["dev"][variant],
                        tok,
                        a,
                        seed,
                        lr,
                        device,
                        test=splits["test"][variant],
                    )
                    result["results"].append(r)
                    out_errors[f"{variant}|{layer}|{seed}"] = sc
                    print(
                        f"{variant}/{layer} seed {seed}: "
                        + " ".join(f"{k}={v['value']:.3f}" for k, v in r["test"].items())
                        + f" ({r['seconds']:.0f}s)",
                        flush=True,
                    )
                    write_json(a.out / "results.json", result)
        np.savez_compressed(a.out / "test_errors.npz", **out_errors)
        (a.out / "test_rows.json").write_text(
            json.dumps(
                [{k: r[k] for k in ("set", "id", "frames", "speaker", "language", "seconds", "sha256")} for r in splits["test"]["adapted"]],
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        result["status"] = "PASS"
        result["known_limits"] = [
            "frozen linear probe, not the 22.2M GRU reader",
            "L12-Proj head has a different size",
            "LoRA training data of the release is unknown",
        ]
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()
    write_json(a.out / "results.json", result)
    print(f"{result['status']}: {a.out / 'results.json'}")
    if result["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
