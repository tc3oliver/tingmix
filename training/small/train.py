#!/usr/bin/env python3
"""Train TingMix-Small (natively streaming GRU-CTC on log-mel) from audio.

  python -m training.small.train --config run.yaml [--key value ...]

Same data format, sampler, fault tolerance and job manager as training/large/train.py: atomic resume.pt / resume.prev.pt,
exact data-order resume, run lock, SIGTERM -> checkpoint + PAUSED, status.json (permanent errors exit 0). The LR is a
cosine over accumulated TRAINING TIME (--hours), so a resumed run continues the same schedule. The best dev checkpoint
is exported as a release directory (<out>/release: small.json, model.safetensors, units.json, tokenizer_bpe.model)."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import signal
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from tingmix.models.small.model import SmallNet
from training.large import checkpoint as C
from training.large import dataset as D

MUTABLE = {"workers", "eval_every", "save_every_min", "resume", "config"}


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path)
    p.add_argument("--out", type=Path)
    p.add_argument("--vocab-from", help="TingMix/Embed2ASR checkpoint whose units + BPE are reused (keys itos, tok)")
    p.add_argument("--train", nargs="+", help="PATH:HOURS[:nd]")
    p.add_argument("--dev", nargs="*", default=[], help="NAME=PATH:METRIC[:nd]")
    p.add_argument("--d", type=int, default=512)
    p.add_argument("--hidden", type=int, default=640)
    p.add_argument("--layers", type=int, default=5)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=0.01)
    p.add_argument("--warmup", type=int, default=2000)
    p.add_argument("--hours", type=float, default=18.0, help="training-time budget = cosine length")
    p.add_argument("--max-seconds", type=float, default=600.0)
    p.add_argument("--max-bs", type=int, default=128)
    p.add_argument("--time-masks", type=int, default=6, help="SpecAugment: time masks per utterance (width <= 10 frames)")
    p.add_argument("--freq-masks", type=int, default=2, help="SpecAugment: frequency masks (width <= 15 bins)")
    p.add_argument("--eval-every", type=int, default=2000)
    p.add_argument("--save-every-min", type=float, default=15.0)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", choices=["auto", "no"], default="auto")
    p.add_argument("--device", default="cuda")
    pre, _ = p.parse_known_args(argv)
    if pre.config:
        import yaml

        cfg = yaml.safe_load(pre.config.read_text()) or {}
        bad = set(cfg) - {a.dest for a in p._actions} - {"allow_code_change"}
        if bad:
            p.error(f"unknown config keys {sorted(bad)}")
        cfg.pop("allow_code_change", None)
        p.set_defaults(**cfg)
    a = p.parse_args(argv)
    a.out = Path(a.out)
    return a


def spec_augment(mel, n_time, n_freq, gen):
    B, T, Fq = mel.shape
    mel = mel.clone()
    for b in range(B):
        for _ in range(n_freq):
            w = int(torch.randint(0, 16, (1,), generator=gen))
            f0 = int(torch.randint(0, max(1, Fq - w), (1,), generator=gen))
            mel[b, :, f0 : f0 + w] = mel[b].mean()
        for _ in range(n_time):
            w = int(torch.randint(0, 11, (1,), generator=gen))
            t0 = int(torch.randint(0, max(1, T - w), (1,), generator=gen))
            mel[b, t0 : t0 + w] = mel[b].mean()
    return mel


def forward(net, wav, lens, aug=None):
    """Like SmallNet.forward, with optional SpecAugment on the log-mel."""
    mel = net.mel(wav)
    if aug:
        mel = spec_augment(mel, *aug)
    T = mel.shape[1] - mel.shape[1] % 4
    h = net.frontend(mel[:, :T])
    fl = torch.tensor([max(0, (int(n) - 512) // 160 + 1) // 4 for n in lens], device=wav.device)
    y, _ = net.gru(net.drop(h))
    return net.out(net.drop(y)).float().log_softmax(-1), fl


class Permanent(Exception):
    pass


def main(argv=None):
    a = parse(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    lock = C.lock_run(a.out)
    if lock is None:
        print(f"another trainer holds {a.out}/.lock; exiting", flush=True)
        return 0
    try:
        return run(a)
    except Permanent as e:
        C.set_status(a.out, "FAILED", str(e), permanent=True)
        print(f"PERMANENT FAILURE: {e}", flush=True)
        return 0
    except torch.OutOfMemoryError as e:
        C.set_status(a.out, "FAILED", f"OOM: {str(e)[:300]}", permanent=True)
        return 0


def run(a):
    from embed2asr.ctc import collapse_ctc, ctc_min_len
    from embed2asr.metrics import corpus
    from embed2asr.tokenizer import Tok
    from safetensors.torch import save_file

    torch.multiprocessing.set_sharing_strategy("file_system")
    dev = torch.device(a.device)
    C.set_status(a.out, "RUNNING", "starting")
    vk = torch.load(a.vocab_from, map_location="cpu", weights_only=False)
    tok = Tok(vk["itos"], vk["tok"].get("sp"))
    pools, paths = [], [a.vocab_from]
    for spec in a.train:
        parts = spec.split(":")
        rows = D.load_rows(parts[0], nodigit=len(parts) > 2 and parts[2] == "nd")
        rows = [r for r in rows if 0 < ctc_min_len(tok.encode(r["text"])) <= r["frames"]]
        print(f"pool {parts[0]}: {len(rows)} rows, {parts[1]} h/epoch", flush=True)
        pools.append((rows, float(parts[1])))
        paths.append(parts[0])
    devsets = {}
    for spec in a.dev:
        name, rest = spec.split("=", 1)
        nd = rest.endswith(":nd")
        path, metric = rest.removesuffix(":nd").rsplit(":", 1)
        devsets[name] = (D.load_rows(path, nodigit=nd), metric)
        paths.append(path)
    h = hashlib.sha256()
    for x in sorted(set(paths)):
        h.update(Path(x).read_bytes())
    data_hash = h.hexdigest()
    cfg_now = json.loads(json.dumps({k: v for k, v in vars(a).items() if k not in MUTABLE}, default=str))

    torch.manual_seed(a.seed)
    net = SmallNet(len(tok.itos), a.d, a.hidden, a.layers, a.dropout).to(dev)
    print(f"TingMix-Small params {sum(p.numel() for p in net.parameters()) / 1e6:.2f}M", flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=a.wd)
    gen = torch.Generator().manual_seed(a.seed)
    st = {"step": 0, "epoch": 1, "batch": 0, "train_s": 0.0, "best": None, "best_step": None}
    budget = a.hours * 3600

    def lr_at():
        warm = min(1.0, (st["step"] + 1) / a.warmup)
        return a.lr * warm * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * min(st["train_s"] / budget, 1.0))))

    if a.resume == "auto":
        r, rp = C.load_resume(a.out)
        if r is not None:
            if r["config"] != cfg_now or r["data_hash"] != data_hash:
                raise Permanent("cannot resume: config or data differ")
            net.load_state_dict(r["net"])
            opt.load_state_dict(r["opt"])
            st = r["state"]
            random.setstate(r["rng"]["py"])
            np.random.set_state(r["rng"]["np"])
            torch.set_rng_state(r["rng"]["torch"])
            if r["rng"]["cuda"] is not None and dev.type == "cuda":
                torch.cuda.set_rng_state(r["rng"]["cuda"])
            gen.set_state(r["rng"]["gen"])
            print(f"[resume] {rp}: step {st['step']} train {st['train_s'] / 3600:.2f} h", flush=True)

    def snapshot():
        return {
            "net": net.state_dict(),
            "opt": opt.state_dict(),
            "state": dict(st),
            "config": cfg_now,
            "data_hash": data_hash,
            "rng": {
                "py": random.getstate(),
                "np": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state() if dev.type == "cuda" else None,
                "gen": gen.get_state(),
            },
        }

    def export(rec):
        rel = a.out / "release"
        rel.mkdir(exist_ok=True)
        save_file({k: v.detach().cpu().contiguous() for k, v in net.state_dict().items()}, str(rel / "model.safetensors.tmp"))
        Path(rel / "model.safetensors.tmp").replace(rel / "model.safetensors")
        (rel / "small.json").write_text(json.dumps(net.cfg))
        (rel / "units.json").write_text(json.dumps(tok.itos, ensure_ascii=False), encoding="utf-8")
        if tok.sp_bytes:
            (rel / "tokenizer_bpe.model").write_bytes(tok.sp_bytes)
        (rel / "metrics.json").write_text(json.dumps(rec, ensure_ascii=False))

    t0 = time.time()
    dev_data = {}
    for name, (rows, metric) in devsets.items():
        order = sorted(range(len(rows)), key=lambda i: rows[i]["seconds"])
        bs = [order[i : i + 64] for i in range(0, len(order), 64)]
        dev_data[name] = (metric, list(D.loader(rows, bs, None, tok.encode, a.workers, 0, "wave")))
    print(f"dev audio ready in {time.time() - t0:.0f}s", flush=True)

    @torch.no_grad()
    def evaluate():
        net.eval()
        out = {}
        for name, (metric, batches) in dev_data.items():
            pairs, mixed = [], []
            for wav, lens, _, _, texts, langs in batches:
                lp, fl = forward(net, wav.to(dev), lens)
                for row, n, ref, lg in zip(lp.argmax(-1).cpu().tolist(), fl.tolist(), texts, langs):
                    hyp = tok.decode(collapse_ctc(row[:n]))
                    pairs.append((ref, hyp))
                    if lg in ("mix", "zh-en"):
                        mixed.append((ref, hyp))
            if metric == "mixmer":
                out[name], out[name + "_all"] = corpus(mixed, "MER")["value"], corpus(pairs, "MER")["value"]
            else:
                out[name] = corpus(pairs, metric.upper())["value"]
        net.train()
        return out

    log = (a.out / "train_log.jsonl").open("a", encoding="utf-8")

    def record(rec):
        log.write(json.dumps(rec, ensure_ascii=False) + "\n")
        log.flush()
        print(json.dumps(rec, ensure_ascii=False), flush=True)

    def do_eval():
        m = evaluate()
        score = float(np.mean([m[k] for k in devsets]))  # from scratch: plain mean of the dev error rates
        rec = {
            "type": "eval",
            "step": st["step"],
            "train_h": round(st["train_s"] / 3600, 3),
            **{k: round(v, 5) for k, v in m.items()},
            "score": round(score, 5),
        }
        if st["best"] is None or score < st["best"]:
            st["best"], st["best_step"] = score, st["step"]
            export(rec)
            rec["new_best"] = True
        record(rec)

    stop_req = []
    signal.signal(signal.SIGTERM, lambda *_: stop_req.append(1))
    sampler = D.RotateSampler(pools, a.seed)
    net.train()
    C.set_status(a.out, "RUNNING", f"step {st['step']}")
    last_save, t_seg = time.time(), time.time()
    agg = {"loss": 0.0, "n": 0, "audio_s": 0.0, "t0": time.time()}
    while st["train_s"] < budget:
        rows = sampler.epoch(st["epoch"])
        batches = D.make_batches(rows, a.max_seconds, a.max_bs, random.Random(a.seed * 7919 + st["epoch"]))
        for k, (wav, lens, tg, tl, _, _) in enumerate(
            D.loader(rows, batches, None, tok.encode, a.workers, st["batch"], "wave"), st["batch"]
        ):
            for g in opt.param_groups:
                g["lr"] = lr_at()
            lp, fl = forward(net, wav.to(dev, non_blocking=True), lens, (a.time_masks, a.freq_masks, gen))
            per = F.ctc_loss(lp.transpose(0, 1), tg.to(dev), fl, tl.to(dev), blank=0, reduction="none", zero_infinity=False)
            fin = torch.isfinite(per)
            if not bool(fin.any()):
                raise Permanent(f"all CTC losses non-finite at step {st['step']}")
            loss = (per / tl.to(dev).clamp(min=1))[fin].mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()
            st["step"] += 1
            st["batch"] = k + 1
            agg["loss"] += float(loss.detach())
            agg["n"] += 1
            agg["audio_s"] += float(lens.sum()) / 16000
            now = time.time()
            st["train_s"] += now - t_seg
            t_seg = now
            if st["step"] % 100 == 0:
                record(
                    {
                        "type": "train",
                        "step": st["step"],
                        "epoch": st["epoch"],
                        "loss": round(agg["loss"] / agg["n"], 4),
                        "lr": round(lr_at(), 7),
                        "audio_h_per_h": round(agg["audio_s"] / max(now - agg["t0"], 1e-9), 1),
                        "train_h": round(st["train_s"] / 3600, 3),
                    }
                )
                agg = {"loss": 0.0, "n": 0, "audio_s": 0.0, "t0": now}
            if st["step"] % a.eval_every == 0:
                do_eval()
                t_seg = time.time()
                C.save_resume(snapshot(), a.out)
                last_save = time.time()
            elif time.time() - last_save > a.save_every_min * 60:
                C.save_resume(snapshot(), a.out)
                last_save = time.time()
                C.set_status(a.out, "RUNNING", f"step {st['step']}")
            if stop_req:
                C.save_resume(snapshot(), a.out)
                C.set_status(a.out, "PAUSED", f"SIGTERM at step {st['step']}")
                return 0
            if st["train_s"] >= budget:
                break
        else:
            st["epoch"] += 1
            st["batch"] = 0
    do_eval()
    C.save_resume(snapshot(), a.out)
    C.set_status(a.out, "DONE", f"time budget: step {st['step']}, best {st['best']} at step {st['best_step']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
