#!/usr/bin/env python3
"""Train TingMix-Large: EmbeddingGemma 2 audio tower (frozen / LoRA / partial top layers) + GRU-CTC reader, from audio.

  python -m training.large.train --config run.yaml [--key value ...]      (CLI overrides YAML)

Warm start from an Embed2ASR reader checkpoint (--init, same vocabulary). The encoder runs from log-mel every step, so
LoRA gradients reach every adapted layer (no cached intermediate states).

Fault tolerance (driven by tools/tingmix_train.py, but the trainer is self-contained):
  * resume.pt / resume.prev.pt (atomic) every --save-every-min minutes and at every eval: model delta, optimizer,
    scheduler, step, epoch, batch position, RNG states, config, code/data hashes, accumulated train time, best metrics;
  * --resume auto continues exactly (data order is a function of seed/epoch/batch index), refuses incompatible
    config/code/data instead of silently starting a different experiment;
  * a run-dir flock prevents two trainers on one run;
  * status.json: DONE / FAILED (permanent: OOM, config mismatch, bad data) -> exit 0 so the supervisor does not restart;
    any other crash exits non-zero and is restarted by the supervisor, resuming from the last checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import signal
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from tingmix.models.large import encoder as E
from training.large import checkpoint as C
from training.large import dataset as D

MUTABLE = {"max_hours", "workers", "save_every_min", "eval_every", "resume", "allow_code_change", "config", "total_steps_cap"}


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path)
    p.add_argument("--out", type=Path)
    p.add_argument("--mode", choices=["frozen", "lora", "partial"], default="lora")
    p.add_argument("--init", help="Embed2ASR reader checkpoint (.pt with model/itos/cfg/tok)")
    p.add_argument("--train", nargs="+", help="PATH:HOURS[:nd] per pool (nd = drop rows with digits)")
    p.add_argument(
        "--dev", nargs="*", default=[], help="NAME=PATH:METRIC[:nd] (cer|wer|mer|mixmer; nd = drop digit rows); all are selection keys"
    )
    p.add_argument("--map-prefix", nargs="*", default=[], help="OLD=NEW audio path rewrites")
    p.add_argument(
        "--features",
        choices=["audio", "cache"],
        default="audio",
        help="cache: read cached encoder states instead of audio (frozen: final 1536-d; partial: layer-10 1024-d)",
    )
    p.add_argument("--feat-root", help="dir that relative cache paths in the manifests are relative to")
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=float, default=32.0)
    p.add_argument("--lora-dropout", type=float, default=0.0)
    p.add_argument("--lora-layers", default="", help="comma list of layer indices (default all 12)")
    p.add_argument("--start", type=int, default=10, help="partial: first trainable layer")
    p.add_argument("--lr-reader", type=float, default=1e-4)
    p.add_argument("--lr-enc", type=float, default=1e-4)
    p.add_argument("--wd", type=float, default=0.01)
    p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--total-steps", type=int, default=20000, help="cosine schedule length")
    p.add_argument("--total-steps-cap", type=int, default=0, help="stop early at this step (pilots), schedule unchanged")
    p.add_argument("--max-seconds", type=float, default=320.0, help="padded audio seconds per batch")
    p.add_argument("--max-bs", type=int, default=48)
    p.add_argument("--time-mask", type=float, default=0.1)
    p.add_argument("--grad-clip", type=float, default=5.0)
    p.add_argument("--eval-every", type=int, default=1000)
    p.add_argument("--save-every-min", type=float, default=15.0)
    p.add_argument("--max-hours", type=float, default=0.0, help="train wall-time budget summed over resumes (0 = none)")
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", choices=["auto", "no"], default="auto")
    p.add_argument("--allow-code-change", action="store_true")
    p.add_argument("--device", default="cuda")
    pre, _ = p.parse_known_args(argv)
    if pre.config:
        import yaml

        cfg = yaml.safe_load(pre.config.read_text()) or {}
        bad = set(cfg) - {a.dest for a in p._actions}
        if bad:
            p.error(f"unknown config keys {sorted(bad)}")
        p.set_defaults(**cfg)
    a = p.parse_args(argv)
    a.out = Path(a.out)
    return a


def sha_files(paths):
    h = hashlib.sha256()
    for x in sorted(set(paths)):
        h.update(x.encode())
        with open(x, "rb") as f:
            for b in iter(lambda: f.read(1 << 22), b""):
                h.update(b)
    return h.hexdigest()


def code_hash():
    root = Path(__file__).resolve().parents[2]
    files = sorted([*root.glob("training/large/*.py"), root / "src/tingmix/models/large/encoder.py"])
    return hashlib.sha256(b"".join(f.read_bytes() for f in files)).hexdigest()[:16]


class Permanent(Exception):
    """Error that a restart cannot fix."""


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
        print(f"PERMANENT FAILURE (OOM): {e}", flush=True)
        return 0


def run(a):
    from embed2asr.ctc import collapse_ctc, ctc_min_len
    from embed2asr.metrics import corpus
    from embed2asr.reader import Reader
    from embed2asr.tokenizer import Tok

    dev = torch.device(a.device)
    C.set_status(a.out, "RUNNING", "starting")
    pm = [tuple(x.split("=", 1)) for x in a.map_prefix]
    init = torch.load(a.init, map_location="cpu", weights_only=False)
    if "reader" in init:  # TingMix export (reader + encoder delta) -> same keys as an Embed2ASR reader checkpoint
        init = {**init, "model": init["reader"], "cfg": init["reader_cfg"]}
    tok = Tok(init["itos"], init["tok"].get("sp"))

    pools, paths = [], [a.init]
    feat_root = a.feat_root if a.features == "cache" else None
    if a.features == "cache" and a.mode == "lora":
        raise Permanent("LoRA adapts every layer: it must run the encoder from audio (features: audio)")
    for spec in a.train:
        parts = spec.split(":")
        path, hours, nd = parts[0], float(parts[1]), len(parts) > 2 and parts[2] == "nd"
        rows = D.load_rows(path, pm, nodigit=nd, feat_root=feat_root)
        n0 = len(rows)
        rows = [r for r in rows if 0 < ctc_min_len(tok.encode(r["text"])) <= r["frames"]]
        if a.features == "cache" and any("bin" not in r for r in rows[:100]):
            raise Permanent(f"{path}: rows have no feature-cache reference")
        print(f"pool {path}: {len(rows)}/{n0} CTC-feasible, {sum(r['seconds'] for r in rows) / 3600:.1f} h, {hours} h/epoch", flush=True)
        if not rows:
            raise Permanent(f"pool {path} has no usable rows")
        pools.append((rows, hours))
        paths.append(path)
    devsets = {}
    for spec in a.dev:
        name, rest = spec.split("=", 1)
        nd = rest.endswith(":nd")
        path, metric = rest.removesuffix(":nd").rsplit(":", 1)
        devsets[name] = (D.load_rows(path, pm, nodigit=nd, feat_root=feat_root), metric)
        paths.append(path)
    data_hash = sha_files(paths)
    cfg_now = {k: v for k, v in vars(a).items() if k not in MUTABLE}
    cfg_now = json.loads(json.dumps(cfg_now, default=str))
    (a.out / "config.json").write_text(json.dumps({**cfg_now, "data_hash": data_hash, "code": code_hash()}, indent=1))

    fe = E.load_feature_extractor()
    tower = E.load_tower().to(dev)
    lora_layers = [int(x) for x in str(a.lora_layers).split(",") if x != ""] or None
    if a.mode == "lora":
        enc_params = E.apply_lora(tower, a.lora_r, a.lora_alpha, a.lora_dropout, lora_layers)
    elif a.mode == "partial":
        enc_params = E.set_partial(tower, a.start)
    else:
        enc_params = []
    enc_cfg = {"mode": a.mode, "lora_r": a.lora_r, "lora_alpha": a.lora_alpha, "lora_layers": lora_layers, "start": a.start}
    if init.get("enc"):  # warm-start the encoder adaptation too; it must be the same kind of delta
        ic = init["enc_cfg"]
        if ic["mode"] != a.mode or (
            a.mode == "lora" and (ic["lora_r"], ic["lora_alpha"], ic["lora_layers"]) != (a.lora_r, a.lora_alpha, lora_layers)
        ):
            raise Permanent(f"--init encoder delta {ic} does not match mode {a.mode} r {a.lora_r} layers {lora_layers}")
        own = dict(tower.named_parameters())
        with torch.no_grad():
            for n, t in init["enc"].items():
                own[n].copy_(t.to(own[n].dtype))
        print(f"[init] encoder delta from {a.init}: {len(init['enc'])} tensors ({ic})", flush=True)
    reader = Reader(**init["cfg"]).to(dev)
    reader.load_state_dict(init["model"])
    reader.train()
    n_enc = sum(p.numel() for p in enc_params)
    print(
        f"mode {a.mode}: encoder trainable {n_enc / 1e6:.2f}M, reader {sum(p.numel() for p in reader.parameters()) / 1e6:.2f}M", flush=True
    )

    groups = [{"params": list(reader.parameters()), "lr": a.lr_reader}]
    if enc_params:
        groups.append({"params": enc_params, "lr": a.lr_enc})
    opt = torch.optim.AdamW(groups, weight_decay=a.wd)

    def lam(s):
        return min(1.0, (s + 1) / a.warmup) * (0.5 * (1 + np.cos(np.pi * min(s / a.total_steps, 1.0))) * 0.95 + 0.05)

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lam)
    gen = torch.Generator().manual_seed(a.seed)
    st = {"step": 0, "epoch": 1, "batch": 0, "train_s": 0.0, "best": None, "base": None, "best_step": None, "bad_rows": 0}

    if a.resume == "auto":
        r, rp = C.load_resume(a.out)
        if r is not None:
            probs = []
            if r["config"] != cfg_now:
                diff = sorted(k for k in set(r["config"]) | set(cfg_now) if r["config"].get(k) != cfg_now.get(k))
                probs.append(f"config differs: {diff}")
            if r["data_hash"] != data_hash:
                probs.append("data hash differs")
            if r["itos"] != tok.itos:
                probs.append("tokenizer differs")
            if r["code"] != code_hash() and not a.allow_code_change:
                probs.append(f"code {r['code']} != {code_hash()} (--allow-code-change to accept)")
            if probs:
                raise Permanent("cannot resume: " + "; ".join(probs))
            reader.load_state_dict(r["reader"])
            own = dict(tower.named_parameters())
            with torch.no_grad():
                for n, t in r["enc"].items():
                    own[n].copy_(t.to(own[n].dtype))
            opt.load_state_dict(r["opt"])
            sched.load_state_dict(r["sched"])
            st = r["state"]
            random.setstate(r["rng"]["py"])
            np.random.set_state(r["rng"]["np"])
            torch.set_rng_state(r["rng"]["torch"])
            if r["rng"]["cuda"] is not None and dev.type == "cuda":
                torch.cuda.set_rng_state(r["rng"]["cuda"])
            gen.set_state(r["rng"]["gen"])
            print(
                f"[resume] {rp}: step {st['step']} epoch {st['epoch']} batch {st['batch']} train {st['train_s'] / 3600:.2f} h", flush=True
            )

    def snapshot():
        return {
            "reader": reader.state_dict(),
            "enc": E.trainable_state(tower),
            "enc_cfg": enc_cfg,
            "reader_cfg": init["cfg"],
            "itos": tok.itos,
            "tok": init["tok"],
            "opt": opt.state_dict(),
            "sched": sched.state_dict(),
            "state": dict(st),
            "rng": {
                "py": random.getstate(),
                "np": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state() if dev.type == "cuda" else None,
                "gen": gen.get_state(),
            },
            "config": cfg_now,
            "data_hash": data_hash,
            "code": code_hash(),
            "commit": os.environ.get("TINGMIX_COMMIT", ""),
        }

    def export(name, extra=None):
        """Inference checkpoint: reader + encoder delta + vocab (no optimizer)."""
        C.save_atomic(
            {
                "reader": reader.state_dict(),
                "reader_cfg": init["cfg"],
                "enc": E.trainable_state(tower),
                "enc_cfg": enc_cfg,
                "itos": tok.itos,
                "tok": init["tok"],
                "step": st["step"],
                "metrics": extra,
                "commit": os.environ.get("TINGMIX_COMMIT", ""),
                "data_hash": data_hash,
            },
            a.out / name,
        )

    # ---- dev: log-mel once in RAM ----
    t0 = time.time()
    dev_mel = {}
    for name, (rows, metric) in devsets.items():
        order = sorted(range(len(rows)), key=lambda i: rows[i]["seconds"])
        bs = [order[i : i + 32] for i in range(0, len(order), 32)]
        dev_mel[name] = (rows, metric, bs, list(D.loader(rows, bs, fe, tok.encode, a.workers, 0, a.features)))
    print(f"dev mel ready in {time.time() - t0:.0f}s: {', '.join(f'{k}={len(v[0])}' for k, v in dev_mel.items())}", flush=True)

    def forward_enc(x, m, mode):
        """-> (features [B,T,1536] fp32, lengths [B] on device) from audio or from cached states."""
        if a.features == "audio":
            return E.encode(tower, x.to(dev, non_blocking=True), m.to(dev, non_blocking=True), mode, a.start)
        lens = m.to(dev)
        x = x.to(dev, non_blocking=True)
        if x.shape[-1] == E.FEATURE_DIM:
            if mode != "frozen":
                raise Permanent("final-feature cache can only train a frozen encoder")
            return x.float(), lens
        with torch.set_grad_enabled(mode == "partial" and torch.is_grad_enabled()):
            y = E.top(tower, x, lens, a.start).float()
        return y * (torch.arange(x.shape[1], device=dev)[None] < lens[:, None])[..., None], lens

    @torch.no_grad()
    def evaluate():
        reader.eval()
        out = {}
        for name, (rows, metric, bs, batches) in dev_mel.items():
            pairs, mixed = [], []
            for b, (mel, mask, _, _, texts, langs) in zip(bs, batches):
                x, lens = forward_enc(mel, mask, a.mode)
                ids = reader(x, lens.cpu()).argmax(-1).cpu().tolist()
                for seq, L, ref, lg in zip(ids, lens.tolist(), texts, langs):
                    hyp = tok.decode(collapse_ctc(seq[:L]))
                    pairs.append((ref, hyp))
                    if lg in ("mix", "zh-en"):
                        mixed.append((ref, hyp))
            m = metric.upper()
            if m == "MIXMER":
                out[name] = corpus(mixed, "MER")["value"]
                out[name + "_all"] = corpus(pairs, "MER")["value"]
            else:
                out[name] = corpus(pairs, m)["value"]
        reader.train()
        return out

    sampler = D.RotateSampler(pools, a.seed)
    log = (a.out / "train_log.jsonl").open("a", encoding="utf-8")

    def record(rec):
        log.write(json.dumps(rec, ensure_ascii=False) + "\n")
        log.flush()
        print(json.dumps(rec, ensure_ascii=False), flush=True)

    def do_eval():
        t = time.time()
        m = evaluate()
        if st["base"] is None:
            st["base"] = m
        keys = [k for k in devsets]
        score = float(np.mean([m[k] / max(st["base"][k], 1e-9) for k in keys]))
        rec = {"type": "eval", "step": st["step"], "epoch": st["epoch"], **{k: round(v, 5) for k, v in m.items()}, "score": round(score, 5)}
        rec["eval_s"] = round(time.time() - t, 1)
        if st["best"] is None or score < st["best"]:
            st["best"], st["best_step"] = score, st["step"]
            export("best.pt", rec)
            rec["new_best"] = True
        if st["step"]:  # every eval: candidates for checkpoint averaging (tools/avg_ckpt.py)
            export(f"eval_{st['step']:06d}.pt", rec)
        record(rec)

    if st["base"] is None:
        do_eval()  # step 0 = the warm-start model under this exact pipeline (reference for selection)
        C.save_resume(snapshot(), a.out)

    stop_req = []
    signal.signal(signal.SIGTERM, lambda *_: stop_req.append(1))  # docker stop: checkpoint, then exit 0 (PAUSED)
    cap = a.total_steps_cap or a.total_steps
    budget = a.max_hours * 3600 if a.max_hours else float("inf")
    C.set_status(a.out, "RUNNING", f"step {st['step']}")
    last_save, t_seg = time.time(), time.time()
    agg = {"loss": 0.0, "n": 0, "inf": 0, "audio_s": 0.0, "wait": 0.0, "gn": 0.0, "gne": 0.0, "t0": time.time()}
    torch.cuda.reset_peak_memory_stats() if dev.type == "cuda" else None
    done = st["step"] >= cap or st["train_s"] >= budget
    while not done:
        rows = sampler.epoch(st["epoch"])
        batches = D.make_batches(rows, a.max_seconds, a.max_bs, random.Random(a.seed * 7919 + st["epoch"]))
        t_wait = time.time()
        for k, (mel, mask, tg, tl, _, _) in enumerate(
            D.loader(rows, batches, fe, tok.encode, a.workers, st["batch"], a.features), st["batch"]
        ):
            agg["wait"] += time.time() - t_wait
            x, lens = forward_enc(mel, mask, a.mode)
            x = D.time_mask(x, lens, a.time_mask, gen)
            logits = reader(x, lens.cpu())
            lp = logits.float().log_softmax(-1).transpose(0, 1)
            per = F.ctc_loss(lp, tg.to(dev), lens, tl.to(dev), blank=0, reduction="none", zero_infinity=False)
            fin = torch.isfinite(per)
            if not bool(fin.any()):
                raise Permanent(f"all CTC losses non-finite at step {st['step']}")
            loss = (per / tl.to(dev).clamp(min=1))[fin].mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            agg["gn"] += float(torch.nn.utils.clip_grad_norm_(reader.parameters(), a.grad_clip))
            if enc_params:
                agg["gne"] += float(torch.nn.utils.clip_grad_norm_(enc_params, a.grad_clip))
            opt.step()
            sched.step()
            st["step"] += 1
            st["batch"] = k + 1
            agg["loss"] += float(loss.detach())
            agg["n"] += 1
            agg["inf"] += int((~fin).sum())
            agg["audio_s"] += float(lens.sum()) / E.FRAME_RATE
            now = time.time()
            st["train_s"] += now - t_seg
            t_seg = now
            if not torch.isfinite(loss):
                raise Permanent(f"non-finite loss at step {st['step']}")
            if st["step"] % 50 == 0:
                n = agg["n"]
                record(
                    {
                        "type": "train",
                        "step": st["step"],
                        "epoch": st["epoch"],
                        "loss": round(agg["loss"] / n, 4),
                        "lr": [round(g, 8) for g in sched.get_last_lr()],
                        "gn": round(agg["gn"] / n, 3),
                        "gne": round(agg["gne"] / n, 3),
                        "inf": agg["inf"],
                        "audio_h_per_h": round(agg["audio_s"] / max(now - agg["t0"], 1e-9), 1),
                        "data_wait_s": round(agg["wait"], 1),
                        "max_alloc_gib": round(torch.cuda.max_memory_allocated() / 2**30, 1) if dev.type == "cuda" else None,
                        "train_h": round(st["train_s"] / 3600, 3),
                    }
                )
                agg = {"loss": 0.0, "n": 0, "inf": 0, "audio_s": 0.0, "wait": 0.0, "gn": 0.0, "gne": 0.0, "t0": now}
            if st["step"] % a.eval_every == 0:
                do_eval()
                t_seg = time.time()  # eval time is not training time
                C.save_resume(snapshot(), a.out)
                last_save = time.time()
            elif time.time() - last_save > a.save_every_min * 60:
                C.save_resume(snapshot(), a.out)
                last_save = time.time()
                C.set_status(a.out, "RUNNING", f"step {st['step']}")
            if stop_req:
                C.save_resume(snapshot(), a.out)
                C.set_status(a.out, "PAUSED", f"SIGTERM at step {st['step']}")
                print(f"PAUSED at step {st['step']} (SIGTERM), checkpoint saved", flush=True)
                return 0
            if st["step"] >= cap or st["train_s"] >= budget:
                done = True
                break
            t_wait = time.time()
        if not done:
            st["epoch"] += 1
            st["batch"] = 0
    if st["step"] % a.eval_every:
        do_eval()
    C.save_resume(snapshot(), a.out)
    export("last.pt")
    why = "step cap" if st["step"] >= cap else "time budget"
    C.set_status(a.out, "DONE", f"{why}: step {st['step']}, best score {st['best']} at step {st['best_step']}")
    print(f"DONE at step {st['step']} ({why}); best {st['best']} @ {st['best_step']}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
