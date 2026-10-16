#!/usr/bin/env python3
"""Average TingMix-Large exports (reader + encoder delta) of ONE run: python tools/avg_ckpt.py OUT.pt IN1.pt IN2.pt ...

Uniform mean of every floating-point tensor; vocabulary, configs and tokenizer must be identical (checked).
LoRA: A and B are averaged separately, so the averaged update is not the mean of the B@A products; use exports of one
run near the end of its schedule, where they are close."""

import sys

import torch


def main(out, paths):
    cks = [torch.load(p, map_location="cpu", weights_only=False) for p in paths]
    ref = cks[0]
    for c, p in zip(cks[1:], paths[1:]):
        for k in ("itos", "reader_cfg", "enc_cfg"):
            if c[k] != ref[k]:
                raise SystemExit(f"{p}: {k} differs from {paths[0]}")
    avg = dict(ref)
    for part in ("reader", "enc"):
        avg[part] = {
            n: torch.stack([c[part][n].float() for c in cks]).mean(0).to(t.dtype) if t.is_floating_point() else t
            for n, t in ref[part].items()
        }
    avg.update(step=[c.get("step") for c in cks], metrics=None, averaged_from=list(paths))
    torch.save(avg, out)
    print(f"averaged {len(cks)} exports -> {out}")


if __name__ == "__main__":
    if len(sys.argv) < 4:
        raise SystemExit(__doc__)
    main(sys.argv[1], sys.argv[2:])
