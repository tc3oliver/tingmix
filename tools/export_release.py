#!/usr/bin/env python3
"""TingMix-Large training export (.pt, pickle) -> release directory (safetensors, no pickle at load time):

  python tools/export_release.py EXPORT.pt OUT_DIR [--note "..."]

OUT_DIR = Embed2ASR release layout (config.json, model.safetensors = reader, units.json, tokenizer_bpe.model) +
tingmix.json (format, encoder delta config) + encoder_delta.safetensors (LoRA / top-layer weights only; the base
encoder is NOT included and is downloaded from its official source). Load with tingmix's `load_model(OUT_DIR)`.
Licence review is a separate, manual step: this tool never uploads anything."""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from embed2asr.pipeline import save_release
from safetensors.torch import save_file


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("export", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--note", default="")
    p.add_argument("--att-weight", type=float, default=1.0, help="default attention rescoring weight (export with a decoder)")
    p.add_argument("--att-mu", type=float, default=0.0)
    a = p.parse_args(argv)
    ck = torch.load(a.export, map_location="cpu", weights_only=False)
    save_release(a.out, ck["reader"], ck["reader_cfg"], ck["itos"], ck["tok"].get("sp"))
    save_file({k: v.contiguous() for k, v in ck["enc"].items()}, str(a.out / "encoder_delta.safetensors"), metadata={"format": "pt"})
    meta = {
        "format": "tingmix-large",
        "format_version": 1,
        "encoder_delta": ck["enc_cfg"],
        "encoder_delta_file": "encoder_delta.safetensors",
        "step": ck.get("step"),
        "commit": ck.get("commit"),
        "note": a.note,
    }
    if "att" in ck:  # attention decoder for N-best rescoring (models/large/attention.py)
        save_file(
            {k: v.contiguous() for k, v in ck["att"].items()}, str(a.out / "attention_decoder.safetensors"), metadata={"format": "pt"}
        )
        meta |= {
            "attention_decoder": ck["att_cfg"],
            "attention_decoder_file": "attention_decoder.safetensors",
            "rescore": {"att_weight": a.att_weight, "att_mu": a.att_mu, "nbest": 8},
        }
    (a.out / "tingmix.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    sums = {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(a.out.iterdir()) if f.is_file() and f.name != "SHA256SUMS"}
    (a.out / "SHA256SUMS").write_text("".join(f"{h}  {n}\n" for n, h in sums.items()))
    print(f"wrote {a.out}: " + ", ".join(sums))


if __name__ == "__main__":
    main()
