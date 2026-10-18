#!/usr/bin/env python3
"""Attention rescoring of CTC N-best lists (DEV sets only): adds field "att" = log P_att(hyp + eos | audio) to every hypothesis.

  python -m evaluation.att_rescore --ckpt runs/X/best.pt --nbest nbest.json --set cv=MANIFEST:cer ... --out scored.json
  python -m evaluation.nbest_llm tune --scored scored.json --field att

--set must repeat the `lm_tune dump` specs that produced the logits (same rows, same order). The encoder + Reader come
from the checkpoint (its LoRA delta and Reader = the CTC model the N-best came from)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--nbest", required=True)
    p.add_argument("--set", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda")
    a = p.parse_args(argv)

    import torch
    from embed2asr.audio import load_audio

    from evaluation.lm_tune import read_rows
    from tingmix.core.registry import load_model
    from tingmix.models.large.attention import AttDecoder, reader_states

    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    cfg = dict(ck["att_cfg"])
    mem = cfg.pop("mem")
    dec = AttDecoder(**cfg)
    dec.load_state_dict(ck["att"])
    dec.to(a.device).eval()
    asr = load_model(a.ckpt, device=a.device).asr
    data = json.loads(Path(a.nbest).read_text())
    with torch.inference_mode():
        for spec in a.set:
            name, _, rows = read_rows(spec)
            nb = data[name]["rows"]
            assert len(nb) == len(rows) and all(r["text"] == n["text"] for r, n in zip(rows, nb)), f"{name}: rows differ from the N-best"
            for r, n in zip(rows, nb):
                feats, lens = asr.encoder([load_audio(r["audio"])])
                x = feats.to(asr.device).float()
                _, h = reader_states(asr.reader, x, lens)
                m = (h if mem == "reader" else x)[0].float()
                for hyp, s in zip(n["nbest"], dec.score(m, int(lens[0]), [hy["ids"] for hy in n["nbest"]])):
                    hyp["att"] = s
            print(name, len(rows), flush=True)
    Path(a.out).write_text(json.dumps(data, ensure_ascii=False))


if __name__ == "__main__":
    main()
