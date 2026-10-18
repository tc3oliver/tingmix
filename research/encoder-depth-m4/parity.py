"""Phase 0: verify published LoRA and parity with the official inference path on MPS."""
from __future__ import annotations

import argparse
import traceback
from pathlib import Path

import numpy as np
import torch
from common import environment, load_adapted, mel_input, stages, sync, write_json


def run(wav, model):
    asr = model.asr
    tower = asr.encoder.tower
    mel, mask = mel_input(wav, asr.encoder.feature_extractor)
    with torch.inference_mode():
        reference = tower(mel, mask, return_dict=True)
        h, lens = stages(tower, mel, mask)
        official_feat, official_len = asr.encoder([wav])
        custom = h["L12-Proj"]
        if not torch.equal(lens.cpu(), official_len.cpu()):
            raise AssertionError("Frame lengths differ from official pipeline")
        if custom.shape != reference.last_hidden_state.shape:
            raise AssertionError("Projected hidden shape differs from full forward")
        valid = torch.arange(custom.shape[1], device=custom.device)[None] < lens[:, None]
        a, b = custom.float()[valid], reference.last_hidden_state.float()[valid]
        rel = (a - b).norm() / b.norm().clamp_min(1e-9)
        ref_rel = ((official_feat.float()[valid] - b).norm() / b.norm().clamp_min(1e-9))
        if rel.item() > 0.003 or ref_rel.item() > 0.003:
            raise AssertionError(f"Full/split forward parity failed: {rel.item()} / {ref_rel.item()}")
        logits = asr.reader(custom.float(), lens)
        official_logits = asr.reader(official_feat.float(), official_len)
        if not torch.isfinite(logits).all().item():
            raise AssertionError("Nonfinite CTC logits")
        if not torch.allclose(logits, official_logits, rtol=0.01, atol=0.03):
            raise AssertionError("CTC logits mismatch")
        greedy = asr.decode(logits, lens, asr.tok)[0]
        official_greedy = asr.transcribe(wav, sr=16000).text
        if greedy != official_greedy:
            raise AssertionError(f"Greedy transcript mismatch: {greedy!r} != {official_greedy!r}")
        sync()
    return {
        "status": "PASS", "feature_shape": list(custom.shape), "lengths": lens.cpu().tolist(),
        "hidden_shapes": {k: list(v.shape) for k, v in h.items()},
        "logit_shape": list(logits.shape), "relative_l2_split": rel.item(),
        "relative_l2_official": ref_rel.item(), "greedy": greedy,
        "transcript_match": True, "nan_inf": False,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--audio", help="a legally obtained, fixed 16-kHz speech sample; else synthetic diagnostic only")
    p.add_argument("--revision", help="pin TingMix-Large HF commit; latest resolved to immutable SHA if omitted")
    p.add_argument("--out", default="research/encoder-depth-m4/runs/parity.json")
    a = p.parse_args()
    result = {"phase": "parity", "environment": environment(), "status": "BLOCKED"}
    try:
        from embed2asr.audio import load_audio
        if a.audio:
            wav = load_audio(a.audio)
            result["audio"] = {"source": str(Path(a.audio).resolve()), "kind": "speech"}
        else:
            t = np.arange(3 * 16000, dtype=np.float32) / 16000
            wav = (0.05 * np.sin(2 * np.pi * 220 * t) +
                   0.01 * np.random.default_rng(17).standard_normal(len(t))).astype(np.float32)
            result["audio"] = {"kind": "synthetic", "seconds": 3, "seed": 17}
        model, origin = load_adapted(a.revision)
        result["origin"] = origin
        result.update(run(wav, model))
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()
    write_json(a.out, result)
    print(f"{result['status']}: {a.out}")
    if result["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
