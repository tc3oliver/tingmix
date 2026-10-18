"""Phase 0: verify published LoRA and parity with the official inference path on MPS."""

from __future__ import annotations

import argparse
import traceback
from pathlib import Path

import numpy as np
import torch
from common import FEATURES, environment, load_adapted, load_base, mel_input, stages, sync, write_json


def rel_l2(a, b):
    return ((a.float().cpu() - b.float().cpu()).norm() / b.float().cpu().norm().clamp_min(1e-9)).item()


def audit_lora(tower):
    """Every one of the 12 layers must carry non-trivial LoRA tensors, and nothing may sit off-device or off-dtype."""
    layers = []
    for i, layer in enumerate(tower.layers):
        a = [p for n, p in layer.named_parameters() if n.endswith(".lora_a")]
        b = [p for n, p in layer.named_parameters() if n.endswith(".lora_b")]
        if not a or len(a) != len(b) or any(float(x.norm()) == 0.0 for x in b):
            raise AssertionError(f"LoRA missing or zero on layer {i + 1}")
        layers.append({"layer": i + 1, "modules": len(a), "lora_b_norm": float(sum(float(x.norm()) ** 2 for x in b) ** 0.5)})
    bad = {(p.device.type, str(p.dtype)) for p in tower.parameters()} - {("mps", "torch.float32")}
    if bad:
        raise AssertionError(f"Device/dtype mismatch in encoder parameters: {sorted(bad)}")
    return layers


@torch.inference_mode(False)
@torch.enable_grad()
def mps_ctc_backward(vocab):
    """CTC loss forward+backward on MPS with CPU fallback disabled (the probe depends on it)."""
    x = torch.randn(1, 50, 64, device="mps", requires_grad=True)
    head = torch.nn.Linear(64, vocab).to("mps")
    logp = head(x)[0].log_softmax(-1)[:, None, :]
    target = torch.randint(1, vocab, (12,), device="mps")
    loss = torch.nn.functional.ctc_loss(logp, target, torch.tensor([50], device="mps"), torch.tensor([12], device="mps"), blank=0)
    loss.backward()
    sync()
    if not (torch.isfinite(loss).item() and torch.isfinite(head.weight.grad).all().item() and head.weight.grad.abs().sum().item() > 0):
        raise AssertionError("MPS CTC loss/backward produced nonfinite or zero gradients")
    return {"loss": loss.item(), "grad_abs_sum": head.weight.grad.abs().sum().item()}


def run(wav, model, cpu_model, base_tower):
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
        ref_rel = (official_feat.float()[valid] - b).norm() / b.norm().clamp_min(1e-9)
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
        lora = audit_lora(tower)
        base_h, _ = stages(base_tower, mel, mask)
        lora_effect = {k: rel_l2(h[k], base_h[k]) for k in FEATURES}
        if min(lora_effect.values()) <= 1e-6:
            raise AssertionError(f"LoRA has no effect on some stage: {lora_effect}")
        cpu_tower = cpu_model.asr.encoder.tower
        cmel, cmask = mel_input(wav, cpu_model.asr.encoder.feature_extractor, device="cpu")
        cpu_h, cpu_lens = stages(cpu_tower, cmel, cmask)
        if not torch.equal(cpu_lens, lens.cpu()):
            raise AssertionError("CPU/MPS frame lengths differ")
        vs_cpu = {k: rel_l2(h[k], cpu_h[k]) for k in FEATURES}
        if max(vs_cpu.values()) > 0.01:
            raise AssertionError(f"MPS vs CPU fp32 stage mismatch: {vs_cpu}")
        cpu_logits = cpu_model.asr.reader(cpu_h["L12-Proj"].float(), cpu_lens)
        logit_diff = (logits.float().cpu() - cpu_logits).abs().max().item()
        cpu_greedy = cpu_model.asr.decode(cpu_logits, cpu_lens, cpu_model.asr.tok)[0]
        if cpu_greedy != greedy:
            raise AssertionError(f"MPS vs CPU transcript mismatch: {greedy!r} != {cpu_greedy!r}")
        ctc = mps_ctc_backward(len(asr.tok.itos))
    return {
        "lora_layers": lora,
        "lora_effect_vs_base_rel_l2": lora_effect,
        "mps_vs_cpu_fp32_rel_l2": vs_cpu,
        "mps_vs_cpu_logit_max_abs": logit_diff,
        "cpu_greedy": cpu_greedy,
        "mps_ctc_backward": ctc,
        "status": "PASS",
        "feature_shape": list(custom.shape),
        "lengths": lens.cpu().tolist(),
        "hidden_shapes": {k: list(v.shape) for k, v in h.items()},
        "logit_shape": list(logits.shape),
        "relative_l2_split": rel.item(),
        "relative_l2_official": ref_rel.item(),
        "greedy": greedy,
        "transcript_match": True,
        "nan_inf": False,
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
            wav = (0.05 * np.sin(2 * np.pi * 220 * t) + 0.01 * np.random.default_rng(17).standard_normal(len(t))).astype(np.float32)
            result["audio"] = {"kind": "synthetic", "seconds": 3, "seed": 17}
        model, origin = load_adapted(a.revision)
        result["origin"] = origin
        result.update(run(wav, model, load_adapted(a.revision or origin["hf_revision"], device="cpu")[0], load_base()))
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()
    write_json(a.out, result)
    print(f"{result['status']}: {a.out}")
    if result["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
