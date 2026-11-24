"""Shared, fail-closed helpers for the M4 encoder-depth study. Never downloads a different ASR release."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

# Must be set before torch is imported anywhere in this process.
if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") != "0":
    raise RuntimeError("PYTORCH_ENABLE_MPS_FALLBACK must be 0 for this study")
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from tingmix.models.large import encoder as E  # noqa: E402

REPO = "tc3oliver/TingMix-Large"
DEPTHS = (3, 6, 9, 12)
FEATURES = ("L3", "L6", "L9", "L12", "L12-Proj")
BASE = "base"
ADAPTED = "adapted"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def require_mps():
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS_UNAVAILABLE: must run on Mac Studio M4 Max; no CPU substitution")
    return torch.device("mps")


def sync():
    torch.mps.synchronize()


def environment():
    import transformers

    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    return {
        "platform": platform.platform(),
        "macos": platform.mac_ver()[0],
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": sys.version,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "mps_available": torch.backends.mps.is_available(),
        "mps_built": torch.backends.mps.is_built(),
        "fallback_env": os.environ["PYTORCH_ENABLE_MPS_FALLBACK"],
        "threads": torch.get_num_threads(),
        "git_commit": commit,
        "device": "mps" if torch.backends.mps.is_available() else "unavailable",
        "precision": "fp32",
    }


def resolve_release(revision=None):
    from huggingface_hub import HfApi, snapshot_download

    info = HfApi().model_info(REPO, revision=revision)
    sha = info.sha
    path = Path(snapshot_download(REPO, revision=sha, ignore_patterns=["lm/*"]))  # the n-gram LM is not used here
    for line in (path / "SHA256SUMS").read_text().splitlines():
        if not line.strip():
            continue
        digest, name = line.split(maxsplit=1)
        if sha256(path / name.strip()) != digest:
            raise RuntimeError(f"RELEASE_MISMATCH: {name} does not match SHA256SUMS")
    meta_file = path / "tingmix.json"
    if not meta_file.is_file():
        raise RuntimeError(f"RELEASE_MISMATCH: {REPO}@{sha} has no tingmix.json; refusing base Embed2ASR-v1")
    meta = json.loads(meta_file.read_text(encoding="utf-8"))
    delta_file = path / meta["encoder_delta_file"]
    if not delta_file.is_file():
        raise RuntimeError(f"RELEASE_MISMATCH: missing LoRA delta {delta_file.name}")
    return path, {
        "hf_repo": REPO,
        "hf_revision": sha,
        "encoder_revision": E.ENCODER_REVISION,
        "delta_name": delta_file.name,
        "delta_sha256": sha256(delta_file),
        "reader_sha256": sha256(path / "model.safetensors"),
        "tokenizer_sha256": sha256(path / "units.json"),
        "delta_config": meta["encoder_delta"],
    }


def load_adapted(revision=None, device="mps"):
    """Load *published* Reader/tokenizer/encoder + complete delta. Fail on incomplete LoRA.
    device="cpu" is only for the fp32 reference side of the parity check; every measurement uses MPS."""
    from safetensors import safe_open

    from tingmix.models.large.model import LargeModel

    device = require_mps() if device == "mps" else torch.device(device)
    path, origin = resolve_release(revision)
    model = LargeModel.from_pretrained(str(path), device=str(device), encoder_revision=E.ENCODER_REVISION)
    tower = model.asr.encoder.tower
    cfg = origin["delta_config"]
    if cfg.get("mode") != "lora" or cfg.get("lora_r") != 16:
        raise RuntimeError(f"Unexpected release adaptation config: {cfg}")
    if len(tower.layers) != 12:
        raise RuntimeError(f"Unexpected layer count: {len(tower.layers)}")
    if model.asr.config["encoder"]["revision"] != E.ENCODER_REVISION:
        raise RuntimeError("Published encoder revision differs from pinned revision")
    if model.asr.reader.cfg["in_dim"] != 1536:
        raise RuntimeError("Reader was not trained on 1536-d projected features")
    expected = {n for n, _ in tower.named_parameters() if n.endswith((".lora_a", ".lora_b"))}
    with safe_open(str(path / origin["delta_name"]), framework="pt", device="cpu") as f:
        got = set(f.keys())
    if not expected or expected != got:
        raise RuntimeError(
            f"LoRA integrity mismatch: expected {len(expected)} tensors, release has {len(got)}; "
            f"missing={sorted(expected - got)[:4]}, extra={sorted(got - expected)[:4]}"
        )
    origin["encoder_params"] = sum(p.numel() for p in tower.parameters())
    origin["encoder_base_params"] = sum(p.numel() for n, p in tower.named_parameters() if ".lora_" not in n)
    origin["lora_params"] = sum(p.numel() for n, p in tower.named_parameters() if ".lora_" in n)
    origin["reader_params"] = sum(p.numel() for p in model.asr.reader.parameters())
    origin["vocab_size"] = len(model.asr.tok.itos)
    if origin["encoder_base_params"] != 304824608:
        raise RuntimeError(f"Base encoder parameter mismatch: {origin['encoder_base_params']}")
    model.asr.encoder.tower.eval().requires_grad_(False)
    model.asr.reader.eval().requires_grad_(False)
    return model, origin


def read_tokenizer(release_path):
    from embed2asr.tokenizer import Tok

    cfg = json.loads((release_path / "config.json").read_text(encoding="utf-8"))
    units = json.loads((release_path / cfg["tokenizer"]["units"]).read_text(encoding="utf-8"))
    sp_name = cfg["tokenizer"].get("sentencepiece")
    return Tok(units, (release_path / sp_name).read_bytes() if sp_name else None)


def load_base():
    device = require_mps()
    tower = E.load_tower(revision=E.ENCODER_REVISION, dtype=torch.float32).to(device).eval()
    tower.requires_grad_(False)
    if len(tower.layers) != 12:
        raise RuntimeError("Wrong base encoder architecture")
    return tower


def mel_input(wav, feature_extractor, device="mps"):
    if wav.dtype != np.float32 or wav.ndim != 1:
        raise ValueError("Expected mono float32 16 kHz")
    inp = feature_extractor([wav], sampling_rate=16000, return_tensors="pt")
    device = require_mps() if device == "mps" else torch.device(device)
    return inp["input_features"].to(device=device, dtype=torch.float32), inp["input_features_mask"].to(device=device)


@torch.inference_mode()
def stages(tower, mel, mask):
    """Single pass, 1-based layer labels; no projected state substituted for L12."""
    h, lengths = E.subsample(tower, mel, mask)
    out = {}
    start = 0
    for end in DEPTHS:
        h = E.run_layers(tower, h, lengths, start, end, proj=False)
        out[f"L{end}"] = h
        start = end
    out["L12-Proj"] = tower.output_proj(h)
    for label, value in out.items():
        dim = 1536 if label == "L12-Proj" else 1024
        if value.shape[0] != 1 or value.shape[2] != dim or not torch.isfinite(value).all().item():
            raise RuntimeError(f"Invalid {label} tensor: {tuple(value.shape)}")
    return out, lengths


@torch.inference_mode()
def early(tower, mel, mask, depth):
    """REAL early exit from input, never a cached L12 state. E12 includes projection."""
    if depth not in (6, 9, 12):
        raise ValueError(depth)
    h, lens = E.subsample(tower, mel, mask)
    return E.run_layers(tower, h, lens, 0, depth, proj=depth == 12), lens


def executed_params(tower, depth, proj):
    modules = [tower.subsample_conv_projection, *list(tower.layers[:depth])]
    if proj:
        modules.append(tower.output_proj)
    return sum(p.numel() for module in modules for p in module.parameters())


def read_manifest(path):
    rows = []
    for i, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        r = json.loads(line)
        for required in ("audio", "text"):
            if not r.get(required):
                raise ValueError(f"{path}:{i + 1} missing {required}")
        if not Path(r["audio"]).is_file():
            raise FileNotFoundError(r["audio"])
        rows.append(r)
    return rows


def no_overlap(train, dev):
    """Exact byte-identical path/content overlap is a hard error; near duplicates still require an external audit."""

    def keys(rows):
        found = set()
        for r in rows:
            path = Path(r["audio"]).resolve()
            found.add(("path", str(path)))
            found.add(("bytes", sha256(path)))
        return found

    overlap = keys(train) & keys(dev)
    if overlap:
        raise RuntimeError(f"TRAIN_DEV_LEAKAGE: {len(overlap)} matching exact audio identifiers")
    return {"exact_duplicates": 0, "near_duplicate_audit": "NOT_VERIFIED"}


def sample_key(row):
    return hashlib.sha256((str(Path(row["audio"]).resolve()) + "\0" + row["text"]).encode()).hexdigest()[:24]
