"""EmbeddingGemma 2 audio tower for TingMix-Large: load, adapt (LoRA / top-layer fine-tuning), run, save deltas.

The tower is `google/embeddinggemma-2` `audio_tower` (Gemma4AudioModel: subsample conv -> 12 conformer layers ->
output_proj 1024->1536, 25 frames/s). Features = output of output_proj (`last_hidden_state`), as in Embed2ASR-v1.

Adaptation modes (only the delta is saved; the base weights always come from the pinned revision):
  frozen   nothing trains
  lora     LoRA on every attention / feed-forward linear of the selected layers (gradients flow through all layers,
           so the encoder must run from audio: no cached intermediate states)
  partial  layers[start:] + output_proj train; layers[:start] are frozen and may be replaced by cached layer-`start` states
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

ENCODER_ID = "google/embeddinggemma-2"
ENCODER_REVISION = "914f7f89142e33e77833254d9c9b90c3cef7303b"
FEATURE_DIM = 1536
HIDDEN_DIM = 1024
FRAME_RATE = 25
LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "post", "ffw_layer_1", "ffw_layer_2")


def load_tower(revision: str = ENCODER_REVISION, dtype=torch.bfloat16):
    from transformers import AutoModel

    full = AutoModel.from_pretrained(ENCODER_ID, revision=revision, dtype=dtype)
    tower = full.audio_tower
    del full
    tower.requires_grad_(False).eval()
    return tower


def load_feature_extractor(revision: str = ENCODER_REVISION):
    from transformers import AutoFeatureExtractor

    return AutoFeatureExtractor.from_pretrained(ENCODER_ID, revision=revision)


class LoRALinear(nn.Module):
    """y = W x + (alpha / r) * B(A(dropout(x))); B starts at zero, so the wrapped layer is exact at init."""

    def __init__(self, base: nn.Linear, r: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        self.base, self.r, self.scale = base, r, alpha / r
        self.lora_a = nn.Parameter(torch.empty(r, base.in_features, dtype=torch.float32))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, r, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        d = (self.drop(x).to(self.lora_a.dtype) @ self.lora_a.T) @ self.lora_b.T
        return self.base(x) + (self.scale * d).to(x.dtype)


def apply_lora(tower, r: int = 16, alpha: float = 32.0, dropout: float = 0.0, layers=None, targets=LORA_TARGETS) -> list[nn.Parameter]:
    """Wrap the inner nn.Linear of every target ClippableLinear (clipping is kept) in the given layer indices."""
    params = []
    for li in layers if layers is not None else range(len(tower.layers)):
        for name, mod in tower.layers[li].named_modules():
            if name.split(".")[-1] in targets and isinstance(getattr(mod, "linear", None), nn.Linear):
                mod.linear = LoRALinear(mod.linear, r, alpha, dropout).to(mod.linear.weight.device)
                params += [mod.linear.lora_a, mod.linear.lora_b]
    if not params:
        raise ValueError("apply_lora matched no linear layers")
    return params


def set_partial(tower, start: int = 10) -> list[nn.Parameter]:
    """layers[start:] + output_proj trainable with fp32 master weights."""
    params = []
    for mod in [*tower.layers[start:], tower.output_proj]:
        for p in mod.parameters():
            p.data = p.data.float()
            p.requires_grad_(True)
            params.append(p)
    return params


def _mask_and_pos(tower, h, lengths):
    from transformers.masking_utils import create_bidirectional_mask
    from transformers.models.gemma4.modeling_gemma4 import sliding_window_mask_function

    c = tower.config
    mask = torch.arange(h.shape[1], device=h.device)[None] < lengths.to(h.device)[:, None]
    am = create_bidirectional_mask(
        config=c,
        inputs_embeds=h,
        attention_mask=mask,
        and_mask_function=sliding_window_mask_function((c.attention_context_left - 1, c.attention_context_right)),
    )
    if am is not None:
        am = tower._convert_4d_mask_to_blocked_5d(am)
    return am, tower.rel_pos_enc(h)


def run_layers(tower, h, lengths, start: int, end: int | None = None, proj: bool = True):
    """Run tower.layers[start:end] (+ output_proj) on padded hidden states h [B,T,1024] with valid frames `lengths`.
    Same mask / relative-position construction as Gemma4AudioModel.forward and eg2_cache.top_layers (tested equal)."""
    am, pe = _mask_and_pos(tower, h, lengths)
    for layer in tower.layers[start:end]:
        h = layer(h, attention_mask=am, position_embeddings=pe)
    return tower.output_proj(h) if proj else h


def subsample(tower, mel, mel_mask):
    """mel [B,F,128], mel_mask [B,F] -> (h [B,T,1024], lengths [B]) after the conv front end."""
    h, m = tower.subsample_conv_projection(mel, mel_mask)
    return h, m.sum(1)


def encode(tower, mel, mel_mask, mode: str = "frozen", start: int = 10):
    """Full encoder from log-mel. Frozen parts run without autograd in the tower dtype (pure bf16, identical to
    Embed2ASR inference); the fp32 top layers of `partial` run under bf16 autocast. Returns (features [B,T,1536] fp32,
    lengths [B]); padded output frames are zeroed."""
    if mode == "lora":
        h, lens = subsample(tower, mel, mel_mask)
        x = run_layers(tower, h, lens, 0)
    else:
        with torch.no_grad():
            h, lens = subsample(tower, mel, mel_mask)
            h = run_layers(tower, h, lens, 0, start if mode == "partial" else None, proj=mode != "partial")
        if mode == "partial":
            x = top(tower, h, lens, start)
        else:
            x = h
    valid = torch.arange(x.shape[1], device=x.device)[None] < lens[:, None]
    return x.float() * valid[..., None], lens


def top(tower, h, lens, start: int = 10):
    """Trainable top of `partial` mode on (cached or live) layer-`start` states."""
    with torch.autocast(h.device.type, dtype=torch.bfloat16, enabled=h.device.type == "cuda"):
        return run_layers(tower, h, lens, start)


def trainable_state(module: nn.Module) -> dict[str, torch.Tensor]:
    """Encoder delta to save: every parameter with requires_grad (LoRA A/B or unfrozen top layers)."""
    return {n: p.detach().float().cpu().clone() for n, p in module.named_parameters() if p.requires_grad}


def load_delta(tower, cfg: dict, state: dict[str, torch.Tensor]) -> None:
    """Rebuild the adaptation described by cfg ({"mode", "lora_r", "lora_alpha", "start"}) and load its weights."""
    mode = cfg.get("mode", "frozen")
    if mode == "lora":
        apply_lora(tower, cfg["lora_r"], cfg["lora_alpha"], 0.0, cfg.get("lora_layers"))
    elif mode == "partial":
        set_partial(tower, cfg.get("start", 10))
    if state:
        missing = set(state) - {n for n, _ in tower.named_parameters()}
        if missing:
            raise ValueError(f"encoder delta has unknown tensors: {sorted(missing)[:5]}")
        own = dict(tower.named_parameters())
        with torch.no_grad():
            for n, t in state.items():
                own[n].copy_(t.to(own[n].dtype))
    tower.requires_grad_(False)
