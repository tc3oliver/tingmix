"""TingMix-Large inference: adapter over Embed2ASR (frozen EmbeddingGemma 2 audio encoder + GRU-CTC reader).

The model is offline (bidirectional, max window 29.5 s), so `streaming_native` is False: streaming re-decodes the open segment."""

from __future__ import annotations

import numpy as np

DEFAULT_REPO = "tc3oliver/Embed2ASR-v1"


class LargeModel:
    name = "tingmix-large"
    sample_rate = 16000
    max_window_s = 29.5
    streaming_native = False

    def __init__(self, asr):
        self.asr = asr

    @classmethod
    def from_pretrained(cls, path_or_repo: str = DEFAULT_REPO, device: str | None = None, encoder_delta=None, **kw) -> LargeModel:
        """Embed2ASR release directory or Hugging Face repo. A TingMix-Large release (tools/export_release.py) adds
        `tingmix.json` + `encoder_delta.safetensors`, applied automatically; `encoder_delta` = (cfg, state) overrides it."""
        import json
        from pathlib import Path

        from embed2asr import Embed2ASR

        path = Path(path_or_repo)
        if not path.exists():
            from huggingface_hub import snapshot_download

            path = Path(snapshot_download(str(path_or_repo), revision=kw.pop("revision", None)))
        asr = Embed2ASR.from_pretrained(path, device=device, **kw)
        if encoder_delta is None and (path / "tingmix.json").is_file():
            from safetensors.torch import load_file

            meta = json.loads((path / "tingmix.json").read_text(encoding="utf-8"))
            encoder_delta = (meta["encoder_delta"], load_file(str(path / meta["encoder_delta_file"])))
        if encoder_delta is not None:
            from .encoder import load_delta

            load_delta(asr.encoder.tower, *encoder_delta)
        return cls(asr)

    @classmethod
    def from_checkpoint(
        cls, path: str, device: str | None = None, lm: str | None = None, lm_alpha: float = 0.25, lm_beta: float = 1.5, beam: int = 16
    ) -> LargeModel:
        """Load a TingMix-Large training export (.pt): reader, tokenizer and encoder delta (trusted file: unpickled).
        `lm`: Embed2ASR character n-gram LM file -> CTC prefix beam search (CJK units scored; English pieces LM-neutral)."""
        import torch
        from embed2asr.encoder import FrozenAudioEncoder, default_device
        from embed2asr.pipeline import Embed2ASR, make_config
        from embed2asr.reader import Reader
        from embed2asr.tokenizer import Tok

        from .encoder import load_delta

        ck = torch.load(path, map_location="cpu", weights_only=False)
        device = torch.device(device or default_device())
        reader = Reader(**ck["reader_cfg"])
        reader.load_state_dict(ck["reader"])
        reader.to(device).eval().requires_grad_(False)
        encoder = FrozenAudioEncoder(device=device)
        load_delta(encoder.tower, ck["enc_cfg"], ck["enc"])
        sp = ck["tok"].get("sp")
        config = make_config(ck["reader_cfg"], len(ck["itos"]), bool(sp))
        lm_args = None
        if lm:
            from embed2asr.lm import CharLM

            lm_args = (CharLM.load(lm), lm_alpha, lm_beta, beam)
        return cls(Embed2ASR(reader, Tok(ck["itos"], sp), encoder, config, device, lm=lm_args))

    def transcribe_batch(self, wavs: list[np.ndarray]) -> list[str]:
        if not wavs:
            return []
        return [t.text for t in self.asr.transcribe(list(wavs), sr=self.sample_rate)]
