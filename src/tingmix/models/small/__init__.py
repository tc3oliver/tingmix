"""TingMix-Small: natively streaming CTC model for CPUs (see model.py). Same `ASRModel` + `StreamingASRModel` interface as
Large; vocabulary format = Embed2ASR units.json + tokenizer_bpe.model.

Release directory: small.json (net config), model.safetensors, units.json, tokenizer_bpe.model."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

DEFAULT_REPO = "tc3oliver/TingMix-Small"


class SmallModel:
    name = "tingmix-small"
    sample_rate = 16000
    max_window_s = float("inf")
    streaming_native = True

    def __init__(self, net, tok, device="cpu"):
        device = device or "cpu"
        self.net, self.tok, self.device = net.eval().to(device), tok, device

    @classmethod
    def from_pretrained(cls, path_or_repo: str = DEFAULT_REPO, device: str = "cpu", threads: int | None = None):
        import torch
        from embed2asr.tokenizer import Tok
        from safetensors.torch import load_file

        from .model import SmallNet

        path = Path(path_or_repo)
        if not path.exists():
            from huggingface_hub import snapshot_download

            path = Path(snapshot_download(str(path_or_repo)))
        if threads:
            torch.set_num_threads(threads)
        cfg = json.loads((path / "small.json").read_text())
        itos = json.loads((path / "units.json").read_text(encoding="utf-8"))
        sp = (path / "tokenizer_bpe.model").read_bytes() if (path / "tokenizer_bpe.model").exists() else None
        net = SmallNet(**cfg)
        net.load_state_dict(load_file(str(path / "model.safetensors")))
        return cls(net, Tok(itos, sp), device)

    def _decode(self, ids):
        from embed2asr.ctc import collapse_ctc

        return self.tok.decode(collapse_ctc(ids))

    def transcribe_batch(self, wavs: list[np.ndarray]) -> list[str]:
        import torch

        if not wavs:
            return []
        lens = torch.tensor([len(w) for w in wavs])
        x = torch.nn.utils.rnn.pad_sequence([torch.as_tensor(w, dtype=torch.float32) for w in wavs], batch_first=True).to(self.device)
        with torch.inference_mode():
            lp, fl = self.net(x, lens)
        return [self._decode(row[:n].argmax(-1).tolist()) for row, n in zip(lp, fl.tolist())]

    def new_stream(self):
        from .model import Stream

        return Stream(self.net, self._decode)
