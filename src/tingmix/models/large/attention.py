"""Attention decoder for joint CTC / attention decoding (WeNet "attention rescoring"): a Transformer decoder over the
Reader's contextual states rescores the CTC beam's N-best for the final text. Partials stay CTC (no extra latency);
a final costs one batched teacher-forced decoder pass over N hypotheses.

Tokens = the CTC vocabulary (CJK characters + English BPE pieces) with blank (id 0) unused; sos = eos = vocab_size."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttDecoder(nn.Module):
    def __init__(self, mem_dim, vocab_size, d=512, layers=6, heads=8, ff=2048, dropout=0.1):
        super().__init__()
        self.cfg = dict(mem_dim=mem_dim, vocab_size=vocab_size, d=d, layers=layers, heads=heads, ff=ff, dropout=dropout)
        self.sos = self.eos = vocab_size
        self.mem = nn.Sequential(nn.LayerNorm(mem_dim), nn.Linear(mem_dim, d))
        self.emb = nn.Embedding(vocab_size + 1, d)
        self.dec = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d, heads, ff, dropout, activation="gelu", batch_first=True, norm_first=True), layers
        )
        self.norm = nn.LayerNorm(d)
        self.out = nn.Linear(d, vocab_size + 1)
        pos = torch.arange(4096)[:, None] * torch.exp(torch.arange(0, d, 2) * (-math.log(10000.0) / d))[None]
        self.register_buffer("pe", torch.stack([pos.sin(), pos.cos()], -1).flatten(1), persistent=False)

    def forward(self, mem, mem_lens, ys_in, ys_lens):
        """mem [B,T,mem_dim], ys_in [B,U] (sos + tokens, padded) -> logits [B,U,V+1]."""
        m = self.mem(mem)
        m = m + self.pe[: m.shape[1]].to(m.dtype)
        U = ys_in.shape[1]
        y = self.emb(ys_in) * math.sqrt(self.emb.embedding_dim) + self.pe[:U].to(m.dtype)
        mem_pad = torch.arange(m.shape[1], device=m.device)[None] >= mem_lens.to(m.device)[:, None]
        y_pad = torch.arange(U, device=m.device)[None] >= ys_lens.to(m.device)[:, None]
        causal = torch.triu(torch.ones(U, U, dtype=torch.bool, device=m.device), 1)
        h = self.dec(y, m, tgt_mask=causal, tgt_key_padding_mask=y_pad, memory_key_padding_mask=mem_pad)
        return self.out(self.norm(h))

    def batch(self, seqs, device):
        """token id lists -> (ys_in, ys_out, lens): ys_in = sos + y, ys_out = y + eos (pad -100)."""
        L = torch.tensor([len(s) + 1 for s in seqs])
        ys_in = torch.full((len(seqs), int(L.max())), self.eos, dtype=torch.long)
        ys_out = torch.full((len(seqs), int(L.max())), -100, dtype=torch.long)
        for i, s in enumerate(seqs):
            ys_in[i, 0] = self.sos
            ys_in[i, 1 : len(s) + 1] = torch.tensor(s, dtype=torch.long)
            ys_out[i, : len(s)] = torch.tensor(s, dtype=torch.long)
            ys_out[i, len(s)] = self.eos
        return ys_in.to(device), ys_out.to(device), L

    def loss(self, mem, mem_lens, seqs, smoothing=0.1):
        ys_in, ys_out, L = self.batch(seqs, mem.device)
        logits = self(mem, mem_lens, ys_in, L).float()
        ce = F.cross_entropy(logits.transpose(1, 2), ys_out, ignore_index=-100, label_smoothing=smoothing)
        valid = ys_out != -100
        acc = ((logits.argmax(-1) == ys_out) & valid).sum() / valid.sum()
        return ce, acc

    @torch.no_grad()
    def score(self, mem, mem_len, hyps):
        """One utterance: mem [T,mem_dim]; hyps = token id lists -> log P(hyp + eos) per hyp (list of floats)."""
        if not hyps:
            return []
        ys_in, ys_out, L = self.batch(hyps, mem.device)
        n = len(hyps)
        logits = self(mem[None].expand(n, -1, -1), torch.full((n,), int(mem_len)), ys_in, L).float().log_softmax(-1)
        lp = logits.gather(-1, ys_out.clamp(min=0)[..., None])[..., 0] * (ys_out != -100)
        return lp.sum(1).tolist()


def reader_states(reader, x, lens):
    """Run the Embed2ASR Reader once -> (CTC logits, contextual states [B,T,2*gru] = the input of its output layer)."""
    box = {}

    def keep(mod, inp, out):  # must return None: a returned value would replace the module output
        box["h"] = inp[0]

    hook = reader.out.register_forward_hook(keep)
    try:
        logits = reader(x, lens)
    finally:
        hook.remove()
    return logits, box["h"]


if __name__ == "__main__":  # shape / masking self-check
    torch.manual_seed(0)
    dec = AttDecoder(16, 10, d=32, layers=2, heads=4, ff=64, dropout=0.0).eval()
    mem = torch.randn(2, 7, 16)
    ce, acc = dec.loss(mem, torch.tensor([7, 5]), [[1, 2, 3], [4]])
    assert torch.isfinite(ce)
    # padding must not change the score of the shorter sequence / utterance
    a = dec.score(mem[1], 5, [[4], [1, 2, 3]])
    b = dec.score(mem[1, :5], 5, [[4]])
    assert abs(a[0] - b[0]) < 1e-4, (a, b)
    from embed2asr.reader import Reader

    r = Reader(16, 10, d=8, gru=4, blocks=1).eval()
    lg, h = reader_states(r, mem, torch.tensor([7, 5]))
    assert lg.shape[-1] == 10 and h.shape[-1] == 8 and torch.allclose(lg, r(mem, torch.tensor([7, 5])))
    print("ok", float(ce.detach()), a)
