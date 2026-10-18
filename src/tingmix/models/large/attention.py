"""Attention decoder for joint CTC / attention decoding (WeNet "attention rescoring"): a Transformer decoder over the
Reader's contextual states rescores the CTC beam's N-best for the final text. Partials stay CTC (no extra latency);
a final costs one batched teacher-forced decoder pass over N hypotheses.

Tokens = the CTC vocabulary (CJK characters + English BPE pieces) with blank (id 0) unused; sos = eos = vocab_size."""

from __future__ import annotations

import collections
import math

import numpy as np
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


class ZeroLM:
    """No n-gram LM: plain CTC prefix beam search (use alpha 0, beta 0)."""

    def logp(self, prefix, w):
        return 0.0


def beam_nbest(lp, tok, lm, alpha, beta, beam, n, units, topk=8):
    """embed2asr.lm.beam_decode, returning the n best (text, score) instead of the best text."""
    from embed2asr.lm import EOS, NEG, PH, lse

    st = {(): ("", 0.0, 0)}

    def state(pf):
        v = st.get(pf)
        if v is None:
            ctx, s, k = state(pf[:-1])
            u = units[pf[-1]]
            if u is None:
                if not ctx.endswith(PH):
                    s += lm.logp(ctx, PH)
                    ctx += PH
                v = (ctx, s, k)
            else:
                for ch in u:
                    s += lm.logp(ctx, ch)
                    ctx += ch
                v = (ctx, s, k + len(u))
            st[pf] = v
        return v

    def score(pf, pb, pnb):
        _, s, k = state(pf)
        return lse(pb, pnb) + alpha * s + beta * k

    beams = {(): (0.0, NEG)}
    for t in range(lp.shape[0]):
        row = lp[t]
        cand = np.argpartition(-row, topk)[:topk] if row.shape[0] > topk else range(row.shape[0])
        cand = [int(c) for c in cand if row[c] > -14 or c == 0]
        new = collections.defaultdict(lambda: [NEG, NEG])
        for pf, (pb, pnb) in beams.items():
            tot = lse(pb, pnb)
            for c in cand:
                p = float(row[c])
                if c == 0:
                    e = new[pf]
                    e[0] = lse(e[0], tot + p)
                    continue
                if pf and c == pf[-1]:
                    e = new[pf + (c,)]
                    e[1] = lse(e[1], pb + p)
                    e = new[pf]
                    e[1] = lse(e[1], pnb + p)
                else:
                    e = new[pf + (c,)]
                    e[1] = lse(e[1], tot + p)
        beams = dict(sorted(((k, tuple(v)) for k, v in new.items()), key=lambda kv: -score(kv[0], *kv[1]))[:beam])
    fin = sorted(((score(k, *v) + alpha * lm.logp(state(k)[0], EOS), k) for k, v in beams.items()), reverse=True)[:n]
    out, seen = [], set()
    for s, k in fin:
        txt = tok.decode(k).replace("臺", "台")
        if txt not in seen:
            seen.add(txt)
            ctx, lm_sum, scored = state(k)
            out.append(
                {
                    "text": txt,
                    "beam": s,
                    "units": len(k),
                    "ids": list(k),
                    "ctc": lse(*beams[k]),
                    "lm": lm_sum + lm.logp(ctx, EOS),
                    "scored": scored,
                }
            )
    return out


def rescoring_decoder(dec, mem_box, lm, alpha, beta, beam=16, n=8, att_weight=0.5, mu=0.0):
    """DECODERS-style fn(logits, lengths, tok) -> texts: CTC(+LM) prefix beam N-best, rescored by the attention decoder.
    mem_box["h"] = decoder memory of the same batch (filled by a forward hook on the Reader output layer)."""
    from embed2asr.lm import lm_units

    lm = lm if lm is not None else ZeroLM()

    def decode(logits, lengths, tok):
        units = lm_units(tok.itos)
        lp = logits.float().log_softmax(-1).cpu().numpy()
        out = []
        for i, L in enumerate(lengths.tolist()):
            hyps = beam_nbest(lp[i, :L], tok, lm, alpha, beta, beam, n, units)
            att = dec.score(mem_box["h"][i].float(), L, [h["ids"] for h in hyps])
            out.append(max(zip(hyps, att), key=lambda p: p[0]["beam"] + att_weight * p[1] + mu * p[0]["units"])[0]["text"] if hyps else "")
        return out

    return decode


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
