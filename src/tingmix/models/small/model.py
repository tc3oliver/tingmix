"""TingMix-Small: a natively streaming CTC model for CPUs.

log-mel (causal, no centering) -> 2 causal strided convs (100 -> 25 frames/s) -> unidirectional GRU stack -> CTC.
Every component is causal, so feeding audio chunk by chunk gives exactly the same frames and states as one offline pass:
`Stream.accept` keeps the STFT sample remainder, the conv input context and the GRU hidden state between chunks."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

SR, N_FFT, WIN, HOP, N_MELS = 16000, 512, 400, 160, 80
TAIL_S = 0.3  # silence flushed through the model at the end of an utterance (dev: NTUML mixed MER 35.8 -> 32.8)


def mel_filters(n_mels=N_MELS, n_fft=N_FFT, sr=SR, fmin=20.0, fmax=7600.0):
    """Slaney-style triangular mel filterbank [n_mels, n_fft//2+1] (HTK mel scale)."""

    def hz2mel(f):
        return 2595.0 * math.log10(1.0 + f / 700.0)

    mels = torch.linspace(hz2mel(fmin), hz2mel(fmax), n_mels + 2)
    hz = 700.0 * (10 ** (mels / 2595.0) - 1.0)
    bins = torch.linspace(0, sr / 2, n_fft // 2 + 1)
    lo, ce, hi = hz[:-2, None], hz[1:-1, None], hz[2:, None]
    fb = torch.clamp(torch.minimum((bins - lo) / (ce - lo), (hi - bins) / (hi - ce)), min=0.0)
    return fb * (2.0 / (hi - lo))  # area normalization


class LogMel(nn.Module):
    """Causal log-mel: frame t covers samples [t*HOP, t*HOP+N_FFT) (Hann window of WIN centred in it); n frames need
    (n-1)*HOP + N_FFT samples."""

    def __init__(self):
        super().__init__()
        self.register_buffer("window", torch.hann_window(WIN), persistent=False)
        self.register_buffer("fb", mel_filters(), persistent=False)

    def forward(self, wav):  # [B, S] -> [B, T, N_MELS]
        if wav.shape[-1] < N_FFT:
            return wav.new_zeros(wav.shape[0], 0, N_MELS)
        spec = torch.stft(wav, N_FFT, HOP, WIN, self.window, center=False, return_complex=True).abs().pow(2)
        return torch.log(torch.clamp(self.fb @ spec, min=1e-6)).transpose(1, 2)

    @staticmethod
    def frames(n_samples):
        return 0 if n_samples < N_FFT else (n_samples - N_FFT) // HOP + 1


class SmallNet(nn.Module):
    def __init__(self, vocab_size, d=512, hidden=640, layers=5, dropout=0.1):
        super().__init__()
        self.cfg = dict(vocab_size=vocab_size, d=d, hidden=hidden, layers=layers, dropout=dropout)
        self.mel = LogMel()
        self.norm = nn.LayerNorm(N_MELS)
        self.conv1 = nn.Conv1d(N_MELS, d, 3, stride=2)  # causal: left context only (inputs padded on the left)
        self.conv2 = nn.Conv1d(d, d, 3, stride=2)
        self.gru = nn.GRU(d, hidden, num_layers=layers, batch_first=True, dropout=0.0)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(hidden, vocab_size)

    def frontend(self, mel):  # [B,T,80] (T multiple of 4 after left-padding) -> [B, T/4, d]
        x = self.norm(mel).transpose(1, 2)
        x = F.gelu(self.conv1(F.pad(x, (1, 0))))
        x = F.gelu(self.conv2(F.pad(x, (1, 0))))
        return x.transpose(1, 2)

    def forward(self, wav, wav_lens):
        """Offline/training pass: wav [B,S] float, wav_lens [B] -> (log-probs [B,T,V], frame lengths [B])."""
        mel = self.mel(wav)
        T = mel.shape[1] - mel.shape[1] % 4
        h = self.frontend(mel[:, :T])
        lens = torch.div(torch.tensor([LogMel.frames(int(n)) for n in wav_lens], device=wav.device), 4, rounding_mode="floor")
        y, _ = self.gru(self.drop(h))
        return self.out(self.drop(y)).log_softmax(-1), lens


class Stream:
    """Incremental decoding of one utterance; output equals SmallNet.forward on the whole audio (greedy CTC)."""

    def __init__(self, net: SmallNet, decode, final=None):
        self.net, self.decode, self.final = net, decode, final
        self.samples = torch.zeros(0)
        self.mel_buf = torch.zeros(0, N_MELS)
        self.state = None
        self.ids: list[int] = []
        self.logp: list[torch.Tensor] = []  # per-frame log-probs (diagnostics / tests)

    @torch.inference_mode()
    def accept(self, pcm) -> str:
        self.samples = torch.cat([self.samples, torch.as_tensor(pcm, dtype=torch.float32)])
        n = LogMel.frames(len(self.samples))
        if n:
            mel = self.net.mel(self.samples[None, : (n - 1) * HOP + N_FFT])[0]
            self.samples = self.samples[n * HOP :]
            self.mel_buf = torch.cat([self.mel_buf, mel])
        k = len(self.mel_buf) // 4 * 4
        if k:
            # conv receptive field: each output step needs 3 inputs; recompute with the previous 4-frame block as context
            ctx = getattr(self, "_ctx", None)
            m = self.mel_buf[:k] if ctx is None else torch.cat([ctx, self.mel_buf[:k]])
            h = self.net.frontend(m[None])[0]
            h = h if ctx is None else h[1:]
            self._ctx = self.mel_buf[k - 4 : k]
            self.mel_buf = self.mel_buf[k:]
            y, self.state = self.net.gru(h[None], self.state)
            lp = self.net.out(y)[0].log_softmax(-1)
            self.logp.append(lp)
            self.ids += lp.argmax(-1).tolist()
        return self.decode(self.ids)

    def finalize(self) -> str:
        """Final text: `final(log-probs)` (e.g. LM beam search) when given, else the greedy hypothesis. Feeds TAIL_S of
        silence first: the causal model emits the last tokens a few frames after the audio that carries them."""
        self.accept(torch.zeros(int(TAIL_S * SR)))
        if self.final is not None and self.logp:
            return self.final(torch.cat(self.logp))
        return self.decode(self.ids)
