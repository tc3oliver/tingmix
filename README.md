# TingMix

Real-time speech recognition for **Taiwanese Mandarin, English and intra-sentence Mandarin-English code-switching**.

Status: early development. No TingMix-Large weights are released yet; test numbers are published only after a single,
pre-registered test run (see [Evaluation](#evaluation)). What works today:

- `tingmix transcribe | listen | serve` with Embed2ASR-v1 or a TingMix-Large training export, optional character
  n-gram LM decoding (`--lm`), WebSocket streaming with bearer-token auth;
- encoder LoRA fine-tuning of the EmbeddingGemma 2 audio tower jointly with the CTC reader, with a restartable,
  session-independent job manager ([docs/training.md](docs/training.md));
- end-to-end accuracy and streaming evaluation scripts (`evaluation/`).

## Models

| family | target | streaming | status |
|---|---|---|---|
| **TingMix-Large** | GPU, accuracy first | buffered re-decoding of an offline encoder (honest label: not native streaming) | in training |
| **TingMix-Small** | CPU, real time on a regular PC | native stateful streaming (planned) | interface only |

Both share one streaming API (partial / final events; a final is never revised).

TingMix-Large starts from [Embed2ASR-v1](https://github.com/tc3oliver/embed2asr) (`tc3oliver/Embed2ASR-v1`): the
EmbeddingGemma 2 audio encoder (~304.8M parameters, 25 frames/s, 1536-d) and a 9.66M-parameter GRU-CTC reader.
TingMix depends on a pinned Embed2ASR release (one-way) and adapts it with encoder LoRA, partial fine-tuning of the
top encoder layers, and reader improvements. The TingMix API does not assume EmbeddingGemma: a model is anything that
implements `tingmix.models.base`.

## Benchmark (TingMix-Large candidate, single test run)

| Test set | Metric | Embed2ASR-v1 | TingMix-Large |
|---|---|---|---|
| Common Voice zh-TW | CER | 15.62 | **12.48** |
| Parliament A | CER | 19.17 | **17.35** |
| ASCEND mixed | MER | 23.90 | **20.86** |
| NTUML2021 mixed (evaluation only) | MER | 29.48 | **26.12** |
| FLEURS en_us | WER | 24.52 | **19.71** |
| LibriSpeech test-clean | WER | 8.99 | **7.91** |

Embed2ASR-v1 = published greedy numbers; TingMix-Large uses its character LM. Full table, protocol and caveats:
[docs/benchmark.md](docs/benchmark.md). Weights are not released yet (licence review pending).

## Install

```bash
pip install torch --index-url https://download.pytorch.org/whl/rocm7.1   # or the CUDA / CPU wheel
pip install "tingmix[large,server] @ git+https://github.com/tc3oliver/tingmix.git"
```

The EmbeddingGemma 2 encoder is downloaded from Hugging Face under Google's terms on first use.

## Use

```bash
tingmix transcribe meeting.wav --model large
tingmix listen --model large                         # microphone, local model
tingmix serve --model large --port 8765              # WebSocket server (binds 127.0.0.1)
tingmix listen --server ws://127.0.0.1:8765          # e.g. a Mac microphone through an SSH tunnel to a GPU host
```

Streaming protocol and latency knobs: [docs/streaming.md](docs/streaming.md).

## Training

[docs/training.md](docs/training.md): LoRA / partial / frozen modes, data format, and the restartable job manager
(`tools/tingmix_train.py`) that keeps long runs alive across session loss, worker crashes and container restarts.
Training data, manifests, feature caches and private checkpoints are not part of this repository.

## Evaluation

Model selection uses development sets only. Test sets are run once per candidate release, and every number in a
report comes from one checkpoint. CER for Mandarin, WER for English, MER (CJK characters + English words) for
code-switching; the metrics are never averaged together.

## License

Code: Apache-2.0. Model weights (when released) carry the terms of their base model and training data; see the model
card of each release.
