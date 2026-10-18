# TingMix

Real-time speech recognition for **Taiwanese Mandarin, English and intra-sentence Mandarin-English code-switching**.

Status: both models are released on Hugging Face: [tc3oliver/TingMix-Large](https://huggingface.co/tc3oliver/TingMix-Large)
(v3, GPU) and [tc3oliver/TingMix-Small](https://huggingface.co/tc3oliver/TingMix-Small) (CPU, native streaming). What works today:

- `tingmix transcribe | listen | serve` with Embed2ASR-v1 or a TingMix-Large training export, optional character
  n-gram LM decoding (`--lm`), WebSocket streaming with bearer-token auth;
- encoder LoRA fine-tuning of the EmbeddingGemma 2 audio tower jointly with the CTC reader, with a restartable,
  session-independent job manager ([docs/training.md](docs/training.md));
- end-to-end accuracy and streaming evaluation scripts (`evaluation/`).

## Models

| family | target | streaming | status |
|---|---|---|---|
| **TingMix-Large** | GPU, accuracy first | buffered re-decoding of an offline encoder (honest label: not native streaming) | released (v3) |
| **TingMix-Small** | CPU, real time on a regular PC | native stateful streaming (causal GRU-CTC) | released |

Both share one streaming API (partial / final events; a final is never revised).

TingMix-Large starts from [Embed2ASR-v1](https://github.com/tc3oliver/embed2asr) (`tc3oliver/Embed2ASR-v1`): the
EmbeddingGemma 2 audio encoder (~304.8M parameters, 25 frames/s, 1536-d) and a 9.66M-parameter GRU-CTC reader.
TingMix depends on a pinned Embed2ASR release (one-way) and adapts it with encoder LoRA, partial fine-tuning of the
top encoder layers, and reader improvements. The TingMix API does not assume EmbeddingGemma: a model is anything that
implements `tingmix.models.base`.

## Benchmark (TingMix-Large, single test run)

Model: [huggingface.co/tc3oliver/TingMix-Large](https://huggingface.co/tc3oliver/TingMix-Large)

| Test set | Metric | Embed2ASR-v1 | TingMix-Large v2 | TingMix-Large v3 |
|---|---|---|---|---|
| Common Voice zh-TW | CER | 15.62 | 12.26 | **12.02** |
| Parliament A | CER | 19.17 | 17.41 | **17.10** |
| ASCEND mixed | MER | 23.90 | 20.76 | **20.02** |
| NTUML2021 mixed (in-domain: train split used) | MER | 29.48 | 15.15 | **13.39** |
| FLEURS en_us | WER | 24.52 | 19.88 | **19.00** |
| LibriSpeech test-clean | WER | 8.99 | 7.89 | **7.05** |
| AMI IHM | WER | 23.59 | 20.90 | **18.76** |

Embed2ASR-v1 = published greedy numbers; v2 = CTC + character LM; v3 = v2 + attention-decoder rescoring of the CTC
N-best (`attention_decoder.safetensors`, on by default; v2 stays available as revision `v2`). Full table, protocol, caveats and the
earlier NTUML-unseen candidate: [docs/benchmark.md](docs/benchmark.md).

## Install

```bash
pip install torch --index-url https://download.pytorch.org/whl/rocm7.1   # or the CUDA / CPU wheel
pip install "tingmix[large,server] @ git+https://github.com/tc3oliver/tingmix.git"
```

The EmbeddingGemma 2 encoder is downloaded from Hugging Face under Google's terms on first use.

## Use

```bash
LM="--lm lm/lmCN_cv_parl_ascend_ntuml.charlm.json.gz --lm-alpha 0.35 --lm-beta 1.5"   # optional, recommended
tingmix transcribe meeting.wav --model tc3oliver/TingMix-Large $LM
tingmix listen --model tc3oliver/TingMix-Large $LM   # microphone, local model
tingmix serve --model tc3oliver/TingMix-Large $LM --port 8765   # WebSocket server (binds 127.0.0.1)
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
