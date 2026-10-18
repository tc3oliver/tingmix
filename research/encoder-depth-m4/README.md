# TingMix-Large / M4 Max Encoder Depth Study

**State: unmeasured on M4 — research harness only.** This directory is separate from published weights and ongoing MI300X training. The question is whether an ASR-adapted 12-layer, 304,824,608-base-parameter encoder has promising 6- or 9-layer early exits; it is **not** a Small/MLX/Core ML/distillation project.

## Source of truth and isolation

- Published ASR: `tc3oliver/TingMix-Large` (NEVER `LargeModel()` without specifying the release).
- Audio tower: `google/embeddinggemma-2` at `914f7f89142e33e77833254d9c9b90c3cef7303b`.
- `parity.py` resolves the ASR release to an immutable HF commit, reads `tingmix.json`, checks the complete LoRA delta tensor names, computes SHA256 of LoRA/Reader/tokenizer, verifies 12 layers and the base encoder parameter count, and compares full-forward vs the production inference path.
- LoRA rank 16 is expected. If the published artifact is different, the experiment stops.
- Source paths remain unchanged. All samples, learned probe heads and JSON outputs are confined to ignored `runs/`. Do not commit any data or weights.

## macOS setup (run **on the Mac Studio**)

```bash
git clone https://github.com/tc3oliver/tingmix.git
cd tingmix
git switch research/encoder-depth-m4
python3.12 -m venv research/encoder-depth-m4/.venv
source research/encoder-depth-m4/.venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[large,train]'
python -m pip install imageio-ffmpeg safetensors sentencepiece
export PYTORCH_ENABLE_MPS_FALLBACK=0
python -c 'import torch; print(torch.__version__, torch.backends.mps.is_available())'
```

Use your real authenticated HF download access; never save tokens in scripts. For a clean experimental snapshot, record `python -m pip freeze > research/encoder-depth-m4/runs/pip-freeze.txt`. Initial runs use fp32 (bf16 only after a separate parity test; fp16 is not a default, per the model's activation constraints).

## Phase 0 — MPS parity (mandatory gate)

```bash
python research/encoder-depth-m4/parity.py --audio /LOCAL/legal_3s_speech.wav
# If no audio path is passed, a fixed 3-second synthetic signal is used for method validation only.
cat research/encoder-depth-m4/runs/parity.json
```

Fail-closed MPS: no CPU fallback, no unsupported-op silent substitution. If the published release is gated/unavailable, or a MPS op fails, save the error; do **not** continue with latency or feature extraction. Compare SHA, shape, lengths, NaNs, CTC logits and greedy transcript. Optional ROCm cross-device results are only used if pre-existing and read-only.

## Phase 1 — extract both original and ASR-LoRA states

Manifests: local UTF-8 JSONL with `{"audio": "/absolute/path.wav", "text": "normalized label", "lang": "zh-TW"}`; no dev/test audio in training. Example invocation:

```bash
python research/encoder-depth-m4/extract.py \
  --train-set CV=/LOCAL/cv-train.jsonl --train-set FLEURS=/LOCAL/fleurs-train.jsonl \
  --train-set ASCEND=/LOCAL/ascend-train.jsonl \
  --dev-set CV=/LOCAL/cv-dev.jsonl --dev-set FLEURS=/LOCAL/fleurs-dev.jsonl \
  --dev-set ASCEND=/LOCAL/ascend-dev.jsonl
```

Extract L3/L6/L9/L12 at 1024 dimensions and L12-Proj at 1536; names are 1-based. Extraction shares each encoder's full forward for efficiency; base and adapted states remain separate. Every feature file contains FP32 states and a `frames` entry in the index. The exact same sample identifiers are required across variants. Audio clips must be <=29.5 seconds for parity with the production single-window path. Clips longer than this must be consistently segmented upstream, then assigned unique stable manifest rows.

Train/dev exact-content/path intersection fails closed; near-duplicates (e.g. differently encoded crops of the same utterance) require an independent acoustic duplicate audit. **Do not call accuracy evidence independent until that audit is done.** For smoke runs, `--limit 8` is permitted and must be reported as exploratory.

## Phase 2 — frozen-feature Linear CTC

```bash
python research/encoder-depth-m4/probe.py \
  --train-sets CV FLEURS ASCEND \
  --dev-set CV=CER --dev-set FLEURS=WER --dev-set ASCEND=MER \
  --epochs 8 --lr 0.0003 --seed 1337
```

Same labels, vocab, order, seed, AdamW, LR, updates and CTC objective for every layer. Head = stateless LayerNorm + Linear(1024, vocab); L12-Proj uses 1536 and is a **different parameter budget**. CTC feasibility is checked before training; invalid labels are excluded identically from every condition. The probe learns only on cached train embeddings. Dev is only scored. No full Reader is trained, no test set is used. If train data is missing, accuracy is **BLOCKED**, not estimated. MPS CTC unsupported errors are captured rather than passed off as MPS results.

The probe procedure has one seed by default. Repeatability and bootstrap CIs should be established before any GO decision; never interpret a probe score as an E6/E9 complete ASR model.

## Phase 3 — uncached MPS latency

```bash
python research/encoder-depth-m4/benchmark.py --audio /LOCAL/legal_25s_speech.wav \
  --warmup 6 --iterations 50
```

Runs E6/E9/E12 from the **input mel** through subsampling and the real first N layers, with MPS synchronization before/after each timing. E12 includes output projection; E6/E9 return 1024-d states without that projection. Separately measure waveform→mel preprocessing and the original full model end-to-end; E6/E9 end-to-end **cannot** be claimed until there is a trained compatible head. Audio durations 5/15/25s, batch 1, P50/P95/RTF, feature shapes, executed parameter count, device and fp32 are logged. If `--audio` is omitted, the workload is explicitly recorded as synthetic and cannot support accuracy claims. No feature-cache timing counts as encoder inference, and `loaded_encoder_params` remains the full 12-layer resident encoder (no pruning).

## Results / gates

- `runs/parity.json`, `runs/features/extract_results.json`, `runs/latency.json`, `runs/probes/results.json`: **measured outputs, gitignored**.
- `results.json`: tracked status inventory; contains **no invented measurements**.
- `REPORT.md`: current evidence, limitations and research-stage gate.

Prospective GO gate: at least 15% lower encoder P50 **and** at most 5% relative probe error increase over L12, no material language-specific regression, stable confidence intervals/repeats, and split integrity. The threshold is a *research screen*, not a release metric. Otherwise INVESTIGATE or NO-GO only with actual evidence.

Do not modify checkpoints, published HF assets, Attention Sprint, or remote MI300X jobs. This experiment makes no outbound uploads.
