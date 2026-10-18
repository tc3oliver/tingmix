# TingMix-Large M4 Encoder Depth Research Report

**Research decision: INVESTIGATE — More Evidence Required.** This is not a product decision and is not evidence against a shallower encoder.

## Hardware / Software Environment

- Target: Mac Studio M4 Max, 64 GB Unified Memory, macOS version to detect at execution.
- Accessible test environment: Linux x86_64, PyTorch CPU-only, MPS unavailable. **The user's Mac is not remotely accessible from this environment.**
- `parity.py` will log actual macOS, Python, PyTorch, Transformers, Git commit, MPS availability, CPU-fallback policy and threads when run on M4.
- Encoder: `google/embeddinggemma-2`, revision `914f7f89142e33e77833254d9c9b90c3cef7303b`.
- Published ASR source: `tc3oliver/TingMix-Large`. Actual Hugging Face release revision and SHA256 of its LoRA, Reader and tokenizer are **pending local download and validation**.

## Pinned Architecture (repository source, not locally verified weight contents)

The existing code describes 12 Gemma4 Conformer layers, 1024 hidden features, 1024→1536 final projection, 25 frames/sec, 304,824,608 audio tower base parameters, LoRA rank 16/alpha32 and ~22.2M Reader parameters. The actual public artifact must pass `load_adapted()` integrity guards before the study can proceed.

## Experiment Design

- Phase 0: official full encoder and staged full-forward parity; shape/length/logits/greedy transcript compare on exactly the same published model.
- Phase 1: L3/L6/L9/L12 hidden states and L12-Proj, for base and ASR LoRA variants, preserving exact sample order.
- Phase 2: frozen 1024→Vocabulary Linear CTC (same seed, tokenizer, train split, loss, optimizer, epochs); 1536-dimensional projected output has a separate parameter budget.
- Phase 3: actual early-exit E6/E9/E12 on 5/15/25-second inputs, 50 synchronized MPS samples after warmup, with CPU preprocessing and full official E2E separately measured.

## ASR Accuracy

**NOT MEASURED.** There are no valid current-run CV zh-TW CER, FLEURS English WER or ASCEND mixed MER values. No legal local train and dev manifests were available in the execution environment. Do not substitute dev/test for training. Earlier published full-model metrics are not cross-depth probe scores.

## Encoder Latency / Memory / Compute

**NOT MEASURED on M4.** No P50/P95/RTF, peak unified memory, or speedup is asserted. E6/E9 return 1024-dimensional hidden states without projection, E12 includes its 1536-dimensional projection. Executed parameter counts can be derived from the model graph only after load. Resident encoder parameters do not decrease merely through early return.

## Failure Cases and Known Limitations

- The M4 environment, MPS operators, published checkpoint and LoRA hashes have not been directly tested here.
- Train/dev near-duplicates require a separate acoustic-content audit even if exact path and file checksums differ.
- A frozen-feature probe is not a fully retrained 6- or 9-layer encoder/Reader; weak linear probing cannot rule out an independently trained early-exit system.
- Cross-language repeatability, confidence intervals and multiple probe seeds remain open research requirements.
- FP16 is not considered a valid default precision for this encoder. Start with FP32; test BF16 only after parity succeeds.

## Reproduce Commands

See [README.md](README.md) for macOS isolated install, mandatory `parity.py`, `extract.py`, `probe.py` and `benchmark.py` commands. Results land in ignored `runs/` JSON files. Tracked [results.json](results.json) contains no fabricated values.

## Decision

**INVESTIGATE — More Evidence Required** solely because direct MPS and independent accuracy measurements are missing. This status must not be interpreted as a measured NO-GO or a GO for MI300X retraining.
