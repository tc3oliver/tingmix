# TingMix-Large benchmark

Released model: [tc3oliver/TingMix-Large](https://huggingface.co/tc3oliver/TingMix-Large) v3 (2026-10-18; v2 = revision `v2`).

## v3: attention-decoder rescoring (single test run)

v2 unchanged (encoder LoRA, Reader, LM) plus a 30.8M-parameter Transformer decoder (6 layers, d 512) trained on the
Reader states with everything else frozen (28k steps, 4 GPU-hours). Decoding: CTC + LM prefix beam (beam 16) -> 8 best
hypotheses -> score = beam score + 1.0 x attention log-probability. The weight was chosen on even-numbered dev rows and
checked on odd-numbered rows; the test sets were run once, after the choice.

| Test set | Metric | v2 | v3 | paired difference, 95% bootstrap CI (points) |
|---|---|---|---|---|
| Common Voice zh-TW 27.0 | CER | 12.26 | **12.02** | -0.23 [-0.41, -0.07] |
| Parliament A | CER | 17.41 | **17.10** | -0.31 [-0.44, -0.18] |
| Parliament B | CER | 17.86 | **17.57** | -0.29 [-0.34, -0.23] |
| ASCEND, mixed sentences | MER | 20.76 | **20.02** | -0.86 [-1.21, -0.50] (all rows) |
| NTUML2021, mixed sentences (in-domain) | MER | 15.15 | **13.39** | -1.24 [-1.33, -1.14] (all rows) |
| FLEURS en_us | WER | 19.88 | **19.00** | -0.88 [-1.11, -0.66] |
| LibriSpeech test-clean | WER | 7.89 | **7.05** | -0.85 [-0.95, -0.74] |
| LibriSpeech test-other | WER | 17.28 | **16.11** | -1.17 [-1.29, -1.05] |
| VoxPopuli en | WER | 13.61 | **12.78** | -0.82 [-0.94, -0.71] |
| AMI IHM | WER | 20.90 | **18.76** | -2.14 [-2.26, -2.01] |

Development sets (full, end-to-end): without any LM the rescoring alone improves every set over greedy CTC
(CV 11.70 -> 10.89, NTUML 24.16 -> 21.14, FLEURS 20.47 -> 18.89, LibriSpeech dev 7.82 -> 6.92, VoxPopuli dev 13.49 -> 12.40;
LibriSpeech and VoxPopuli dev were not used for any choice). Streaming (50 dev clips per set): final CER/MER CV 10.24 -> 9.43,
ASCEND 19.70 -> 18.81, NTUML 14.92 -> 13.31; per-update compute p50 93 -> 108 ms, final latency p50 0.743 -> 0.757 s.
CPU inference works but is not real time for Large (RTF about 7 for v2 and v3 alike).

Ablations (dev): a decoder reading the encoder features instead of the Reader states is on par at equal steps; joint
CTC + attention fine-tuning of the Reader and LoRA (0.7/0.3) is within seed noise of a matched control without the
attention loss, i.e. the attention loss did not improve the CTC model itself; the gain comes from rescoring.

## v2 (single test run)

Fine-tuned from the earlier candidate below with the NTUML2021 **train** split added (official lecture-disjoint
train/test split: 0 shared lectures; 2 train lectures held out as NTUML dev for selection). NTUML2021 test is therefore
an in-domain test for this model. Decoder: character 5-gram LM (CV + Parliament + ASCEND + NTUML-train transcripts,
all dev/test sentences removed), alpha 0.35, beta 1.5, beam 16.

| Test set | Metric | Embed2ASR-v1 greedy | TingMix-Large (released) |
|---|---|---|---|
| Common Voice zh-TW 27.0 | CER | 15.62 | **12.26** |
| Parliament A | CER | 19.17 | **17.41** |
| Parliament B | CER | 19.31 | **17.86** |
| ASCEND, mixed sentences | MER | 23.90 | **20.76** |
| NTUML2021, mixed sentences (in-domain) | MER | 29.48 | **15.15** |
| FLEURS en_us | WER | 24.52 | **19.88** |
| LibriSpeech test-clean | WER | 8.99 | **7.89** |
| LibriSpeech test-other | WER | 19.72 | **17.28** |
| VoxPopuli en | WER | 15.31 | **13.61** |
| AMI IHM | WER | 23.59 | **20.90** |

Streaming (released model, buffered re-decoding, 50 dev clips per set + a 61.7 s concatenated stream): final MER/CER
stream vs offline — Common Voice 10.24 vs 10.51, ASCEND mixed 19.40 vs 18.81, NTUML dev mixed 14.72 vs 17.34; per-update
compute p50 55 ms / p95 73 ms on one MI300X; first partial p50 0.55 s; final 0.71 s (p50) after the segment ends;
13/13 segments found in the concatenated stream.

## Earlier candidate (NTUML never seen)

The numbers below are kept as the unseen-domain reference: NTUML2021 was not used for training, the LM or selection.

One checkpoint, one test run. All model, decoder and LM-weight choices were made on development sets only; the test
sets below were run once on the final candidate. Scoring is identical to the Embed2ASR v0.1.0 benchmark
(`embed2asr.metrics.score` per language: CER for Mandarin rows, WER for English rows, MER for code-switched rows;
rows whose reference contains digits are dropped), so the Embed2ASR-v1 columns are its published numbers. Both TingMix
columns come from the same checkpoint (two decoders, each run once). Re-scoring Embed2ASR-v1's stored test outputs with
`evaluation/accuracy.py`'s scorer reproduces them exactly.

## Model

- Encoder: `google/embeddinggemma-2` audio tower (frozen base weights) + LoRA (r 16, alpha 32) on q/k/v/post and both
  feed-forward blocks of all 12 layers (~5.5M parameters).
- Reader: GRU-CTC, d 768, BiGRU 384 x 2, 4 conv blocks (22.2M parameters), first trained on frozen encoder features
  (80 epochs, Embed2ASR v0.1.0 trainer), then fine-tuned jointly with the LoRA adapters (12k steps, cosine, encoder
  LR 1e-4, reader LR 2e-5).
- Decoder: CTC prefix beam search (beam 16) with a character 5-gram LM (alpha 0.35, beta 1.5; LM text from the
  training sets' Mandarin transcripts, all dev/test sentences removed). English pieces are LM-neutral.

## Results (error rates in %, lower is better)

| Test set | Metric | Embed2ASR-v1 greedy | Embed2ASR-v1 + LM | TingMix-Large greedy | TingMix-Large + LM |
|---|---|---|---|---|---|
| Common Voice zh-TW 27.0 | CER | 15.62 | 14.72 | 14.39 | **12.48** |
| Parliament A | CER | 19.17 | 19.01 | 18.49 | **17.35** |
| Parliament B | CER | 19.31 | 19.22 | 19.04 | **17.79** |
| ASCEND, mixed sentences | MER | 23.90 | 22.93 | 22.21 | **20.86** |
| NTUML2021, mixed sentences (evaluation only) | MER | 29.48 | 28.92 | 28.21 | **26.12** |
| FLEURS en_us | WER | 24.52 | – | 20.06 | **19.71** |
| LibriSpeech test-clean | WER | 8.99 | – | 8.03 | **7.91** |
| LibriSpeech test-other | WER | 19.72 | – | 17.53 | **17.38** |
| VoxPopuli en | WER | 15.31 | – | 13.97 | **13.69** |
| AMI IHM | WER | 23.59 | – | 21.37 | **21.05** |

Reproduce: `python -m evaluation.accuracy --model <release dir> --lm <lm> --lm-alpha 0.35 --lm-beta 1.5 --by-lang
--set NAME=MANIFEST:METRIC:nd ...` (manifests are not distributed; see docs/training.md for the expected format).

## Caveats

- NTUML2021 was never used for training or model selection; it is the unseen-domain Taiwanese-accent code-switching
  test. Its licence is not stated, so it is evaluation-only.
- ASCEND speakers are from Hong Kong; Parliament references are not verbatim transcripts.
- The LM includes Parliament (CC BY-NC 4.0) and ASCEND (CC BY-SA 4.0) text; the reader and adapters were trained on
  Parliament speech. See the licence review in [model-card-large.md](model-card-large.md). No weights are released yet.
- Streaming (buffered re-decoding, Locked Dev clips): final text matches offline decoding on 90% of Common Voice and
  82% of ASCEND clips; per-update compute p50 82 ms / p95 117 ms on one shared MI300X; final result 0.74 s (p50) after
  the speech segment ends.
