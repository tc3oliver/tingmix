# TingMix-Large / M4 Max Encoder Depth Study

**State: measured on a Mac Studio M4 Max. Decision: INVESTIGATE** (see [REPORT.md](REPORT.md); numbers in [results.json](results.json)).
The question: can the 12-layer, 304,824,608-parameter audio encoder of the released TingMix-Large (ASR LoRA r16) be cut to 6 or 9 layers and keep most of
its ASR ability while saving real MPS inference time? This is a research directory only. It does not modify published weights, the MI300X
training, or the Attention Sprint, and it is not a Small / MLX / Core ML / distillation project. Nothing here uploads data.

## What was done

| Phase | Script | Output (gitignored `runs/`) |
|---|---|---|
| 2a data | `prepare_data.py` | `manifests/*.jsonl`, `manifests/report.json` |
| 0 parity gate | `parity.py` | `parity.json` |
| 1 features | `extract.py` | `features/{base,adapted}/…`, `features/extract_results.json` |
| 3 latency | `benchmark.py` | `latency.json` |
| 2b frozen Linear CTC probe | `probe.py` → `analyze.py` | `probes/*`, `analysis.json` |
| summary | `collect.py` | tracked `results.json` |

`common.py` holds the shared loader (verifies every file of the release against its `SHA256SUMS`, never the default `Embed2ASR-v1`) and the early-exit helpers.

## Data (public corpora, none redistributed)

| Set | Source (Hugging Face dataset @ revision) | Licence | Metric | Used |
|---|---|---|---|---|
| Common Voice zh-TW | `fsicoli/common_voice_22_0` @ `ae911de250fe9375d08e2daa2105b671d659d446` | cc0-1.0 | CER | 2,000 train / 200 dev / 500 test |
| FLEURS en_us | `google/fleurs` @ `70bb2e84b976b7e960aa89f1c648e09c59f894dd` | cc-by-4.0 | WER | 1,000 / 200 / 400 |
| ASCEND (zh/en code-switching) | `CAiRE/ASCEND` @ `737e9800ae31be9932ba8464c80366559bd28424` | cc-by-sa-4.0 | MER | 3,000 / 300 / 600 |

- Official train/dev(validation)/test splits are kept. Each is shuffled with `random.Random(f"1337-{set}-{split}")`; the first eligible rows are kept. Eligible = 1–29.5 s, no digits left after the
  Embed2ASR `numnorm` + `norm` pipeline, and losslessly representable by the release tokenizer. Dev/test were over-drawn ×2 and rows whose normalised text occurs in train were removed (ASCEND dev 13, test 25).
  Speaker overlap, identical audio bytes and acoustic near-duplicates (mean-pooled L6 cosine > 0.98) between splits are audited; all are zero.
- ASCEND transcripts are Simplified Chinese; they are converted to Traditional with OpenCC `s2t` to match the tokenizer.
- Audio, features, manifests and checkpoints stay under the gitignored `runs/`. **Do not upload or commit them.**
- Limitations of the data: Common Voice is a third-party mirror and was not compared with the official Mozilla release; the original data of the LoRA is unknown and may overlap these corpora;
  ASCEND scoring normalisation differs from the others so its absolute numbers are not comparable to CV/FLEURS; ASCEND test has ~2 speakers; the TingMix original manifests were not available.

## Reproduce (on the Mac, from the repo root)

```bash
git switch research/encoder-depth-m4
uv sync --extra large --extra train --extra dev
uv pip install pyarrow opencc-python-reimplemented num2words
export HF_HOME=$PWD/research/encoder-depth-m4/runs/hf    # keep caches inside the ignored runs/ dir
export PYTORCH_ENABLE_MPS_FALLBACK=0 PYTHONPATH=research/encoder-depth-m4
```

1. Download the public data into `runs/data` (anonymous access works, a token only raises the rate limit):

   ```bash
   .venv/bin/python - <<'PY'
   import os
   from huggingface_hub import hf_hub_download
   D = "research/encoder-depth-m4/runs/data"
   files = {
     "google/fleurs": ["data/en_us/%s.tsv" % s for s in ("train", "dev", "test")] + ["data/en_us/audio/%s.tar.gz" % s for s in ("train", "dev", "test")],
     "CAiRE/ASCEND": ["main/train-0000%d-of-00003.parquet" % i for i in range(3)] + ["main/validation-00000-of-00001.parquet", "main/test-00000-of-00001.parquet"],
     "fsicoli/common_voice_22_0": ["transcript/zh-TW/%s.tsv" % s for s in ("train", "dev", "test")] + ["audio/zh-TW/%s/zh-TW_%s_0.tar" % (s, s) for s in ("train", "dev", "test")],
   }
   rev = {"google/fleurs": "70bb2e84b976b7e960aa89f1c648e09c59f894dd", "CAiRE/ASCEND": "737e9800ae31be9932ba8464c80366559bd28424",
          "fsicoli/common_voice_22_0": "ae911de250fe9375d08e2daa2105b671d659d446"}
   for repo, names in files.items():
       for f in names:
           hf_hub_download(repo, f, repo_type="dataset", revision=rev[repo], local_dir=os.path.join(D, {"google/fleurs": "fleurs"}.get(repo, repo.split("/")[1])))
   PY
   ```

2. `.venv/bin/python research/encoder-depth-m4/prepare_data.py` (≈ tens of minutes, one sequential pass over each tar).
3. `.venv/bin/python research/encoder-depth-m4/parity.py --audio <a legal 3–10 s speech wav>`. Fail-closed: MPS only, CPU used solely as the reference. **Must print PASS before anything below.**
4. `.venv/bin/python research/encoder-depth-m4/extract.py` (about 20 GB of float16 features for both variants; float16 is storage only, the probe computes in fp32).
5. `.venv/bin/python research/encoder-depth-m4/benchmark.py --audio <≥4 real speech wav files, e.g. runs/manifests/audio/FLEURS/test/*.wav>` (first four files were used; 5/15/25 s, 6 warm-ups, 50 interleaved iterations, batch 1, fp32).
   Do not run it concurrently with anything else on the GPU.
6. `.venv/bin/python research/encoder-depth-m4/probe.py --epochs 20` (picks one LR on dev from `--pilot-lrs 1e-3 3e-3 1e-2`, then 3 seeds × 2 variants × 5 layers; ≈ 70–80 min; `--lr` skips the pilot).
7. `.venv/bin/python research/encoder-depth-m4/analyze.py` then `.venv/bin/python research/encoder-depth-m4/collect.py`.
8. `.venv/bin/python -m pytest -q` and `ruff check . && ruff format --check .`.

## Probe protocol (identical for every condition)

Frozen cached features → parameter-free LayerNorm → `Linear(dim, 5025)` → CTC (blank 0, `zero_infinity=False`). AdamW (wd 0.01), 5 % warm-up + cosine, batch 16, 20 epochs, seeds 1337/1338/1339, same data order and targets.
L12-Proj (1536-d) has a larger head (7.72 M vs 5.15 M parameters) and is labelled as a different budget. The test split is scored once after training; its labels were never used for the learning-rate choice.
Statistics: utterance-level paired bootstrap (2000 draws) for CIs and contrasts, plus a speaker-cluster bootstrap for Common Voice. Probe results measure how linearly decodable each layer is;
they are **not** the accuracy of a retrained 6- or 9-layer encoder or of the existing Reader on an early exit.

## Decision screen (a research screen, not a release gate)

Encoder P50 at least 15 % lower **and** probe error at most 5 % worse than L12, no material per-language regression, stable intervals, clean splits.
GO would only mean "worth investing in next-stage early-exit training". Early exit here skips compute only: all layers stay resident, so no model-shrink claim is made;
the Reader (52–68 % of end-to-end time) is untouched and its optimisation is Future Work.
