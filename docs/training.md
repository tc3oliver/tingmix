# Training TingMix-Large

TingMix-Large starts from [Embed2ASR-v1](https://github.com/tc3oliver/embed2asr) (EmbeddingGemma 2 audio encoder,
~304.8M parameters, + a 9.66M-parameter GRU-CTC reader, 25 frames/s) and adapts it. One trainer covers three modes:

| mode | trainable | encoder input | notes |
|---|---|---|---|
| `frozen` | reader | audio, or cached final features (`features: cache`) | control / reader experiments |
| `lora` | LoRA on q/k/v/post and both feed-forward blocks of every layer + reader | **audio only** | gradients reach layer 1; a cached intermediate state would silently cut them, so the trainer refuses `features: cache` |
| `partial` | layers 11-12 + `output_proj` (fp32 master weights) + reader | audio (layers 1-10 run without autograd), or cached layer-10 states | |

The reader is warm-started from an Embed2ASR checkpoint (`init:`) and keeps its vocabulary (CJK characters +
English BPE pieces), so step 0 of every run reproduces the starting model under the same pipeline; that evaluation
is the reference for model selection (`score` = mean of each dev metric divided by its step-0 value).

```bash
python -m training.large.train --config configs/large-lora.example.yaml
```

## Data

Manifests are jsonl, one utterance per line: `audio` (path), `text` (normalized: Traditional Chinese, lowercase
English), `lang` (`zh-TW`, `en-US`, `mix`), `seconds`, `frames` (= encoder frames at 25 fps). Each training pool is
`PATH:HOURS_PER_EPOCH`; every epoch takes the next unused rows of each pool (rotate sampler), so the data order is a
pure function of the seed and the epoch. Dev sets are `NAME=PATH:METRIC` with `cer`, `wer`, `mer` or `mixmer`
(MER on code-switched rows only). Never put test sets in `dev`.

No training data, manifests or checkpoints are distributed with this repository.

## Long runs that survive the session: `tools/tingmix_train.py`

```bash
tingmix-train launch <exp>      # $TINGMIX_HOME/experiments/<exp>.yaml = trainer options + job: {gpu, restarts, memory, cpus}
tingmix-train status            # container state, trainer state, last step / eval, free GPU memory
tingmix-train logs <exp> -f
tingmix-train stop <exp>        # SIGTERM: the trainer writes a checkpoint and exits (PAUSED)
tingmix-train resume <exp>      # same frozen code; --new-code to deliberately switch code
tingmix-train reconcile         # supervisor tick (run it from a systemd timer or cron)
```

How it stays up:

- every run is one detached Docker container with a fixed name (`tingmix-<exp>`; a second launch is refused) and
  `--restart=on-failure:N`; it runs a `git archive` snapshot of the code in the run dir, never the live working tree;
- the trainer holds an exclusive lock on its run dir, so two trainers can never write the same run;
- `resume.pt` (+ `resume.prev.pt`) is written atomically (tmp + fsync + rename) every `save_every_min` and at every
  eval: model / adapter weights, optimizer, LR scheduler, step, epoch, position in the epoch, Python / NumPy / torch /
  GPU RNG states, config, code and data hashes, accumulated training time (the `max_hours` budget is not reset by a
  restart) and best dev metrics. On restart it continues the same data order exactly;
- resume refuses a different config, tokenizer, data or code version (unless `--new-code` was asked for) and marks the
  run FAILED instead of pretending to continue the same experiment;
- permanent errors (out of memory, incompatible checkpoint, non-finite loss) set `status.json` to FAILED and exit 0,
  so nothing restarts in a loop; other crashes exit non-zero and Docker restarts them;
- `reconcile` restarts runs that should be running but are not (host or Docker daemon restart, `docker kill`,
  exhausted restart budget), at most 5 times.

Host settings (`TINGMIX_HOME`, image, GPU render nodes, read-only data mounts) come from environment variables or
`$TINGMIX_HOME/host.env`; see the docstring of `tools/tingmix_train.py`.
