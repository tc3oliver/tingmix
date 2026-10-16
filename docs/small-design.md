# TingMix-Small design (interface only; not trained yet)

Goal: real-time Taiwanese Mandarin / English / code-switching ASR on a regular PC CPU, behind the same streaming API as
TingMix-Large (`StreamingSession` events, WebSocket protocol, CLI).

## Why a different model

TingMix-Large re-decodes the whole open segment with an offline, bidirectional encoder every hop (~0.5 s). Compute per
update grows with segment length, and the 304.8M-parameter encoder needs a GPU. Small must consume each audio frame
once and keep state, so its cost per second of audio is constant.

## Interface (src/tingmix/models/base.py)

- `ASRModel`: `name`, `sample_rate`, `max_window_s`, `streaming_native`, `transcribe_batch(wavs) -> texts` (offline use,
  shared with Large).
- `StreamingASRModel.new_stream() -> Stream`; `Stream.accept(pcm) -> current hypothesis`, `Stream.finalize() -> final text`.
  With `streaming_native = True` the session will open one stream per VAD segment, call `accept` on every chunk (partials
  from its return value, stability from the same `StablePrefix`), and `finalize` at the endpoint (final event).
  Events, revision numbers and the "final is immutable" rule are unchanged, so clients cannot tell Large from Small.

## Candidate model (to be decided with data, not now)

- Chunked / causal conformer or Zipformer encoder with limited right context (e.g. 160-320 ms), CTC or transducer
  head, int8 ONNX on CPU; 20-80M parameters.
- Vocabulary: may differ from Large (the API does not share tokenizers).
- Training signal: the same licensed data; distillation from TingMix-Large hypotheses is an option for untranscribed
  audio. Acceptance: real-time factor < 0.3 on a 4-core laptop CPU, first partial < 0.5 s, dev CER/MER reported next to
  Large on the same Locked Dev sets.
