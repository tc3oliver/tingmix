# Streaming

Both model families share one streaming API: `StreamingSession(model).feed(float32 chunk) -> events`, `.flush() -> events`.

## Buffered streaming (TingMix-Large) vs native streaming (TingMix-Small)

TingMix-Large is an **offline, bidirectional** model (EmbeddingGemma 2 encoder + GRU-CTC reader, 29.5 s window). It is **not**
a native streaming model. The session emulates streaming by *re-decoding*: while a segment is open, the whole segment audio is
run through the model every `hop_ms`, and the result is emitted as a partial. At an endpoint the segment is decoded once more,
emitted as a final, and its audio is dropped. Consequences:

- compute per update grows with segment length (bounded by the 29.5 s window; the scheduler stretches the hop if decoding is slow);
- partials can change anywhere in the text, not only at the tail; `stable_prefix_len` marks what agreed across the last two partials;
- a segment forced to end at `max_segment_s` (default 25 s) can split a word;
- a final can differ slightly from decoding the same clip offline (different padding / preroll; the encoder is not padding-invariant).

TingMix-Small (CPU) is planned as a natively stateful streaming model (`streaming_native = True`): audio is consumed once, no
re-decoding. Only the interface exists today (`tingmix.models.small`).

## Pipeline

audio -> `RingBuffer` -> `EnergyVAD` (20 ms frames, hysteresis; any object with `frame_ms` and `update(frame) -> bool` can replace it)
-> `Endpointer` (segment opens on speech; closes after `silence_ms` of non-speech or at `max_segment_s`)
-> `Scheduler` (decode every `hop_ms` of new segment audio) -> model -> `StablePrefix` -> `Transcript` (revision numbers; finals immutable).

## Latency knobs

| knob | default | effect |
|---|---|---|
| `hop_ms` | 480 | partial update rate; lower = more GPU work per second of audio |
| `silence_ms` | 600 | wait before a segment is finalized; lower = faster finals, more mid-sentence cuts |
| `max_segment_s` | 25 | forced cut (capped below the model window minus preroll) |
| `preroll_ms` / `tail_pad_ms` | 800 / 0 | audio kept before the detected onset / extra kept after the VAD's last speech frame (the 200 ms VAD hangover is kept anyway) |
| `EnergyVAD(threshold_db)` | -50 dBFS | fixed threshold; no noise-floor tracking |

The defaults come from a sweep over preroll {300,500,800} ms x tail pad {0..700} ms x VAD threshold {-45,-50} dBFS on 100 Locked-Dev clips
(50 ASCEND + 50 CV) plus a 61.7 s concatenated stream (all settings segmented it 13/13 and kept the Partial/Revision/Final order).
The encoder is sensitive to the audio around the speech, so the final matches offline decoding best when the segment keeps the same
leading context (long preroll) and does not add extra trailing audio: final-vs-offline MER went from 6.7% to 4.6% (ASCEND) and
2.7% to 0.3% (CV); final-vs-reference MER is unchanged within +-0.5 pt. Pad length does not change latency from the end of speech
(that is `silence_ms` plus compute). Remaining ASCEND difference is because its clips are cut tightly, so any context differs from the clip.

## Events (JSON)

```json
{"type": "partial", "segment_id": 0, "revision": 2, "text": "...", "start": 0.0, "end": 1.9, "stable_prefix_len": 4}
{"type": "final",   "segment_id": 0, "revision": 3, "text": "...", "start": 0.0, "end": 2.4}
```

`revision` counts up per segment; the final is the last revision and is never revised. Times are seconds from the stream start.

## WebSocket protocol

```
client -> {"type": "start", "sample_rate": 16000, "hop_ms": 480, "silence_ms": 600}    # knobs optional, 100..5000
server -> {"type": "ready"}
client -> binary frames: little-endian PCM16 mono, 16 kHz (<= 256 KiB per message)
server -> partial / final events
client -> {"type": "stop"}
server -> remaining finals, {"type": "done"}
```

Errors are `{"type": "error", "message": ...}` followed by a close. Sessions are limited to 3600 s of audio and wall-clock time.
One model is shared by all connections; model calls are serialized by a lock.

```
tingmix serve --model large --port 8765 [--token T]      # or TINGMIX_TOKEN
tingmix listen --server ws://127.0.0.1:8765 [--file a.wav] [--token T]
```

Security: the server binds to 127.0.0.1 and refuses a non-loopback `--host` without a token. The bearer token
(`Authorization: Bearer ...`, constant-time compare) travels in clear over `ws://`, so reach a remote GPU host through an SSH tunnel
(`ssh -L 8765:127.0.0.1:8765 gpu-host`) or a TLS reverse proxy (`wss://`). `examples/mic_client.py` streams a Mac microphone.

## Measuring

`evaluation/streaming.py MANIFEST.jsonl --model-path ... --out result.json` replays clips (and a concatenated stream) through a
session and reports per-update compute, first-partial and finalization latency. Finalization latency is measured from the end
of the kept segment audio, so it is roughly `silence_ms - tail_pad_ms` plus compute.
