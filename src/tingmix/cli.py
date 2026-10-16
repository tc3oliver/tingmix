"""`tingmix transcribe | listen | serve`."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time

from .core.audio import SAMPLE_RATE

CHUNK_MS = 100


async def file_source(path: str, realtime: bool = True):
    """Chunks of a file, paced like a live microphone."""
    from .core.audio import load_audio

    wav = load_audio(path)
    n = SAMPLE_RATE * CHUNK_MS // 1000
    t0 = time.monotonic()
    for i in range(0, len(wav), n):
        if realtime:
            await asyncio.sleep(max(0.0, t0 + i / SAMPLE_RATE - time.monotonic()))
        yield wav[i : i + n]


async def mic_source():
    try:
        import sounddevice as sd
    except ImportError as e:
        raise SystemExit("microphone input needs sounddevice (pip install 'tingmix[mic]'); or use --file") from e
    loop, q = asyncio.get_running_loop(), asyncio.Queue()
    n = SAMPLE_RATE * CHUNK_MS // 1000
    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        blocksize=n,
        callback=lambda x, *_: loop.call_soon_threadsafe(q.put_nowait, x[:, 0].copy()),
    ):
        while True:
            yield await q.get()


def show(ev, as_json: bool) -> None:
    if as_json:
        print(json.dumps(ev.to_dict(), ensure_ascii=False), flush=True)
    elif ev.type == "partial":
        print(f"\r\033[K{ev.text}", end="", flush=True)
    else:
        print(f"\r\033[K{ev.text}", flush=True)


def _model(args):
    import numpy as np

    from .core.registry import load_model

    model = load_model(args.model, **({"lm": args.lm} if args.lm else {}))
    model.transcribe_batch([np.zeros(model.sample_rate, dtype=np.float32)])  # warm-up: the first call is slow (kernel tuning)
    return model


async def listen(args) -> None:
    source = file_source(args.file) if args.file else mic_source()
    if args.server:
        from .server.websocket import stream

        events = stream(args.server, source, token=args.token or os.environ.get("TINGMIX_TOKEN"))
    else:
        from .streaming.session import StreamingSession

        session = StreamingSession(_model(args))

        async def local():
            async for chunk in source:
                for ev in session.feed(chunk):
                    yield ev
            for ev in session.flush():
                yield ev

        events = local()
    async for ev in events:
        show(ev, args.json)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="tingmix")
    sub = p.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("transcribe", help="offline transcription of an audio file")
    t.add_argument("file")
    t.add_argument("--model", default="large", help="large | TingMix .pt export | Embed2ASR release dir")
    t.add_argument("--lm", help="character n-gram LM file (Embed2ASR format): CTC prefix beam search with the LM")
    t.add_argument("--json", action="store_true")
    li = sub.add_parser("listen", help="live transcription from the microphone (or --file, paced in real time)")
    li.add_argument("--model", default="large", help="large | TingMix .pt export | Embed2ASR release dir")
    li.add_argument("--lm", help="character n-gram LM file (Embed2ASR format): CTC prefix beam search with the LM")
    li.add_argument("--file")
    li.add_argument("--server", help="ws:// or wss:// URL of a `tingmix serve` instance instead of a local model")
    li.add_argument("--token", help="bearer token for --server (default: $TINGMIX_TOKEN)")
    li.add_argument("--json", action="store_true", help="one JSON event per line")
    s = sub.add_parser("serve", help="WebSocket streaming server")
    s.add_argument("--model", default="large", help="large | TingMix .pt export | Embed2ASR release dir")
    s.add_argument("--lm", help="character n-gram LM file (Embed2ASR format): CTC prefix beam search with the LM")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8765)
    s.add_argument("--token", help="bearer token (default: $TINGMIX_TOKEN)")
    args = p.parse_args(argv)

    if args.cmd == "transcribe":
        from .api import transcribe

        text = transcribe(args.file, _model(args))
        print(json.dumps({"file": args.file, "text": text}, ensure_ascii=False) if args.json else text)
    elif args.cmd == "listen":
        try:
            asyncio.run(listen(args))
        except KeyboardInterrupt:
            sys.exit(130)
    else:
        from .server.websocket import serve

        async def run():
            async with serve(_model(args), args.host, args.port, args.token or os.environ.get("TINGMIX_TOKEN")) as server:
                print(f"serving on ws://{args.host}:{args.port}", file=sys.stderr, flush=True)
                await server.serve_forever()

        asyncio.run(run())


if __name__ == "__main__":
    main()
