"""WebSocket streaming server and client.

Protocol (one StreamingSession per connection):
  client -> {"type": "start", "sample_rate": 16000, "hop_ms": 480, "silence_ms": 600}   (hop_ms / silence_ms optional)
  server -> {"type": "ready"}
  client -> binary frames of little-endian PCM16 mono audio
  server -> {"type": "partial" | "final", "segment_id", "revision", "text", "start", "end", ...}   (see core.events)
  client -> {"type": "stop"}
  server -> remaining finals, then {"type": "done"}; errors are {"type": "error", "message"} followed by a close.

Security: binds 127.0.0.1 by default and refuses a non-loopback bind without a token. The token is sent as
`Authorization: Bearer <token>` (compared in constant time) but the transport is plain ws://; for remote use put the
server behind an SSH tunnel (`ssh -L 8765:127.0.0.1:8765 gpu-host`) or a TLS reverse proxy and connect with wss://."""

from __future__ import annotations

import asyncio
import hmac
import json
import threading
from http import HTTPStatus

from websockets.asyncio.client import connect
from websockets.asyncio.server import serve as _serve

from ..core.audio import float_to_pcm16, pcm16_to_float
from ..core.events import event_from_dict
from ..streaming.session import StreamingSession

MAX_MESSAGE_BYTES = 256 * 1024  # ~8 s of PCM16 at 16 kHz
MAX_SESSION_S = 3600
KNOBS = ("hop_ms", "silence_ms")


class _SharedModel:
    """One model for all sessions; GPU calls are serialized."""

    def __init__(self, model):
        self._model, self._lock = model, threading.Lock()
        self.sample_rate, self.max_window_s = model.sample_rate, model.max_window_s

    def transcribe_batch(self, wavs):
        with self._lock:
            return self._model.transcribe_batch(wavs)


async def _send(ws, obj) -> None:
    await ws.send(json.dumps(obj, ensure_ascii=False))


def serve(model, host: str = "127.0.0.1", port: int = 8765, token: str | None = None, max_session_s: float = MAX_SESSION_S):
    """-> websockets server, usable as `async with serve(...) as server: await server.serve_forever()`."""
    if host not in ("127.0.0.1", "localhost", "::1") and not token:
        raise ValueError("refusing a non-loopback bind without a token (--token / TINGMIX_TOKEN)")
    shared = _SharedModel(model)

    def authorize(connection, request):
        if token and not hmac.compare_digest(request.headers.get("Authorization", "").encode(), f"Bearer {token}".encode()):
            return connection.respond(HTTPStatus.UNAUTHORIZED, "unauthorized\n")

    async def handler(ws):
        session = None
        audio_s = 0.0
        try:
            async with asyncio.timeout(max_session_s):
                async for msg in ws:
                    if isinstance(msg, str):
                        d = json.loads(msg)
                        if d.get("type") == "start" and session is None:
                            if d.get("sample_rate") != shared.sample_rate:
                                raise ValueError(f"sample_rate must be {shared.sample_rate}")
                            knobs = {k: float(d[k]) for k in KNOBS if k in d}
                            if any(not 100 <= v <= 5000 for v in knobs.values()):
                                raise ValueError("hop_ms / silence_ms must be within 100..5000")
                            session = StreamingSession(shared, **knobs)
                            await _send(ws, {"type": "ready"})
                        elif d.get("type") == "stop" and session is not None:
                            for ev in await asyncio.to_thread(session.flush):
                                await _send(ws, ev.to_dict())
                            await _send(ws, {"type": "done"})
                            break
                        else:
                            raise ValueError(f"unexpected message {d.get('type')!r}")
                    else:
                        if session is None or len(msg) % 2:
                            raise ValueError("audio must follow 'start' and be whole PCM16 samples")
                        audio_s += len(msg) / 2 / shared.sample_rate
                        if audio_s > max_session_s:
                            raise ValueError("session audio limit reached")
                        for ev in await asyncio.to_thread(session.feed, pcm16_to_float(msg)):
                            await _send(ws, ev.to_dict())
        except (ValueError, TimeoutError) as e:  # json.JSONDecodeError is a ValueError
            await _send(ws, {"type": "error", "message": str(e) or type(e).__name__})
        await ws.close()

    return _serve(
        handler, host, port, max_size=MAX_MESSAGE_BYTES, process_request=authorize, close_timeout=1
    )  # a client that stopped reading must not hold the connection for 10 s


async def stream(url: str, chunks, token: str | None = None, sample_rate: int = 16000, **knobs):
    """Async generator: send float32 chunks (an async iterator) to a server, yield its events until it is done."""
    headers = {"Authorization": f"Bearer {token}"} if token else None
    async with connect(url, additional_headers=headers, max_size=MAX_MESSAGE_BYTES) as ws:
        await _send(ws, {"type": "start", "sample_rate": sample_rate, **knobs})

        async def send_audio():
            async for chunk in chunks:
                await ws.send(float_to_pcm16(chunk))
            await _send(ws, {"type": "stop"})

        sender = asyncio.create_task(send_audio())
        try:
            async for msg in ws:
                d = json.loads(msg)
                if d["type"] == "error":
                    raise RuntimeError(d["message"])
                if d["type"] == "done":
                    break
                if d["type"] != "ready":
                    yield event_from_dict(d)
        finally:
            sender.cancel()
            await asyncio.gather(sender, return_exceptions=True)
