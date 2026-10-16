import asyncio

import numpy as np
import pytest
from conftest import FakeModel, chunks, silence, tone

pytest.importorskip("websockets")
from websockets.exceptions import InvalidStatus  # noqa: E402

from tingmix.server.websocket import serve, stream  # noqa: E402

AUDIO = np.concatenate([tone(1.5), silence(1.5)])


async def source():
    for c in chunks(AUDIO):
        yield c


async def collect(port, token=None):
    return [e async for e in stream(f"ws://127.0.0.1:{port}", source(), token=token)]


def port_of(server):
    return server.sockets[0].getsockname()[1]


def test_round_trip():
    async def go():
        async with serve(FakeModel(), port=0) as server:
            return await collect(port_of(server))

    events = asyncio.run(go())
    assert [e.type for e in events][-1] == "final" and events[0].type == "partial"
    assert events[-1].text.startswith("a")


def test_token_required():
    async def go():
        async with serve(FakeModel(), port=0, token="secret") as server:
            ok = await collect(port_of(server), "secret")
            with pytest.raises(InvalidStatus):
                await collect(port_of(server), "wrong")
            with pytest.raises(InvalidStatus):
                await collect(port_of(server))
            return ok

    assert asyncio.run(go())[-1].type == "final"


def test_bad_sample_rate_and_public_bind():
    async def go():
        async with serve(FakeModel(), port=0) as server:
            with pytest.raises(RuntimeError, match="sample_rate"):
                [e async for e in stream(f"ws://127.0.0.1:{port_of(server)}", source(), sample_rate=8000)]

    asyncio.run(go())
    with pytest.raises(ValueError):
        serve(FakeModel(), host="0.0.0.0")
