"""Stream the microphone to a TingMix server and print the transcript (same client code as `tingmix listen --server`).

pip install 'tingmix[server,mic]'
ssh -L 8765:127.0.0.1:8765 gpu-host        # or reach the server through a TLS reverse proxy (wss://)
python mic_client.py ws://127.0.0.1:8765 [token]
"""

import asyncio
import sys

from tingmix.cli import mic_source, show
from tingmix.server.websocket import stream


async def main(url: str, token: str | None) -> None:
    async for ev in stream(url, mic_source(), token=token):
        show(ev, as_json=False)


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None))
