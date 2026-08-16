"""Record a provider once, replay it forever, and refuse the request nobody recorded.

Run it with `uv run python examples/cassettes.py`.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from tempfile import TemporaryDirectory

from tesserix_adk.core import Message, ModelRequest, TextPart
from tesserix_adk.testing import (
    CassetteHarness,
    CassetteMismatchError,
    CassetteMode,
    FakeModelProvider,
    MatchOn,
    ScriptedTurn,
    mode_from_env,
)


def asked(text: str) -> ModelRequest:
    """One user turn, as the runtime would assemble it."""
    turn = Message(role="user", content=[TextPart(text=text)])
    return ModelRequest(model="fake-1", messages=(turn,))


async def main() -> None:
    """Record against a stand-in provider, replay it, then diverge from the recording."""
    with TemporaryDirectory() as directory:
        path = Path(directory) / "refunds.json"
        print(f"the environment asks for {mode_from_env().value}")  # noqa: T201

        live = FakeModelProvider(ScriptedTurn.saying("refunded 12.00", output_tokens=6))
        recorder = CassetteHarness(path, mode=CassetteMode.RECORD, provider="openai")
        answer = await recorder.provider(live).complete(asked("refund the charge"))
        recorder.save()
        print(f"recorded: {answer.content}")  # noqa: T201

        replay = CassetteHarness(path).provider()
        again = await replay.complete(asked("refund the charge"))
        print(f"replayed without a socket: {again.content}")  # noqa: T201

        try:
            await replay.complete(asked("refund a different charge"))
        except CassetteMismatchError as err:
            print(f"a miss is never papered over: {err}")  # noqa: T201

        loose = CassetteHarness(path, match=MatchOn(messages=False)).provider()
        wider = await loose.complete(asked("refund a different charge"))
        print(f"ignoring the prompt, the same recording serves it: {wider.content}")  # noqa: T201


if __name__ == "__main__":
    asyncio.run(main())
