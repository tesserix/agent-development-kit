"""A scripted conversation, a rate limit, and a loop that asks one question too many.

Run it with `uv run python examples/fake_model_provider.py`.
"""

from __future__ import annotations

import asyncio
from decimal import Decimal

from tesserix_adk.core import Cost, Message, ModelRequest, RateLimitError, TextPart
from tesserix_adk.testing import (
    FakeModelProvider,
    Fault,
    ScriptedTurn,
    ScriptExhaustedError,
)


def asked(text: str) -> ModelRequest:
    """One user turn, as the runtime would assemble it."""
    turn = Message(role="user", content=[TextPart(text=text)])
    return ModelRequest(model="fake-1", messages=(turn,))


async def main() -> None:
    """Replay a script, recover from a rate limit, then run off the end of it."""
    provider = FakeModelProvider(
        ScriptedTurn.calling("lookup_charge", {"id": "ch_1"}, input_tokens=40, output_tokens=8),
        ScriptedTurn.failing(Fault.RATE_LIMIT, payload="120 requests in 60s"),
        ScriptedTurn.returning(
            {"status": "refunded", "amount": "12.00"},
            input_tokens=90,
            output_tokens=20,
            cost=Cost(input=Decimal("0.0001"), output=Decimal("0.0002")),
        ),
    )

    first = await provider.complete(asked("refund the charge"))
    print(f"asked for {first.tool_calls[0].name}({first.tool_calls[0].arguments})")  # noqa: T201

    try:
        await provider.complete(asked("tool said: charge found"))
    except RateLimitError as err:
        print(f"retrying after {type(err).__name__}: {err}")  # noqa: T201

    answer = await provider.complete(asked("tool said: charge found"))
    spent = answer.usage.cost
    money = f"{spent.total} {spent.currency}" if spent else "nothing anybody counted"
    print(f"answered {answer.content} for {money}")  # noqa: T201
    print(f"{provider.calls} calls made, {provider.remaining} turns unused")  # noqa: T201

    try:
        await provider.complete(asked("and again"))
    except ScriptExhaustedError as err:
        print(f"the loop kept going: {err}")  # noqa: T201


if __name__ == "__main__":
    asyncio.run(main())
