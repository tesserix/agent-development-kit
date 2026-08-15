"""A tool minting a scoped, short-lived credential for one downstream service.

Run it with `uv run python examples/tool_credentials.py`.
"""

from __future__ import annotations

import asyncio

from tesserix_adk.core import AgentIdentity, AuthorisationError, Principal
from tesserix_adk.testing import FakeClock
from tesserix_adk.tools import (
    CachingCredentials,
    CredentialBroker,
    CredentialRequest,
    ExchangedCredentials,
)

READ = "payments:read"
WRITE = "payments:write"
PAYMENTS = "https://payments.internal"


class CountingExchange:
    """Stands in for a token endpoint, and says how often it was asked."""

    def __init__(self) -> None:
        self.calls = 0

    async def exchange(self, request: CredentialRequest) -> tuple[str, float]:
        """Mint a token naming the scopes it was asked for, live for five minutes."""
        self.calls += 1
        scopes = "+".join(sorted(request.scopes))
        return f"tok-{self.calls}-{scopes}", 300.0


async def main() -> None:
    """Show what a tool gets, what it cannot ask for, and what a fan-out costs."""
    clock = FakeClock()
    exchange = CountingExchange()
    broker = CredentialBroker(
        CachingCredentials(ExchangedCredentials(exchange, clock=clock), clock=clock),
        clock=clock,
    )
    identity = AgentIdentity.resolve(
        agent="desk",
        declared=(READ, WRITE),
        principal=Principal(subject="ada", tenant="acme", scopes=frozenset({READ})),
    )

    credential = await broker.for_tool(
        identity=identity, audience=PAYMENTS, needs=(READ,), run_id="run_1", agent_version="2.1.0"
    )
    print(f"minted for: {sorted(credential.scopes)} at {credential.audience}")  # noqa: T201
    print(f"expires in: {credential.expires_at - clock.now()}s")  # noqa: T201
    print(f"the credential itself renders as: {credential.token}")  # noqa: T201
    for name, value in credential.headers().items():
        shown = "Bearer ***" if name == "Authorization" else value
        print(f"  {name}: {shown}")  # noqa: T201

    try:
        await broker.for_tool(
            identity=identity, audience=PAYMENTS, needs=(READ, WRITE), run_id="run_1"
        )
    except AuthorisationError as refused:
        print(f"a tool asking for more than the run holds: {refused}")  # noqa: T201

    await asyncio.gather(
        *(
            broker.for_tool(identity=identity, audience=PAYMENTS, needs=(READ,), run_id="run_1")
            for _ in range(6)
        )
    )
    print(f"six concurrent tool calls cost {exchange.calls} mint(s)")  # noqa: T201

    clock.advance(300.0)
    await broker.for_tool(identity=identity, audience=PAYMENTS, needs=(READ,), run_id="run_1")
    print(f"once it lapses, a call costs {exchange.calls} mint(s) in total")  # noqa: T201


if __name__ == "__main__":
    asyncio.run(main())
