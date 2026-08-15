"""Two tenants calling one configured MCP server through one process.

Run it with `uv run python examples/mcp_credentials.py`.
"""

from __future__ import annotations

import asyncio

from tesserix_adk.core import AgentIdentity, AuthorisationError, McpAuthError, Principal
from tesserix_adk.mcp import McpAuthorizer, McpServerAuth, ServerSessions
from tesserix_adk.testing import FakeClock
from tesserix_adk.tools import (
    CachingCredentials,
    CredentialBroker,
    CredentialRequest,
    ExchangedCredentials,
)

READ = "bookings:read"
WRITE = "bookings:write"
ADMIN = "bookings:admin"
SERVER = "bookings-mcp"

BOOKINGS = McpServerAuth(server=SERVER, audience="https://bookings.mcp", scopes=(READ, WRITE))


class CountingExchange:
    """Stands in for a token endpoint, and says how often it was asked."""

    def __init__(self) -> None:
        self.calls = 0

    async def exchange(self, request: CredentialRequest) -> tuple[str, float]:
        """Mint a token naming the tenant it is for, live for five minutes."""
        self.calls += 1
        return f"tok-{self.calls}-{request.attribution.tenant}", 300.0


def caller(tenant: str, subject: str, *held: str) -> AgentIdentity:
    """One tenant's caller, holding what their grant says and no more."""
    return AgentIdentity.resolve(
        agent="desk",
        declared=(READ, WRITE, ADMIN),
        principal=Principal(subject=subject, tenant=tenant, scopes=frozenset(held)),
    )


async def main() -> None:
    """Show per-call credentials, scope narrowing, and a pooled session refusing a leak."""
    clock = FakeClock()
    exchange = CountingExchange()
    authorizer = McpAuthorizer(
        CredentialBroker(
            CachingCredentials(ExchangedCredentials(exchange, clock=clock), clock=clock),
            clock=clock,
        ),
        servers={SERVER: BOOKINGS},
    )
    acme = caller("acme", "ada", READ)
    globex = caller("globex", "bob", READ, WRITE, ADMIN)

    for identity, run_id in ((acme, "run_1"), (globex, "run_2")):
        call = await authorizer.authorise(
            server=SERVER, identity=identity, needs=(READ, WRITE, ADMIN), run_id=run_id
        )
        print(f"{identity.principal.tenant}: presents {sorted(call.scopes)}")  # noqa: T201
        print(f"  metadata: {call.meta()}")  # noqa: T201
        print(f"  span:     {call.span_attributes()}")  # noqa: T201
        print(f"  header:   {call.headers()['Authorization']}")  # noqa: T201

    print(f"the server declares {list(BOOKINGS.scopes)}, so admin never reaches it")  # noqa: T201

    granted = authorizer.narrowed_for(server=SERVER, identity=globex, requested=(ADMIN,))
    print(f"a server asking for admin during negotiation gets: {sorted(granted)}")  # noqa: T201

    try:
        await authorizer.authorise(
            server=SERVER, identity=caller("acme", "ada", ADMIN), needs=(ADMIN,), run_id="run_3"
        )
    except AuthorisationError as refused:
        print(f"a call that would present nothing: {refused}")  # noqa: T201

    sessions = ServerSessions()
    lease = sessions.lease(server=SERVER, identity=acme)
    try:
        lease.check(globex)
    except McpAuthError as refused:
        print(f"a pooled connection reused across tenants: {refused}")  # noqa: T201

    await asyncio.gather(
        *(
            authorizer.authorise(server=SERVER, identity=acme, needs=(READ,), run_id="run_1")
            for _ in range(4)
        )
    )
    print(f"two tenants and a fan-out cost {exchange.calls} mint(s)")  # noqa: T201


if __name__ == "__main__":
    asyncio.run(main())
