"""One agent calling another as the person who started the run.

Run it with `uv run python examples/peer_delegation.py`.
"""

from __future__ import annotations

from tesserix_adk.a2a import (
    AgentCard,
    DelegationClaims,
    PeerDelegator,
    PeerVerifier,
)
from tesserix_adk.core import AgentIdentity, AuthorisationError, Principal
from tesserix_adk.testing import FakeClock
from tesserix_adk.tools import CredentialBroker, CredentialRequest, ExchangedCredentials

READ = "itinerary:read"
WRITE = "payments:write"

PAYMENTS = AgentCard(
    agent="payments-agent",
    audience="https://payments.peer",
    declared=(READ, WRITE),
    accepted_issuers=("desk",),
)


class CountingExchange:
    """Stands in for a token endpoint."""

    def __init__(self) -> None:
        self.calls = 0

    async def exchange(self, request: CredentialRequest) -> tuple[str, float]:
        """Mint a token for the peer's audience, live for five minutes."""
        self.calls += 1
        return f"tok-{self.calls}-for-{request.audience}", 300.0


async def main() -> None:
    """Show a peer receiving less than it declared, and refusing what it was not sent."""
    clock = FakeClock()
    delegator = PeerDelegator(
        CredentialBroker(ExchangedCredentials(CountingExchange(), clock=clock), clock=clock),
        clock=clock,
    )
    caller = AgentIdentity.resolve(
        agent="desk",
        declared=(READ, WRITE),
        principal=Principal(subject="ada", tenant="acme", scopes=frozenset({READ})),
    )

    delegation = await delegator.delegate(
        identity=caller, peer=PAYMENTS, needs=(READ, WRITE), run_id="run_1"
    )
    print(f"the caller holds:  {sorted(caller.effective)}")  # noqa: T201
    print(f"the peer declares: {list(PAYMENTS.declared)}")  # noqa: T201
    print(f"the delegation carries: {sorted(delegation.claims.scopes)}")  # noqa: T201
    print(f"span attribution: {delegation.span_attributes()}")  # noqa: T201
    print(f"headers: {delegation.headers()['Authorization']}")  # noqa: T201

    arrived = DelegationClaims.from_meta(delegation.meta())
    peer = PeerVerifier(PAYMENTS, clock=clock).accept(arrived)
    print(f"the peer runs as {peer.principal.subject} of {peer.principal.tenant}")  # noqa: T201
    print(f"  holding {sorted(peer.effective)}, through {peer.chain}")  # noqa: T201

    try:
        peer.check((WRITE,), where="payments")
    except AuthorisationError as refused:
        print(f"  and its payment tool: {refused}")  # noqa: T201

    clock.advance(300.0)
    try:
        PeerVerifier(PAYMENTS, clock=clock).accept(arrived)
    except AuthorisationError as refused:
        print(f"once it lapses: {refused}")  # noqa: T201


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
