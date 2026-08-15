"""A run that outlives its token, and carries on anyway.

Run it with `uv run python examples/credential_refresh.py`.
"""

from __future__ import annotations

from tesserix_adk.core import (
    AgentIdentity,
    AuthorityRevokedError,
    CredentialExpiredError,
    ExpiringCredential,
    Principal,
)
from tesserix_adk.runtime import RefreshPolicy, RunCredentials
from tesserix_adk.testing import FakeClock
from tesserix_adk.tools import CredentialBroker, CredentialRequest, ExchangedCredentials

READ = "payments:read"
WRITE = "payments:write"
PAYMENTS = "https://payments.internal"


class CountingExchange:
    """Stands in for a token endpoint, minting five-minute tokens."""

    def __init__(self) -> None:
        self.calls = 0

    async def exchange(self, request: CredentialRequest) -> tuple[str, float]:
        """Mint a token for one audience."""
        self.calls += 1
        return f"tok-{self.calls}-for-{request.audience}", 300.0


class Directory:
    """Answers what the caller holds now, which is not always what they held."""

    def __init__(self) -> None:
        self.revoked = False

    async def reauthorise(self, identity: AgentIdentity) -> AgentIdentity:
        """Re-derive the caller's authority, refusing once the grant is withdrawn."""
        if self.revoked:
            raise AuthorityRevokedError("ada's payments grant was withdrawn")
        return identity


def caller() -> AgentIdentity:
    """The person the run acts for."""
    return AgentIdentity.resolve(
        agent="desk",
        declared=(READ, WRITE),
        principal=Principal(subject="ada", tenant="acme", scopes=frozenset({READ, WRITE})),
    )


async def main() -> None:
    """Run past two token lifetimes, then have the grant withdrawn underneath."""
    clock = FakeClock()
    exchange = CountingExchange()
    directory = Directory()
    credentials = RunCredentials(
        CredentialBroker(ExchangedCredentials(exchange, clock=clock), clock=clock),
        identity=caller(),
        clock=clock,
        policy=RefreshPolicy(skew_seconds=30.0),
        reauthorise=directory,
    )

    async def read(credential: ExpiringCredential, key: str) -> str:
        return f"{credential.token.get_secret_value()} under {key or 'no key'}"

    for step in range(6):
        answer = await credentials.call(
            read,
            audience=PAYMENTS,
            needs=(READ,),
            run_id="run-1",
            idempotency_key=f"step-{step}",
        )
        print(f"step {step}: {answer}")  # noqa: T201
        clock.advance(120.0)

    print(f"\nsix steps over 12 minutes cost {exchange.calls} mint(s)")  # noqa: T201

    directory.revoked = True
    clock.advance(300.0)
    try:
        await credentials.call(read, audience=PAYMENTS, needs=(READ,), run_id="run-1")
    except AuthorityRevokedError as halted:
        print(f"halted: {halted}")  # noqa: T201
    print(f"halted={credentials.halted}, credentials held={credentials.held}")  # noqa: T201

    try:
        await credentials.call(read, audience=PAYMENTS, needs=(READ,), run_id="run-1")
    except (AuthorityRevokedError, CredentialExpiredError) as refused:
        print(f"and stays halted: {type(refused).__name__}")  # noqa: T201


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
