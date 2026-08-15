"""What a tool call authenticates with, and what it may ask for."""

from __future__ import annotations

import asyncio

import pytest
from pydantic import SecretStr

from tesserix_adk.core import AgentIdentity, AuthorisationError, Principal, SecretRef
from tesserix_adk.core.errors import CredentialError
from tesserix_adk.core.secrets import EnvironmentSecrets
from tesserix_adk.testing import FakeClock
from tesserix_adk.tools.credentials import (
    MAX_TTL_SECONDS,
    Attribution,
    CachingCredentials,
    Credential,
    CredentialBroker,
    CredentialRequest,
    ExchangedCredentials,
    StaticCredentials,
)

pytestmark = pytest.mark.anyio

READ = "payments:read"
WRITE = "payments:write"
PAYMENTS = "https://payments.internal"


def _identity(
    *held: str, agent: str = "desk", declared: tuple[str, ...] = (READ, WRITE)
) -> AgentIdentity:
    return AgentIdentity.resolve(
        agent=agent,
        declared=declared,
        principal=Principal(subject="ada", tenant="acme", scopes=frozenset(held)),
    )


def _attribution() -> Attribution:
    return Attribution(run_id="run_1", agent="desk", agent_version="2.1.0", tenant="acme")


def _request(*scopes: str, audience: str = PAYMENTS, ttl: float = 300.0) -> CredentialRequest:
    return CredentialRequest(
        audience=audience,
        scopes=frozenset(scopes),
        attribution=_attribution(),
        ttl_seconds=ttl,
    )


class _Exchange:
    """A token endpoint that mints, counts, and can refuse or stall."""

    def __init__(self, *, grants: float = 300.0, refuses: str = "", delay: float = 0.0) -> None:
        self.grants = grants
        self.refuses = refuses
        self.delay = delay
        self.calls: list[CredentialRequest] = []

    async def exchange(self, request: CredentialRequest) -> tuple[str, float]:
        self.calls.append(request)
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.refuses:
            raise CredentialError(self.refuses, audience=request.audience)
        return f"tok-{len(self.calls)}", self.grants


class TestAttribution:
    def test_carries_the_run_the_agent_and_its_version(self) -> None:
        headers = _attribution().headers()
        assert headers["X-Tesserix-Run"] == "run_1"
        assert headers["X-Tesserix-Agent"] == "desk"
        assert headers["X-Tesserix-Agent-Version"] == "2.1.0"

    def test_omits_what_it_was_not_told(self) -> None:
        headers = Attribution(run_id="run_1", agent="desk").headers()
        assert "X-Tesserix-Tenant" not in headers

    def test_a_run_it_cannot_name_is_refused(self) -> None:
        with pytest.raises(ValueError, match="names the run"):
            Attribution(run_id=" ", agent="desk")


class TestACredential:
    def _credential(self, *, expires_at: float = 300.0) -> Credential:
        return Credential(
            token=SecretStr("tok-1"),
            audience=PAYMENTS,
            scopes=frozenset({READ}),
            expires_at=expires_at,
            attribution=_attribution(),
        )

    def test_does_not_render_its_token(self) -> None:
        credential = self._credential()
        assert "tok-1" not in repr(credential)
        assert "tok-1" not in credential.model_dump_json()

    def test_is_revealed_only_in_the_headers_that_make_the_call(self) -> None:
        headers = self._credential().headers()
        assert headers["Authorization"] == "Bearer tok-1"
        assert headers["X-Tesserix-Run"] == "run_1"

    def test_is_spent_before_it_expires_so_skew_does_not_bite(self) -> None:
        credential = self._credential(expires_at=300.0)
        assert not credential.expired(240.0)
        assert credential.expired(299.0)

    def test_a_credential_that_expires_in_the_past_is_refused(self) -> None:
        with pytest.raises(ValueError, match="already expired"):
            Credential(
                token=SecretStr("tok-1"),
                audience=PAYMENTS,
                scopes=frozenset({READ}),
                expires_at=-1.0,
                attribution=_attribution(),
            )


class TestARequest:
    def test_may_not_ask_for_longer_than_the_ceiling(self) -> None:
        with pytest.raises(ValueError, match="ceiling"):
            _request(READ, ttl=MAX_TTL_SECONDS + 1)

    def test_an_audience_it_cannot_name_is_refused(self) -> None:
        with pytest.raises(ValueError, match="names the audience"):
            _request(READ, audience="")


class TestExchangingForAShortLivedToken:
    async def test_mints_a_credential_scoped_to_what_was_asked(self) -> None:
        clock = FakeClock()
        credentials = ExchangedCredentials(_Exchange(), clock=clock)
        minted = await credentials.issue(_request(READ))
        assert minted.scopes == frozenset({READ})
        assert minted.audience == PAYMENTS
        assert minted.expires_at == clock.now() + 300.0

    async def test_a_grant_shorter_than_asked_for_is_what_expires(self) -> None:
        credentials = ExchangedCredentials(_Exchange(grants=60.0), clock=FakeClock())
        minted = await credentials.issue(_request(READ, ttl=300.0))
        assert minted.expires_at == 60.0

    async def test_a_grant_longer_than_the_ceiling_is_cut_to_it(self) -> None:
        credentials = ExchangedCredentials(_Exchange(grants=MAX_TTL_SECONDS * 4), clock=FakeClock())
        minted = await credentials.issue(_request(READ, ttl=MAX_TTL_SECONDS))
        assert minted.expires_at == MAX_TTL_SECONDS

    async def test_the_attribution_travels_to_the_endpoint(self) -> None:
        exchange = _Exchange()
        await ExchangedCredentials(exchange, clock=FakeClock()).issue(_request(READ))
        assert exchange.calls[0].attribution.run_id == "run_1"

    async def test_an_endpoint_that_refuses_fails_the_call(self) -> None:
        credentials = ExchangedCredentials(_Exchange(refuses="scope denied"), clock=FakeClock())
        with pytest.raises(CredentialError, match="scope denied"):
            await credentials.issue(_request(READ))

    async def test_an_endpoint_that_breaks_is_a_credential_error_not_a_broader_fallback(
        self,
    ) -> None:
        class _Broken:
            async def exchange(self, request: CredentialRequest) -> tuple[str, float]:  # noqa: ARG002
                raise TimeoutError("no route to host")

        credentials = ExchangedCredentials(_Broken(), clock=FakeClock())
        with pytest.raises(CredentialError, match=PAYMENTS):
            await credentials.issue(_request(READ))


class TestTheStaticKeyException:
    def _static(self) -> StaticCredentials:
        return StaticCredentials(
            EnvironmentSecrets(environ={"PAYMENTS_KEY": "sk-test-static"}),
            keys={PAYMENTS: SecretRef(name="payments-key")},
            clock=FakeClock(),
        )

    async def test_reads_the_reference_for_that_audience(self) -> None:
        minted = await self._static().issue(_request(READ))
        assert minted.headers()["Authorization"] == "Bearer sk-test-static"

    async def test_says_it_is_long_lived_so_the_exception_can_be_counted(self) -> None:
        assert (await self._static().issue(_request(READ))).long_lived

    async def test_an_audience_with_no_documented_key_is_refused(self) -> None:
        with pytest.raises(CredentialError, match="no static key"):
            await self._static().issue(_request(READ, audience="https://other.internal"))


class TestCaching:
    def _caching(self, inner: _Exchange, clock: FakeClock) -> CachingCredentials:
        return CachingCredentials(ExchangedCredentials(inner, clock=clock), clock=clock)

    async def test_the_same_ask_reuses_the_live_credential(self) -> None:
        exchange = _Exchange()
        credentials = self._caching(exchange, FakeClock())
        first = await credentials.issue(_request(READ))
        second = await credentials.issue(_request(READ))
        assert first.token.get_secret_value() == second.token.get_secret_value()
        assert len(exchange.calls) == 1

    async def test_a_different_audience_is_a_different_credential(self) -> None:
        exchange = _Exchange()
        credentials = self._caching(exchange, FakeClock())
        await credentials.issue(_request(READ))
        await credentials.issue(_request(READ, audience="https://other.internal"))
        assert len(exchange.calls) == 2

    async def test_a_different_scope_set_is_a_different_credential(self) -> None:
        exchange = _Exchange()
        credentials = self._caching(exchange, FakeClock())
        await credentials.issue(_request(READ))
        await credentials.issue(_request(READ, WRITE))
        assert len(exchange.calls) == 2

    async def test_one_tenants_credential_is_never_served_to_another(self) -> None:
        exchange = _Exchange()
        credentials = self._caching(exchange, FakeClock())
        await credentials.issue(_request(READ))
        other = CredentialRequest(
            audience=PAYMENTS,
            scopes=frozenset({READ}),
            attribution=Attribution(run_id="run_2", agent="desk", tenant="globex"),
        )
        await credentials.issue(other)
        assert len(exchange.calls) == 2

    async def test_one_principals_credential_is_never_served_to_another(self) -> None:
        exchange = _Exchange()
        credentials = self._caching(exchange, FakeClock())
        await credentials.issue(_request(READ))
        other = CredentialRequest(
            audience=PAYMENTS,
            scopes=frozenset({READ}),
            attribution=Attribution(run_id="run_2", agent="desk", tenant="acme", subject="bob"),
        )
        await credentials.issue(other)
        assert len(exchange.calls) == 2

    async def test_an_expiring_credential_is_reminted_rather_than_reused(self) -> None:
        exchange = _Exchange()
        clock = FakeClock()
        credentials = self._caching(exchange, clock)
        await credentials.issue(_request(READ))
        clock.advance(299.0)
        await credentials.issue(_request(READ))
        assert len(exchange.calls) == 2

    async def test_a_fan_out_does_not_stampede_the_endpoint(self) -> None:
        exchange = _Exchange(delay=0.01)
        credentials = self._caching(exchange, FakeClock())
        await asyncio.gather(*(credentials.issue(_request(READ)) for _ in range(6)))
        assert len(exchange.calls) == 1

    async def test_a_cancelled_mint_leaves_nothing_usable_behind(self) -> None:
        exchange = _Exchange(delay=0.05)
        credentials = self._caching(exchange, FakeClock())
        minting = asyncio.ensure_future(credentials.issue(_request(READ)))
        await asyncio.sleep(0.01)
        minting.cancel()
        with pytest.raises(asyncio.CancelledError):
            await minting
        assert credentials.cached == 0

    async def test_a_retry_after_a_partial_failure_reuses_rather_than_reminting(self) -> None:
        """A second credential for a non-idempotent call is a second side effect."""
        exchange = _Exchange()
        credentials = self._caching(exchange, FakeClock())
        first = await credentials.issue(_request(READ))
        second = await credentials.issue(_request(READ))
        assert first.token.get_secret_value() == second.token.get_secret_value()

    async def test_a_revoked_credential_can_be_dropped_before_it_expires(self) -> None:
        exchange = _Exchange()
        credentials = self._caching(exchange, FakeClock())
        await credentials.issue(_request(READ))
        credentials.invalidate_all()
        await credentials.issue(_request(READ))
        assert len(exchange.calls) == 2


class TestTheBroker:
    def _broker(self, exchange: _Exchange, clock: FakeClock | None = None) -> CredentialBroker:
        clock = clock or FakeClock()
        return CredentialBroker(
            CachingCredentials(ExchangedCredentials(exchange, clock=clock), clock=clock),
            clock=clock,
        )

    async def test_a_tool_gets_a_credential_for_exactly_what_it_needs(self) -> None:
        exchange = _Exchange()
        minted = await self._broker(exchange).for_tool(
            identity=_identity(READ, WRITE),
            audience=PAYMENTS,
            needs=(READ,),
            run_id="run_1",
        )
        assert minted.scopes == frozenset({READ})
        assert exchange.calls[0].scopes == frozenset({READ})

    async def test_the_credential_names_the_run_and_the_agent_version(self) -> None:
        minted = await self._broker(_Exchange()).for_tool(
            identity=_identity(READ),
            audience=PAYMENTS,
            needs=(READ,),
            run_id="run_1",
            agent_version="3.0.0",
        )
        assert minted.attribution.run_id == "run_1"
        assert minted.attribution.agent_version == "3.0.0"
        assert minted.attribution.tenant == "acme"

    async def test_a_tool_cannot_ask_for_more_than_the_run_holds(self) -> None:
        with pytest.raises(AuthorisationError, match=WRITE):
            await self._broker(_Exchange()).for_tool(
                identity=_identity(READ),
                audience=PAYMENTS,
                needs=(READ, WRITE),
                run_id="run_1",
            )

    async def test_asking_for_nothing_is_refused_rather_than_minting_everything(self) -> None:
        with pytest.raises(AuthorisationError, match="names no scope"):
            await self._broker(_Exchange()).for_tool(
                identity=_identity(READ), audience=PAYMENTS, needs=(), run_id="run_1"
            )

    async def test_a_ttl_past_the_ceiling_is_refused_at_construction(self) -> None:
        clock = FakeClock()
        with pytest.raises(ValueError, match="ceiling"):
            CredentialBroker(
                ExchangedCredentials(_Exchange(), clock=clock),
                clock=clock,
                ttl_seconds=MAX_TTL_SECONDS * 2,
            )

    async def test_a_refusal_downstream_reaches_the_caller_as_a_credential_error(self) -> None:
        broker = self._broker(_Exchange(refuses="audience not permitted"))
        with pytest.raises(CredentialError, match="audience not permitted"):
            await broker.for_tool(
                identity=_identity(READ), audience=PAYMENTS, needs=(READ,), run_id="run_1"
            )

    async def test_two_tools_needing_the_same_access_share_one_credential(self) -> None:
        exchange = _Exchange()
        broker = self._broker(exchange)
        identity = _identity(READ)
        for _ in range(2):
            await broker.for_tool(
                identity=identity, audience=PAYMENTS, needs=(READ,), run_id="run_1"
            )
        assert len(exchange.calls) == 1
