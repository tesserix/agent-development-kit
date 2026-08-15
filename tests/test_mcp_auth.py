"""What an MCP server is told about the caller, and what authority it receives."""

from __future__ import annotations

import asyncio
from collections.abc import Iterable  # noqa: TC003 — used at runtime by the doubles

import pytest

from tesserix_adk.core import AgentIdentity, AuthorisationError, Principal
from tesserix_adk.core.errors import McpAuthError, McpAuthReason
from tesserix_adk.mcp.auth import (
    META_PREFIX,
    CallCredential,
    CredentialSource,
    McpAuthorizer,
    McpServerAuth,
    ServerSessions,
)
from tesserix_adk.testing import FakeClock
from tesserix_adk.tools import CachingCredentials, CredentialBroker, ExchangedCredentials

pytestmark = pytest.mark.anyio

READ = "bookings:read"
WRITE = "bookings:write"
ADMIN = "bookings:admin"
SERVER = "bookings-mcp"

AUTH = McpServerAuth(server=SERVER, audience="https://bookings.mcp", scopes=(READ, WRITE))


class _Exchange:
    """A token endpoint that mints one token per call and counts them."""

    def __init__(self, *, delay: float = 0.0) -> None:
        self.delay = delay
        self.requests: list[object] = []

    async def exchange(self, request: object) -> tuple[str, float]:
        self.requests.append(request)
        if self.delay:
            await asyncio.sleep(self.delay)
        return f"tok-{len(self.requests)}", 300.0


class _Failing:
    """A credential source that reads the request and then fails."""

    def __init__(self, failure: Exception) -> None:
        self.failure = failure
        self.seen: list[tuple[str, str, tuple[str, ...], str, str]] = []

    async def for_tool(
        self,
        *,
        identity: AgentIdentity,
        audience: str,
        needs: Iterable[str],
        run_id: str,
        agent_version: str = "1.0.0",
    ) -> CallCredential:
        self.seen.append((identity.agent, audience, tuple(needs), run_id, agent_version))
        raise self.failure


def _identity(tenant: str = "acme", *held: str, subject: str = "ada") -> AgentIdentity:
    return AgentIdentity.resolve(
        agent="desk",
        declared=(READ, WRITE, ADMIN),
        principal=Principal(subject=subject, tenant=tenant, scopes=frozenset(held or (READ,))),
    )


def _authorizer(exchange: _Exchange | None = None) -> McpAuthorizer:
    clock = FakeClock()
    broker = CredentialBroker(
        CachingCredentials(ExchangedCredentials(exchange or _Exchange(), clock=clock), clock=clock),
        clock=clock,
    )
    return McpAuthorizer(broker, servers={SERVER: AUTH})


class TestTheServerContract:
    def test_a_server_may_not_be_declared_without_an_audience(self) -> None:
        with pytest.raises(ValueError, match="audience"):
            McpServerAuth(server=SERVER, audience=" ", scopes=(READ,))

    def test_a_server_declaring_no_scopes_is_declaring_it_receives_none(self) -> None:
        assert McpServerAuth(server=SERVER, audience="https://x.invalid").scopes == ()

    def test_the_kits_own_broker_is_a_credential_source(self) -> None:
        """This package states what it needs rather than importing it: they are siblings."""
        clock = FakeClock()
        broker = CredentialBroker(ExchangedCredentials(_Exchange(), clock=clock), clock=clock)
        assert isinstance(broker, CredentialSource)


class TestWhatTheServerIsTold:
    async def test_the_metadata_names_the_tenant_the_caller_and_the_run(self) -> None:
        call = await _authorizer().authorise(
            server=SERVER, identity=_identity(), needs=(READ,), run_id="run_1"
        )
        meta = call.meta()
        assert meta[f"{META_PREFIX}/tenant"] == "acme"
        assert meta[f"{META_PREFIX}/subject"] == "ada"
        assert meta[f"{META_PREFIX}/run"] == "run_1"
        assert meta[f"{META_PREFIX}/agent"] == "desk"

    async def test_the_metadata_carries_no_credential(self) -> None:
        call = await _authorizer().authorise(
            server=SERVER, identity=_identity(), needs=(READ,), run_id="run_1"
        )
        assert "tok-1" not in str(call.meta())
        assert "tok-1" not in repr(call)

    async def test_the_credential_travels_in_the_headers_only(self) -> None:
        call = await _authorizer().authorise(
            server=SERVER, identity=_identity(), needs=(READ,), run_id="run_1"
        )
        assert call.headers()["Authorization"] == "Bearer tok-1"

    async def test_what_a_span_may_record_holds_no_credential(self) -> None:
        call = await _authorizer().authorise(
            server=SERVER, identity=_identity(), needs=(READ,), run_id="run_1"
        )
        attributes = call.span_attributes()
        assert "tok-1" not in str(attributes)
        assert attributes["mcp.server"] == SERVER
        assert attributes["mcp.scopes"] == READ


class TestScopeNarrowing:
    async def test_a_server_receives_only_what_it_declared(self) -> None:
        call = await _authorizer().authorise(
            server=SERVER,
            identity=_identity("acme", READ, WRITE, ADMIN),
            needs=(READ, ADMIN),
            run_id="run_1",
        )
        assert call.scopes == frozenset({READ})

    async def test_a_server_receives_only_what_the_run_holds(self) -> None:
        call = await _authorizer().authorise(
            server=SERVER, identity=_identity("acme", READ), needs=(READ, WRITE), run_id="run_1"
        )
        assert call.scopes == frozenset({READ})

    async def test_a_call_needing_nothing_the_server_may_have_is_refused(self) -> None:
        with pytest.raises(AuthorisationError, match="nothing"):
            await _authorizer().authorise(
                server=SERVER, identity=_identity("acme", ADMIN), needs=(ADMIN,), run_id="run_1"
            )

    async def test_a_server_asking_for_more_during_negotiation_gets_no_more(self) -> None:
        authorizer = _authorizer()
        granted = authorizer.narrowed_for(
            server=SERVER, identity=_identity("acme", READ), requested=(READ, WRITE, ADMIN)
        )
        assert granted == frozenset({READ})

    async def test_a_capability_change_carries_no_implied_authority(self) -> None:
        """A server announcing a new capability announces nothing about what it may do."""
        authorizer = _authorizer()
        assert (
            authorizer.narrowed_for(
                server=SERVER, identity=_identity("acme", READ), requested=("bookings:refund",)
            )
            == frozenset()
        )

    async def test_an_undeclared_server_is_refused_rather_than_defaulted(self) -> None:
        with pytest.raises(McpAuthError, match="not configured") as refused:
            await _authorizer().authorise(
                server="unknown-mcp", identity=_identity(), needs=(READ,), run_id="run_1"
            )
        assert refused.value.reason is McpAuthReason.UNAUTHENTICATED


class TestTenantIsolation:
    async def test_two_tenants_get_two_credentials(self) -> None:
        exchange = _Exchange()
        authorizer = _authorizer(exchange)
        first = await authorizer.authorise(
            server=SERVER, identity=_identity("acme"), needs=(READ,), run_id="run_1"
        )
        second = await authorizer.authorise(
            server=SERVER, identity=_identity("globex"), needs=(READ,), run_id="run_2"
        )
        assert first.headers()["Authorization"] != second.headers()["Authorization"]
        assert len(exchange.requests) == 2

    async def test_two_callers_in_one_tenant_get_two_credentials(self) -> None:
        exchange = _Exchange()
        authorizer = _authorizer(exchange)
        await authorizer.authorise(
            server=SERVER,
            identity=_identity("acme", READ, subject="ada"),
            needs=(READ,),
            run_id="run_1",
        )
        await authorizer.authorise(
            server=SERVER,
            identity=_identity("acme", READ, subject="bob"),
            needs=(READ,),
            run_id="run_2",
        )
        assert len(exchange.requests) == 2

    async def test_a_fan_out_for_one_tenant_mints_once(self) -> None:
        exchange = _Exchange(delay=0.01)
        authorizer = _authorizer(exchange)
        identity = _identity("acme")
        await asyncio.gather(
            *(
                authorizer.authorise(
                    server=SERVER, identity=identity, needs=(READ,), run_id="run_1"
                )
                for _ in range(5)
            )
        )
        assert len(exchange.requests) == 1

    async def test_a_credential_is_minted_per_call_not_per_session(self) -> None:
        """A long-lived stdio session outlives any one caller's authority."""
        exchange = _Exchange()
        authorizer = _authorizer(exchange)
        sessions = ServerSessions()
        lease = sessions.lease(server=SERVER, identity=_identity("acme"))
        first = await authorizer.authorise(
            server=SERVER, identity=_identity("acme"), needs=(READ,), run_id="run_1"
        )
        assert lease.tenant == "acme"
        assert first.headers()["Authorization"] == "Bearer tok-1"


class TestPooledConnections:
    def test_a_pooled_session_is_keyed_by_server_tenant_and_caller(self) -> None:
        sessions = ServerSessions()
        first = sessions.lease(server=SERVER, identity=_identity("acme"))
        second = sessions.lease(server=SERVER, identity=_identity("globex"))
        assert first.key != second.key

    def test_the_same_caller_reuses_one_lease(self) -> None:
        sessions = ServerSessions()
        identity = _identity("acme")
        assert (
            sessions.lease(server=SERVER, identity=identity).key
            == sessions.lease(server=SERVER, identity=identity).key
        )

    def test_a_lease_refuses_to_serve_another_tenant(self) -> None:
        sessions = ServerSessions()
        lease = sessions.lease(server=SERVER, identity=_identity("acme"))
        with pytest.raises(McpAuthError, match="globex"):
            lease.check(_identity("globex"))

    def test_a_lease_serves_the_caller_it_was_opened_for(self) -> None:
        lease = ServerSessions().lease(server=SERVER, identity=_identity("acme"))
        lease.check(_identity("acme"))

    def test_the_pool_says_how_many_sessions_are_open(self) -> None:
        sessions = ServerSessions()
        sessions.lease(server=SERVER, identity=_identity("acme"))
        sessions.lease(server=SERVER, identity=_identity("globex"))
        sessions.lease(server=SERVER, identity=_identity("acme"))
        assert sessions.open == 2


class TestFailures:
    def test_an_unauthenticated_response_is_typed(self) -> None:
        refused = McpAuthError.from_status(401, server=SERVER, scopes=(READ,))
        assert refused.reason is McpAuthReason.UNAUTHENTICATED
        assert refused.server == SERVER

    def test_an_insufficient_scope_response_names_what_was_needed(self) -> None:
        refused = McpAuthError.from_status(403, server=SERVER, scopes=(WRITE,))
        assert refused.reason is McpAuthReason.INSUFFICIENT_SCOPE
        assert WRITE in str(refused)

    def test_an_expired_credential_is_distinguished_from_a_missing_one(self) -> None:
        refused = McpAuthError.from_status(
            401, server=SERVER, scopes=(READ,), description="token expired"
        )
        assert refused.reason is McpAuthReason.EXPIRED

    def test_any_other_refusal_is_still_unauthenticated_rather_than_silent(self) -> None:
        assert McpAuthError.from_status(400, server=SERVER).reason is McpAuthReason.UNAUTHENTICATED

    async def test_a_provider_failure_leaves_no_half_open_session(self) -> None:
        provider = _Failing(TimeoutError("credential endpoint down"))
        authorizer = McpAuthorizer(provider, servers={SERVER: AUTH})
        with pytest.raises(McpAuthError, match=SERVER) as refused:
            await authorizer.authorise(
                server=SERVER, identity=_identity(), needs=(READ,), run_id="run_1"
            )
        assert refused.value.reason is McpAuthReason.UNAUTHENTICATED
        assert provider.seen == [("desk", AUTH.audience, (READ,), "run_1", "1.0.0")]

    async def test_a_typed_refusal_from_the_provider_travels_unchanged(self) -> None:
        provider = _Failing(McpAuthError("lapsed", reason=McpAuthReason.EXPIRED, server=SERVER))
        with pytest.raises(McpAuthError) as refused:
            await McpAuthorizer(provider, servers={SERVER: AUTH}).authorise(
                server=SERVER, identity=_identity(), needs=(READ,), run_id="run_1"
            )
        assert refused.value.reason is McpAuthReason.EXPIRED

    async def test_an_authorisation_refusal_is_not_reshaped_into_an_auth_error(self) -> None:
        """A run that does not hold the scope is a different problem from a server refusing."""
        with pytest.raises(AuthorisationError):
            await _authorizer().authorise(
                server=SERVER,
                identity=_identity("acme", READ),
                needs=(WRITE,),
                run_id="run_1",
                strict=True,
            )
