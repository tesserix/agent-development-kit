"""Who an MCP call is for, carried on every request and never inferred at the far side."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

import httpx
import pytest
from pydantic import SecretStr

from tesserix_adk.adapters.mcp import McpServerInfo
from tesserix_adk.adapters.mcp_context import (
    AuthorisingSession,
    CallerContext,
    IncomingCall,
    TenantAuthority,
    arriving_call,
    redacted,
)
from tesserix_adk.adapters.mcp_transport import HttpTransport, TransportSession
from tesserix_adk.core.config import McpServerConfig
from tesserix_adk.core.errors import AuthorisationError, McpAuthError, McpAuthReason
from tesserix_adk.core.identity import AgentIdentity, Principal
from tesserix_adk.core.propagation import HEADER, carried
from tesserix_adk.core.tenancy import TenantContext, tenant_scope
from tesserix_adk.mcp import (
    META_PREFIX,
    GatewayToolResult,
    McpAuthorizer,
    McpServerAuth,
    McpToolDescriptor,
)
from tesserix_adk.observability.propagation import TRACEPARENT, W3CContext
from tesserix_adk.tools.credentials import Credential, CredentialBroker, CredentialRequest

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from pydantic import JsonValue


_HANDBOOK = McpServerAuth(server="handbook", audience="handbook.svc", scopes=("hb:read",))


class _Clock:
    """A clock a test moves by hand, so expiry happens without waiting for it."""

    def __init__(self, now: float = 0.0) -> None:
        self.reading = now

    def now(self) -> float:
        return self.reading

    async def sleep(self, seconds: float) -> None:
        self.reading += seconds


class _Mint:
    """A credential provider that mints a distinct short-lived token per request."""

    def __init__(self, *, clock: _Clock, lifetime: float = 600.0) -> None:
        self.clock = clock
        self.lifetime = lifetime
        self.issued: list[CredentialRequest] = []

    async def issue(self, request: CredentialRequest) -> Credential:
        self.issued.append(request)
        minted = f"tok-{request.attribution.tenant}-{len(self.issued)}"
        return Credential(
            token=SecretStr(minted),
            audience=request.audience,
            scopes=request.scopes,
            expires_at=self.clock.now() + self.lifetime,
            attribution=request.attribution,
        )


class _Refuses:
    """A provider that cannot mint, which is the fail-closed path."""

    async def issue(self, request: CredentialRequest) -> Credential:
        raise ConnectionError(f"the mint for {request.audience} is unreachable")


def _identity(
    tenant: str = "acme", subject: str = "ada", scopes: tuple[str, ...] = ("hb:read",)
) -> AgentIdentity:
    return AgentIdentity.resolve(
        agent="desk",
        declared=scopes,
        principal=Principal(subject=subject, tenant=tenant, scopes=frozenset(scopes)),
    )


def _authority(
    mint: _Mint | _Refuses,
    *,
    clock: _Clock,
    identity: AgentIdentity | None = None,
    run_id: str = "run-1",
    servers: Mapping[str, McpServerAuth] | None = None,
) -> TenantAuthority:
    who = identity or _identity()
    broker = CredentialBroker(mint, clock=clock)
    authorizer = McpAuthorizer(broker, servers=servers or {"handbook": _HANDBOOK})
    return TenantAuthority(
        authorizer,
        caller=lambda: CallerContext.current(identity=who, run_id=run_id),
        clock=clock,
    )


class TestTheCallerIsRead:
    """The context a call travels under comes from the run, not from the call site."""

    def test_the_bound_tenant_is_the_callers_tenant(self) -> None:
        with tenant_scope(TenantContext(tenant="acme", user="ada")):
            context = CallerContext.current(identity=_identity(), run_id="run-1")
        assert context.tenant.tenant == "acme"
        assert context.run_id == "run-1"

    def test_a_call_with_no_bound_tenant_is_refused(self) -> None:
        with pytest.raises(McpAuthError) as refused:
            CallerContext.current(identity=_identity(), run_id="run-1")
        assert refused.value.reason is McpAuthReason.UNAUTHENTICATED

    def test_a_tenant_that_is_not_the_callers_is_refused(self) -> None:
        with tenant_scope(TenantContext(tenant="other")), pytest.raises(McpAuthError) as refused:
            CallerContext.current(identity=_identity(tenant="acme"), run_id="run-1")
        assert "other" in str(refused.value)

    def test_a_run_started_outside_a_trace_starts_one(self) -> None:
        with tenant_scope(TenantContext(tenant="acme")):
            context = CallerContext.current(identity=_identity(), run_id="run-1")
        assert context.trace.trace_id == W3CContext.rooted("run-1").trace_id
        assert context.headers()[TRACEPARENT].endswith("-01")

    def test_an_arriving_trace_is_continued_rather_than_replaced(self) -> None:
        upstream = W3CContext.rooted("caller-run")
        with tenant_scope(TenantContext(tenant="acme")):
            context = CallerContext.current(
                identity=_identity(), run_id="run-1", headers=upstream.carried()
            )
        assert context.trace.trace_id == upstream.trace_id
        assert context.trace.parent_span_id == upstream.span_id


class TestEveryRequestCarriesTheCaller:
    """The primary scenario: authority and context on the request, with no plumbing."""

    @pytest.mark.asyncio
    async def test_the_request_carries_a_credential_and_the_context(self) -> None:
        clock = _Clock()
        authority = _authority(_Mint(clock=clock), clock=clock)
        with tenant_scope(TenantContext(tenant="acme", user="ada")):
            headers = await authority.headers_for(server="handbook")
            meta = await authority.meta_for(server="handbook")
        assert headers["Authorization"] == "Bearer tok-acme-1"
        assert headers[HEADER] == carried(TenantContext(tenant="acme", user="ada"))[HEADER]
        assert headers[TRACEPARENT].startswith("00-")
        assert meta[f"{META_PREFIX}/tenant"] == "acme"
        assert meta[f"{META_PREFIX}/run"] == "run-1"
        assert meta[f"{META_PREFIX}/traceparent"] == headers[TRACEPARENT]

    @pytest.mark.asyncio
    async def test_the_credential_is_minted_for_the_server_and_its_scopes(self) -> None:
        clock = _Clock()
        mint = _Mint(clock=clock)
        authority = _authority(mint, clock=clock)
        with tenant_scope(TenantContext(tenant="acme")):
            await authority.headers_for(server="handbook")
        assert mint.issued[0].audience == "handbook.svc"
        assert mint.issued[0].scopes == frozenset({"hb:read"})

    @pytest.mark.asyncio
    async def test_one_credential_serves_repeated_calls_for_the_same_caller(self) -> None:
        clock = _Clock()
        mint = _Mint(clock=clock)
        authority = _authority(mint, clock=clock)
        with tenant_scope(TenantContext(tenant="acme")):
            first = await authority.headers_for(server="handbook")
            second = await authority.headers_for(server="handbook")
        assert first["Authorization"] == second["Authorization"]
        assert len(mint.issued) == 1


class TestTwoTenantsInOneProcess:
    """Nothing minted or held for one tenant can be reached by another."""

    @pytest.mark.asyncio
    async def test_each_tenant_gets_its_own_credential(self) -> None:
        clock = _Clock()
        mint = _Mint(clock=clock)
        broker = CredentialBroker(mint, clock=clock)
        authorizer = McpAuthorizer(broker, servers={"handbook": _HANDBOOK})
        held: dict[str, str] = {}

        async def called(tenant: str) -> None:
            who = _identity(tenant=tenant, subject=f"user-{tenant}")
            authority = TenantAuthority(
                authorizer,
                caller=lambda: CallerContext.current(identity=who, run_id=f"run-{tenant}"),
                clock=clock,
            )
            with tenant_scope(TenantContext(tenant=tenant)):
                headers = await authority.headers_for(server="handbook")
            held[tenant] = headers["Authorization"]

        await asyncio.gather(called("acme"), called("globex"))
        assert held["acme"] != held["globex"]
        assert "acme" in held["acme"]
        assert "globex" in held["globex"]

    @pytest.mark.asyncio
    async def test_a_second_tenant_on_one_authority_does_not_reuse_the_first(self) -> None:
        clock = _Clock()
        mint = _Mint(clock=clock)
        broker = CredentialBroker(mint, clock=clock)
        authorizer = McpAuthorizer(broker, servers={"handbook": _HANDBOOK})
        acting: list[AgentIdentity] = [_identity(tenant="acme")]
        authority = TenantAuthority(
            authorizer,
            caller=lambda: CallerContext.current(identity=acting[0], run_id="run-1"),
            clock=clock,
        )
        with tenant_scope(TenantContext(tenant="acme")):
            first = await authority.headers_for(server="handbook")
        acting[0] = _identity(tenant="globex", subject="bob")
        with tenant_scope(TenantContext(tenant="globex")):
            second = await authority.headers_for(server="handbook")
        assert first["Authorization"] != second["Authorization"]
        assert authority.held == 2


class TestFailingClosed:
    """Nothing leaves the process without a credential and a tenant."""

    @pytest.mark.asyncio
    async def test_a_mint_that_cannot_answer_refuses_the_call(self) -> None:
        clock = _Clock()
        authority = _authority(_Refuses(), clock=clock)
        with tenant_scope(TenantContext(tenant="acme")), pytest.raises(McpAuthError) as refused:
            await authority.headers_for(server="handbook")
        assert refused.value.reason is McpAuthReason.UNAUTHENTICATED

    @pytest.mark.asyncio
    async def test_nothing_is_sent_when_the_credential_cannot_be_minted(self) -> None:
        clock = _Clock()
        sent: list[httpx.Request] = []

        async def endpoint(request: httpx.Request) -> httpx.Response:
            sent.append(request)
            return httpx.Response(200, json={"jsonrpc": "2.0", "id": 1, "result": {}})

        config = McpServerConfig(name="handbook", endpoint="https://mcp.example/rpc")
        transport = HttpTransport(
            config,
            client=httpx.AsyncClient(transport=httpx.MockTransport(endpoint)),
            authority=_authority(_Refuses(), clock=clock),
        )
        with tenant_scope(TenantContext(tenant="acme")), pytest.raises(McpAuthError):
            await transport.request("tools/list", {}, timeout_seconds=5.0)
        await transport.close()
        assert sent == []

    @pytest.mark.asyncio
    async def test_an_unconfigured_server_gets_no_default_credential(self) -> None:
        clock = _Clock()
        authority = _authority(_Mint(clock=clock), clock=clock)
        with tenant_scope(TenantContext(tenant="acme")), pytest.raises(McpAuthError):
            await authority.headers_for(server="payroll")

    @pytest.mark.asyncio
    async def test_a_server_asking_beyond_the_caller_is_not_escalated(self) -> None:
        clock = _Clock()
        wide = McpServerAuth(server="handbook", audience="handbook.svc", scopes=("hb:write",))
        authority = _authority(
            _Mint(clock=clock),
            clock=clock,
            identity=_identity(scopes=("hb:read",)),
            servers={"handbook": wide},
        )
        with tenant_scope(TenantContext(tenant="acme")), pytest.raises(AuthorisationError):
            await authority.headers_for(server="handbook")


class TestCredentialsThatExpire:
    """A short-lived credential is replaced before it is spent, never after."""

    @pytest.mark.asyncio
    async def test_a_credential_near_expiry_is_replaced(self) -> None:
        clock = _Clock()
        mint = _Mint(clock=clock, lifetime=120.0)
        authority = _authority(mint, clock=clock)
        with tenant_scope(TenantContext(tenant="acme")):
            first = await authority.headers_for(server="handbook")
            clock.reading += 100.0
            second = await authority.headers_for(server="handbook")
        assert first["Authorization"] != second["Authorization"]
        assert len(mint.issued) == 2

    @pytest.mark.asyncio
    async def test_a_call_longer_than_the_credential_refreshes_before_it_starts(self) -> None:
        clock = _Clock()
        mint = _Mint(clock=clock, lifetime=300.0)
        authority = _authority(mint, clock=clock)
        with tenant_scope(TenantContext(tenant="acme")):
            await authority.headers_for(server="handbook", holding_for=10.0)
            clock.reading += 200.0
            await authority.headers_for(server="handbook", holding_for=120.0)
        assert len(mint.issued) == 2

    @pytest.mark.asyncio
    async def test_a_call_no_credential_can_outlive_is_refused(self) -> None:
        clock = _Clock()
        authority = _authority(_Mint(clock=clock, lifetime=60.0), clock=clock)
        with tenant_scope(TenantContext(tenant="acme")), pytest.raises(McpAuthError) as refused:
            await authority.headers_for(server="handbook", holding_for=600.0)
        assert refused.value.reason is McpAuthReason.EXPIRED


class _Forever:
    """A credential source whose credentials say nothing about expiry."""

    async def for_tool(
        self,
        *,
        identity: AgentIdentity,
        audience: str,
        needs: Iterable[str],
        run_id: str,
        agent_version: str = "1.0.0",
    ) -> _Static:
        del identity, audience, run_id, agent_version
        return _Static(scopes=frozenset(needs))


@dataclass(frozen=True, slots=True)
class _Static:
    """A credential with no expiry, which is the documented static-key exception."""

    scopes: frozenset[str]

    @property
    def token(self) -> SecretStr:
        return SecretStr("static-token-value")

    def headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.token.get_secret_value()}"}


class TestACredentialThatNeverExpires:
    """A static key is spent as it is, rather than refreshed on every call."""

    @pytest.mark.asyncio
    async def test_it_is_held_rather_than_reminted(self) -> None:
        clock = _Clock()
        authorizer = McpAuthorizer(_Forever(), servers={"handbook": _HANDBOOK})
        who = _identity()
        authority = TenantAuthority(
            authorizer,
            caller=lambda: CallerContext.current(identity=who, run_id="run-1"),
            clock=clock,
        )
        with tenant_scope(TenantContext(tenant="acme")):
            first = await authority.headers_for(server="handbook")
            clock.reading += 10_000.0
            second = await authority.headers_for(server="handbook", holding_for=10_000.0)
        assert first["Authorization"] == second["Authorization"]
        assert authority.held == 1


class TestNothingSecretEscapes:
    """A credential appears in one place only: the headers of the request it authenticates."""

    @pytest.mark.asyncio
    async def test_the_meta_and_the_span_carry_no_credential(self) -> None:
        clock = _Clock()
        authority = _authority(_Mint(clock=clock), clock=clock)
        with tenant_scope(TenantContext(tenant="acme")):
            authorised = await authority.authorised(server="handbook")
        rendered = " ".join(
            [repr(authorised), str(authorised.meta()), str(authorised.span_attributes())]
        )
        assert "tok-acme-1" not in rendered

    @pytest.mark.asyncio
    async def test_a_refusal_names_no_credential(self) -> None:
        clock = _Clock()
        authority = _authority(_Refuses(), clock=clock)
        with tenant_scope(TenantContext(tenant="acme")), pytest.raises(McpAuthError) as refused:
            await authority.headers_for(server="handbook")
        assert "unreachable" not in str(refused.value)

    def test_a_server_echoing_a_credential_back_has_it_redacted(self) -> None:
        answered = GatewayToolResult(
            content=({"type": "text", "text": "you sent Bearer tok-acme-1"},),
            structured_content={"seen": {"authorization": "Bearer tok-acme-1"}},
        )
        clean = redacted(answered, secrets=("tok-acme-1",))
        assert "tok-acme-1" not in str(clean.content)
        assert "tok-acme-1" not in str(clean.structured_content)

    def test_a_credential_that_reads_as_a_pattern_is_masked_literally(self) -> None:
        answered = GatewayToolResult(
            content=({"type": "text", "text": "sent tok.(a+b)* and tokXaXbXX"},)
        )
        clean = redacted(answered, secrets=("tok.(a+b)*",))
        assert "tok.(a+b)*" not in str(clean.content)
        assert "tokXaXbXX" in str(clean.content)

    def test_a_credential_nested_in_a_list_is_masked_too(self) -> None:
        answered = GatewayToolResult(
            content=({"type": "text", "text": "ok"},),
            structured_content={"seen": ["Bearer tok-acme-1", 7, None]},
        )
        clean = redacted(answered, secrets=("tok-acme-1",))
        assert "tok-acme-1" not in str(clean.structured_content)
        assert clean.structured_content is not None
        seen = clean.structured_content["seen"]
        assert isinstance(seen, list)
        assert seen[1] == 7

    def test_a_credential_shaped_string_is_redacted_without_being_named(self) -> None:
        answered = GatewayToolResult(
            content=({"type": "text", "text": "leaked sk-live-0123456789abcdef"},)
        )
        assert "sk-live-0123456789abcdef" not in str(redacted(answered).content)


class TestTheServerSide:
    """What a hosted server reads back off the call, refusing what it cannot attribute."""

    def test_the_context_is_read_from_the_headers(self) -> None:
        trace = W3CContext.rooted("run-1")
        headers = {
            **carried(TenantContext(tenant="acme", user="ada")),
            **trace.carried(),
        }
        meta = {
            f"{META_PREFIX}/tenant": "acme",
            f"{META_PREFIX}/subject": "ada",
            f"{META_PREFIX}/run": "run-1",
            f"{META_PREFIX}/agent": "desk",
            f"{META_PREFIX}/scopes": "hb:read",
        }
        arrived = arriving_call(headers=headers, meta=meta)
        assert arrived.tenant.tenant == "acme"
        assert arrived.subject == "ada"
        assert arrived.run_id == "run-1"
        assert arrived.scopes == frozenset({"hb:read"})
        assert arrived.trace.trace_id == trace.trace_id

    def test_a_stdio_call_is_read_from_its_metadata(self) -> None:
        meta = {
            f"{META_PREFIX}/tenant": "acme",
            f"{META_PREFIX}/subject": "ada",
            f"{META_PREFIX}/run": "run-1",
            f"{META_PREFIX}/agent": "desk",
        }
        assert arriving_call(meta=meta).tenant.tenant == "acme"

    def test_a_call_naming_no_tenant_is_refused(self) -> None:
        with pytest.raises(McpAuthError):
            arriving_call(meta={f"{META_PREFIX}/run": "run-1"})

    def test_a_header_that_cannot_be_read_is_refused_rather_than_guessed(self) -> None:
        with pytest.raises(McpAuthError):
            arriving_call(headers={HEADER: "adk/9 tenant=acme"})

    def test_a_header_disagreeing_with_the_authenticated_tenant_is_refused(self) -> None:
        with pytest.raises(McpAuthError):
            arriving_call(headers=carried(TenantContext(tenant="acme")), authenticated="globex")

    def test_a_call_disagreeing_with_the_authenticated_tenant_is_refused(self) -> None:
        meta = {f"{META_PREFIX}/tenant": "acme", f"{META_PREFIX}/run": "run-1"}
        with pytest.raises(McpAuthError):
            arriving_call(meta=meta, authenticated="globex")

    def test_handling_a_call_runs_under_its_tenant(self) -> None:
        meta = {f"{META_PREFIX}/tenant": "acme", f"{META_PREFIX}/run": "run-1"}
        arrived = arriving_call(meta=meta)
        with arrived.bound() as here:
            assert here.tenant == "acme"

    def test_an_untraced_call_is_attributed_rather_than_dropped(self) -> None:
        meta = {f"{META_PREFIX}/tenant": "acme", f"{META_PREFIX}/run": "run-1"}
        arrived: IncomingCall = arriving_call(meta=meta)
        assert arrived.trace.trace_id == W3CContext.rooted("run-1").trace_id


class TestOneCallEndToEnd:
    """The whole path: an agent's tool call reaching a server that can attribute it."""

    @pytest.mark.asyncio
    async def test_the_server_sees_the_tenant_the_run_acts_for(self) -> None:
        clock = _Clock()
        seen: list[IncomingCall] = []

        async def endpoint(request: httpx.Request) -> httpx.Response:
            body = request.read().decode()
            arrived = arriving_call(
                headers=dict(request.headers),
                meta=_meta_of(body),
            )
            seen.append(arrived)
            return httpx.Response(
                200,
                json={
                    "jsonrpc": "2.0",
                    "id": 1,
                    "result": {
                        "content": [{"type": "text", "text": f"for {arrived.tenant.tenant}"}]
                    },
                },
            )

        config = McpServerConfig(name="handbook", endpoint="https://mcp.example/rpc")
        authority = _authority(_Mint(clock=clock), clock=clock)
        transport = HttpTransport(
            config,
            client=httpx.AsyncClient(transport=httpx.MockTransport(endpoint)),
            authority=authority,
        )
        session = AuthorisingSession(
            TransportSession(transport, config=config), authority=authority, server="handbook"
        )
        with tenant_scope(TenantContext(tenant="acme", user="ada")):
            result = await session.call_tool(
                "search", {"query": "leave"}, meta={}, timeout_seconds=5.0
            )
        await session.close()
        assert seen[0].tenant.tenant == "acme"
        assert seen[0].run_id == "run-1"
        assert "for acme" in str(result.content)

    @pytest.mark.asyncio
    async def test_discovery_and_close_pass_through(self) -> None:
        clock = _Clock()
        inner = _Session()
        session = AuthorisingSession(
            inner, authority=_authority(_Mint(clock=clock), clock=clock), server="handbook"
        )
        with tenant_scope(TenantContext(tenant="acme")):
            info = await session.initialize()
            tools = await session.list_tools()
        await session.close()
        assert info.capabilities == ("tools",)
        assert [tool.name for tool in tools] == ["search"]
        assert inner.closed


class _Session:
    """An in-process session that records nothing but that it was reached."""

    def __init__(self) -> None:
        self.closed = False
        self.meta: dict[str, str] = {}

    async def initialize(self) -> McpServerInfo:
        return McpServerInfo(name="handbook", capabilities=("tools",))

    async def list_tools(self) -> tuple[McpToolDescriptor, ...]:
        return (
            McpToolDescriptor(
                name="search", description="Search.", input_schema={"type": "object"}
            ),
        )

    async def call_tool(
        self,
        name: str,
        arguments: Mapping[str, JsonValue],
        *,
        meta: Mapping[str, str],
        timeout_seconds: float,
    ) -> GatewayToolResult:
        del name, arguments, timeout_seconds
        self.meta = dict(meta)
        return GatewayToolResult(content=({"type": "text", "text": "ok"},))

    async def close(self) -> None:
        self.closed = True


def _meta_of(body: str) -> dict[str, str]:
    """The `_meta` an MCP request carried, read back the way a server would."""
    params = json.loads(body).get("params", {})
    carried_meta = params.get("_meta", {})
    return {key: str(value) for key, value in carried_meta.items()}
