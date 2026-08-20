"""Who an MCP call is for, decided by the run rather than by the code making the call.

An MCP server reached with one long-lived service credential sees the platform, not the
caller: it cannot scope its answer to a tenant, and an agent acting for one customer holds
whatever the platform holds. The fix is not a header a tool remembers to set. It is that
the authority and the context are assembled here, per call, from the run — the bound
tenant, the resolved identity, the trace this process sits in — and that a call which
cannot be attributed does not leave the process at all.

A credential is minted for one tenant, one subject and one audience, held under a key that
names all three, and replaced before it expires rather than after. It is rendered in
exactly one place, the headers of the request it authenticates: never in `_meta`, never in
a span attribute, never in a refusal, and scrubbed out of anything a server sends back.

The server half reads the same fields the client half wrote, and refuses a call naming no
tenant instead of running it under its own.
"""

from __future__ import annotations

import asyncio
import re
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, Self

from tesserix_adk.core import (
    ExpiringCredential,
    McpAuthError,
    McpAuthReason,
    TenantContext,
    TenantContextError,
    scrub,
    tenant_here,
    tenant_scope,
)
from tesserix_adk.core.propagation import HEADER, carried, restored
from tesserix_adk.mcp import META_PREFIX, AuthorisedCall, GatewayToolResult, McpAuthorizer
from tesserix_adk.observability.propagation import TRACEPARENT, TRACESTATE, W3CContext
from tesserix_adk.tools.credentials import EXPIRY_SKEW_SECONDS

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator, Mapping

    from pydantic import JsonValue

    from tesserix_adk.adapters.mcp import McpServerInfo, McpSession
    from tesserix_adk.core import AgentIdentity, Clock
    from tesserix_adk.mcp import McpToolDescriptor

__all__ = [
    "AuthorisingSession",
    "CallAuthority",
    "CallerContext",
    "IncomingCall",
    "TenantAuthority",
    "arriving_call",
    "redacted",
]

_MIN_SECRET = 8
"""Below this a header value is not a secret, it is a flag, and masking it hides nothing."""


@dataclass(frozen=True, slots=True)
class CallerContext:
    """The run's own answer to who a call is for and where it sits in a trace.

    Built from what is bound rather than from arguments, so a tool cannot state a tenant
    other than the one its run is executing under.

    Args:
        tenant: The bound isolation boundary.
        identity: What the run holds, resolved when it started.
        run_id: The run, carried downstream as attribution.
        trace: Where this process sits in the trace the call continues.
    """

    tenant: TenantContext
    identity: AgentIdentity
    run_id: str
    trace: W3CContext

    @classmethod
    def current(
        cls,
        *,
        identity: AgentIdentity,
        run_id: str,
        headers: Mapping[str, str] | None = None,
        sampled: bool = True,
    ) -> Self:
        """The context for a call made here and now.

        Args:
            identity: Who the run acts for and what it holds.
            run_id: The run making the call.
            headers: What arrived with the work, where the run was started by a hop. The
                trace is continued from it; absent one a trace is started for the run, so
                a call is attributed rather than untraced.
            sampled: What to decide only where no upstream decision arrived.

        Returns:
            The context, whose tenant is the bound one.

        Raises:
            McpAuthError: If no tenant is bound, or the bound tenant is not the one the
                identity was resolved for. Both are refused rather than resolved in favour
                of one: an unscoped call and a cross-tenant call are the two things this
                exists to make impossible.
        """
        here = tenant_here()
        if here is None:
            raise McpAuthError(
                "no tenant is bound, so there is nobody this MCP call could be made for",
                reason=McpAuthReason.UNAUTHENTICATED,
            )
        if here.tenant != identity.principal.tenant:
            raise McpAuthError(
                f"the run acts for {identity.principal.tenant!r} but {here.tenant!r} is bound "
                f"here, and a call cannot be made for both",
                reason=McpAuthReason.UNAUTHENTICATED,
            )
        upstream = W3CContext.extracted(headers or {})
        trace = (
            upstream.child(run_id)
            if upstream is not None
            else W3CContext.rooted(run_id, sampled=sampled)
        )
        return cls(tenant=here, identity=identity, run_id=run_id, trace=trace)

    def headers(self) -> dict[str, str]:
        """The context as headers: the tenant and the trace, and nothing secret.

        Example:
            >>> from tesserix_adk.core import Principal, tenant_scope
            >>> from tesserix_adk.core.identity import AgentIdentity
            >>> who = AgentIdentity.resolve(
            ...     agent="desk", declared=(), principal=Principal(subject="ada", tenant="acme")
            ... )
            >>> with tenant_scope(TenantContext(tenant="acme")):
            ...     sorted(CallerContext.current(identity=who, run_id="run-1").headers())
            ['adk-tenant', 'traceparent', 'tracestate']
        """
        return {**carried(self.tenant), **self.trace.carried()}

    def meta(self) -> dict[str, str]:
        """The trace as MCP metadata, for a transport with no headers to put it on."""
        rendered = self.trace.carried()
        return {
            f"{META_PREFIX}/traceparent": rendered[TRACEPARENT],
            **(
                {f"{META_PREFIX}/tracestate": rendered[TRACESTATE]}
                if rendered.get(TRACESTATE)
                else {}
            ),
        }


class CallAuthority(Protocol):
    """Whatever decides what one MCP request may present. `TenantAuthority` does."""

    async def headers_for(self, *, server: str, holding_for: float = ...) -> Mapping[str, str]:
        """The headers for one request: the credential, the tenant and the trace.

        Raises:
            McpAuthError: If nothing can be presented. Raised before the request is built,
                so an unauthorised call never reaches the wire.
        """
        ...

    async def meta_for(self, *, server: str, holding_for: float = ...) -> Mapping[str, str]:
        """The MCP `_meta` entries for one call, which never carry the credential."""
        ...

    async def secrets_for(self, *, server: str) -> tuple[str, ...]:
        """The values a result must not contain, so an echoed credential can be scrubbed."""
        ...


class TenantAuthority:
    """Per-call authority for MCP servers, keyed by the tenant and subject it was minted for.

    One process serving two tenants holds two credentials for the same server and cannot
    hand either to the other: the key names the tenant, and the caller is read from what is
    bound at the moment of the call rather than from when the authority was constructed.

    Args:
        authorizer: What mints the credential and narrows the scopes to the server.
        caller: How to read the run's context. Called per request, because a long-lived
            authority outlives any one run.
        clock: What decides whether a held credential is still worth presenting.
        skew_seconds: How far ahead of expiry a credential is replaced, for a downstream
            clock running fast.
    """

    __slots__ = ("_authorizer", "_caller", "_clock", "_held", "_locks", "_skew")

    def __init__(
        self,
        authorizer: McpAuthorizer,
        *,
        caller: Callable[[], CallerContext],
        clock: Clock,
        skew_seconds: float = EXPIRY_SKEW_SECONDS,
    ) -> None:
        self._authorizer = authorizer
        self._caller = caller
        self._clock = clock
        self._skew = skew_seconds
        self._held: dict[tuple[str, str, str], tuple[AuthorisedCall, CallerContext]] = {}
        self._locks: dict[tuple[str, str, str], asyncio.Lock] = {}

    @property
    def held(self) -> int:
        """How many credentials are cached, one per tenant, subject and server."""
        return len(self._held)

    async def authorised(self, *, server: str, holding_for: float = 0.0) -> AuthorisedCall:
        """The authority for one call to `server`, minted or refreshed as needed.

        Args:
            server: Which configured server the call is to.
            holding_for: How long the call may run. A credential that would expire inside
                that window is replaced before the call starts rather than during it.

        Returns:
            The call's credential and its metadata.

        Raises:
            McpAuthError: If the server is not configured, if no credential could be
                minted, or if no credential can be minted that outlives the call. Never a
                broader or an ambient credential.
            AuthorisationError: If the run holds nothing the server was configured for.
        """
        return (await self._resolved(server=server, holding_for=holding_for))[0]

    async def _resolved(
        self, *, server: str, holding_for: float
    ) -> tuple[AuthorisedCall, CallerContext]:
        """The authority for one call and the context it was minted under, which agree."""
        context = self._caller()
        principal = context.identity.principal
        key = (principal.tenant, principal.subject, server)
        async with self._locks.setdefault(key, asyncio.Lock()):
            existing = self._held.get(key)
            if existing is not None and not self._stale(existing[0], holding_for):
                return existing
            minted = await self._minted(server=server, context=context)
            if self._stale(minted, holding_for):
                raise McpAuthError(
                    f"a credential for {server} expires inside the {holding_for}s this call may "
                    f"take, and a call is not started on one that will not outlive it",
                    reason=McpAuthReason.EXPIRED,
                    server=server,
                )
            self._held[key] = (minted, context)
            return minted, context

    async def headers_for(self, *, server: str, holding_for: float = 0.0) -> dict[str, str]:
        """The headers one request carries: the credential, the tenant and the trace.

        Raises:
            McpAuthError: As `authorised`. Nothing is returned half-authorised.
            AuthorisationError: As `authorised`.
        """
        authority, context = await self._resolved(server=server, holding_for=holding_for)
        return {**context.headers(), **authority.headers()}

    async def meta_for(self, *, server: str, holding_for: float = 0.0) -> dict[str, str]:
        """The `_meta` entries one call carries, which are the context and never the token."""
        authority, context = await self._resolved(server=server, holding_for=holding_for)
        return {**authority.meta(), **context.meta()}

    async def secrets_for(self, *, server: str) -> tuple[str, ...]:
        """The values a result must not contain, so an echoed credential can be scrubbed."""
        authority = await self.authorised(server=server)
        return tuple(value for value in authority.headers().values() if len(value) >= _MIN_SECRET)

    def _stale(self, authority: AuthorisedCall, holding_for: float) -> bool:
        """Whether the credential will still be good when the call it starts finishes."""
        credential = authority.credential
        if not isinstance(credential, ExpiringCredential):
            return False
        return credential.expired(self._clock.now() + holding_for, skew=self._skew)

    async def _minted(self, *, server: str, context: CallerContext) -> AuthorisedCall:
        """One credential for this caller and this server, narrowed to what the server has."""
        return await self._authorizer.authorise(
            server=server,
            identity=context.identity,
            needs=self._authorizer.declared(server).scopes,
            run_id=context.run_id,
        )


class AuthorisingSession:
    """An MCP session that attributes every call and redacts what comes back.

    Wraps any session, so the same context reaches a server over stdio, where the metadata
    is the only channel, as over HTTP, where the headers carry it too.

    Args:
        session: The connected server.
        authority: What mints the call's authority.
        server: The server as configured here.
    """

    __slots__ = ("_authority", "_server", "_session")

    def __init__(self, session: McpSession, *, authority: CallAuthority, server: str) -> None:
        self._session = session
        self._authority = authority
        self._server = server

    async def initialize(self) -> McpServerInfo:
        """Negotiate the session, which carries no caller of its own."""
        return await self._session.initialize()

    async def list_tools(self) -> tuple[McpToolDescriptor, ...]:
        """The server's tool list, which is the same for every caller."""
        return await self._session.list_tools()

    async def call_tool(
        self,
        name: str,
        arguments: Mapping[str, JsonValue],
        *,
        meta: Mapping[str, str],
        timeout_seconds: float,
    ) -> GatewayToolResult:
        """Call one tool as the bound caller, and scrub the answer of credential material.

        Raises:
            McpAuthError: If the call cannot be attributed or authorised. Raised before
                the tool is called.
        """
        carried_meta = await self._authority.meta_for(
            server=self._server, holding_for=timeout_seconds
        )
        result = await self._session.call_tool(
            name,
            arguments,
            meta={**meta, **carried_meta},
            timeout_seconds=timeout_seconds,
        )
        return redacted(result, secrets=await self._authority.secrets_for(server=self._server))

    async def close(self) -> None:
        """Release the wrapped session."""
        await self._session.close()


@dataclass(frozen=True, slots=True)
class IncomingCall:
    """What a hosted MCP server reads off a call before it does any work.

    Args:
        tenant: Who the call is for, as the caller stated and the transport allowed.
        subject: The acting principal within that tenant, where one was named.
        run_id: The caller's run, so both sides' logs join on one identifier.
        agent: Which agent called.
        scopes: What the caller presented, for a server that narrows its own answer.
        trace: The trace to record under, continued from the caller or started here.
    """

    tenant: TenantContext
    run_id: str
    trace: W3CContext
    subject: str = ""
    agent: str = ""
    scopes: frozenset[str] = field(default_factory=frozenset)

    @contextmanager
    def bound(self) -> Iterator[TenantContext]:
        """Handle the call under its tenant, so nothing below reads the server's own."""
        with tenant_scope(self.tenant) as context:
            yield context


def arriving_call(
    *,
    headers: Mapping[str, str] | None = None,
    meta: Mapping[str, str] | None = None,
    authenticated: str | None = None,
) -> IncomingCall:
    """Read who an arriving MCP call is for, or refuse it.

    The ingress half of `TenantAuthority`: the headers where the transport has them, the
    call's own metadata where it does not, and the same refusal either way.

    Args:
        headers: What the transport carried, where it carries headers.
        meta: The call's `_meta`, which every transport carries.
        authenticated: The tenant the caller proved it is, where the edge authenticated
            one. A call claiming another is refused rather than overridden.

    Returns:
        The caller's tenant, attribution and trace.

    Raises:
        McpAuthError: If the call names no tenant, or names one the caller did not
            authenticate as. A server's own tenant is never the default.

    Example:
        >>> arriving_call(meta={f"{META_PREFIX}/tenant": "acme"}).tenant.tenant
        'acme'
    """
    stated = dict(meta or {})
    run_id = stated.get(f"{META_PREFIX}/run", "")
    tenant = _tenant_of(headers or {}, stated, authenticated=authenticated)
    trace = W3CContext.extracted(headers or {}) or _traced(stated, run_id=run_id)
    scopes = stated.get(f"{META_PREFIX}/scopes", "")
    return IncomingCall(
        tenant=tenant,
        run_id=run_id,
        trace=trace,
        subject=stated.get(f"{META_PREFIX}/subject", "") or tenant.user or "",
        agent=stated.get(f"{META_PREFIX}/agent", ""),
        scopes=frozenset(scopes.split()) if scopes else frozenset(),
    )


def redacted(result: GatewayToolResult, *, secrets: Iterable[str] = ()) -> GatewayToolResult:
    """The same result with credential material masked out of it.

    A server that echoes a request header back in its content puts a live token in the
    place a transcript, a span and a memory all read from. Masking here is the last point
    before that happens.

    Args:
        result: What the server answered.
        secrets: Exact values to mask as well as the shapes a credential usually has.

    Returns:
        A result whose text and structured content carry neither.

    Example:
        >>> answered = GatewayToolResult(content=({"type": "text", "text": "sent tok-acme-1"},))
        >>> redacted(answered, secrets=("tok-acme-1",)).content[0]["text"]
        'sent [redacted]'
    """
    exact = tuple(re.escape(value) for value in secrets if len(value) >= _MIN_SECRET)
    content = tuple(_clean_part(part, exact) for part in result.content)
    structured = result.structured_content
    return GatewayToolResult(
        content=content,
        structured_content=_clean_object(structured, exact) if structured is not None else None,
        is_error=result.is_error,
    )


def _tenant_of(
    headers: Mapping[str, str], meta: Mapping[str, str], *, authenticated: str | None
) -> TenantContext:
    """The tenant the call named, from wherever this transport could carry it."""
    if HEADER in {name.lower() for name in headers}:
        try:
            return restored(headers, authenticated=authenticated)
        except TenantContextError as unusable:
            raise McpAuthError(
                "the call's tenant could not be read, so it was not handled",
                reason=McpAuthReason.UNAUTHENTICATED,
            ) from unusable
    named = meta.get(f"{META_PREFIX}/tenant", "")
    if not named:
        raise McpAuthError(
            "the call names no tenant, and the server's own tenant is not a safe default",
            reason=McpAuthReason.UNAUTHENTICATED,
        )
    if authenticated is not None and named != authenticated:
        raise McpAuthError(
            f"the call claims tenant {named!r} but the caller authenticated as {authenticated!r}",
            reason=McpAuthReason.UNAUTHENTICATED,
        )
    return TenantContext(tenant=named, user=meta.get(f"{META_PREFIX}/subject") or None)


def _traced(meta: Mapping[str, str], *, run_id: str) -> W3CContext:
    """The trace the metadata carried, or one started here so the call is not untraced."""
    parent = meta.get(f"{META_PREFIX}/traceparent", "")
    state = meta.get(f"{META_PREFIX}/tracestate", "")
    extracted = W3CContext.extracted({TRACEPARENT: parent, TRACESTATE: state}) if parent else None
    return extracted or W3CContext.rooted(run_id or "untraced")


def _clean_part(part: Mapping[str, JsonValue], secrets: tuple[str, ...]) -> dict[str, JsonValue]:
    """One content part with its text scrubbed, leaving its shape alone."""
    return _clean_object(dict(part), secrets)


def _clean_object(value: Mapping[str, JsonValue], secrets: tuple[str, ...]) -> dict[str, JsonValue]:
    """A JSON object with every string inside it masked."""
    return {key: _clean(nested, secrets) for key, nested in value.items()}


def _clean(value: JsonValue, secrets: tuple[str, ...]) -> JsonValue:
    """Every string anywhere inside `value`, masked."""
    if isinstance(value, str):
        return scrub(value, secrets)
    if isinstance(value, dict):
        return {key: _clean(nested, secrets) for key, nested in value.items()}
    if isinstance(value, list):
        return [_clean(nested, secrets) for nested in value]
    return value
