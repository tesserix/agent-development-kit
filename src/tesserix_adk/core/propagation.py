"""One wire contract for carrying the tenant across a hop that leaves the process.

In-process the tenant is a contextvar (see `tesserix_adk.core.tenancy`). Across a hop it
is whatever the producer remembered to put in the message, and left to itself every
integration invents its own field: `x-tenant` on one transport, `tenantId` in a workflow
input, `org` in a tool's metadata. A shared service cannot honour three spellings, so the
one it does not know about is the one that silently runs under the worker's own tenant.

So there is one field name, one encoding and one version, used by every carrier — headers
for a peer call or a broker, a payload key for durable work whose input is the message.
Egress writes it; ingress reads it and refuses rather than guesses:

- nothing sent → refused, never the consumer's own tenant, because a default is one typo
  away from being every tenant;
- unreadable, or a version this side does not know → refused, never read field by field;
- disagreeing with the caller's authenticated claim → refused, because the payload never
  outranks the credential.

Durable work carries the context as input rather than as ambient state, so a workflow
replayed on another worker a week later reconstructs the tenant it started under instead
of inheriting the replayer's. Redelivery reads the same bytes and lands on the same tenant.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING
from urllib.parse import quote, unquote

from tesserix_adk.core.errors import TenantContextError
from tesserix_adk.core.tenancy import TenantContext, tenant_scope

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping

__all__ = [
    "HEADER",
    "MAX_HEADER_BYTES",
    "PAYLOAD_KEY",
    "VERSION",
    "arriving",
    "carried",
    "header_of",
    "in_payload",
    "of_payload",
    "restored",
]

HEADER = "adk-tenant"
"""The header every transport uses, lower-case because brokers normalise case anyway."""

PAYLOAD_KEY = "adk_tenant"
"""The key for carriers whose message is a flat string map — a queued item, a workflow input."""

VERSION = "adk/1"
"""Leads the value, so a reader that does not know the encoding refuses instead of parsing it."""

MAX_HEADER_BYTES = 1024
"""What a header is assumed to have room for, below the smallest ceiling in common use."""

_FIELDS = ("tenant", "user", "locale", "region", "correlation_id", "crossing")

# Dropped in this order when the context does not fit, least load-bearing first. `user`
# goes last because an audit entry without it names nobody; `tenant` is never dropped.
_SHEDDABLE = ("crossing", "correlation_id", "region", "locale", "user")


def header_of(context: TenantContext) -> str:
    """Encode `context` as the value of the `adk-tenant` header.

    Args:
        context: What to carry.

    Returns:
        A versioned, percent-encoded value that fits `MAX_HEADER_BYTES`. Optional fields
        are dropped to make it fit, and the result is then marked partial so the far side
        can tell a field that was absent from one that was shed.

    Raises:
        TenantContextError: With reason `oversized`, where even the tenant alone does not
            fit. Truncating is not an option: half a tenant name is a different tenant.

    Example:
        >>> header_of(TenantContext(tenant="acme"))
        'adk/1 tenant=acme'
    """
    sheddable = list(_SHEDDABLE)
    dropped: set[str] = set()
    while True:
        value = _encoded(context, dropped)
        if len(value.encode()) <= MAX_HEADER_BYTES:
            return value
        if not sheddable:
            raise TenantContextError(
                f"the tenant context for {context.tenant!r} does not fit in "
                f"{MAX_HEADER_BYTES} bytes even on its own",
                reason="oversized",
                tenant=context.tenant,
            )
        dropped.add(sheddable.pop(0))


def carried(context: TenantContext) -> dict[str, str]:
    """The headers to merge into an outgoing request.

    Args:
        context: What to carry.

    Returns:
        A single-entry mapping, so a caller merges rather than choosing a field name.

    Example:
        >>> carried(TenantContext(tenant="acme"))
        {'adk-tenant': 'adk/1 tenant=acme'}
    """
    return {HEADER: header_of(context)}


def restored(
    headers: Mapping[str, str],
    *,
    authenticated: str | None = None,
) -> TenantContext:
    """Read the context off an incoming message, or refuse it.

    Args:
        headers: What arrived. Looked up case-insensitively, since brokers and HTTP
            stacks disagree about which case they hand back.
        authenticated: The tenant the caller proved it is, where the transport
            authenticated one. A context naming a different tenant is refused rather
            than overridden — the disagreement is the finding, and a silent override
            would hide it.

    Returns:
        The context the producer sent.

    Raises:
        TenantContextError: With reason `missing`, `malformed`, `version` or
            `contradicted`. Never a default context.

    Example:
        >>> restored(carried(TenantContext(tenant="acme"))).tenant
        'acme'
    """
    value = _header_in(headers)
    if value is None:
        raise TenantContextError(
            f"no {HEADER} header on this message; the producer has to send one, because "
            f"the consumer's own tenant is not a safe default",
            reason="missing",
        )
    context = _parsed(value)
    if authenticated is not None and context.tenant != authenticated:
        raise TenantContextError(
            f"the message claims tenant {context.tenant!r} but the caller authenticated "
            f"as {authenticated!r}",
            reason="contradicted",
            tenant=authenticated,
        )
    return context


def in_payload(context: TenantContext, payload: Mapping[str, str]) -> dict[str, str]:
    """`payload` with the context added under `PAYLOAD_KEY`.

    For work whose message is its input rather than a request with headers: a queued item,
    a durable workflow argument. Recording it in the input is what makes a replay
    deterministic — ambient state on the worker that happens to run the replay is not.

    Args:
        context: What to carry.
        payload: The caller's own fields, which are copied rather than mutated.

    Returns:
        A new mapping carrying both.

    Example:
        >>> in_payload(TenantContext(tenant="acme"), {"booking": "AB-1"})["booking"]
        'AB-1'
    """
    return {**payload, PAYLOAD_KEY: header_of(context)}


def of_payload(payload: Mapping[str, str]) -> TenantContext:
    """Read the context back out of a payload, or refuse it.

    Args:
        payload: The item's input, as it was recorded.

    Returns:
        The context the work was enqueued under — the same one on every redelivery and
        every replay, whatever is bound on the worker reading it.

    Raises:
        TenantContextError: With reason `missing`, `malformed` or `version`.

    Example:
        >>> of_payload(in_payload(TenantContext(tenant="acme"), {})).tenant
        'acme'
    """
    value = payload.get(PAYLOAD_KEY)
    if value is None:
        raise TenantContextError(
            f"no {PAYLOAD_KEY} in this payload; work enqueued without a tenant cannot be "
            f"run under the worker's own",
            reason="missing",
        )
    return _parsed(value)


@contextmanager
def arriving(
    headers: Mapping[str, str],
    *,
    authenticated: str | None = None,
) -> Iterator[TenantContext]:
    """Bind the incoming message's tenant for the length of handling it.

    The ingress half of `carried`: what a queue consumer, a peer endpoint or a tool server
    wraps its handler in, so everything below reads the producer's tenant from the context
    rather than being handed it.

    Args:
        headers: What arrived.
        authenticated: The tenant the caller proved it is, where there is one.

    Yields:
        The bound context.

    Raises:
        TenantContextError: Where the message cannot be trusted or read. Raised before
            anything is bound, so a refused message never runs under a partial context.
        TenantCrossingError: Where the consumer already holds a different tenant. A worker
            with its own tenant bound picking up another's work is the failure this exists
            to catch, so it is loud rather than a silent rebind.

    Example:
        >>> with arriving(carried(TenantContext(tenant="acme"))) as here:
        ...     here.tenant
        'acme'
    """
    with tenant_scope(restored(headers, authenticated=authenticated)) as context:
        yield context


def _encoded(context: TenantContext, dropped: set[str]) -> str:
    """`context` as a header value, without the fields in `dropped`."""
    parts = [
        f"{field}={quote(value, safe='')}"
        for field in _FIELDS
        if field not in dropped
        for value in (getattr(context, field),)
        if value is not None
    ]
    if context.partial or dropped:
        parts.append("partial=1")
    return f"{VERSION} {';'.join(parts)}"


def _header_in(headers: Mapping[str, str]) -> str | None:
    """The header's value whatever case the stack handed it back in."""
    for name, value in headers.items():
        if name.lower() == HEADER:
            return value
    return None


def _parsed(value: str) -> TenantContext:
    """Decode a header value, refusing anything this side cannot read exactly."""
    version, _, rest = value.partition(" ")
    if version != VERSION:
        raise TenantContextError(
            f"tenant context version {version!r} is not one this build reads ({VERSION!r}); "
            f"reading it field by field is how a tenant becomes a locale",
            reason="version",
        )
    fields = {
        name: unquote(raw)
        for name, sep, raw in (pair.partition("=") for pair in rest.split(";"))
        if sep and name in {*_FIELDS, "partial"}
    }
    tenant = fields.get("tenant", "")
    if not tenant.strip():
        raise TenantContextError(
            f"tenant context {value!r} names no tenant",
            reason="malformed",
        )
    return TenantContext(
        tenant=tenant,
        user=fields.get("user"),
        locale=fields.get("locale"),
        region=fields.get("region"),
        correlation_id=fields.get("correlation_id"),
        crossing=fields.get("crossing"),
        partial=fields.get("partial") == "1",
    )
