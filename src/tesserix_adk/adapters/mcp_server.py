"""Kit tools published over MCP, under the guarantees the in-process path already gives.

Capability written as kit tools is reusable inside one process and nowhere else, so the
second consumer reimplements it as a standalone MCP server: the schema copied by hand, the
validation rewritten, the tenant check remembered or not. The rewrite is where a tool ends
up enforcing less than the tool it was copied from.

So publishing is a view of what already exists rather than a second implementation. The
descriptors are generated from the `@tool` definitions, so a published schema cannot drift
from the body behind it. The call runs through the same registry view an in-process caller
uses, so the ceiling, the lanes and the typed errors are the same ones. What is added here
is what only a remote caller needs: an export allowlist narrower than the agent's own, the
tenant read from the request rather than assumed, per-tenant lanes so one caller cannot
take the process, and an approval gate that answers `pending` instead of running because
the caller is far away.

Nothing here widens under load: an unexported name is refused exactly as a name nobody
registered is, and a session serves the allowlist it was opened with.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
from enum import StrEnum
from typing import TYPE_CHECKING, Any, cast

from pydantic import BaseModel

from tesserix_adk.adapters.mcp import McpServerInfo
from tesserix_adk.adapters.mcp_context import arriving_call, redacted
from tesserix_adk.adapters.mcp_surface import namespaced
from tesserix_adk.adapters.mcp_transport import PROTOCOL_VERSION
from tesserix_adk.core import AdkError, McpAuthError, McpAuthReason
from tesserix_adk.core.errors import ToolTimedOutError
from tesserix_adk.core.hooks import ApprovalRecord
from tesserix_adk.core.redaction import scrub
from tesserix_adk.mcp import META_PREFIX, GatewayToolResult, McpToolDescriptor
from tesserix_adk.runtime.blocking import Ambient, carrying
from tesserix_adk.tools.errors import ToolErrorMap, ToolRefusal, permanent

if TYPE_CHECKING:
    from collections.abc import Collection, Sequence
    from contextlib import AbstractContextManager

    from pydantic import JsonValue

    from tesserix_adk.adapters.mcp_context import IncomingCall
    from tesserix_adk.tools import AgentToolView, Tool

__all__ = [
    "APPROVAL_REQUIRED",
    "ExportedSession",
    "McpExportError",
    "McpExportReason",
    "McpServer",
    "published",
]

APPROVAL_REQUIRED = "approval_required"
"""The refusal code a call held for a human is answered with."""

_OBJECT = "object"
_IDEMPOTENCY_KEYS = ("idempotency-key", f"{META_PREFIX}/idempotency-key")
_ERRORS = ToolErrorMap({ToolTimedOutError: permanent("tool_timed_out")})


class McpExportReason(StrEnum):
    """Why a published server refused a request before any tool body ran."""

    NOT_FOUND = "not_found"
    """Nothing is published under that name, whether or not anything is registered."""

    INVALID_ARGUMENTS = "invalid_arguments"
    """The arguments are not what the published schema allows."""

    UNSUPPORTED_PROTOCOL = "unsupported_protocol"
    """The client speaks a revision this server does not."""


class McpExportError(AdkError):
    """A request the published server would not serve, with nothing of the request in it.

    Args:
        reason: What was wrong, in terms a caller can branch on.
        tool: The name the caller used, where the refusal is about one.
    """

    def __init__(self, message: str, *, reason: McpExportReason, tool: str = "") -> None:
        self.reason = reason
        self.tool = tool
        super().__init__(message, details={"reason": reason.value, "tool": tool})


def published(tool: Tool[Any, Any]) -> McpToolDescriptor:
    """The MCP descriptor for `tool`, generated from its definition rather than written.

    The result schema is published only where the tool returns an object, because that is
    the only shape MCP structured content can be held to. A tool returning a string is
    published without one rather than with one no answer could satisfy.

    Example:
        >>> from tesserix_adk.tools import tool as as_tool
        >>> @as_tool(name="fare_for")
        ... def fare_for(leg: str) -> dict[str, str]:
        ...     '''Price a leg.'''
        ...     return {"leg": leg}
        >>> published(fare_for).name
        'fare_for'
    """
    returns = tool.returns_schema
    return McpToolDescriptor(
        name=namespaced(tool.name),
        description=tool.description,
        input_schema=tool.parameters_schema,
        output_schema=returns if returns and returns.get("type") == _OBJECT else None,
    )


class McpServer:
    """The kit tools one process publishes, and the sessions callers hold them through.

    Args:
        view: The agent's own tool view. What it may call bounds what may be exported: a
            server cannot publish authority the agent behind it does not have.
        exports: The names published, which is normally narrower than the view. Explicit
            rather than "everything the agent has", because the agent's allowlist is
            written for the agent and read by nobody thinking about remote callers.
        name: What the server calls itself when a session is initialised.
        version: The server's own version, which is not the protocol's.
        per_tenant_calls: How many calls one tenant may have in flight here. The registry
            bounds the process; this bounds one caller's share of it.
        secrets: Exact values to mask out of every result, beside the shapes always masked.
        protocol_versions: The MCP revisions this server serves.
        require_authenticated_tenant: Whether every session must carry a tenant identity
            established by the transport. Agent exports enable this; general local MCP
            servers retain the backward-compatible metadata-only option.

    Raises:
        McpExportError: If `exports` names something the view cannot call.

    Example:
        >>> from tesserix_adk.tools import ToolRegistry, tool as as_tool
        >>> @as_tool(name="fare_for")
        ... def fare_for(leg: str) -> dict[str, str]:
        ...     '''Price a leg.'''
        ...     return {"leg": leg}
        >>> registry = ToolRegistry((fare_for,))
        >>> server = McpServer(registry.view(allow=("fare_for",)), exports=("fare_for",))
        >>> server.exports
        ('fare_for',)
    """

    __slots__ = (
        "_exports",
        "_lanes",
        "_name",
        "_require_authentication",
        "_secrets",
        "_version",
        "_versions",
        "_view",
    )

    def __init__(
        self,
        view: AgentToolView,
        *,
        exports: Sequence[str],
        name: str = "",
        version: str = "0",
        per_tenant_calls: int = 8,
        secrets: Sequence[str] = (),
        protocol_versions: Collection[str] = (PROTOCOL_VERSION,),
        require_authenticated_tenant: bool = False,
    ) -> None:
        self._view = view
        self._name = name
        self._version = version
        self._secrets = tuple(secrets)
        self._versions = frozenset(protocol_versions)
        self._require_authentication = require_authenticated_tenant
        self._lanes = _TenantLanes(per_tenant_calls)
        self._exports = self._checked(exports)

    @property
    def exports(self) -> tuple[str, ...]:
        """What is published right now, which is what the next session will serve."""
        return self._exports

    def reload(self, exports: Sequence[str]) -> None:
        """Publish `exports` from now on, leaving open sessions serving what they have.

        Raises:
            McpExportError: If `exports` names something the view cannot call. The
                previous allowlist stays in force: a reload that half-applied would
                publish a set nobody wrote.
        """
        self._exports = self._checked(exports)

    def connect(
        self,
        *,
        headers: Mapping[str, str] | None = None,
        meta: Mapping[str, str] | None = None,
        authenticated: str | None = None,
        protocol_version: str = PROTOCOL_VERSION,
    ) -> ExportedSession:
        """Open a session over the allowlist as it stands now.

        Args:
            headers: What the transport carried, where it carries headers.
            meta: Session-level metadata, overridden per call by the call's own.
            authenticated: The tenant the edge proved the caller to be. A call claiming
                another is refused rather than believed.
            protocol_version: The revision the client asked for, checked on `initialize`.

        Raises:
            McpAuthError: If this server requires transport authentication and none was
                supplied. Tenant metadata alone is a claim, not proof.
        """
        if self._require_authentication and not authenticated:
            raise McpAuthError(
                "this MCP server requires a transport-authenticated tenant",
                reason=McpAuthReason.UNAUTHENTICATED,
            )
        return ExportedSession(
            self._view,
            exports=self._exports,
            name=self._name,
            version=self._version,
            secrets=self._secrets,
            lanes=self._lanes,
            headers=dict(headers or {}),
            meta=dict(meta or {}),
            authenticated=authenticated,
            protocol_version=protocol_version,
            supported=self._versions,
        )

    def _checked(self, exports: Sequence[str]) -> tuple[str, ...]:
        """`exports` as a stable tuple, refusing anything the view cannot call."""
        named = tuple(dict.fromkeys(exports))
        for name in named:
            try:
                self._view.resolve(name)
            except AdkError as unavailable:
                raise McpExportError(
                    f"{name} cannot be published: this agent cannot call it",
                    reason=McpExportReason.NOT_FOUND,
                    tool=name,
                ) from unavailable
        return named


class ExportedSession:
    """One caller's session: the tools it was opened with, and the calls it makes.

    Built by `McpServer.connect`. It implements the same session protocol the kit's MCP
    client consumes, so a published server can be driven in process by the very client
    that would reach it over a socket — which is how the two paths are held to the same
    behaviour rather than asserted to have it.
    """

    __slots__ = (
        "_authenticated",
        "_by_published_name",
        "_descriptors",
        "_headers",
        "_lanes",
        "_meta",
        "_name",
        "_protocol_version",
        "_secrets",
        "_supported",
        "_version",
        "_view",
    )

    def __init__(
        self,
        view: AgentToolView,
        *,
        exports: tuple[str, ...],
        name: str,
        version: str,
        secrets: tuple[str, ...],
        lanes: _TenantLanes,
        headers: dict[str, str],
        meta: dict[str, str],
        authenticated: str | None,
        protocol_version: str,
        supported: frozenset[str],
    ) -> None:
        self._view = view
        self._name = name
        self._version = version
        self._secrets = secrets
        self._lanes = lanes
        self._headers = headers
        self._meta = meta
        self._authenticated = authenticated
        self._protocol_version = protocol_version
        self._supported = supported
        self._descriptors = tuple(published(view.resolve(each)) for each in exports)
        self._by_published_name = {
            descriptor.name: registered
            for descriptor, registered in zip(self._descriptors, exports, strict=True)
        }

    async def initialize(self) -> McpServerInfo:
        """Negotiate the session, refusing a revision this server does not speak.

        Raises:
            McpExportError: If the client asked for an unsupported protocol revision. A
                best-effort session is one where neither side knows what the other meant.
        """
        if self._protocol_version not in self._supported:
            raise McpExportError(
                f"this server speaks {', '.join(sorted(self._supported))} and the client "
                f"asked for {self._protocol_version}",
                reason=McpExportReason.UNSUPPORTED_PROTOCOL,
            )
        return McpServerInfo(
            name=self._name,
            version=self._version,
            protocol_version=self._protocol_version,
            capabilities=("tools",),
        )

    async def list_tools(self) -> tuple[McpToolDescriptor, ...]:
        """The published tools, as this session was opened with them."""
        return self._descriptors

    async def call_tool(
        self,
        name: str,
        arguments: Mapping[str, JsonValue],
        *,
        meta: Mapping[str, str],
        timeout_seconds: float,
    ) -> GatewayToolResult:
        """Run one published tool for whoever the request says it is for.

        Args:
            name: The published name. One that is not published is refused as not found,
                whether or not the process registers it.
            arguments: What the caller chose, validated against the published schema
                before the body is entered.
            meta: The call's `_meta`, carrying the tenant, the subject and the run.
            timeout_seconds: Ignored: the ceiling is the registry's, so a remote caller
                cannot ask for a longer one than a local caller gets.

        Returns:
            The tool's answer, redacted, or an error result carrying a typed code.

        Raises:
            McpExportError: If the name is not published, or the arguments are outside
                its schema.
            McpAuthError: If the call names no tenant, or names one it did not
                authenticate as.
        """
        del timeout_seconds
        registered = self._by_published_name.get(name)
        if registered is None:
            raise McpExportError(
                f"no tool named {name} is published here",
                reason=McpExportReason.NOT_FOUND,
                tool=name,
            )
        stated = {**self._meta, **meta}
        incoming = arriving_call(
            headers=self._headers, meta=stated, authenticated=self._authenticated
        )
        tool = self._view.resolve(registered)
        checked = self._validated(tool, arguments)
        if tool.requires_approval(checked):
            return self._pending(tool, checked, incoming)
        async with self._lanes.held(incoming.tenant.tenant):
            return self._answer(await self._ran(registered, checked, incoming, stated))

    async def close(self) -> None:
        """Release the session. There is nothing to release: the tools outlive it."""

    def _validated(
        self, tool: Tool[Any, Any], arguments: Mapping[str, JsonValue]
    ) -> dict[str, Any]:
        """The arguments the body would receive, or a refusal that never reaches it."""
        try:
            return dict(tool.validator.arguments(dict(arguments)))
        except AdkError as outside:
            raise McpExportError(
                scrub(str(outside)),
                reason=McpExportReason.INVALID_ARGUMENTS,
                tool=tool.name,
            ) from None

    async def _ran(
        self,
        name: str,
        arguments: Mapping[str, Any],
        incoming: IncomingCall,
        stated: Mapping[str, str],
    ) -> GatewayToolResult:
        """One call, under the caller's tenant and inside the registry's own bounds."""
        with incoming.bound(), _ambient(incoming, name, stated):
            try:
                produced = await self._view.invoke(name, dict(arguments))
            except Exception as failure:
                return _failed(_ERRORS.classify(failure, tool=name))
        return _rendered(produced)

    def _pending(
        self, tool: Tool[Any, Any], arguments: Mapping[str, Any], incoming: IncomingCall
    ) -> GatewayToolResult:
        """The answer to a call a human has to decide about, with no arguments in it."""
        record = ApprovalRecord.for_call(
            run_id=incoming.run_id or f"mcp-{tool.name}",
            tenant=incoming.tenant.tenant,
            agent_name=incoming.agent or self._name or "mcp",
            tool_name=tool.name,
            arguments=arguments,
            reason=tool.approval.reason or "this tool requires approval",
        )
        return GatewayToolResult(
            structured_content={
                "refusal": {"code": APPROVAL_REQUIRED},
                "approval": record.model_dump(mode="json"),
            },
            is_error=True,
        )

    def _answer(self, result: GatewayToolResult) -> GatewayToolResult:
        """The last thing that happens before anything leaves the process."""
        return redacted(result, secrets=self._secrets)


class _TenantLanes:
    """One tenant's share of the process, rebuilt if the event loop changes."""

    def __init__(self, width: int) -> None:
        self._width = width
        self._loop: asyncio.AbstractEventLoop | None = None
        self._lanes: dict[str, asyncio.Semaphore] = {}

    def held(self, tenant: str) -> asyncio.Semaphore:
        """The lane calls for `tenant` stand in."""
        running = asyncio.get_running_loop()
        if self._loop is not running:
            self._loop = running
            self._lanes = {}
        lane = self._lanes.get(tenant)
        if lane is None:
            lane = asyncio.Semaphore(self._width)
            self._lanes[tenant] = lane
        return lane


def _ambient(
    incoming: IncomingCall, name: str, stated: Mapping[str, str]
) -> AbstractContextManager[None]:
    """The run a tool taking a `ToolContext` sees, built from the request."""
    return carrying(
        Ambient(
            run_id=incoming.run_id or f"mcp-{name}",
            tenant=incoming.tenant.tenant,
            user=incoming.subject or None,
            scopes=tuple(sorted(incoming.scopes)),
            trace=incoming.trace.carried(),
            idempotency_key=next(
                (stated[key] for key in _IDEMPOTENCY_KEYS if stated.get(key)), None
            ),
        )
    )


def _rendered(produced: object) -> GatewayToolResult:
    """A tool's own return value as MCP content, structured where it has structure."""
    if isinstance(produced, BaseModel):
        produced = produced.model_dump(mode="json")
    if isinstance(produced, Mapping):
        structured = cast(
            "dict[str, JsonValue]",
            {str(key): value for key, value in cast("Mapping[object, object]", produced).items()},
        )
        text = json.dumps(structured, ensure_ascii=False, sort_keys=True, default=str)
        return GatewayToolResult(
            content=({"type": "text", "text": text},), structured_content=structured
        )
    text = produced if isinstance(produced, str) else json.dumps(produced, default=str)
    return GatewayToolResult(content=({"type": "text", "text": text},))


def _failed(failure: AdkError) -> GatewayToolResult:
    """A typed error as an error result, carrying its code and none of its words.

    The code is the tool's own vocabulary and is safe to publish; the detail is whatever
    the body said, and a body's words reaching a remote model outside the untrusted-data
    envelope is the thing every other guard here exists to prevent.
    """
    code = getattr(failure, "code", "") or "unmapped_failure"
    key = "refusal" if isinstance(failure, ToolRefusal) else "failure"
    return GatewayToolResult(structured_content={key: {"code": code}}, is_error=True)
