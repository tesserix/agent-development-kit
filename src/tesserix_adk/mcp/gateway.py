"""One tenant-safe MCP tool surface routed through AgentGateway."""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Protocol
from urllib.parse import unquote, urlsplit, urlunsplit

from pydantic import Field, JsonValue, field_validator, model_validator

from tesserix_adk.core import AdkError, AdkModel, AgentIdentity, CredentialSource, ToolNotFoundError
from tesserix_adk.mcp.auth import McpAuthorizer, McpServerAuth, ServerSessions

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = [
    "AgentGatewayConfig",
    "AgentGatewayRoute",
    "AgentGatewayRouter",
    "AgentGatewayToolConfig",
    "AgentGatewayTools",
    "AgentGatewayTransport",
    "GatewayTool",
    "GatewayToolResult",
    "McpGatewayError",
    "McpGatewayReason",
    "McpToolDescriptor",
]

_NAME = re.compile(r"^[A-Za-z0-9_-]+$")


class McpGatewayReason(StrEnum):
    """Stable reasons an AgentGateway operation failed."""

    DISCOVERY = "discovery"
    LIMIT = "limit"
    PAYLOAD = "payload"
    TOOL_ERROR = "tool_error"
    UNAVAILABLE = "unavailable"


class McpGatewayError(AdkError):
    """An AgentGateway failure that never repeats credentials or remote content."""

    def __init__(
        self,
        message: str,
        *,
        reason: McpGatewayReason,
        server: str = "",
        tool: str = "",
        run_id: str | None = None,
        tenant: str | None = None,
    ) -> None:
        self.reason = reason
        self.server = server
        self.tool = tool
        super().__init__(
            message,
            run_id=run_id,
            tenant=tenant,
            details={"reason": reason.value, "server": server, "tool": tool},
        )

    @property
    def retryable(self) -> bool:
        """Only a dependency outage may change when the same safe operation is retried."""
        return self.reason is McpGatewayReason.UNAVAILABLE


class AgentGatewayToolConfig(AdkModel):
    """Operator-owned policy for one remote MCP tool."""

    name: str
    description: str = Field(min_length=1, max_length=512)
    scopes: tuple[str, ...] = Field(min_length=1)
    requires_approval: bool = True

    @field_validator("name")
    @classmethod
    def _portable_name(cls, value: str) -> str:
        if not _NAME.fullmatch(value):
            raise ValueError("an MCP tool name may contain only letters, digits, '_' and '-'")
        return value


class AgentGatewayRoute(AdkModel):
    """One registry-rendered AgentGateway MCP route and its local authority policy."""

    server: str
    audience: str = Field(min_length=1)
    scopes: tuple[str, ...] = Field(min_length=1)
    tools: tuple[AgentGatewayToolConfig, ...] = Field(min_length=1)

    @field_validator("server")
    @classmethod
    def _portable_server(cls, value: str) -> str:
        if not _NAME.fullmatch(value):
            raise ValueError("an MCP server name may contain only letters, digits, '_' and '-'")
        return value

    @model_validator(mode="after")
    def _tool_policy_is_narrower(self) -> AgentGatewayRoute:
        names = [tool.name for tool in self.tools]
        if len(names) != len(set(names)):
            raise ValueError(f"the {self.server} route declares a tool more than once")
        allowed = set(self.scopes)
        for tool in self.tools:
            outside = sorted(set(tool.scopes) - allowed)
            if outside:
                raise ValueError(
                    f"{self.server}/{tool.name} asks for scopes outside its route: "
                    f"{', '.join(outside)}"
                )
        return self


class AgentGatewayConfig(AdkModel):
    """The one AgentGateway origin and the MCP routes an ADK process may reach."""

    base_url: str
    routes: tuple[AgentGatewayRoute, ...] = Field(min_length=1)
    route_prefix: str = "/mcp"
    timeout_seconds: float = Field(default=15.0, gt=0, le=300)
    max_tools: int = Field(default=40, ge=1, le=128)
    max_result_bytes: int = Field(default=256 * 1024, ge=1, le=4 * 1024 * 1024)

    @field_validator("base_url")
    @classmethod
    def _http_origin(cls, value: str) -> str:
        parsed = urlsplit(value)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("AgentGateway needs an absolute http or https URL")
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("AgentGateway URL may not carry credentials, a query, or a fragment")
        if ".." in unquote(parsed.path).split("/") or "\\" in unquote(parsed.path):
            raise ValueError("AgentGateway URL may not contain path traversal")
        path = parsed.path.rstrip("/")
        return urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))

    @field_validator("route_prefix")
    @classmethod
    def _route_prefix(cls, value: str) -> str:
        decoded = unquote(value)
        if (
            not value.startswith("/")
            or ".." in decoded.split("/")
            or "\\" in decoded
            or "?" in value
            or "#" in value
        ):
            raise ValueError("route_prefix must be an absolute path without traversal or a query")
        return "/" + value.strip("/")

    @model_validator(mode="after")
    def _one_route_per_server(self) -> AgentGatewayConfig:
        servers = [route.server for route in self.routes]
        if len(servers) != len(set(servers)):
            raise ValueError("an AgentGateway server route may be declared only once")
        return self

    def endpoint(self, server: str) -> str:
        """Return the configured gateway URL for one already-validated server name."""
        if not _NAME.fullmatch(server):
            raise ValueError("an MCP server name may contain only letters, digits, '_' and '-'")
        return f"{self.base_url}{self.route_prefix}/{server}"

    def route_for(self, server: str) -> AgentGatewayRoute:
        """Return the configured route for `server`."""
        route = next((route for route in self.routes if route.server == server), None)
        if route is None:
            raise ValueError("the MCP server is not a configured AgentGateway route")
        return route


class McpToolDescriptor(AdkModel):
    """One tool returned by MCP discovery, before local policy is applied."""

    name: str
    description: str = ""
    input_schema: dict[str, JsonValue]
    output_schema: dict[str, JsonValue] | None = None

    @field_validator("name")
    @classmethod
    def _portable_name(cls, value: str) -> str:
        if not _NAME.fullmatch(value):
            raise ValueError("an MCP tool name may contain only letters, digits, '_' and '-'")
        return value


class GatewayToolResult(AdkModel):
    """A bounded MCP result with vendor types removed at the transport boundary."""

    content: tuple[dict[str, JsonValue], ...] = ()
    structured_content: dict[str, JsonValue] | None = None
    is_error: bool = False


class AgentGatewayTransport(Protocol):
    """The vendor-neutral operations the router needs from an MCP transport."""

    async def list_tools(
        self,
        *,
        endpoint: str,
        headers: Mapping[str, str],
        meta: Mapping[str, str],
        timeout_seconds: float,
        max_result_bytes: int,
        max_tools: int,
    ) -> tuple[McpToolDescriptor, ...]:
        """List one routed MCP server's tools."""

    async def call_tool(
        self,
        *,
        endpoint: str,
        tool: str,
        arguments: Mapping[str, JsonValue],
        headers: Mapping[str, str],
        meta: Mapping[str, str],
        timeout_seconds: float,
        max_result_bytes: int,
    ) -> GatewayToolResult:
        """Call one routed MCP tool."""


class GatewayTool(AdkModel):
    """A remote tool after operator policy and stable namespacing are applied."""

    name: str
    server: str
    remote_name: str
    description: str
    scopes: tuple[str, ...]
    input_schema: dict[str, JsonValue]
    output_schema: dict[str, JsonValue] | None = None
    requires_approval: bool = True


@dataclass(frozen=True, slots=True)
class AgentGatewayTools:
    """The immutable AgentGateway tool set pinned for one run and one identity."""

    declarations: tuple[GatewayTool, ...]
    _router: AgentGatewayRouter = field(repr=False)
    _identity: AgentIdentity = field(repr=False)
    _run_id: str = field(repr=False)
    _agent_version: str = field(repr=False)

    @property
    def names(self) -> tuple[str, ...]:
        """The names exposed to the agent, in configured route order."""
        return tuple(tool.name for tool in self.declarations)

    async def invoke(self, name: str, arguments: Mapping[str, JsonValue]) -> GatewayToolResult:
        """Call one pinned tool with the identity this tool set was built for."""
        tool = next((candidate for candidate in self.declarations if candidate.name == name), None)
        if tool is None:
            raise ToolNotFoundError(name, known=self.names)
        return await self._router._invoke(
            tool,
            arguments,
            identity=self._identity,
            run_id=self._run_id,
            agent_version=self._agent_version,
        )


class AgentGatewayRouter:
    """Discover an agent's fixed MCP surface through the single AgentGateway egress."""

    def __init__(
        self,
        config: AgentGatewayConfig,
        *,
        credentials: CredentialSource,
        transport: AgentGatewayTransport,
        sessions: ServerSessions | None = None,
    ) -> None:
        self._config = config
        self._transport = transport
        self._sessions = sessions or ServerSessions()
        self._authorizer = McpAuthorizer(
            credentials,
            servers={
                route.server: McpServerAuth(
                    server=route.server, audience=route.audience, scopes=route.scopes
                )
                for route in config.routes
            },
        )

    async def tools_for(
        self,
        *,
        identity: AgentIdentity,
        run_id: str,
        agent_version: str = "1.0.0",
    ) -> AgentGatewayTools:
        """Discover and pin the allowlisted tools this run may call."""
        discovered_by_route = await asyncio.gather(
            *(
                self._discover_route(
                    route,
                    identity=identity,
                    run_id=run_id,
                    agent_version=agent_version,
                )
                for route in self._config.routes
            )
        )
        found = [tool for route_tools in discovered_by_route for tool in route_tools]
        if len(found) > self._config.max_tools:
            raise McpGatewayError(
                "AgentGateway discovered more tools than the configured ceiling",
                reason=McpGatewayReason.LIMIT,
                run_id=run_id,
                tenant=identity.principal.tenant,
            )
        return AgentGatewayTools(tuple(found), self, identity, run_id, agent_version)

    async def _discover_route(
        self,
        route: AgentGatewayRoute,
        *,
        identity: AgentIdentity,
        run_id: str,
        agent_version: str,
    ) -> tuple[GatewayTool, ...]:
        eligible = tuple(tool for tool in route.tools if not identity.missing(tool.scopes))
        if not eligible:
            return ()
        policies = {tool.name: tool for tool in eligible}
        discovery_scopes = tuple(dict.fromkeys(scope for tool in eligible for scope in tool.scopes))
        lease = self._sessions.lease(server=route.server, identity=identity)
        lease.check(identity)
        call = await self._authorizer.authorise(
            server=route.server,
            identity=identity,
            needs=discovery_scopes,
            run_id=run_id,
            agent_version=agent_version,
        )
        try:
            discovered = await self._transport.list_tools(
                endpoint=self._config.endpoint(route.server),
                headers=call.headers(),
                meta=call.meta(),
                timeout_seconds=self._config.timeout_seconds,
                max_result_bytes=self._config.max_result_bytes,
                max_tools=self._config.max_tools,
            )
        except McpGatewayError as failure:
            raise McpGatewayError(
                "AgentGateway rejected the MCP discovery response",
                reason=failure.reason,
                server=route.server,
                run_id=run_id,
                tenant=identity.principal.tenant,
            ) from failure
        except Exception as failure:
            raise McpGatewayError(
                "AgentGateway was unavailable during MCP discovery",
                reason=McpGatewayReason.UNAVAILABLE,
                server=route.server,
                run_id=run_id,
                tenant=identity.principal.tenant,
            ) from failure
        remote_names = [remote.name for remote in discovered]
        if len(remote_names) != len(set(remote_names)) or not set(policies).issubset(remote_names):
            raise McpGatewayError(
                "the configured tool allowlist and AgentGateway discovery disagree",
                reason=McpGatewayReason.DISCOVERY,
                server=route.server,
                run_id=run_id,
                tenant=identity.principal.tenant,
            )
        return tuple(
            GatewayTool(
                name=f"{route.server}__{remote.name}",
                server=route.server,
                remote_name=remote.name,
                description=policy.description,
                scopes=policy.scopes,
                input_schema=remote.input_schema,
                output_schema=remote.output_schema,
                requires_approval=policy.requires_approval,
            )
            for remote in discovered
            if (policy := policies.get(remote.name)) is not None
        )

    async def _invoke(
        self,
        tool: GatewayTool,
        arguments: Mapping[str, JsonValue],
        *,
        identity: AgentIdentity,
        run_id: str,
        agent_version: str,
    ) -> GatewayToolResult:
        """Authorise and route one call from a pinned tool set."""
        route = self._config.route_for(tool.server)
        lease = self._sessions.lease(server=route.server, identity=identity)
        lease.check(identity)
        call = await self._authorizer.authorise(
            server=route.server,
            identity=identity,
            needs=tool.scopes,
            run_id=run_id,
            agent_version=agent_version,
        )
        try:
            result = await self._transport.call_tool(
                endpoint=self._config.endpoint(route.server),
                tool=tool.remote_name,
                arguments=arguments,
                headers=call.headers(),
                meta=call.meta(),
                timeout_seconds=self._config.timeout_seconds,
                max_result_bytes=self._config.max_result_bytes,
            )
        except McpGatewayError as failure:
            raise McpGatewayError(
                "AgentGateway rejected the MCP tool response",
                reason=failure.reason,
                server=route.server,
                tool=tool.remote_name,
                run_id=run_id,
                tenant=identity.principal.tenant,
            ) from failure
        except Exception as failure:
            raise McpGatewayError(
                "AgentGateway was unavailable during an MCP tool call",
                reason=McpGatewayReason.UNAVAILABLE,
                server=route.server,
                tool=tool.remote_name,
                run_id=run_id,
                tenant=identity.principal.tenant,
            ) from failure
        if result.is_error:
            raise McpGatewayError(
                "the MCP tool reported a failure",
                reason=McpGatewayReason.TOOL_ERROR,
                server=route.server,
                tool=tool.remote_name,
                run_id=run_id,
                tenant=identity.principal.tenant,
            )
        return result
