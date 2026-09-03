"""Official MCP Streamable HTTP transport for the AgentGateway router."""

from __future__ import annotations

import asyncio
import json
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from datetime import timedelta
from typing import TYPE_CHECKING, Any, Protocol, cast

import httpx
from pydantic import JsonValue

from tesserix_adk.adapters._mcp_schema import (
    SchemaArguments,
    bounded_schema,
    without_remote_prose,
)
from tesserix_adk.core import ApprovalPolicy, require_extra
from tesserix_adk.mcp import (
    AgentGatewayTools,
    GatewayTool,
    GatewayToolResult,
    McpGatewayError,
    McpGatewayReason,
    McpToolDescriptor,
)
from tesserix_adk.tools import Tool

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable, Mapping

__all__ = ["McpStreamableHttpTransport", "agent_gateway_tools"]


class _Dumpable(Protocol):
    def model_dump(self, *, mode: str, by_alias: bool) -> dict[str, Any]:
        """Return one vendor SDK model as JSON data."""


class _ClientSession(Protocol):
    async def discover(self) -> object:
        """Discover the modern stateless MCP surface."""

    async def list_tools(self, cursor: str | None = None) -> _Dumpable:
        """Return one page of tools."""

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, JsonValue] | None = None,
        read_timeout_seconds: timedelta | None = None,
        *,
        meta: dict[str, Any] | None = None,
    ) -> _Dumpable:
        """Call one MCP tool."""


type _SessionFactory = Callable[..., AbstractAsyncContextManager[_ClientSession]]
type _StreamableClient = Callable[
    ..., AbstractAsyncContextManager[tuple[object, object, Callable[[], str | None]]]
]


class McpStreamableHttpTransport:
    """Open one bounded official-SDK MCP connection for each gateway operation."""

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
        """Discover a routed server and return all of its MCP tool pages."""
        del meta
        async with asyncio.timeout(timeout_seconds):
            async with self._session(
                endpoint=endpoint, headers=headers, timeout_seconds=timeout_seconds
            ) as session:
                cursor: str | None = None
                found: list[McpToolDescriptor] = []
                seen: set[str] = set()
                pages = 0
                while True:
                    pages += 1
                    if pages > max_tools + 1:
                        raise McpGatewayError(
                            "MCP discovery exceeded the configured page ceiling",
                            reason=McpGatewayReason.LIMIT,
                        )
                    page = await session.list_tools(cursor)
                    page_data = page.model_dump(mode="json", by_alias=True)
                    raw_tools = page_data.get("tools")
                    if not isinstance(raw_tools, list):
                        raise McpGatewayError(
                            "the MCP server returned an invalid tool page",
                            reason=McpGatewayReason.DISCOVERY,
                        )
                    for raw_tool in raw_tools:
                        if not isinstance(raw_tool, dict):
                            raise McpGatewayError(
                                "the MCP server returned an invalid tool descriptor",
                                reason=McpGatewayReason.DISCOVERY,
                            )
                        found.append(_tool_descriptor(raw_tool))
                        if len(found) > max_tools:
                            raise McpGatewayError(
                                "MCP discovery exceeded the configured tool ceiling",
                                reason=McpGatewayReason.LIMIT,
                            )
                    next_cursor = page_data.get("nextCursor")
                    if next_cursor is None:
                        break
                    if not isinstance(next_cursor, str) or next_cursor in seen:
                        raise McpGatewayError(
                            "the MCP server repeated a discovery cursor",
                            reason=McpGatewayReason.DISCOVERY,
                        )
                    seen.add(next_cursor)
                    cursor = next_cursor
        _within_limit(
            [tool.model_dump(mode="json") for tool in found], max_result_bytes=max_result_bytes
        )
        return tuple(found)

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
        """Discover a routed server and call one tool through the official SDK."""
        async with asyncio.timeout(timeout_seconds):
            async with self._session(
                endpoint=endpoint, headers=headers, timeout_seconds=timeout_seconds
            ) as session:
                raw = await session.call_tool(
                    tool,
                    dict(arguments),
                    read_timeout_seconds=timedelta(seconds=timeout_seconds),
                    meta=dict(meta),
                )
        result = _tool_result(raw.model_dump(mode="json", by_alias=True))
        _within_limit(result.model_dump(mode="json"), max_result_bytes=max_result_bytes)
        return result

    @asynccontextmanager
    async def _session(
        self, *, endpoint: str, headers: Mapping[str, str], timeout_seconds: float
    ) -> AsyncIterator[_ClientSession]:
        """Open one stateless SDK connection without ambient proxies or redirects."""
        mcp = require_extra("mcp", "mcp")
        streamable_module = require_extra("mcp", "mcp.client.streamable_http")
        session_factory = cast("_SessionFactory", mcp.ClientSession)
        streamable = cast("_StreamableClient", streamable_module.streamable_http_client)
        timeout = httpx.Timeout(timeout_seconds)
        async with (
            httpx.AsyncClient(
                headers=dict(headers),
                timeout=timeout,
                follow_redirects=False,
                trust_env=False,
            ) as client,
            streamable(endpoint, http_client=client) as (read, write, _),
            session_factory(
                read, write, read_timeout_seconds=timedelta(seconds=timeout_seconds)
            ) as session,
        ):
            await session.discover()
            yield session


def _within_limit(value: object, *, max_result_bytes: int) -> None:
    """Refuse a parsed MCP response that exceeds the configured transport ceiling."""
    try:
        size = len(json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode())
    except (RecursionError, TypeError, ValueError) as failure:
        raise McpGatewayError(
            "the MCP server returned a response outside the JSON boundary",
            reason=McpGatewayReason.PAYLOAD,
        ) from failure
    if size > max_result_bytes:
        raise McpGatewayError(
            f"the MCP response was {size} bytes, over the ceiling of {max_result_bytes}",
            reason=McpGatewayReason.PAYLOAD,
        )


def _tool_descriptor(raw: dict[str, Any]) -> McpToolDescriptor:
    name = raw.get("name")
    description = raw.get("description") or ""
    input_schema = raw.get("inputSchema")
    output_schema = raw.get("outputSchema")
    if (
        not isinstance(name, str)
        or not isinstance(description, str)
        or not isinstance(input_schema, dict)
        or (output_schema is not None and not isinstance(output_schema, dict))
    ):
        raise McpGatewayError(
            "the MCP server returned an invalid tool descriptor",
            reason=McpGatewayReason.DISCOVERY,
        )
    try:
        return McpToolDescriptor(
            name=name,
            description=description,
            input_schema=cast("dict[str, JsonValue]", input_schema),
            output_schema=cast("dict[str, JsonValue] | None", output_schema),
        )
    except (RecursionError, TypeError, ValueError) as failure:
        raise McpGatewayError(
            "the MCP server returned an invalid tool descriptor",
            reason=McpGatewayReason.DISCOVERY,
        ) from failure


def _tool_result(raw: dict[str, Any]) -> GatewayToolResult:
    content = raw.get("content")
    structured = raw.get("structuredContent")
    is_error = raw.get("isError", False)
    if (
        not isinstance(content, list)
        or any(not isinstance(item, dict) for item in content)
        or (structured is not None and not isinstance(structured, dict))
        or not isinstance(is_error, bool)
    ):
        raise McpGatewayError(
            "the MCP server returned an invalid tool result",
            reason=McpGatewayReason.PAYLOAD,
        )
    try:
        return GatewayToolResult(
            content=tuple(cast("list[dict[str, JsonValue]]", content)),
            structured_content=cast("dict[str, JsonValue] | None", structured),
            is_error=is_error,
        )
    except (RecursionError, TypeError, ValueError) as failure:
        raise McpGatewayError(
            "the MCP server returned an invalid tool result",
            reason=McpGatewayReason.PAYLOAD,
        ) from failure


def agent_gateway_tools(routed: AgentGatewayTools) -> tuple[Tool[..., JsonValue], ...]:
    """Adapt a pinned AgentGateway tool set into tools a ToolRegistry can adopt."""
    return tuple(_gateway_tool(routed, declaration) for declaration in routed.declarations)


def _gateway_tool(routed: AgentGatewayTools, declaration: GatewayTool) -> Tool[..., JsonValue]:
    schema = bounded_schema(declaration.input_schema, tool=declaration.name)
    advertised = cast("dict[str, Any]", without_remote_prose(schema))
    output = (
        cast(
            "dict[str, Any]",
            without_remote_prose(bounded_schema(declaration.output_schema, tool=declaration.name)),
        )
        if declaration.output_schema is not None
        else None
    )

    async def invoke(**arguments: JsonValue) -> JsonValue:
        result = await routed.invoke(declaration.name, arguments)
        if result.structured_content is not None:
            return result.structured_content
        return {"content": list(result.content)}

    return Tool[..., JsonValue](
        name=declaration.name,
        description=declaration.description,
        parameters_schema=advertised,
        returns_schema=output,
        is_async=True,
        function=invoke,
        validator=SchemaArguments(schema, tool=declaration.name),
        timeout=None,
        parallel_safe=False,
        approval=ApprovalPolicy(required=declaration.requires_approval),
        returns_type=None,
    )
