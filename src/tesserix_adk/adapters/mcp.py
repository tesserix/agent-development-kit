"""An MCP server's tools, adopted as kit tools rather than run as a second tool system.

Capability already sits behind MCP servers, and every product that wants it writes the
same client twice: once to translate the server's schema into whatever the model client
wants, and once to paste the server's answer into the conversation. Both halves are where
the bugs are. A schema keyword nobody translated loses a required field, so the call fails
at the server instead of at validation; a response pasted in as text is remote instruction
sitting in the same position as the operator's own words.

Here a discovered tool becomes an ordinary `Tool`: its arguments are held to the server's
own JSON Schema before the call leaves the process, its call is bounded by the declared
timeout, its failures land in the kit's taxonomy, and its content comes back inside the
untrusted-data envelope. A tool whose schema the kit cannot hold a call to is not
registered at all — the remaining tools still load, and nothing is guessed.

Transports live behind `McpSession`, so this module knows nothing about sockets and its
tests run entirely in process.
"""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, cast

from tesserix_adk.adapters._mcp_schema import (
    MAX_SCHEMA_BYTES,
    MAX_SCHEMA_DEPTH,
    SchemaArguments,
    schema_shape,
    schema_violations,
)
from tesserix_adk.adapters.mcp_surface import (
    SurfaceEntry,
    ToolSurface,
    fingerprint,
    namespaced,
)
from tesserix_adk.core import (
    AdkError,
    AdkModel,
    ConfigurationError,
    ContentSource,
    Idempotency,
    IdempotencyPolicy,
    Origin,
    ToolFailure,
    ToolRefusal,
    ToolTimedOutError,
    screen,
    sealed,
)
from tesserix_adk.mcp import GatewayToolResult, McpGatewayError, McpToolDescriptor
from tesserix_adk.tools import STRICT, ArgumentPolicy, Tool, ToolContext

if TYPE_CHECKING:
    from collections.abc import Collection, Mapping
    from types import TracebackType

    from pydantic import JsonValue

    from tesserix_adk.adapters.mcp_surface import SurfacePin
    from tesserix_adk.core.config import McpServerConfig

__all__ = [
    "McpClient",
    "McpDiscovery",
    "McpProtocolError",
    "McpRejection",
    "McpSchemaError",
    "McpServerInfo",
    "McpSession",
]

_MAX_PAYLOAD_BYTES = 4 * 1024

_CODE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_TOOLS_CAPABILITY = "tools"
_EVERYTHING = "*"


class McpServerInfo(AdkModel):
    """What a server said about itself when the session was initialised.

    Args:
        name: What the server calls itself, which is not what the operator calls it.
        version: The server's own version.
        protocol_version: The MCP revision it negotiated.
        capabilities: What it offers. A server that does not offer tools is one this
            client has nothing to do with.
    """

    name: str = ""
    version: str = ""
    protocol_version: str = ""
    capabilities: tuple[str, ...] = ()


class McpSession(Protocol):
    """The vendor-neutral operations this client needs from a connected MCP server.

    A transport implements it; a test implements it in process. Nothing here mentions a
    socket, so nothing about adopting a server's tools depends on how it is reached.
    """

    async def initialize(self) -> McpServerInfo:
        """Negotiate the session and return what the server said about itself."""

    async def list_tools(self) -> tuple[McpToolDescriptor, ...]:
        """Return the tools the server advertises right now."""

    async def call_tool(
        self,
        name: str,
        arguments: Mapping[str, JsonValue],
        *,
        meta: Mapping[str, str],
        timeout_seconds: float,
    ) -> GatewayToolResult:
        """Call one tool with arguments that have already been validated locally."""

    async def close(self) -> None:
        """Release the session. Called once, whatever happened to it."""


class McpSchemaError(AdkError):
    """A tool whose input schema the kit cannot hold a call to, and why.

    Raised during discovery and reported per tool: one unrepresentable schema costs that
    tool, not the server. Never carries the schema itself, which is remote input.

    Args:
        server: The server as configured here.
        tool: The tool it advertised.
        construct: What could not be held — the offending keyword or bound.
    """

    def __init__(self, message: str, *, server: str, tool: str, construct: str) -> None:
        self.server = server
        self.tool = tool
        self.construct = construct
        super().__init__(message, details={"server": server, "tool": tool, "construct": construct})


class McpProtocolError(AdkError):
    """A server answered with something that is not the answer it promised.

    A truncated stream, a protocol violation, or structured content the tool's own result
    schema does not allow. Nothing partially parsed is accepted as a result, and the raw
    payload travels with the error, bounded, so the failure can be read without being
    reproduced.

    Args:
        server: The server as configured here.
        tool: The tool that was called, under the server's own name for it.
        payload: What actually came back, truncated to a size a log can carry.
    """

    def __init__(self, message: str, *, server: str, tool: str, payload: str = "") -> None:
        self.server = server
        self.tool = tool
        self.payload = payload[:_MAX_PAYLOAD_BYTES]
        super().__init__(message, details={"server": server, "tool": tool, "payload": self.payload})


class McpRejection(AdkModel):
    """One advertised tool this process will not register, and the reason it will not.

    Args:
        tool: The name the server advertised.
        reason: Why it was not adopted, in terms of the rule rather than the content.
    """

    tool: str
    reason: str


@dataclass(frozen=True, slots=True)
class McpDiscovery:
    """What one pass over a server's tool list produced.

    Args:
        server: The server as configured here.
        tools: The tools adopted, in the order the allowlist or the server gave them.
        rejected: Every tool not adopted, each with its reason. Discovery reports rather
            than throws: one broken schema does not cost the rest of the server.
        surface: What was adopted, under both names, with a fingerprint of each schema.
        truncated: Whether the discovery cap held, which is a fact a reader needs.
    """

    server: str
    tools: tuple[Tool[..., str], ...] = ()
    rejected: tuple[McpRejection, ...] = ()
    surface: ToolSurface = field(default_factory=ToolSurface)
    truncated: bool = False


class McpClient:
    """One MCP server, adopted into the kit's own tool contract.

    The client owns the session for as long as the consuming agent needs it and gives it
    back on `aclose`, which `async with` does for you.

    Args:
        session: The connected server, behind the transport-neutral protocol.
        config: The declared server: its name here, its allowlist, and its bounds. It is
            ordinary configuration, so a server can be added by an environment variable.
        arguments: How strictly arguments are read before a call leaves the process.
        pin: The tool surface this server is expected to have. Given one, a server that
            has since added, dropped or reshaped a tool fails discovery instead of
            quietly changing what the model can do.
    """

    __slots__ = ("_arguments", "_closed", "_config", "_discovery", "_info", "_pin", "_session")

    def __init__(
        self,
        session: McpSession,
        *,
        config: McpServerConfig,
        arguments: ArgumentPolicy = STRICT,
        pin: SurfacePin | None = None,
    ) -> None:
        self._session = session
        self._config = config
        self._arguments = arguments
        self._pin = pin
        self._info: McpServerInfo | None = None
        self._discovery: McpDiscovery | None = None
        self._closed = False

    @property
    def server(self) -> str:
        """What this server is called here, which is what its tools' content is sealed as."""
        return self._config.name

    @property
    def required(self) -> bool:
        """Whether an agent can be assembled at all without this server's tools."""
        return self._config.required

    async def connect(self) -> McpServerInfo:
        """Initialise the session once and negotiate that the server offers tools at all.

        Returns:
            What the server said about itself, cached for the life of the client.

        Raises:
            ConfigurationError: The client is closed, or the server does not advertise the
                tools capability, which makes declaring it here a configuration mistake.
        """
        self._open()
        if self._info is None:
            info = await self._session.initialize()
            if _TOOLS_CAPABILITY not in info.capabilities:
                raise ConfigurationError(
                    f"the MCP server {self._config.name} does not advertise tools"
                )
            self._info = info
        return self._info

    async def tools(self) -> tuple[Tool[..., str], ...]:
        """The adopted tools, discovered once and then held still.

        A server may add a tool or change a schema at any time; a run that is already
        under way keeps the surface it started with. `refresh` is how the view widens.
        """
        return (await self._discovered()).tools

    async def discover(self, *, known: Collection[str] = ()) -> McpDiscovery:
        """Read the server's tool list and adopt what this process can validate calls for.

        Args:
            known: Tool names already taken locally. A remote tool answering to one of
                them is a conflict, not something to let shadow the local tool.

        Returns:
            The tools adopted, those rejected and why, the resolved surface, and whether
            the cap held.

        Raises:
            ConfigurationError: The client is closed, the server offers no tools, or the
                allowlist names a tool this server does not advertise.
            McpToolConflictError: Two tools would answer to one name.
            McpSurfaceDriftError: The server's tools are not what the pin says.
        """
        info = await self.connect()
        _ = info
        advertised = await self._session.list_tools()
        selected, rejected = self._selected(advertised)
        capped, over = selected[: self._config.max_tools], selected[self._config.max_tools :]
        rejected.extend(
            McpRejection(
                tool=descriptor.name,
                reason=f"over the discovery cap of {self._config.max_tools} tools",
            )
            for descriptor in over
        )
        adopted: list[Tool[..., str]] = []
        entries: list[SurfaceEntry] = []
        spent = 0
        for descriptor in capped:
            size = self._size(descriptor.input_schema)
            if spent + size > self._config.max_schema_bytes:
                rejected.append(
                    McpRejection(
                        tool=descriptor.name,
                        reason=f"over the schema budget of {self._config.max_schema_bytes} bytes",
                    )
                )
                continue
            name = namespaced(descriptor.name, prefix=self._config.prefix)
            try:
                adopted.append(self._adapted(descriptor, name))
            except McpSchemaError as unrepresentable:
                rejected.append(McpRejection(tool=descriptor.name, reason=str(unrepresentable)))
                continue
            spent += size
            entries.append(
                SurfaceEntry(
                    server=self._config.name,
                    tool=descriptor.name,
                    name=name,
                    fingerprint=fingerprint(descriptor),
                )
            )
        surface = ToolSurface.of(entries, local=known)
        if self._pin is not None:
            surface.check(self._pin)
        self._discovery = McpDiscovery(
            server=self._config.name,
            tools=tuple(adopted),
            rejected=tuple(rejected),
            surface=surface,
            truncated=bool(over),
        )
        return self._discovery

    async def refresh(self, *, known: Collection[str] = ()) -> McpDiscovery:
        """Discover again on purpose, replacing the held tool view with what is there now."""
        self._discovery = None
        return await self.discover(known=known)

    async def aclose(self) -> None:
        """Close the session. Doing it twice does nothing; using the client after it fails."""
        if self._closed:
            return
        self._closed = True
        await self._session.close()

    async def __aenter__(self) -> McpClient:
        """Return the client. The session is initialised on first use, not here."""
        return self

    async def __aexit__(
        self,
        kind: type[BaseException] | None,
        error: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Give the session back however the block ended."""
        await self.aclose()

    async def _discovered(self) -> McpDiscovery:
        """The held discovery, running one if there has not been one yet."""
        self._open()
        if self._discovery is None:
            return await self.discover()
        return self._discovery

    def _open(self) -> None:
        """Refuse to work through a session that has already been given back."""
        if self._closed:
            raise ConfigurationError(f"the MCP client for {self._config.name} is closed")

    def _selected(
        self, advertised: tuple[McpToolDescriptor, ...]
    ) -> tuple[list[McpToolDescriptor], list[McpRejection]]:
        """Default-deny: the allowlist minus the denylist, in the order it was declared."""
        by_name = {descriptor.name: descriptor for descriptor in advertised}
        missing = [
            name for name in self._config.allow if name != _EVERYTHING and name not in by_name
        ]
        if missing:
            raise ConfigurationError(
                f"{self._config.name} does not advertise {', '.join(missing)}, "
                f"which its allowlist names"
            )
        denied = set(self._config.deny)
        allowed = (
            [name for name in by_name if name not in denied]
            if _EVERYTHING in self._config.allow
            else [name for name in self._config.allow if name not in denied]
        )
        rejected = [
            McpRejection(
                tool=descriptor.name,
                reason=(
                    f"{self._config.name} denies {descriptor.name}"
                    if descriptor.name in denied
                    else f"{self._config.name} does not allow {descriptor.name}"
                ),
            )
            for descriptor in advertised
            if descriptor.name not in allowed
        ]
        return [by_name[name] for name in allowed], rejected

    def _adapted(self, descriptor: McpToolDescriptor, name: str) -> Tool[..., str]:
        """One advertised tool as a native tool, or a typed rejection of its schema."""
        schema = _closed(descriptor.input_schema)
        validator = self._validator(descriptor, schema)
        context_parameter = self._context_parameter(descriptor)

        async def invoke(**arguments: Any) -> str:  # noqa: ANN401 — whatever the schema declares
            context = cast("ToolContext | None", arguments.pop(context_parameter, None))
            return await self._call(descriptor, arguments, context, name)

        return Tool[..., str](
            name=name,
            description=self._described(descriptor),
            parameters_schema=dict(schema),
            returns_schema=dict(descriptor.output_schema) if descriptor.output_schema else None,
            is_async=True,
            function=invoke,
            validator=validator,
            context_parameter=context_parameter,
            context_required=False,
            timeout=self._config.timeout_seconds,
            parallel_safe=False,
            idempotency=IdempotencyPolicy(kind=Idempotency.EFFECTFUL),
            returns_type=str,
        )

    def _described(self, descriptor: McpToolDescriptor) -> str:
        """A tool description a server wrote, fenced where it reads as an instruction.

        The description sits in the tool list, which is the most privileged place remote
        text reaches. One that screens as instructions is kept — the operator allowed the
        tool — but inside the untrusted envelope, where it is data rather than a line.
        """
        if not screen(descriptor.description):
            return descriptor.description
        return sealed(
            descriptor.description,
            source=ContentSource(
                origin=Origin.MCP_RESULT, name=f"{self._config.name}/{descriptor.name}"
            ),
        )

    def _validator(
        self, descriptor: McpToolDescriptor, schema: dict[str, JsonValue]
    ) -> SchemaArguments:
        """The server's own schema, bounded, compiled, and made the contract for the call."""
        if schema.get("type") != "object":
            raise self._unrepresentable(
                descriptor, "type", "its root is not an object, and arguments are an object"
            )
        depth, references = schema_shape(cast("JsonValue", schema))
        if depth > MAX_SCHEMA_DEPTH:
            raise self._unrepresentable(
                descriptor, "nesting depth", f"it nests deeper than {MAX_SCHEMA_DEPTH}"
            )
        if self._size(schema) > MAX_SCHEMA_BYTES:
            raise self._unrepresentable(
                descriptor, "schema size", f"it is larger than {MAX_SCHEMA_BYTES} bytes"
            )
        if any(not isinstance(ref, str) or not ref.startswith("#") for ref in references):
            raise self._unrepresentable(descriptor, "$ref", "it points outside its own document")
        try:
            return SchemaArguments(schema, tool=descriptor.name, policy=self._arguments)
        except McpGatewayError as invalid:
            raise self._unrepresentable(
                descriptor, "schema", "it is not a schema anything can be validated against"
            ) from invalid

    def _size(self, schema: Mapping[str, JsonValue]) -> int:
        """The encoded size, where the schema is JSON at all."""
        try:
            return len(json.dumps(schema, ensure_ascii=False, separators=(",", ":")).encode())
        except (RecursionError, TypeError, ValueError):
            return MAX_SCHEMA_BYTES + 1

    def _unrepresentable(
        self, descriptor: McpToolDescriptor, construct: str, why: str
    ) -> McpSchemaError:
        """One rejection naming the server, the tool and the construct, never the schema."""
        return McpSchemaError(
            f"{self._config.name} advertises {descriptor.name} with a schema the kit cannot "
            f"validate a call against: {why} ({construct})",
            server=self._config.name,
            tool=descriptor.name,
            construct=construct,
        )

    def _context_parameter(self, descriptor: McpToolDescriptor) -> str:
        """A name for the injected context that the server's own arguments cannot take."""
        properties = descriptor.input_schema.get("properties")
        taken = set(properties) if isinstance(properties, dict) else set()
        name = "context"
        while name in taken:
            name = f"_{name}"
        return name

    async def _call(
        self,
        descriptor: McpToolDescriptor,
        arguments: Mapping[str, Any],
        context: ToolContext | None,
        name: str,
    ) -> str:
        """One bounded call, with the answer fenced as the untrusted data it is.

        Failures answer to `name`, which is what the model called, while the wire carries
        the server's own name for the tool.
        """
        timeout = self._config.timeout_seconds
        try:
            async with asyncio.timeout(timeout):
                result = await self._session.call_tool(
                    descriptor.name,
                    cast("Mapping[str, JsonValue]", arguments),
                    meta=self._meta(context),
                    timeout_seconds=timeout,
                )
        except TimeoutError as overran:
            raise ToolTimedOutError(name, timeout) from overran
        except AdkError:
            raise
        except Exception as unreachable:
            raise ToolFailure(
                name,
                "mcp_unavailable",
                transient=True,
                detail="the MCP server could not be reached",
            ) from unreachable
        if result.is_error:
            raise self._error(name, result)
        self._promised(descriptor, name, result)
        return sealed(
            self._rendered(result),
            source=ContentSource(
                origin=Origin.MCP_RESULT, name=f"{self._config.name}/{descriptor.name}"
            ),
        )

    def _promised(
        self, descriptor: McpToolDescriptor, name: str, result: GatewayToolResult
    ) -> None:
        """Hold a result to the schema the tool advertised, where it advertised one.

        A truncated answer that still parses is the dangerous one: it reads as a result and
        is half of one. Nothing partially parsed is accepted, and the payload travels with
        the error so the failure can be read without reproducing it.
        """
        if not descriptor.output_schema:
            return
        content = result.structured_content
        if content is None:
            raise McpProtocolError(
                f"{name} promises structured content and {self._config.name} returned none",
                server=self._config.name,
                tool=descriptor.name,
                payload=self._rendered(result),
            )
        violations = schema_violations(descriptor.output_schema, content)
        if violations:
            raise McpProtocolError(
                f"{name} answered outside its own result schema at {', '.join(violations)}",
                server=self._config.name,
                tool=descriptor.name,
                payload=json.dumps(content, ensure_ascii=False, sort_keys=True),
            )

    def _meta(self, context: ToolContext | None) -> dict[str, str]:
        """What the call carries besides its arguments: the run, and the key for repeating it."""
        if context is None:
            return {}
        meta = {"run-id": context.run_id}
        if context.idempotency_key:
            meta["idempotency-key"] = context.idempotency_key
        return meta

    def _error(self, tool: str, result: GatewayToolResult) -> ToolFailure | ToolRefusal:
        """A server's decline is an answer; anything else it failed at is a failure.

        Neither carries the server's own words. Remote text in an error message is remote
        text a model reads outside the envelope, which is the thing this module exists to
        prevent.
        """
        declined = (result.structured_content or {}).get("refusal")
        if isinstance(declined, dict):
            code = declined.get("code")
            return ToolRefusal(
                tool,
                code if isinstance(code, str) and _CODE.fullmatch(code) else "declined",
                "the MCP server declined the call",
            )
        return ToolFailure(
            tool, "mcp_tool_error", transient=False, detail="the MCP server reported a failure"
        )

    def _rendered(self, result: GatewayToolResult) -> str:
        """Multi-part content as one bounded block: no bytes, no vendor shapes, no surprises."""
        lines = [self._part(part) for part in result.content]
        if result.structured_content is not None:
            lines.append(json.dumps(result.structured_content, ensure_ascii=False, sort_keys=True))
        text = "\n".join(line for line in lines if line)
        ceiling = self._config.max_result_bytes
        encoded = text.encode()
        if len(encoded) <= ceiling:
            return text
        kept = encoded[:ceiling].decode(errors="ignore")
        return f"{kept}\n[truncated at {ceiling} bytes]"

    def _part(self, part: Mapping[str, JsonValue]) -> str:
        """One content part as text a model can read without being handed a payload."""
        kind = part.get("type")
        if kind == "text":
            text = part.get("text")
            return text if isinstance(text, str) else ""
        if kind == "image":
            data = part.get("data")
            size = len(data) if isinstance(data, str) else 0
            return f"[image {part.get('mimeType') or 'unknown'}, {size} bytes elided]"
        if kind in {"resource", "resource_link"}:
            return f"[resource {self._uri(part)}]"
        return f"[{kind if isinstance(kind, str) else 'unknown'} content elided]"

    def _uri(self, part: Mapping[str, JsonValue]) -> str:
        """The link a resource part carries, however that server nested it."""
        uri = part.get("uri")
        if isinstance(uri, str):
            return uri
        resource = part.get("resource")
        nested = resource.get("uri") if isinstance(resource, dict) else None
        return nested if isinstance(nested, str) else "unknown"


def _closed(schema: dict[str, JsonValue]) -> dict[str, JsonValue]:
    """A root object that says nothing about extra fields is read as forbidding them.

    A model that sends a field the server never advertised has invented it, and inventing
    it is the failure. The reading is put back into the advertised schema so that what the
    model is shown and what the call is held to are the same document.
    """
    if schema.get("type") != "object" or "additionalProperties" in schema:
        return dict(schema)
    return {**schema, "additionalProperties": False}
