"""Choose RTK before a command and Headroom after a content boundary.

The two optimisers do not compete for the same input. RTK understands commands and must
wrap one before it runs. Headroom understands content and receives it after an API, MCP,
retrieval, conversation, or gateway boundary. The caller declares that origin; guessing
from bytes would turn a data-handling decision into a classifier result.
"""

from __future__ import annotations

import json
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from pydantic import ConfigDict, Field, ValidationError

from tesserix_adk.core.errors import AdkError
from tesserix_adk.core.models import AdkModel
from tesserix_adk.memory.compression import estimate_tokens

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = [
    "HeadroomMcpOptimizer",
    "OptimizationBackend",
    "OptimizationChannel",
    "OptimizationDecision",
    "OptimizationError",
    "OptimizationPolicy",
    "OptimizationResult",
    "RtkCommandPlan",
    "TokenOptimizer",
]


class OptimizationBackend(StrEnum):
    """The component selected for one optimisation decision."""

    NONE = "none"
    RTK = "rtk"
    HEADROOM = "headroom"


class OptimizationChannel(StrEnum):
    """Where content came from, which is the safe routing signal."""

    SHELL = "shell"
    CLI = "cli"
    JSON = "json"
    API = "api"
    MCP = "mcp"
    RAG = "rag"
    CONVERSATION = "conversation"
    GATEWAY = "gateway"
    MULTI_AGENT = "multi_agent"
    UNKNOWN = "unknown"


_HEADROOM_CHANNELS = frozenset(
    {
        OptimizationChannel.JSON,
        OptimizationChannel.API,
        OptimizationChannel.MCP,
        OptimizationChannel.RAG,
        OptimizationChannel.CONVERSATION,
        OptimizationChannel.GATEWAY,
        OptimizationChannel.MULTI_AGENT,
    }
)

_RTK_COMMANDS = frozenset(
    {
        "aws",
        "cargo",
        "docker",
        "dotnet",
        "gh",
        "git",
        "glab",
        "go",
        "golangci-lint",
        "gradlew",
        "jest",
        "kubectl",
        "mvn",
        "mypy",
        "npm",
        "npx",
        "oc",
        "phpstan",
        "phpunit",
        "playwright",
        "pnpm",
        "psql",
        "pytest",
        "rg",
        "rspec",
        "rubocop",
        "ruff",
        "sbt",
        "tsc",
        "uv",
        "vitest",
    }
)


class OptimizationError(AdkError):
    """An optimiser response was unavailable, invalid, or outside its bound scope."""


class OptimizationPolicy(AdkModel):
    """The explicit matrix used for every optimiser decision.

    Args:
        rtk_commands: Executables RTK knows how to filter.
        headroom_channels: Origins whose content Headroom understands.
        min_headroom_tokens: Inputs below this stay local because an MCP call would cost
            more than the context it could save.
        rtk_executable: The wrapper placed before an eligible command.
    """

    rtk_commands: frozenset[str] = _RTK_COMMANDS
    headroom_channels: frozenset[OptimizationChannel] = _HEADROOM_CHANNELS
    min_headroom_tokens: int = Field(default=256, ge=0)
    rtk_executable: str = Field(default="rtk", min_length=1)


class OptimizationDecision(AdkModel):
    """Which backend applies and why."""

    backend: OptimizationBackend
    reason: str


class RtkCommandPlan(AdkModel):
    """A command argv after policy selection, never a shell string."""

    backend: OptimizationBackend
    argv: tuple[str, ...]
    reason: str


class OptimizationResult(AdkModel):
    """Content after optimisation, with enough data to meter the decision."""

    content: str
    backend: OptimizationBackend
    reason: str
    original_tokens: int = Field(ge=0)
    optimized_tokens: int = Field(ge=0)
    saved_tokens: int = Field(ge=0)
    handle: str = ""
    transforms: tuple[str, ...] = ()
    untrusted: bool = False


class _McpToolSession(Protocol):
    async def call_tool(self, name: str, arguments: dict[str, str]) -> object:
        """Call one MCP tool and return its SDK result object."""


class _McpTextBlock(AdkModel):
    model_config = ConfigDict(from_attributes=True, extra="ignore")

    type: str
    text: str


class _McpResult(AdkModel):
    model_config = ConfigDict(from_attributes=True, extra="ignore")

    content: list[_McpTextBlock]


class _HeadroomCompression(AdkModel):
    compressed: str
    hash: str = Field(min_length=1)
    original_tokens: int = Field(ge=0)
    compressed_tokens: int = Field(ge=0)
    tokens_saved: int = Field(ge=0)
    savings_percent: float
    transforms: list[str]
    note: str


class _HeadroomRetrieval(AdkModel):
    hash: str = Field(min_length=1)
    source: str
    original_content: str
    original_item_count: int = Field(ge=0)
    compressed_item_count: int = Field(ge=0)
    retrieval_count: int = Field(ge=0)


class HeadroomMcpOptimizer:
    """Headroom's MCP contract bound to exactly one tenant and run.

    The Headroom wire API identifies retained content by a hash but carries no tenant or
    run on a call. Binding the session here prevents that hash becoming a capability that
    another run can redeem through the same object. Construct one instance per run.
    """

    def __init__(self, session: _McpToolSession, *, tenant: str, run_id: str) -> None:
        if not tenant or not run_id:
            raise ValueError("tenant and run_id must be non-empty to scope Headroom retrieval")
        self._session = session
        self._tenant = tenant
        self._run_id = run_id

    async def compress(
        self, content: str, *, tenant: str, run_id: str, untrusted: bool = False
    ) -> OptimizationResult:
        """Compress content through Headroom inside this instance's scope."""
        self._within_scope(tenant, run_id)
        try:
            result = await self._session.call_tool("headroom_compress", {"content": content})
            payload = _HeadroomCompression.model_validate_json(_text_of(result))
        except Exception as failed:
            raise OptimizationError("Headroom compression is unavailable") from failed
        saved = max(0, payload.original_tokens - payload.compressed_tokens)
        return OptimizationResult(
            content=payload.compressed,
            backend=OptimizationBackend.HEADROOM,
            reason="content origin is assigned to Headroom",
            original_tokens=payload.original_tokens,
            optimized_tokens=payload.compressed_tokens,
            saved_tokens=saved,
            handle=payload.hash,
            transforms=tuple(payload.transforms),
            untrusted=untrusted,
        )

    async def retrieve(self, handle: str, *, tenant: str, run_id: str) -> str:
        """Retrieve exact content, refusing a hash outside the bound tenant and run."""
        self._within_scope(tenant, run_id)
        if not handle:
            raise OptimizationError("a Headroom retrieval hash must not be empty")
        try:
            result = await self._session.call_tool("headroom_retrieve", {"hash": handle})
            payload = _HeadroomRetrieval.model_validate_json(_text_of(result))
        except Exception as failed:
            raise OptimizationError("Headroom retrieval is unavailable") from failed
        if payload.hash != handle:
            raise OptimizationError("Headroom returned content for a different retrieval hash")
        return payload.original_content

    def _within_scope(self, tenant: str, run_id: str) -> None:
        if tenant != self._tenant or run_id != self._run_id:
            raise OptimizationError("Headroom content is unavailable outside its tenant and run")


class TokenOptimizer:
    """Select RTK for commands and Headroom for shared content.

    Headroom is opt-in on every call because choosing it can cross a deployment boundary.
    The caller knows whether its configured MCP server is local, private, or external; the
    ADK cannot safely infer that from an endpoint string.
    """

    def __init__(
        self,
        *,
        headroom: HeadroomMcpOptimizer | None = None,
        policy: OptimizationPolicy | None = None,
    ) -> None:
        self._headroom = headroom
        self._policy = policy or OptimizationPolicy()

    def select(
        self,
        channel: OptimizationChannel,
        *,
        command: Sequence[str] = (),
        headroom_allowed: bool = False,
    ) -> OptimizationDecision:
        """Choose a backend from declared origin and configured availability."""
        if channel in {OptimizationChannel.SHELL, OptimizationChannel.CLI}:
            return self._for_command(command)
        if channel in self._policy.headroom_channels:
            if not headroom_allowed:
                return OptimizationDecision(
                    backend=OptimizationBackend.NONE,
                    reason="content was not permitted to cross the Headroom boundary",
                )
            if self._headroom is None:
                return OptimizationDecision(
                    backend=OptimizationBackend.NONE,
                    reason="Headroom is selected by policy but no connector is configured",
                )
            return OptimizationDecision(
                backend=OptimizationBackend.HEADROOM,
                reason="content origin is assigned to Headroom",
            )
        return OptimizationDecision(
            backend=OptimizationBackend.NONE,
            reason="content origin has no safe optimiser",
        )

    def plan_command(self, argv: Sequence[str]) -> RtkCommandPlan:
        """Return RTK-prefixed argv for a supported command, otherwise the original."""
        if not argv:
            raise ValueError("a command must contain an executable")
        command = tuple(argv)
        decision = self._for_command(command)
        if decision.backend is not OptimizationBackend.RTK or Path(command[0]).name == "rtk":
            return RtkCommandPlan(backend=decision.backend, argv=command, reason=decision.reason)
        return RtkCommandPlan(
            backend=OptimizationBackend.RTK,
            argv=(self._policy.rtk_executable, Path(command[0]).name, *command[1:]),
            reason=decision.reason,
        )

    async def optimize(
        self,
        content: str,
        *,
        channel: OptimizationChannel,
        tenant: str,
        run_id: str,
        headroom_allowed: bool = False,
        untrusted: bool = False,
    ) -> OptimizationResult:
        """Compress eligible content or return it unchanged with the bypass reason."""
        counted = estimate_tokens(content)
        decision = self.select(channel, headroom_allowed=headroom_allowed)
        if decision.backend is not OptimizationBackend.HEADROOM:
            return _unchanged(content, counted, decision.reason, untrusted)
        if counted < self._policy.min_headroom_tokens:
            return _unchanged(
                content,
                counted,
                f"content is below the Headroom threshold of "
                f"{self._policy.min_headroom_tokens} tokens",
                untrusted,
            )
        if self._headroom is None:  # narrowed by `select`; kept explicit for mypy.
            return _unchanged(content, counted, "Headroom is not configured", untrusted)
        try:
            return await self._headroom.compress(
                content, tenant=tenant, run_id=run_id, untrusted=untrusted
            )
        except OptimizationError as unavailable:
            return _unchanged(content, counted, str(unavailable), untrusted)

    async def retrieve(self, handle: str, *, tenant: str, run_id: str) -> str:
        """Retrieve Headroom content, failing closed when no scoped connector exists."""
        if self._headroom is None:
            raise OptimizationError("Headroom retrieval is not configured")
        return await self._headroom.retrieve(handle, tenant=tenant, run_id=run_id)

    def _for_command(self, command: Sequence[str]) -> OptimizationDecision:
        if not command:
            return OptimizationDecision(
                backend=OptimizationBackend.NONE,
                reason="RTK needs the command before it runs",
            )
        supplied = command[0]
        executable = Path(supplied).name
        if executable == Path(self._policy.rtk_executable).name:
            return OptimizationDecision(
                backend=OptimizationBackend.RTK,
                reason="command is already routed through RTK",
            )
        if executable != supplied:
            return OptimizationDecision(
                backend=OptimizationBackend.NONE,
                reason="RTK does not replace an explicitly resolved executable",
            )
        if executable in self._policy.rtk_commands:
            return OptimizationDecision(
                backend=OptimizationBackend.RTK,
                reason=f"RTK has a command-aware filter for {executable}",
            )
        return OptimizationDecision(
            backend=OptimizationBackend.NONE,
            reason=f"RTK has no allowlisted filter for {executable}",
        )


def _text_of(result: object) -> str:
    if _is_mcp_error(result):
        raise OptimizationError("Headroom returned an MCP error")
    try:
        parsed = _McpResult.model_validate(result, from_attributes=True)
    except ValidationError as invalid:
        raise OptimizationError("Headroom returned an invalid MCP result") from invalid
    text = [block.text for block in parsed.content if block.type == "text"]
    if len(text) != 1:
        raise OptimizationError("Headroom must return exactly one text result")
    try:
        json.loads(text[0])
    except json.JSONDecodeError as invalid:
        raise OptimizationError("Headroom returned invalid JSON") from invalid
    return text[0]


def _is_mcp_error(result: object) -> bool:
    if isinstance(result, dict):
        return result.get("isError") is True
    return getattr(result, "isError", False) is True


def _unchanged(content: str, tokens: int, reason: str, untrusted: bool) -> OptimizationResult:
    return OptimizationResult(
        content=content,
        backend=OptimizationBackend.NONE,
        reason=reason,
        original_tokens=tokens,
        optimized_tokens=tokens,
        saved_tokens=0,
        untrusted=untrusted,
    )
