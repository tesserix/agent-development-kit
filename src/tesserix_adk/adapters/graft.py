"""Graft transports behind the ADK code-intelligence contract."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from pydantic import ConfigDict, ValidationError

from tesserix_adk.code_intelligence import (
    CodeContextOperation,
    CodeContextRequest,
    CodeContextResult,
    CodeIntelligenceError,
    CodeIntelligenceUnavailableError,
    CodeWorkspace,
    CodeWorkspaceNotFoundError,
)
from tesserix_adk.core.models import AdkModel

__all__ = ["GraftMcpBackend", "GraftSubprocessBackend"]


class _McpSession(Protocol):
    async def call_tool(self, name: str, arguments: dict[str, object]) -> object:
        """Call one MCP tool."""


class _McpTextBlock(AdkModel):
    model_config = ConfigDict(from_attributes=True, extra="ignore")

    type: str
    text: str


class _McpResult(AdkModel):
    model_config = ConfigDict(from_attributes=True, extra="ignore")

    content: list[_McpTextBlock]


_MCP_TOOLS = {
    CodeContextOperation.FIND: "graft_find_code",
    CodeContextOperation.FILE_API: "graft_file_api",
    CodeContextOperation.TRACE: "graft_trace_calls",
    CodeContextOperation.FIND_ALL: "graft_find_all",
    CodeContextOperation.REPO_MAP: "graft_repo_map",
    CodeContextOperation.FRESHNESS: "graft_check_freshness",
}


class GraftMcpBackend:
    """A tenant-bound Graft MCP session."""

    def __init__(
        self,
        session: _McpSession,
        *,
        workspace: CodeWorkspace,
        max_output_chars: int = 256_000,
    ) -> None:
        if max_output_chars < 1:
            raise ValueError("max_output_chars must be positive")
        self._session = session
        self._workspace = workspace
        self._max_output = max_output_chars

    @property
    def workspace(self) -> CodeWorkspace:
        """The workspace assigned when the MCP session was authorized."""
        return self._workspace

    async def execute(self, request: CodeContextRequest) -> CodeContextResult:
        """Call the matching Graft tool without placing identity in model arguments."""
        _within_workspace(request, self._workspace)
        try:
            answer = await self._session.call_tool(
                _MCP_TOOLS[request.operation], _mcp_arguments(request)
            )
            content = _mcp_text(answer)
        except CodeIntelligenceError:
            raise
        except Exception as failure:
            raise CodeIntelligenceUnavailableError(
                "the Graft MCP backend is unavailable"
            ) from failure
        _within_output_limit(content, self._max_output)
        return CodeContextResult(
            operation=request.operation,
            content=content,
            backend="graft-mcp",
        )


@dataclass(frozen=True, slots=True)
class _Captured:
    content: bytes
    too_large: bool = False


class GraftSubprocessBackend:
    """A local Graft CLI bound to one canonical checkout."""

    def __init__(
        self,
        command: tuple[str, ...] = ("graft",),
        *,
        workspace: CodeWorkspace,
        timeout_seconds: float = 15.0,
        max_output_chars: int = 256_000,
    ) -> None:
        if not command or not command[0]:
            raise ValueError("the Graft command must name an executable")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if max_output_chars < 1:
            raise ValueError("max_output_chars must be positive")
        root = Path(workspace.root).resolve()
        if not root.is_dir():
            raise ValueError("the Graft workspace root must be an existing directory")
        self._command = command
        self._workspace = workspace.model_copy(update={"root": str(root)})
        self._timeout = timeout_seconds
        self._max_output = max_output_chars

    @property
    def workspace(self) -> CodeWorkspace:
        """The canonical checkout used as the subprocess working directory."""
        return self._workspace

    async def execute(self, request: CodeContextRequest) -> CodeContextResult:
        """Run one structural Graft command with no shell interpolation."""
        _within_workspace(request, self._workspace)
        argv = (*self._command, *_cli_arguments(request))
        try:
            process = await asyncio.create_subprocess_exec(
                *argv,
                cwd=self._workspace.root,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except OSError as failure:
            raise CodeIntelligenceUnavailableError(
                "the configured Graft executable could not be started"
            ) from failure

        try:
            async with asyncio.timeout(self._timeout):
                async with asyncio.TaskGroup() as tasks:
                    stdout = tasks.create_task(
                        _read_bounded(process.stdout, self._max_output, process)
                    )
                    stderr = tasks.create_task(_read_bounded(process.stderr, 16_384, process))
                    tasks.create_task(process.wait())
        except TimeoutError as failure:
            await _stop(process)
            raise CodeIntelligenceUnavailableError(
                "the Graft subprocess exceeded its timeout"
            ) from failure
        except BaseException:
            await _stop(process)
            raise

        captured = stdout.result()
        if captured.too_large or stderr.result().too_large:
            raise CodeIntelligenceError("Graft exceeded the configured output ceiling")
        accepted = request.operation is CodeContextOperation.FRESHNESS and process.returncode == 1
        if process.returncode != 0 and not accepted:
            raise CodeIntelligenceUnavailableError(
                f"the Graft subprocess exited with status {process.returncode}"
            )
        return CodeContextResult(
            operation=request.operation,
            content=captured.content.decode("utf-8", errors="replace").strip(),
            backend="graft-subprocess",
            stale=accepted,
        )


async def _read_bounded(
    stream: asyncio.StreamReader | None,
    limit: int,
    process: asyncio.subprocess.Process,
) -> _Captured:
    if stream is None:
        return _Captured(b"")
    parts: list[bytes] = []
    size = 0
    while chunk := await stream.read(8192):
        size += len(chunk)
        if size > limit:
            if process.returncode is None:
                process.kill()
            return _Captured(b"", too_large=True)
        parts.append(chunk)
    return _Captured(b"".join(parts))


async def _stop(process: asyncio.subprocess.Process) -> None:
    if process.returncode is None:
        process.kill()
    await process.wait()


def _within_workspace(request: CodeContextRequest, workspace: CodeWorkspace) -> None:
    if request.tenant == workspace.tenant and request.workspace == workspace.id:
        return
    raise CodeWorkspaceNotFoundError("code workspace not found")


def _is_mcp_error(result: object) -> bool:
    if isinstance(result, dict):
        return result.get("isError") is True
    return getattr(result, "isError", False) is True


def _mcp_text(result: object) -> str:
    if _is_mcp_error(result):
        raise CodeIntelligenceUnavailableError("Graft returned an MCP error")
    try:
        parsed = _McpResult.model_validate(result, from_attributes=True)
    except ValidationError as failure:
        raise CodeIntelligenceError("Graft returned an invalid MCP result") from failure
    text = [block.text for block in parsed.content if block.type == "text"]
    if len(text) != 1:
        raise CodeIntelligenceError("Graft must return exactly one text result")
    return text[0]


def _within_output_limit(content: str, limit: int) -> None:
    if len(content) > limit:
        raise CodeIntelligenceError("Graft exceeded the configured output ceiling")


def _mcp_arguments(request: CodeContextRequest) -> dict[str, object]:
    args = request.arguments
    match request.operation:
        case CodeContextOperation.FIND:
            result: dict[str, object] = {"query": args.query, "limit": args.limit}
            if args.full:
                result["full"] = True
        case CodeContextOperation.FILE_API:
            result = {"file": args.file}
        case CodeContextOperation.TRACE:
            result = {
                "symbol": args.symbol,
                "direction": args.direction,
                "depth": args.depth,
            }
        case CodeContextOperation.FIND_ALL:
            result = {"pattern": args.pattern}
            if args.ignore_case:
                result["ignore_case"] = True
            if args.fixed:
                result["fixed"] = True
        case CodeContextOperation.REPO_MAP:
            result = {"max_dirs": args.max_dirs}
        case CodeContextOperation.FRESHNESS:
            result = {}
    if args.path_prefix and request.operation in {
        CodeContextOperation.FIND,
        CodeContextOperation.TRACE,
        CodeContextOperation.FIND_ALL,
    }:
        result["in"] = args.path_prefix
    return result


def _cli_arguments(request: CodeContextRequest) -> tuple[str, ...]:
    args = request.arguments
    match request.operation:
        case CodeContextOperation.FIND:
            command = ["ask", args.query, ".", "--source", "-n", str(args.limit)]
            if args.full:
                command.append("--full")
        case CodeContextOperation.FILE_API:
            command = ["skeleton", args.file, "."]
        case CodeContextOperation.TRACE:
            command = [
                "callers",
                args.symbol,
                ".",
                "--direction",
                args.direction,
                "--depth",
                str(args.depth),
            ]
        case CodeContextOperation.FIND_ALL:
            command = ["grep", args.pattern, "."]
            if args.ignore_case:
                command.append("--ignore-case")
            if args.fixed:
                command.append("--fixed")
        case CodeContextOperation.REPO_MAP:
            command = ["map", ".", "--max-dirs", str(args.max_dirs)]
        case CodeContextOperation.FRESHNESS:
            command = ["check", "."]
    if args.path_prefix and request.operation in {
        CodeContextOperation.FIND,
        CodeContextOperation.TRACE,
        CodeContextOperation.FIND_ALL,
    }:
        command.extend(("--in", args.path_prefix))
    return tuple(command)
