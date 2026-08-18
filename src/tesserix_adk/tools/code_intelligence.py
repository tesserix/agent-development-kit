"""Typed agent tools backed by one authorized code workspace."""

from __future__ import annotations

from typing import Literal

from tesserix_adk.code_intelligence import (
    CodeContextArguments,
    CodeContextBackend,
    CodeContextOperation,
    CodeContextRequest,
    CodeWorkspaceNotFoundError,
)
from tesserix_adk.tools.context import ToolContext  # noqa: TC001  # resolved at registration
from tesserix_adk.tools.decorator import Tool, tool

__all__ = ["code_intelligence_tools"]


def code_intelligence_tools(backend: CodeContextBackend) -> tuple[Tool[..., str], ...]:
    """Build the six read-only tools for one workspace-bound backend."""

    async def execute(
        operation: CodeContextOperation,
        arguments: CodeContextArguments,
        context: ToolContext,
    ) -> str:
        workspace = backend.workspace
        if context.tenant != workspace.tenant:
            raise CodeWorkspaceNotFoundError("code workspace not found")
        result = await backend.execute(
            CodeContextRequest(
                tenant=context.tenant,
                workspace=workspace.id,
                operation=operation,
                arguments=arguments,
            )
        )
        return result.content

    @tool(name="code_find", timeout=15.0, idempotency="read_only")
    async def code_find(
        query: str,
        context: ToolContext,
        limit: int = 5,
        full: bool = False,
        path_prefix: str = "",
    ) -> str:
        """Find relevant code with exact locations and source excerpts.

        Args:
            query: What to understand or change.
            context: The authorized run, supplied by the runtime.
            limit: Maximum number of ranked results, from one to twenty.
            full: Return full definitions instead of compact excerpts.
            path_prefix: Optional workspace-relative area to search.
        """
        return await execute(
            CodeContextOperation.FIND,
            CodeContextArguments(
                query=query,
                limit=limit,
                full=full,
                path_prefix=path_prefix,
            ),
            context,
        )

    @tool(name="code_file_api", timeout=15.0, idempotency="read_only")
    async def code_file_api(file: str, context: ToolContext) -> str:
        """Return every signature and span in one source file without its bodies.

        Args:
            file: Workspace-relative source file.
            context: The authorized run, supplied by the runtime.
        """
        return await execute(
            CodeContextOperation.FILE_API, CodeContextArguments(file=file), context
        )

    @tool(name="code_trace", timeout=15.0, idempotency="read_only")
    async def code_trace(
        symbol: str,
        context: ToolContext,
        direction: Literal["in", "out"] = "in",
        depth: int | Literal["all"] = 1,
        path_prefix: str = "",
    ) -> str:
        """Trace callers, dependencies, and a symbol's transitive blast radius.

        Args:
            symbol: Symbol name, qualified name, or source file.
            context: The authorized run, supplied by the runtime.
            direction: In for callers and dependents; out for dependencies.
            depth: Traversal depth or all for the connected closure.
            path_prefix: Optional workspace-relative area to search.
        """
        return await execute(
            CodeContextOperation.TRACE,
            CodeContextArguments(
                symbol=symbol,
                direction=direction,
                depth=depth,
                path_prefix=path_prefix,
            ),
            context,
        )

    @tool(name="code_find_all", timeout=15.0, idempotency="read_only")
    async def code_find_all(
        pattern: str,
        context: ToolContext,
        path_prefix: str = "",
        ignore_case: bool = False,
        fixed: bool = False,
    ) -> str:
        """Search every indexed file and group matches by enclosing symbol.

        Args:
            pattern: Regular expression, or literal text when fixed is true.
            context: The authorized run, supplied by the runtime.
            path_prefix: Optional workspace-relative area to search.
            ignore_case: Match without case sensitivity.
            fixed: Treat the pattern as literal text.
        """
        return await execute(
            CodeContextOperation.FIND_ALL,
            CodeContextArguments(
                pattern=pattern,
                path_prefix=path_prefix,
                ignore_case=ignore_case,
                fixed=fixed,
            ),
            context,
        )

    @tool(name="code_repo_map", timeout=15.0, idempotency="read_only")
    async def code_repo_map(context: ToolContext, max_dirs: int = 16) -> str:
        """Orient an agent with directory clusters, hubs, and hotspots.

        Args:
            context: The authorized run, supplied by the runtime.
            max_dirs: Maximum directory clusters to return.
        """
        return await execute(
            CodeContextOperation.REPO_MAP,
            CodeContextArguments(max_dirs=max_dirs),
            context,
        )

    @tool(name="code_check_freshness", timeout=15.0, idempotency="read_only")
    async def code_check_freshness(context: ToolContext) -> str:
        """Report whether the structural graph matches the current working tree.

        Args:
            context: The authorized run, supplied by the runtime.
        """
        return await execute(
            CodeContextOperation.FRESHNESS,
            CodeContextArguments(),
            context,
        )

    return (
        code_find,
        code_file_api,
        code_trace,
        code_find_all,
        code_repo_map,
        code_check_freshness,
    )
