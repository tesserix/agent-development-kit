"""Code intelligence stays scoped while local and MCP backends remain interchangeable."""

from __future__ import annotations

import json
import sys
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest

from tesserix_adk.adapters import GraftMcpBackend, GraftSubprocessBackend
from tesserix_adk.code_intelligence import (
    CodeContextArguments,
    CodeContextBackend,
    CodeContextOperation,
    CodeContextRequest,
    CodeContextResult,
    CodeIntelligenceContributor,
    CodeIntelligenceError,
    CodeWorkspace,
    CodeWorkspaceNotFoundError,
)
from tesserix_adk.runtime import ContextRequest
from tesserix_adk.tools import ToolContext, code_intelligence_tools

if TYPE_CHECKING:
    from pathlib import Path

WORKSPACE = CodeWorkspace(id="checkout-1", tenant="acme", root="/work/acme/repo")


class RecordingBackend:
    def __init__(self) -> None:
        self.workspace = WORKSPACE
        self.requests: list[CodeContextRequest] = []

    async def execute(self, request: CodeContextRequest) -> CodeContextResult:
        self.requests.append(request)
        return CodeContextResult(
            operation=request.operation,
            content=f"{request.operation}: precise context",
            backend="recording",
        )


class McpSession:
    def __init__(
        self,
        content: str = "precise MCP context",
        *,
        is_error: bool = False,
    ) -> None:
        self.content = content
        self.is_error = is_error
        self.calls: list[tuple[str, dict[str, object]]] = []

    async def call_tool(self, name: str, arguments: dict[str, object]) -> object:
        self.calls.append((name, arguments))
        block = SimpleNamespace(type="text", text=self.content)
        return SimpleNamespace(content=[block], isError=self.is_error)


def request(
    operation: CodeContextOperation,
    arguments: CodeContextArguments | None = None,
) -> CodeContextRequest:
    return CodeContextRequest(
        tenant=WORKSPACE.tenant,
        workspace=WORKSPACE.id,
        operation=operation,
        arguments=arguments or CodeContextArguments(),
    )


def test_code_paths_are_relative_to_the_bound_workspace() -> None:
    with pytest.raises(ValueError, match="relative"):
        CodeContextArguments(file="/etc/passwd")

    with pytest.raises(ValueError, match="relative"):
        CodeContextArguments(file=r"C:\other-tenant\source.py")

    with pytest.raises(ValueError, match="parent"):
        CodeContextArguments(file="../../other-tenant/source.py")


def test_each_operation_requires_its_own_argument() -> None:
    with pytest.raises(ValueError, match="query"):
        request(CodeContextOperation.FIND)

    with pytest.raises(ValueError, match="file"):
        request(CodeContextOperation.FILE_API)


async def test_backend_protocol_is_structural() -> None:
    backend = RecordingBackend()

    assert isinstance(backend, CodeContextBackend)


async def test_code_context_contributor_builds_a_deduplicated_pointer_pack() -> None:
    backend = RecordingBackend()
    contributor = CodeIntelligenceContributor(backend, limit=3)

    contribution = await contributor.contribute(
        ContextRequest(
            run_id="run-1",
            tenant="acme",
            agent_name="developer",
            query="fix authorization",
        )
    )

    assert contributor.name == "code-intelligence"
    assert contributor.required is False
    assert contribution.content == ("find: precise context",)
    assert len(contribution.keys) == 1
    assert backend.requests == [
        request(
            CodeContextOperation.FIND,
            CodeContextArguments(query="fix authorization", limit=3),
        )
    ]


async def test_tools_translate_model_arguments_without_exposing_identity() -> None:
    backend = RecordingBackend()
    tools = code_intelligence_tools(backend)
    by_name = {declared.name: declared for declared in tools}
    try:
        result = await by_name["code_find"].invoke(
            {"query": "where is authorization checked?", "limit": 3},
            context=ToolContext(run_id="run-1", tenant="acme"),
        )
    finally:
        for declared in tools:
            declared.release()

    assert result == "find: precise context"
    assert backend.requests == [
        request(
            CodeContextOperation.FIND,
            CodeContextArguments(query="where is authorization checked?", limit=3),
        )
    ]
    assert "tenant" not in by_name["code_find"].parameters_schema["properties"]
    assert "workspace" not in by_name["code_find"].parameters_schema["properties"]


async def test_tools_hide_a_workspace_from_another_tenant() -> None:
    backend = RecordingBackend()
    tools = code_intelligence_tools(backend)
    by_name = {declared.name: declared for declared in tools}
    try:
        with pytest.raises(CodeWorkspaceNotFoundError, match="not found"):
            await by_name["code_find"].invoke(
                {"query": "secrets"},
                context=ToolContext(run_id="run-2", tenant="globex"),
            )
    finally:
        for declared in tools:
            declared.release()

    assert backend.requests == []


async def test_mcp_backend_maps_to_graft_without_sending_workspace_identity() -> None:
    session = McpSession()
    backend = GraftMcpBackend(session, workspace=WORKSPACE)

    result = await backend.execute(
        request(
            CodeContextOperation.TRACE,
            CodeContextArguments(
                symbol="Authorizer.check",
                direction="out",
                depth=2,
                path_prefix="src/auth",
            ),
        )
    )

    assert result.content == "precise MCP context"
    assert session.calls == [
        (
            "graft_trace_calls",
            {
                "symbol": "Authorizer.check",
                "direction": "out",
                "depth": 2,
                "in": "src/auth",
            },
        )
    ]


async def test_mcp_errors_and_oversized_results_fail_at_the_boundary() -> None:
    session = McpSession("x" * 65)
    backend = GraftMcpBackend(session, workspace=WORKSPACE, max_output_chars=64)

    with pytest.raises(CodeIntelligenceError, match="output ceiling"):
        await backend.execute(
            request(CodeContextOperation.FIND, CodeContextArguments(query="auth"))
        )

    failed = GraftMcpBackend(McpSession(is_error=True), workspace=WORKSPACE)
    with pytest.raises(CodeIntelligenceError, match="MCP error"):
        await failed.execute(request(CodeContextOperation.FIND, CodeContextArguments(query="auth")))


async def test_graft_backend_hides_another_workspace_before_transport() -> None:
    session = McpSession()
    backend = GraftMcpBackend(session, workspace=WORKSPACE)

    with pytest.raises(CodeWorkspaceNotFoundError, match="not found"):
        await backend.execute(
            CodeContextRequest(
                tenant="acme",
                workspace="another-checkout",
                operation=CodeContextOperation.REPO_MAP,
            )
        )

    assert session.calls == []


async def test_subprocess_backend_passes_queries_as_one_argv_value(tmp_path: Path) -> None:
    script = tmp_path / "graft_stub.py"
    script.write_text("import json, sys\nprint(json.dumps(sys.argv[1:]))\n")
    marker = tmp_path / "must-not-exist"
    query = f"auth $(touch {marker})"
    workspace = CodeWorkspace(id="local", tenant="acme", root=str(tmp_path))
    backend = GraftSubprocessBackend((sys.executable, str(script)), workspace=workspace)

    result = await backend.execute(
        CodeContextRequest(
            tenant="acme",
            workspace="local",
            operation=CodeContextOperation.FIND,
            arguments=CodeContextArguments(query=query, limit=4, full=True),
        )
    )

    assert json.loads(result.content) == ["ask", query, ".", "--source", "-n", "4", "--full"]
    assert not marker.exists()


async def test_subprocess_backend_bounds_output_before_returning_it(tmp_path: Path) -> None:
    script = tmp_path / "graft_stub.py"
    script.write_text("print('x' * 65)\n")
    workspace = CodeWorkspace(id="local", tenant="acme", root=str(tmp_path))
    backend = GraftSubprocessBackend(
        (sys.executable, str(script)), workspace=workspace, max_output_chars=64
    )

    with pytest.raises(CodeIntelligenceError, match="output ceiling"):
        await backend.execute(
            CodeContextRequest(
                tenant="acme",
                workspace="local",
                operation=CodeContextOperation.REPO_MAP,
            )
        )
