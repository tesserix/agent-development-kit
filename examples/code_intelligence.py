"""Push compact code context, then let an agent pull a call trace on demand.

The backend is in-process so the example needs no Graft installation or credentials. A
deployment swaps it for `GraftSubprocessBackend` or `GraftMcpBackend` without changing the
agent, tools, or contributor. Run it with `python examples/code_intelligence.py`.
"""

from __future__ import annotations

import asyncio

from tesserix_adk.code_intelligence import (
    CodeContextOperation,
    CodeContextRequest,
    CodeContextResult,
    CodeIntelligenceContributor,
    CodeWorkspace,
)
from tesserix_adk.core import Agent, RunEventKind, TextPart, ToolCall, Usage
from tesserix_adk.runtime import AgentRunner, ModelResponse
from tesserix_adk.testing import ScriptedProvider
from tesserix_adk.tools import ToolRegistry, code_intelligence_tools


class DemoCodeBackend:
    """One tenant-bound checkout, shaped like either production adapter."""

    workspace = CodeWorkspace(
        id="payments-main",
        tenant="acme",
        root="/srv/checkouts/acme/payments",
    )

    def __init__(self) -> None:
        self.operations: list[CodeContextOperation] = []

    async def execute(self, request: CodeContextRequest) -> CodeContextResult:
        """Return a compact answer while recording which surface requested it."""
        self.operations.append(request.operation)
        content = {
            CodeContextOperation.FIND: "Authorizer.verify — src/auth.py:40-61",
            CodeContextOperation.TRACE: "PaymentHandler -> Authorizer.verify",
        }.get(request.operation, "structural code context")
        return CodeContextResult(
            operation=request.operation,
            content=content,
            backend="demo",
        )


async def main() -> None:
    """Run the automatic push and one model-selected pull query."""
    backend = DemoCodeBackend()

    registry = ToolRegistry(code_intelligence_tools(backend))
    tools = registry.view(allow=registry.names, agent="developer")
    provider = ScriptedProvider(
        ModelResponse(
            tool_calls=(
                ToolCall(
                    id="trace-1",
                    name="code_trace",
                    arguments={"symbol": "Authorizer.verify", "depth": 2},
                ),
            ),
            usage=Usage(input_tokens=40, output_tokens=8),
        ),
        ModelResponse(
            content="The payment handler is in the authorization blast radius.",
            usage=Usage(input_tokens=70, output_tokens=12),
        ),
    )
    agent = Agent(
        name="developer",
        instructions="Trace affected code before proposing a change.",
        model="scripted",
        tools=tools.names,
        free_text=True,
    )
    run = await AgentRunner(
        provider=provider,
        tools=tools,
        context_contributors=(CodeIntelligenceContributor(backend),),
    ).run(agent, "Fix authorization caching", tenant="acme")
    answer = next(
        part.text
        for message in reversed(run.messages)
        if message.role == "assistant"
        for part in message.content
        if isinstance(part, TextPart)
    )

    print("operations:", [operation.value for operation in backend.operations])  # noqa: T201
    print("retrieved:", any(e.kind is RunEventKind.CONTEXT_RETRIEVED for e in run.events))  # noqa: T201
    print("answer:", answer)  # noqa: T201


if __name__ == "__main__":
    asyncio.run(main())
