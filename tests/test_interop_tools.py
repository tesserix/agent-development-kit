"""Foreign tools cross one validated, policy-bearing and attributable boundary."""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from tesserix_adk.adapters.interop import (
    ForeignTool,
    ToolImportPolicy,
    ToolTranslationError,
    import_tool,
    import_toolset,
)
from tesserix_adk.core import Idempotency, ToolArgumentValidationError
from tesserix_adk.testing import scoped_run
from tesserix_adk.tools import ToolCallSpan, ToolContext, ToolRegistry

READ_ONLY = ToolImportPolicy(
    timeout_seconds=2,
    max_concurrency=2,
    requires_approval=False,
    idempotency=Idempotency.READ_ONLY,
)


def descriptor(name: str = "foreign_search") -> dict[str, object]:
    """An OpenAI-compatible function descriptor."""
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": "Search a foreign index.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    }


class SearchArguments(BaseModel):
    """Arguments declared by a class-based foreign framework tool."""

    query: str


class FrameworkSearch:
    """A class-based tool using the common ``ainvoke(input, config)`` shape."""

    name = "framework_search"
    description = "Search through another framework."
    args_schema = SearchArguments

    def __init__(self) -> None:
        self.calls: list[tuple[dict[str, object], dict[str, object]]] = []

    async def ainvoke(
        self, payload: dict[str, object], *, config: dict[str, object]
    ) -> dict[str, str]:
        self.calls.append((payload, config))
        return {"answer": str(payload["query"])}


async def search(*, query: str, context: ToolContext) -> dict[str, str]:
    """A foreign implementation that receives delegated caller context explicitly."""
    return {"answer": query, "tenant": context.tenant}


async def test_an_openai_descriptor_becomes_a_validated_kit_tool() -> None:
    imported = import_tool(descriptor(), implementation=search, policy=READ_ONLY)

    result = await imported.invoke(
        {"query": "Kyoto"}, ToolContext(run_id="run-1", tenant="acme", user="ada")
    )

    assert result == {"answer": "Kyoto", "tenant": "acme"}
    assert imported.parameters_schema["additionalProperties"] is False
    assert imported.timeout == 2
    assert imported.max_concurrency == 2
    assert imported.origin == "openai-function:foreign_search"


async def test_arguments_are_refused_before_the_foreign_body() -> None:
    calls = 0

    async def counted(*, query: str, context: ToolContext) -> str:
        nonlocal calls
        calls += 1
        return f"{context.tenant}:{query}"

    imported = import_tool(descriptor(), implementation=counted, policy=READ_ONLY)

    with pytest.raises(ToolArgumentValidationError):
        await imported.invoke({"query": 7}, ToolContext(run_id="run-1", tenant="acme"))

    assert calls == 0


async def test_a_class_tool_gets_tenant_user_scopes_and_trace_in_config() -> None:
    source = FrameworkSearch()
    imported = import_tool(source, policy=READ_ONLY)
    context = ToolContext(
        run_id="run-1",
        tenant="acme",
        user="ada",
        scopes=("search:read",),
        trace={"traceparent": "00-abc-def-01"},
    )

    assert await imported.invoke({"query": "Kyoto"}, context) == {"answer": "Kyoto"}
    _, config = source.calls[0]
    assert config["metadata"] == {
        "run_id": "run-1",
        "tenant": "acme",
        "user": "ada",
        "scopes": ("search:read",),
        "trace": {"traceparent": "00-abc-def-01"},
    }


def test_a_foreign_tool_without_a_schema_is_refused_at_import() -> None:
    async def untyped(query, context):  # type: ignore[no-untyped-def]
        return f"{context.tenant}:{query}"

    with pytest.raises(ToolTranslationError, match="schema") as raised:
        import_tool(untyped, policy=READ_ONLY)

    assert raised.value.construct == "schema"


def test_policy_must_state_what_repeating_the_foreign_call_does() -> None:
    undeclared = ToolImportPolicy(timeout_seconds=2)

    with pytest.raises(ToolTranslationError, match="idempotency"):
        import_tool(descriptor(), implementation=search, policy=undeclared)


def test_an_unsupported_construct_is_named_and_other_tools_are_retained() -> None:
    unsupported = descriptor("unsafe_schema")
    function = unsupported["function"]
    assert isinstance(function, dict)
    parameters = function["parameters"]
    assert isinstance(parameters, dict)
    parameters["patternProperties"] = {".*": {"type": "string"}}

    entries = (
        ForeignTool(descriptor(), implementation=search),
        ForeignTool(descriptor("second"), implementation=search),
        ForeignTool(unsupported, implementation=search),
    )
    with pytest.raises(ToolTranslationError) as raised:
        import_toolset(entries, policy=READ_ONLY)

    assert [tool.name for tool in raised.value.translated] == ["foreign_search", "second"]
    assert raised.value.failures[0].tool == "unsafe_schema"
    assert raised.value.failures[0].construct == "patternProperties"


async def test_registry_spans_retain_import_provenance() -> None:
    imported = import_tool(descriptor(), implementation=search, policy=READ_ONLY)
    registry = ToolRegistry((imported,))
    spans: list[ToolCallSpan] = []
    registry.observe(spans.append)

    await imported.invoke({"query": "Kyoto"}, ToolContext(run_id="run-1", tenant="acme"))
    with scoped_run(tenant="acme"):
        await registry.view(allow=(imported.name,), agent="planner").invoke(
            imported.name, {"query": "Osaka"}
        )

    assert spans[-1].origin == "openai-function:foreign_search"


def test_a_name_collision_is_refused_before_registration_order_can_decide() -> None:
    entries = (
        ForeignTool(descriptor("same"), implementation=search),
        ForeignTool(descriptor("same"), implementation=search),
    )

    with pytest.raises(ToolTranslationError, match="same") as raised:
        import_toolset(entries, policy=READ_ONLY)

    assert raised.value.construct == "name"
