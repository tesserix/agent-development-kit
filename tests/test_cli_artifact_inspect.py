"""Recorded failures are inspectable, comparable and replayable without a network."""

from __future__ import annotations

import io
import json
from typing import TYPE_CHECKING, Any

from tesserix_adk import Agent, ToolRegistry, tool
from tesserix_adk.cli.artifact_inspect import DIVERGED, UNREADABLE
from tesserix_adk.cli.artifact_inspect import main as inspect_main
from tesserix_adk.cli.artifacts import ArtifactHeader, ArtifactWriter
from tesserix_adk.cli.run_agent import LocalAgent
from tesserix_adk.cli.run_agent import main as run_main
from tesserix_adk.core import (
    ApprovalGate,
    BudgetLimits,
    Run,
    RunState,
    ToolCall,
)
from tesserix_adk.runtime import ToolCallStarted
from tesserix_adk.testing import FakeGuardrail, FakeModelProvider, ScriptedTurn

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


@tool(idempotency="read_only")
def inspect_lookup(query: str) -> str:
    """Return deterministic context.

    Args:
        query: The phrase to find.
    """
    return f"found {query}"


def inspected_agent() -> Agent[Any]:
    """A guarded declaration that exhausts its budget before a second model call."""
    return Agent(
        name="inspected-agent",
        instructions="Use inspect_lookup and answer.",
        model="fake",
        free_text=True,
        tools=("inspect_lookup",),
        idempotent_tools=("inspect_lookup",),
        guardrails=("safe",),
        budget=BudgetLimits(max_model_calls=1, max_tool_calls=1),
    )


def target(
    provider: FakeModelProvider,
) -> Callable[[str, ApprovalGate], LocalAgent]:
    """Resolve the same current declaration for recording and replay."""

    def resolve(reference: str, approvals: ApprovalGate) -> LocalAgent:
        assert reference == "demo:inspected"
        assert approvals is not None
        return LocalAgent(
            agent=inspected_agent(),
            provider=provider,
            tools=ToolRegistry((inspect_lookup,)),
            guardrails={"safe": FakeGuardrail("safe")},
        )

    return resolve


async def recorded_failure(path: Path) -> Callable[[str, ApprovalGate], LocalAgent]:
    """Write one cassette-backed failure and return current target wiring."""
    provider = FakeModelProvider(
        ScriptedTurn.calling("inspect_lookup", {"query": "fare"}, input_tokens=3)
    )
    resolve = target(provider)
    assert (
        await run_main(
            ["demo:inspected", "--input", "find fare", "--record", str(path)],
            resolve=resolve,
            out=io.StringIO(),
            stdin=io.StringIO(),
        )
        == 1
    )
    return target(FakeModelProvider())


async def test_errors_only_keeps_failure_context_and_replays_the_terminal_state(
    tmp_path: Path,
) -> None:
    artefact = tmp_path / "failed.jsonl"
    resolve = await recorded_failure(artefact)
    output = io.StringIO()

    code = await inspect_main(
        [str(artefact), "--errors-only", "--replay"],
        resolve=resolve,
        out=output,
    )

    assert code == 0
    rendered = output.getvalue()
    assert "guardrail_decision" in rendered
    assert "tool_call_started" in rendered
    assert 'arguments={"query": "fare"}' in rendered
    assert "run_failed" in rendered
    assert "terminal state=budget_exhausted" in rendered
    assert "replay terminal=budget_exhausted" in rendered
    assert "matched=yes" in rendered


async def test_a_truncated_file_is_rejected_before_any_partial_event_is_rendered(
    tmp_path: Path,
) -> None:
    artefact = tmp_path / "partial.jsonl"
    writer = ArtifactWriter(artefact, header())
    writer.append(
        ToolCallStarted(
            run_id="run-1",
            sequence=0,
            call_id="call-1",
            tool="inspect_lookup",
            arguments='{"query":"partial"}',
        )
    )
    writer.close()
    output = io.StringIO()

    assert await inspect_main([str(artefact)], out=output) == UNREADABLE

    assert "truncated" in output.getvalue()
    assert "tool_call_started" not in output.getvalue()
    assert 'query":"partial' not in output.getvalue()


async def test_diff_highlights_the_first_tool_sequence_divergence(tmp_path: Path) -> None:
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    write_complete(first, tool="search")
    write_complete(second, tool="refund")
    output = io.StringIO()

    code = await inspect_main([str(first), "--diff", str(second)], out=output)

    assert code == DIVERGED
    assert "first divergence: tool_calls[0]" in output.getvalue()
    assert "search" in output.getvalue()
    assert "refund" in output.getvalue()


async def test_export_writes_one_self_contained_replay_backed_eval_case(tmp_path: Path) -> None:
    artefact = tmp_path / "failed.jsonl"
    await recorded_failure(artefact)
    exported = tmp_path / "regression.json"
    output = io.StringIO()

    assert (
        await inspect_main(
            [str(artefact), "--export-eval-case", str(exported)],
            out=output,
        )
        == 0
    )

    document = json.loads(exported.read_text(encoding="utf-8"))
    assert document["format"] == "tesserix-adk-eval-case"
    assert document["expected_state"] == "budget_exhausted"
    assert len(document["cassette"]["interactions"]) == 1


def header() -> ArtifactHeader:
    """One stable header for manually assembled comparison artefacts."""
    return ArtifactHeader(
        version=1,
        kit_version="0.52.0",
        target="demo:inspected",
        input="same input",
        tenant="local-dev",
        user="local-user",
        agent="inspected-agent",
    )


def write_complete(path: Path, *, tool: str) -> None:
    """Write a committed artefact whose first tool differs."""
    writer = ArtifactWriter(path, header())
    writer.append(
        ToolCallStarted(
            run_id="run-1",
            sequence=0,
            call_id="call-1",
            tool=tool,
            arguments="{}",
        )
    )
    writer.finish(
        Run(
            id="run-1",
            tenant="local-dev",
            agent_name="inspected-agent",
            agent_version="1.0.0",
            model="fake",
            state=RunState.FAILED,
            tool_calls=[ToolCall(id="call-1", name=tool, arguments={})],
        )
    )
