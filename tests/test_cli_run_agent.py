"""The local runner is scoped, streamed, redacted, cancellable and recordable."""

from __future__ import annotations

import io
import json
from decimal import Decimal
from typing import TYPE_CHECKING, Any

from tesserix_adk import Agent, ToolRegistry, tool
from tesserix_adk.cli.artifacts import scan_artifact
from tesserix_adk.cli.run_agent import LocalAgent, main
from tesserix_adk.core import ApprovalGate, BudgetLimits, Cost
from tesserix_adk.testing import FakeModelProvider, Fault, ScriptedTurn

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


@tool(idempotency="read_only")
def local_lookup(query: str) -> str:
    """Return local test context.

    Args:
        query: The phrase to look up.
    """
    return f"context for {query}"


def declaration(*, budget: BudgetLimits | None = None, approval: bool = False) -> Agent[Any]:
    """One free-text tool-using local declaration."""
    return Agent(
        name="local-agent",
        instructions="Use local_lookup and answer.",
        model="fake",
        free_text=True,
        tools=("local_lookup",),
        idempotent_tools=("local_lookup",),
        approval_required_tools=("local_lookup",) if approval else (),
        budget=budget,
    )


def resolver(
    provider: FakeModelProvider, *, agent: Agent[Any] | None = None
) -> Callable[[str, ApprovalGate], LocalAgent]:
    """Return injectable target wiring and retain the CLI gate boundary."""

    def resolve(reference: str, approvals: ApprovalGate) -> LocalAgent:
        assert reference == "demo:local"
        assert approvals is not None
        return LocalAgent(
            agent=agent or declaration(),
            provider=provider,
            tools=ToolRegistry((local_lookup,)),
        )

    return resolve


async def test_a_budget_failure_is_streamed_and_committed_with_usage_and_ceiling(
    tmp_path: Path,
) -> None:
    artefact = tmp_path / "budget.jsonl"
    provider = FakeModelProvider(
        ScriptedTurn.calling(
            "local_lookup",
            {"query": "fares"},
            input_tokens=4,
            output_tokens=2,
            cost=Cost(output=Decimal("0.01")),
        )
    )
    agent = declaration(
        budget=BudgetLimits(
            max_model_calls=1,
            max_tool_calls=1,
            max_input_tokens=10,
            max_output_tokens=10,
        )
    )
    output = io.StringIO()

    code = await main(
        ["demo:local", "--input", "find fares", "--record", str(artefact)],
        resolve=resolver(provider, agent=agent),
        out=output,
        stdin=io.StringIO(),
    )

    assert code == 1
    assert "budget" in output.getvalue().lower()
    assert "running_cost=USD 0.01" in output.getvalue()
    summary = scan_artifact(artefact)
    assert summary.run["state"] == "budget_exhausted"
    usage = summary.run["usage"]
    assert isinstance(usage, dict)
    assert usage["input_tokens"] == 4
    budget = summary.run["budget"]
    assert isinstance(budget, dict)
    limits = budget["limits"]
    assert isinstance(limits, dict)
    assert limits["max_model_calls"] == 1
    assert summary.cassette is not None
    assert len(summary.cassette.interactions) == 1


async def test_json_output_and_artifact_apply_the_same_redaction_path(tmp_path: Path) -> None:
    artefact = tmp_path / "private.jsonl"
    provider = FakeModelProvider(
        ScriptedTurn.saying(
            "email developer@example.com and use sk-live-0123456789",
            input_tokens=3,
            output_tokens=4,
        )
    )
    output = io.StringIO()

    code = await main(
        [
            "demo:local",
            "--input",
            "Bearer opaque-secret",
            "--json",
            "--record",
            str(artefact),
        ],
        resolve=resolver(provider),
        out=output,
        stdin=io.StringIO(),
    )

    assert code == 0
    lines = [json.loads(line) for line in output.getvalue().splitlines()]
    assert lines[0] == {
        "default": True,
        "kind": "local_scope",
        "tenant": "local-dev",
        "user": "local-user",
    }
    rendered = output.getvalue()
    recorded = artefact.read_text(encoding="utf-8")
    for secret in (
        "developer@example.com",
        "sk-live-0123456789",
        "opaque-secret",
    ):
        assert secret not in rendered
        assert secret not in recorded


async def test_no_interactive_denies_an_approval_before_the_tool_runs() -> None:
    provider = FakeModelProvider(ScriptedTurn.calling("local_lookup", {"query": "private"}))
    output = io.StringIO()

    code = await main(
        ["demo:local", "--input", "look it up", "--no-interactive"],
        resolve=resolver(provider, agent=declaration(approval=True)),
        out=output,
        stdin=io.StringIO("yes\n"),
    )

    assert code == 1
    assert "approval" in output.getvalue().lower()
    assert "failed" in output.getvalue().lower()


async def test_interactive_approval_allows_only_the_exact_held_call() -> None:
    provider = FakeModelProvider(
        ScriptedTurn.calling("local_lookup", {"query": "public"}),
        ScriptedTurn.saying("done"),
    )
    output = io.StringIO()

    code = await main(
        ["demo:local", "--input", "look it up"],
        resolve=resolver(provider, agent=declaration(approval=True)),
        out=output,
        stdin=io.StringIO("yes\n"),
    )

    assert code == 0
    assert "approval required" in output.getvalue()
    assert "approval_granted" in output.getvalue()


async def test_a_provider_transport_failure_has_no_reply_and_points_to_doctor() -> None:
    provider = FakeModelProvider(
        ScriptedTurn.failing(
            Fault.TRANSPORT,
            payload="endpoint https://unreachable.invalid elapsed timeout 30s",
        )
    )
    output = io.StringIO()

    code = await main(
        ["demo:local", "--input", "hello", "--tenant", "acme", "--user", "ada"],
        resolve=resolver(provider),
        out=output,
        stdin=io.StringIO(),
    )

    assert code == 1
    assert "provider" in output.getvalue().lower()
    assert "tesserix-adk doctor" in output.getvalue()
    assert "answer_delta" not in output.getvalue()
    assert "local development default" not in output.getvalue()
