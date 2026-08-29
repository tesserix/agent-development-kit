"""The complete reference agents demonstrate production composition offline."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

from tests.ci_config import load_yaml, triggers

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"
GUIDE = ROOT / "docs" / "reference-agents.md"
WORKFLOW = ROOT / ".github" / "workflows" / "reference-agents.yml"


def _run(name: str) -> str:
    result = subprocess.run(  # noqa: S603
        [sys.executable, str(EXAMPLES / name)],
        capture_output=True,
        text=True,
        check=False,
        cwd=ROOT,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout


def test_research_agent_treats_hostile_retrieval_as_data_and_cites_evidence() -> None:
    output = _run("reference_research_agent.py")
    assert "hostile retrieval stayed untrusted: True" in output
    assert "grounded citation: handbook@v3" in output
    assert "tenant: acme" in output
    assert "eval exit: 0" in output
    assert "telemetry redacted: True" in output


def test_booking_agent_suspends_before_an_idempotent_approved_transaction() -> None:
    output = _run("reference_booking_agent.py")
    assert "before approval: suspended, bookings=0" in output
    assert "after approval: completed, bookings=1" in output
    assert "idempotency key:" in output
    assert "replayed approval refused; bookings=1" in output
    assert "eval exit: 0" in output


def test_durable_agent_keeps_partial_state_checkpoint_and_cancellation_inspectable() -> None:
    output = _run("reference_durable_agent.py")
    assert "ProviderUnavailableError: attempts=3" in output
    assert "partial journal: 2 completed activities" in output
    assert "checkpoint: run=trip-42 tenant=acme iteration=1" in output
    assert "resumed answer: Rebooked on the 18:40." in output
    assert "cancelled checkpoint remains inspectable: True" in output
    assert "eval exit: 0" in output


def test_reference_agent_guide_covers_deployment_real_provider_and_rollback() -> None:
    guide = GUIDE.read_text(encoding="utf-8")
    assert "reference_research_agent.py" in guide
    assert "reference_booking_agent.py" in guide
    assert "reference_durable_agent.py" in guide
    assert "manual-reference-agents" in guide
    assert "cost ceiling" in guide.lower()
    assert "cluster chart" in guide.lower()
    assert "rollback" in guide.lower()


def test_real_provider_reference_run_is_manual_protected_and_cost_bounded() -> None:
    workflow = load_yaml(WORKFLOW)
    declared = triggers(WORKFLOW)
    assert set(declared) == {"workflow_dispatch"}
    assert "inputs" in declared["workflow_dispatch"]
    assert workflow["permissions"] == {"contents": "read"}

    jobs: dict[str, Any] = workflow["jobs"]
    live = jobs["live"]
    assert live["environment"] == "reference-agents-live"
    assert "confirm_live" in live["if"]
    command = next(step["run"] for step in live["steps"] if "--live" in step.get("run", ""))
    assert "--max-input-tokens" in command
    assert "--max-output-tokens" in command
    assert live["env"] == {"ADK_PROVIDER_API_KEY": "${{ secrets.REFERENCE_PROVIDER_API_KEY }}"}
