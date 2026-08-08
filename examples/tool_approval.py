"""A tool that moves money declaring it, and the grant covering exactly what was shown.

Four scenarios: what holds a call and what does not; what the approver is shown; what a grant
will not stretch to cover; and what a denial leaves the agent with. Run it with
`python examples/tool_approval.py`.
"""

from __future__ import annotations

import asyncio

from tesserix_adk.core import (
    Agent,
    ApprovalBindingError,
    ApprovalDecision,
    ApprovalRecord,
    ModelCapabilities,
    RunEventKind,
    ToolCall,
    Usage,
)
from tesserix_adk.runtime import AgentRunner, ApprovalLedger, ModelResponse
from tesserix_adk.testing import FakeClock, ScriptedProvider
from tesserix_adk.tools import ToolRegistry, tool

APPROVED = {"amount": 500, "iban": "GB33BUKB20201555555555", "urgent": True}


@tool(requires_approval=True)
async def wire_funds(amount: int, iban: str, urgent: bool = False) -> str:
    """Send a payment.

    Args:
        amount: Minor units.
        iban: Where the money goes.
        urgent: Whether to pay the same day.
    """
    return f"sent {amount} to {iban}{' today' if urgent else ''}"


@tool(requires_approval=lambda arguments: arguments["amount"] > 100)
async def issue_refund(amount: int) -> str:
    """Refund a booking.

    Args:
        amount: Minor units.
    """
    return f"refunded {amount}"


class Desk:
    """An approval backend that declines anything it is asked."""

    async def request(self, record: ApprovalRecord) -> ApprovalDecision:
        """Decline, with a reason the agent can work with."""
        return ApprovalDecision(
            record_id=record.id,
            granted=False,
            decided_by="ada",
            decided_at=0.0,
            reason="the amount is above the desk limit",
        )


def declaring() -> None:
    """What a declaration holds, and what it lets through."""
    print("wire_funds always:", wire_funds.requires_approval({"amount": 1, "iban": "GB33"}))  # noqa: T201
    print("refund over the desk limit:", issue_refund.requires_approval({"amount": 500}))  # noqa: T201
    print("refund under it:", issue_refund.requires_approval({"amount": 5}))  # noqa: T201
    print("arguments it cannot read:", issue_refund.requires_approval({"amount": "lots"}))  # noqa: T201


def raised() -> ApprovalRecord:
    """The record a gate would be shown for one wire."""
    return ApprovalRecord.for_call(
        run_id="run_1",
        tenant="acme",
        agent_name="planner",
        tool_name="wire_funds",
        arguments=APPROVED,
        reason="wire_funds is declared to require approval",
    )


def bound_to_the_payload() -> None:
    """What one grant covers, and what it will not stretch to."""
    ledger, record = ApprovalLedger(), raised()
    print("summary:", record.summary)  # noqa: T201
    ledger.bind(record)
    try:
        ledger.spend(record, {**APPROVED, "amount": 5000})
    except ApprovalBindingError as unbound:
        print("altered arguments:", unbound)  # noqa: T201
    ledger.spend(record, APPROVED)
    try:
        ledger.spend(record, APPROVED)
    except ApprovalBindingError as unbound:
        print("replayed decision:", unbound)  # noqa: T201


async def denied() -> None:
    """What a run does when the desk says no."""
    registry = ToolRegistry((wire_funds, issue_refund), clock=FakeClock())
    runner = AgentRunner(
        provider=ScriptedProvider(
            ModelResponse(
                content="",
                tool_calls=(ToolCall(id="c1", name="wire_funds", arguments=APPROVED),),
                usage=Usage(input_tokens=1, output_tokens=1),
            ),
            ModelResponse(
                content="Asked the desk; they said no.",
                usage=Usage(input_tokens=1, output_tokens=1),
            ),
            capabilities=ModelCapabilities(tool_calling=True, context_window_tokens=200_000),
        ),
        clock=FakeClock(),
        tools=registry.view(allow=("wire_funds", "issue_refund"), agent="planner"),
        approvals=Desk(),
    )
    agent: Agent[str] = Agent(
        name="planner",
        instructions="Settle the booking.",
        free_text=True,
        model="scripted-1",
        tools=("wire_funds", "issue_refund"),
    )
    run = await runner.run(agent, "pay the deposit", tenant="acme", run_id="run_1")
    refusals = [e.detail for e in run.events if e.kind is RunEventKind.TOOL_REFUSED]
    print("run state:", run.state.value, "| refusal:", refusals[0])  # noqa: T201


async def main() -> None:
    """Run every scenario in order."""
    declaring()
    bound_to_the_payload()
    await denied()
    wire_funds.release()
    issue_refund.release()


if __name__ == "__main__":
    asyncio.run(main())
