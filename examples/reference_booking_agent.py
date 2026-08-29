"""Complete offline booking agent: approval first, idempotent transaction second.

Run with ``uv run python examples/reference_booking_agent.py``. The model, approval
transport, stores, clock, telemetry exporter and evaluation are deterministic fakes.
"""

from __future__ import annotations

import asyncio
from typing import Any

from tesserix_adk.core import (
    Agent,
    ApprovalRecord,
    ApprovalToken,
    ApprovalTokenError,
    BudgetLimits,
    Idempotency,
    IdempotencyPolicy,
    ModelCapabilities,
    Run,
    ToolCall,
    Usage,
)
from tesserix_adk.evals import EvalCase, EvalSuite, SuiteRunner
from tesserix_adk.guardrails import Guard, GuardResult
from tesserix_adk.observability import PendingSpan, RedactingSpanProcessor
from tesserix_adk.runtime import (
    AgentRunner,
    Checkpointer,
    DeferringGate,
    MemoryCheckpointStore,
    MemoryIdempotencyStore,
    MemorySuspensionStore,
    ModelResponse,
)
from tesserix_adk.testing import FakeClock, ScriptedProvider
from tesserix_adk.tools import ToolContext, ToolRegistry, tool

CAPABLE = ModelCapabilities(tool_calling=True, context_window_tokens=16_000)
BOOKINGS: list[str] = []
KEYS: list[str] = []


@tool(
    requires_approval=True,
    idempotency=IdempotencyPolicy(
        Idempotency.EFFECTFUL,
        key_arguments=("flight", "traveller"),
    ),
)
async def confirm_booking(flight: str, traveller: str, context: ToolContext) -> str:
    """Commit a confirmed flight booking.

    Args:
        flight: Flight selected by the user.
        traveller: Traveller the approval showed.
        context: Authenticated run context carrying the idempotency key.
    """
    # Deterministic transaction boundary: reasoning has stopped; only approved arguments
    # and the runtime-minted idempotency key reach the side-effecting booking client.
    if not context.idempotency_key:
        raise RuntimeError("an effectful booking must carry an idempotency key")
    BOOKINGS.append(f"{flight}:{traveller}")
    KEYS.append(context.idempotency_key)
    return f"confirmed {flight} for {traveller}"


class BookingGuard(Guard):
    """Refuse instruction leakage while leaving ordinary booking text unchanged."""

    name = "booking_output"

    async def check_output(self, content: str) -> GuardResult:
        """Block a response that echoes hidden instructions."""
        if "system prompt" in content.lower():
            return GuardResult.blocked(code="prompt_leak", detail="hidden instructions")
        return GuardResult.allow()


class Posting:
    """Offline approval transport retaining only the safe summary."""

    def __init__(self) -> None:
        self.summary = ""

    async def deliver(self, record: ApprovalRecord) -> None:
        """Record what an approver would see."""
        self.summary = record.summary


class Desk:
    """Offline approver inbox holding the single-use decision token."""

    def __init__(self) -> None:
        self.token = ""

    async def hand_to(self, token: ApprovalToken) -> None:
        """Accept the opaque token without inspecting or logging it."""
        self.token = token.value


class World:
    """Production-shaped dependencies for one completely offline booking run."""

    def __init__(self) -> None:
        self.clock = FakeClock(start=1_000.0, auto_advance=False)
        self.posting = Posting()
        self.desk = Desk()
        self.suspensions = MemorySuspensionStore()
        self.gate = DeferringGate(
            self.posting,
            self.suspensions,
            hand_to=self.desk.hand_to,
            clock=self.clock,
        )
        self.checkpoints = Checkpointer(MemoryCheckpointStore(), None, self.clock)
        self.idempotency = MemoryIdempotencyStore()
        self.registry = ToolRegistry((confirm_booking,), clock=self.clock)
        self.agent = Agent(
            name="booking-reference",
            instructions=(
                "Propose a booking, but never claim it is confirmed before the tool result."
            ),
            model="scripted",
            free_text=True,
            tools=("confirm_booking",),
            approval_required_tools=("confirm_booking",),
            guardrails=("booking_output",),
            budget=BudgetLimits(max_model_calls=3, max_tool_calls=1, max_input_tokens=2_000),
        )

    def runner(self, *responses: ModelResponse) -> AgentRunner:
        """Build a worker; persistent state remains outside the worker process."""
        return AgentRunner(
            provider=ScriptedProvider(*responses, capabilities=CAPABLE),
            clock=self.clock,
            tools=self.registry.view(allow=("confirm_booking",), agent=self.agent.name),
            guardrails={"booking_output": BookingGuard()},
            approvals=self.gate,
            checkpoints=self.checkpoints,
            idempotency=self.idempotency,
        )


async def main() -> None:
    """Suspend, approve, transact once, replay-refuse, redact and evaluate."""
    BOOKINGS.clear()
    KEYS.clear()
    world = World()
    proposal = ModelResponse(
        tool_calls=(
            ToolCall(
                id="book-1",
                name="confirm_booking",
                arguments={"flight": "QF9", "traveller": "Ada"},
            ),
        ),
        usage=Usage(input_tokens=30, output_tokens=8),
    )
    answer = ModelResponse(
        content="QF9 is confirmed for Ada.",
        usage=Usage(input_tokens=20, output_tokens=7),
    )

    held = await world.runner(proposal, answer).run(
        world.agent,
        "Confirm QF9 for Ada",
        tenant="acme",
        user="ada",
        run_id="booking-42",
    )
    print(f"before approval: {held.state.value}, bookings={len(BOOKINGS)}")  # noqa: T201

    completed = await world.runner(answer).resume_with_decision(
        world.agent,
        "booking-42",
        tenant="acme",
        token=world.desk.token,
        granted=True,
        decided_by="ada",
        user="ada",
    )
    print(f"after approval: {completed.state.value}, bookings={len(BOOKINGS)}")  # noqa: T201
    print(f"idempotency key: {KEYS[0][:12]}")  # noqa: T201

    try:
        await world.runner().resume_with_decision(
            world.agent,
            "booking-42",
            tenant="acme",
            token=world.desk.token,
            granted=True,
            decided_by="mallory",
            user="ada",
        )
    except ApprovalTokenError:
        print(f"replayed approval refused; bookings={len(BOOKINGS)}")  # noqa: T201

    exported = RedactingSpanProcessor().process(
        PendingSpan(
            name="booking.confirmed",
            attributes={"adk.tenant": "acme", "adk.prompt": "ada@example.com booked QF9"},
        )
    )
    print(f"telemetry redacted: {'ada@example.com' not in exported.model_dump_json()}")  # noqa: T201

    async def replay(case: EvalCase, *, run_id: str) -> Run[Any]:
        del case
        return completed.model_copy(update={"id": run_id})

    report = await SuiteRunner(replay).run(
        EvalSuite(
            name="booking-reference",
            version="1",
            cases=(EvalCase(id="approval-then-confirm", input="Confirm QF9", tenant="acme"),),
        )
    )
    print(f"tenant: {completed.tenant}")  # noqa: T201
    print(f"eval exit: {report.exit_code}")  # noqa: T201
    confirm_booking.release()


if __name__ == "__main__":
    asyncio.run(main())
