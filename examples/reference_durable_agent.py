"""Complete offline durable agent with partial-state failure, resume and cancellation.

Run with ``uv run python examples/reference_durable_agent.py``. The journal stands in for
durable workflow history; a deployment supplies the Temporal adapter and durable stores.
"""

from __future__ import annotations

import asyncio
from typing import Any

from tesserix_adk.core import (
    BudgetLimits,
    BudgetScope,
    CancelledError,
    Message,
    NoOutput,
    ProviderUnavailableError,
    Run,
    RunBudget,
    RunState,
    ScopedLimits,
    TextPart,
    ToolCall,
    Usage,
    most_restrictive,
)
from tesserix_adk.evals import EvalCase, EvalSuite, SuiteRunner
from tesserix_adk.guardrails import Guard, GuardrailPipeline, GuardResult
from tesserix_adk.observability import PendingSpan, RedactingSpanProcessor
from tesserix_adk.runtime import CancellationToken, ModelResponse
from tesserix_adk.testing import FakeClock
from tesserix_adk.workflows import (
    ActivityContext,
    AgentWorkflow,
    ModelCallInput,
    ModelCallResult,
    ToolCallInput,
    ToolCallResult,
    WorkflowState,
    continued,
)

CONTEXT = ActivityContext(
    run_id="trip-42",
    tenant="acme",
    user="ada",
    scopes=("flights:read",),
    trace_id="trace-42",
)
SCRIPT = (
    ModelResponse(
        tool_calls=(ToolCall(id="find-1", name="find_flights"),),
        usage=Usage(input_tokens=90, output_tokens=12),
    ),
    ModelResponse(
        content="Rebooked on the 18:40.",
        usage=Usage(input_tokens=110, output_tokens=10),
    ),
)
EFFECTS: dict[str, str] = {}


class NoSpeculation(Guard):
    """Block plausible-sounding uncertainty from becoming a durable answer."""

    name = "no_speculation"

    async def check_output(self, content: str) -> GuardResult:
        """Refuse language this workflow cannot persist as a confirmed outcome."""
        if "probably" in content.lower():
            return GuardResult.blocked(code="unconfirmed", detail="speculative outcome")
        return GuardResult.allow()


class Worker:
    """Activity worker with tenant checks, budget, guards and idempotent effects."""

    def __init__(
        self,
        *,
        unavailable_step: str = "",
        cancellation: CancellationToken | None = None,
    ) -> None:
        self.unavailable_step = unavailable_step
        self.cancellation = cancellation
        self.ran: list[str] = []
        self.guards = GuardrailPipeline((NoSpeculation(),))
        self.budget = RunBudget(
            most_restrictive(
                ScopedLimits(
                    scope=BudgetScope.RUN,
                    limits=BudgetLimits(
                        max_model_calls=4,
                        max_tool_calls=1,
                        max_input_tokens=2_000,
                    ),
                )
            ),
            FakeClock(),
        )

    async def model_call(self, request: ModelCallInput) -> ModelCallResult:
        """Return the scripted model result or a typed repeatable outage."""
        self._same_tenant(request.context)
        self.ran.append(f"{request.step}:attempt-{request.attempt}")
        await self.budget.reserve(40)
        if request.step == self.unavailable_step:
            await self.budget.record(Usage(input_tokens=1, output_tokens=0), model_calls=1)
            raise ProviderUnavailableError(
                "provider timed out",
                provider="scripted",
                status=503,
            )
        index = int(request.step.split(":")[1])
        response = SCRIPT[index]
        if response.content:
            await self.guards.check_output(response.content)
        await self.budget.record(response.usage, model_calls=1, iterations=1)
        return ModelCallResult(response=response, history=f"history-{index + 1}")

    async def tool_call(self, request: ToolCallInput) -> ToolCallResult:
        """Execute one replay-safe effect using the workflow step as its key."""
        self._same_tenant(request.context)
        self.ran.append(request.step)
        key = f"{request.context.run_id}:{request.step}"
        # Deterministic transaction boundary: a replay computes the same key and returns
        # the first outcome instead of asking the side-effecting dependency twice.
        EFFECTS.setdefault(key, "3 flight options")
        await self.budget.record(Usage(input_tokens=0, output_tokens=0), tool_calls=1)
        if self.cancellation is not None:
            self.cancellation.cancel("caller disconnected after retrieval")
        return ToolCallResult(call_id=request.call_id, content=EFFECTS[key], history="history-2")

    @staticmethod
    def _same_tenant(context: ActivityContext) -> None:
        if context.tenant != CONTEXT.tenant:
            raise RuntimeError("activity crossed tenant boundary")


def partial_state() -> WorkflowState:
    """Reconstruct the state fully determined by the two journalled activities."""
    return WorkflowState(
        run_id="trip-42",
        history="history-2",
        iteration=1,
        usage=SCRIPT[0].usage,
    )


async def main() -> None:
    """Exhaust retries, checkpoint, resume without duplicate work, cancel and evaluate."""
    EFFECTS.clear()
    failing_worker = Worker(unavailable_step="model:1")
    failing = AgentWorkflow(
        activities=failing_worker,
        model="scripted",
        attempts=3,
    )
    initial = WorkflowState(run_id="trip-42", history="history-0")
    try:
        await failing.run(initial, context=CONTEXT)
    except ProviderUnavailableError as failure:
        print(  # noqa: T201
            f"ProviderUnavailableError: attempts={failure.details['attempts']} "
            f"step={failure.details['step']}"
        )

    print(f"partial journal: {failing.journal.steps} completed activities")  # noqa: T201
    checkpoint = continued(
        partial_state(),
        tenant=CONTEXT.tenant,
        agent_name="durable-reference",
        model="scripted",
        user=CONTEXT.user,
        scopes=CONTEXT.scopes,
    ).checkpoint
    print(  # noqa: T201
        f"checkpoint: run={checkpoint.run_id} tenant={checkpoint.tenant} "
        f"iteration={checkpoint.iterations}"
    )

    resumed_worker = Worker()
    final = await AgentWorkflow(
        activities=resumed_worker,
        model="scripted",
        journal=failing.journal,
    ).run(initial, context=CONTEXT)
    print(f"resumed answer: {final.answer}")  # noqa: T201
    print(f"resume executed only: {resumed_worker.ran}")  # noqa: T201
    print(f"idempotency key: {next(iter(EFFECTS))}")  # noqa: T201

    token = CancellationToken()
    cancelling = AgentWorkflow(
        activities=Worker(cancellation=token),
        model="scripted",
        token=token,
    )
    try:
        await cancelling.run(initial, context=CONTEXT)
    except CancelledError:
        cancellation_checkpoint = continued(
            partial_state(),
            tenant=CONTEXT.tenant,
            agent_name="durable-reference",
        ).checkpoint
        inspectable = cancelling.journal.steps == 2 and cancellation_checkpoint.tenant == "acme"
        print(f"cancelled checkpoint remains inspectable: {inspectable}")  # noqa: T201

    exported = RedactingSpanProcessor().process(
        PendingSpan(
            name="workflow.resumed",
            attributes={"adk.tenant": "acme", "adk.prompt": "ada@example.com rebooked"},
        )
    )
    print(f"telemetry redacted: {'ada@example.com' not in exported.model_dump_json()}")  # noqa: T201

    evaluated = Run[NoOutput](
        id="eval-source",
        tenant="acme",
        user="ada",
        agent_name="durable-reference",
        agent_version="1",
        model="scripted",
        state=RunState.COMPLETED,
        messages=[Message(role="assistant", content=[TextPart(text=final.answer)])],
        usage=final.usage,
    )

    async def replay(case: EvalCase, *, run_id: str) -> Run[Any]:
        del case
        return evaluated.model_copy(update={"id": run_id})

    report = await SuiteRunner(replay).run(
        EvalSuite(
            name="durable-reference",
            version="1",
            cases=(EvalCase(id="resume-after-outage", input="Rebook", tenant="acme"),),
        )
    )
    print(f"eval exit: {report.exit_code}")  # noqa: T201


if __name__ == "__main__":
    asyncio.run(main())
