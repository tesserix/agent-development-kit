"""A run whose worker dies halfway, and the resumed run that does not pay for it twice.

Runs an agent through AgentWorkflow with a worker that fails after two activities, then
resumes with the journal the first attempt left behind.

Run it with `python examples/durable_run.py`.
"""

from __future__ import annotations

import asyncio

from tesserix_adk.core import ModelResponse, ToolCall, Usage
from tesserix_adk.workflows import (
    ActivityContext,
    AgentWorkflow,
    Journal,
    ModelCallInput,
    ModelCallResult,
    ToolCallInput,
    ToolCallResult,
    WorkflowState,
)

CONTEXT = ActivityContext(run_id="trip-42", tenant="tripbaba", user="ada", trace_id="t-9")

SCRIPT = (
    ModelResponse(
        tool_calls=(ToolCall(id="c0", name="find_flights"),),
        usage=Usage(input_tokens=900, output_tokens=120),
    ),
    ModelResponse(
        content="Rebooked on the 18:40.", usage=Usage(input_tokens=1400, output_tokens=60)
    ),
)


class Worker:
    """A worker that records what it ran, and can be made to die."""

    def __init__(self, *, dies_after: int = 0) -> None:
        self.ran: list[str] = []
        self.dies_after = dies_after

    async def model_call(self, request: ModelCallInput) -> ModelCallResult:
        """Call the provider, or die if this is the activity the pod roll lands on."""
        self._alive()
        self.ran.append(request.step)
        iteration = int(request.step.split(":")[1])
        return ModelCallResult(response=SCRIPT[iteration], history=f"{request.history}+{iteration}")

    async def tool_call(self, request: ToolCallInput) -> ToolCallResult:
        """Run the tool, which is where the money and the side effects are."""
        self._alive()
        self.ran.append(request.step)
        return ToolCallResult(call_id=request.call_id, content="3 options", history="h2")

    def _alive(self) -> None:
        """Die once the worker has run as many activities as the pod roll allowed."""
        if self.dies_after and len(self.ran) >= self.dies_after:
            message = "SIGKILL: the node was drained"
            raise RuntimeError(message)


async def main() -> None:
    """Lose a run, then resume it."""
    dying = Worker(dies_after=2)
    first = AgentWorkflow(activities=dying, model="claude-opus-5")
    try:
        await first.run(WorkflowState(run_id="trip-42", history="h0"), context=CONTEXT)
    except RuntimeError as killed:
        print(f"worker lost: {killed}")  # noqa: T201
    print(f"  it ran: {dying.ran}")  # noqa: T201
    print(f"  journal holds {first.journal.steps} completed activities")  # noqa: T201

    resumed = Worker()
    second = AgentWorkflow(activities=resumed, model="claude-opus-5", journal=first.journal)
    final = await second.run(WorkflowState(run_id="trip-42", history="h0"), context=CONTEXT)

    print(f"\nresumed worker ran: {resumed.ran}")  # noqa: T201
    print(f"answer: {final.answer}")  # noqa: T201
    print(f"usage across both attempts: {final.usage.input_tokens} input tokens")  # noqa: T201

    straight = Worker()
    once = await AgentWorkflow(activities=straight, model="claude-opus-5").run(
        WorkflowState(run_id="trip-42", history="h0"), context=CONTEXT
    )
    print(f"an uninterrupted run: {once.answer!r}, {once.usage.input_tokens} input tokens")  # noqa: T201

    fresh = AgentWorkflow(activities=Worker(), model="claude-opus-5", journal=Journal())
    print(f"a fresh journal skips nothing: {fresh.journal.steps} steps")  # noqa: T201


if __name__ == "__main__":
    asyncio.run(main())
