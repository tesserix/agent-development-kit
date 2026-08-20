"""A watched run's progress, republished as events.

Progress is for whoever is watching this run right now; an event is for systems that were
not watching — a dashboard, cost reporting, support tooling. The two carry different things
on purpose: progress carries the answer as it arrives, and an event carries none of it.

Republishing here rather than inside the loop keeps eventing optional. A caller that wants
it wraps the iteration; a caller that does not pays nothing, and the loop has no publisher
to fail on.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel

from tesserix_adk.core import events
from tesserix_adk.runtime import progress

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from tesserix_adk.core.events import Eventing, EventPayload
    from tesserix_adk.runtime.progress import ProgressEvent, RunStream

__all__ = ["payload_of", "publishing"]


def payload_of(event: ProgressEvent) -> EventPayload | None:
    """The event a consumer downstream should hear about, or None where there is not one.

    A delta, an iteration and a usage update are for whoever is watching the run; none of
    them is a fact another system acts on, and publishing a token delta to a broker is how
    a queue ends up holding the answer.
    """
    match event:
        case progress.RunStarted():
            return events.RunStarted(run_id=event.run_id, agent=event.agent, model=event.model)
        case progress.ToolCallStarted():
            return events.ToolCallRequested(
                run_id=event.run_id, tool=event.tool, tool_call_id=event.call_id
            )
        case progress.ToolCallFinished():
            return events.ToolCallCompleted(
                run_id=event.run_id, tool=event.tool, tool_call_id=event.call_id, state="ok"
            )
        case progress.ToolCallFailed():
            return events.ToolCallCompleted(
                run_id=event.run_id,
                tool=event.tool,
                tool_call_id=event.call_id,
                state="failed",
                error_code=event.error,
            )
        case progress.ToolCallIndeterminate():
            return events.ToolCallCompleted(
                run_id=event.run_id,
                tool=event.tool,
                tool_call_id=event.call_id,
                state="indeterminate",
            )
        case progress.ApprovalRequired():
            return events.ApprovalRequested(
                run_id=event.run_id, approval_id=event.call_id, tool=event.tool
            )
        case progress.RunCompleted():
            return events.RunCompleted(
                run_id=event.run_id,
                input_tokens=event.usage.input_tokens,
                output_tokens=event.usage.output_tokens,
            )
        case progress.RunFailed():
            return events.RunFailed(run_id=event.run_id, error_code=event.error)
        case progress.RunCancelled():
            return events.RunCancelled(run_id=event.run_id, reason_code=event.reason or "stopped")
        case _:
            return None


async def publishing[OutputT: BaseModel](
    stream: RunStream[OutputT], eventing: Eventing
) -> AsyncIterator[ProgressEvent]:
    """Iterate a run's progress, publishing the events other systems act on as it goes.

    Each published event names the one before it as its cause, so a consumer can rebuild
    the run's order without trusting the broker to have kept it.

    Args:
        stream: The run being watched. Iterating it is what drives the run, here as
            anywhere else — this yields every event through untouched.
        eventing: Where the events go, and what a publisher being down means.

    Yields:
        Every progress event, in order, unchanged.

    Raises:
        EventPublishError: Under guaranteed delivery, where the publisher could not
            deliver. The run stops rather than diverging from what was reported.
    """
    caused_by = None
    async for event in stream:
        payload = payload_of(event)
        if payload is not None:
            caused_by = await eventing.emit(payload, caused_by=caused_by) or caused_by
        yield event
