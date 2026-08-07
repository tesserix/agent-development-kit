"""Watch a run as it happens: typed events, not text chunks.

Three scenarios: a run with a tool call rendered from structure alone; the sequence
numbering a consumer checks for loss; and a stream that drops mid-answer, which ends as a
failure rather than as a short answer. Scripted providers stand in for a vendor, so nothing
here reaches the network and no key is needed.

Run it with `python examples/run_progress.py`.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from tesserix_adk.core import Agent, ModelCapabilities, TextDelta, ToolCall, Usage
from tesserix_adk.runtime import (
    AgentRunner,
    AnswerDelta,
    ModelResponse,
    ProgressEvent,
    RunCompleted,
    RunFailed,
    SequenceCheck,
    ToolCallFinished,
    ToolCallStarted,
    UsageUpdated,
)
from tesserix_adk.testing import FakeToolRegistry, ScriptedProvider, estimate_tokens

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Sequence

    from tesserix_adk.core import Message, ModelRequest, StreamEvent

CAPABLE = ModelCapabilities(tool_calling=True, streaming=True, context_window_tokens=200_000)

AGENT = Agent(
    name="planner",
    instructions="Plan trips. Cite the timetable before recommending a leg.",
    model="claude-sonnet-5",
    free_text=True,
    tools=("timetable",),
)


class Dropping:
    """A provider whose connection dies mid-answer: some text, and then nothing."""

    name = "dropping"
    capabilities = CAPABLE

    def count_tokens(self, messages: Sequence[Message]) -> int:
        """Count by characters, as a provider without a tokeniser would."""
        return estimate_tokens(messages)

    async def complete(self, request: ModelRequest) -> ModelResponse:  # noqa: ARG002 — scripted
        """The answer the buffered path would have received."""
        return ModelResponse(content="Kyoto, four nights.")

    async def stream(self, request: ModelRequest) -> AsyncIterator[StreamEvent]:  # noqa: ARG002
        """Some text, and then the connection is gone."""
        return _cut_short()


async def _cut_short() -> AsyncIterator[StreamEvent]:
    yield TextDelta(text="Kyoto, ")


def rendered(event: ProgressEvent) -> str | None:
    """One event as a UI would show it, or nothing where it has nothing to show."""
    match event:
        case AnswerDelta(text=text):
            return text
        case ToolCallStarted(tool=tool, arguments=arguments):
            return f"\n  [{tool} {arguments}]"
        case ToolCallFinished(tool=tool):
            return f"\n  [{tool} done]\n  "
        case UsageUpdated(usage=usage):
            return f"\n  [{usage.input_tokens + usage.output_tokens} tokens so far]\n  "
        case _:
            return None


def runner(*responses: ModelResponse, **overrides: object) -> AgentRunner:
    """A runner over a scripted provider and one timetable tool."""
    fields: dict[str, object] = {
        "provider": ScriptedProvider(*responses, capabilities=CAPABLE),
        "tools": FakeToolRegistry({"timetable": lambda leg: f"{leg}: 09:12, 11:40"}),
    }
    return AgentRunner(**{**fields, **overrides})  # type: ignore[arg-type]


async def a_run_rendered_from_structure() -> None:
    """A tool call, an answer and a running total, without parsing a single string."""
    call = ToolCall(id="c1", name="timetable", arguments={"leg": "Osaka to Kyoto"})
    stream = runner(
        ModelResponse(
            content="", tool_calls=(call,), usage=Usage(input_tokens=812, output_tokens=24)
        ),
        ModelResponse(
            content="Take the 09:12 from Osaka; four nights in Kyoto.",
            usage=Usage(input_tokens=901, output_tokens=63),
        ),
    ).stream(AGENT, "Four nights near Kyoto.", tenant="acme")

    print("\na run, as a consumer sees it\n  ", end="")  # noqa: T201
    async for event in stream:
        shown = rendered(event)
        if shown is not None:
            print(shown, end="")  # noqa: T201
    print(f"\n  state: {stream.run.state}")  # noqa: T201


async def loss_is_detectable() -> None:
    """Contiguous numbering is what lets a consumer tell a slow stream from a lossy one."""
    stream = runner(ModelResponse(content="Kyoto, four nights.")).stream(
        AGENT, "Four nights near Kyoto.", tenant="acme"
    )
    check = SequenceCheck()
    events = [event async for event in stream]
    dropped = [event for event in events if event.sequence != 2]

    print("\nwhat a consumer can detect")  # noqa: T201
    print(f"  events:   {len(events)}, numbered {events[0].sequence} to {events[-1].sequence}")  # noqa: T201
    for event in dropped:
        check.accept(event)
    print(f"  missing after one event is lost: {check.missing}")  # noqa: T201


async def a_dropped_stream_is_not_an_answer() -> None:
    """Accumulated text from a dropped connection is a failure, never a completed run."""
    stream = runner(provider=Dropping()).stream(AGENT, "Four nights near Kyoto.", tenant="acme")
    events = [event async for event in stream]

    print("\na connection that died mid-answer")  # noqa: T201
    print(f"  partial text: {''.join(e.text for e in events if isinstance(e, AnswerDelta))!r}")  # noqa: T201
    print(f"  terminal:     {events[-1].kind}")  # noqa: T201
    assert isinstance(events[-1], RunFailed)  # noqa: S101
    assert not [event for event in events if isinstance(event, RunCompleted)]  # noqa: S101


async def main() -> None:
    """Run all three scenarios."""
    await a_run_rendered_from_structure()
    await loss_is_detectable()
    await a_dropped_stream_is_not_an_answer()


if __name__ == "__main__":
    asyncio.run(main())
