"""What a consumer watching a run is told while it is still going.

A stream of raw text chunks makes every consumer guess: which tool is running, whether the
answer finished or the connection died, what it has cost so far. Each product guessed
differently, so the guessing is replaced here by a discriminated union — one variant per
thing that happens, carrying the same typed primitives the buffered `Run` carries.

Three properties hold whatever the run does. Every event is numbered from zero on one run,
so a gap is detectable rather than invisible. Exactly one terminal event is emitted, and it
is last, so a truncated stream can never read as a finished answer. And redaction happens
here rather than at a transport, because a transport that redacts has already handed the
value to whatever it logs to.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from collections import deque
from contextvars import ContextVar
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Annotated, Any, Literal

from pydantic import Field, ValidationError

from tesserix_adk.core import AdkModel, RunState, StreamInterruptedError, Usage

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Awaitable, Callable, Generator, Mapping

    from pydantic import BaseModel

    from tesserix_adk.core import Clock, Run

__all__ = [
    "AnswerDelta",
    "ApprovalRequired",
    "Backpressure",
    "GuardrailDecision",
    "IterationStarted",
    "Pressure",
    "ProgressEvent",
    "Provisional",
    "RunCancelled",
    "RunCompleted",
    "RunFailed",
    "RunStarted",
    "RunStream",
    "SequenceCheck",
    "StructuredDelta",
    "ToolCallFailed",
    "ToolCallFinished",
    "ToolCallStarted",
    "UsageUpdated",
    "decode_progress",
]


class ProgressEvent(AdkModel):
    """One thing that happened, as a consumer watching the run sees it.

    Args:
        kind: Which variant this is. The discriminator, so a consumer switches on a value
            rather than on the shape of the payload.
        run_id: The run it belongs to. Carried on every event so a multiplexed transport
            needs no envelope of its own.
        sequence: Position in this run's stream, from zero, with no gaps. A consumer that
            cannot detect a gap cannot tell a slow stream from a lossy one.
        at: Unix seconds, where the runner has a clock.
    """

    kind: str
    run_id: str = ""
    sequence: int = Field(default=0, ge=0)
    at: float | None = None


class RunStarted(ProgressEvent):
    """The run is live. Always first."""

    kind: Literal["run_started"] = "run_started"
    agent: str
    model: str
    tenant: str


class IterationStarted(ProgressEvent):
    """Another model call is about to be made, counted from one."""

    kind: Literal["iteration_started"] = "iteration_started"
    iteration: int = Field(ge=1)


class AnswerDelta(ProgressEvent):
    """More of a free-text answer. Concatenating them in order gives the buffered answer.

    Not scrubbed: deltas that no longer reassemble to the answer are a corrupted answer.
    What the model says about the caller's own data is the product, and a consumer that
    must not see it must not be given the run.
    """

    kind: Literal["answer_delta"] = "answer_delta"
    text: str
    coalesced: int = Field(default=0, ge=0)


class StructuredDelta(ProgressEvent):
    """More of a structured answer, as the JSON text arrives.

    Distinct from `AnswerDelta` because a consumer rendering a form needs to know it is
    being sent JSON rather than prose — and that a half-arrived object is not yet valid.
    """

    kind: Literal["structured_delta"] = "structured_delta"
    fragment: str
    coalesced: int = Field(default=0, ge=0)


class ToolCallStarted(ProgressEvent):
    """A tool call cleared policy and is about to run.

    Args:
        call_id: The provider's id for this call. Two parallel calls to one tool differ
            only in their arguments, so the id is what pairs a finish with its start.
        tool: What is being called.
        arguments: The arguments as compact JSON, scrubbed. For display: a consumer that
            re-executes from these re-executes a redacted call.
    """

    kind: Literal["tool_call_started"] = "tool_call_started"
    call_id: str
    tool: str
    arguments: str = ""


class ToolCallFinished(ProgressEvent):
    """A tool returned.

    Args:
        truncated: Whether the result was cut to the runner's ceiling. A consumer that
            renders it whole would be showing an answer the model never saw.
    """

    kind: Literal["tool_call_finished"] = "tool_call_finished"
    call_id: str
    tool: str
    truncated: bool = False


class ToolCallFailed(ProgressEvent):
    """A tool call did not return a result, and why.

    Its own variant rather than a finish with a flag: a consumer that renders a failure as
    a completed step is the reason products report failed runs as successful ones.
    """

    kind: Literal["tool_call_failed"] = "tool_call_failed"
    call_id: str
    tool: str
    error: str
    detail: str = ""


class GuardrailDecision(ProgressEvent):
    """A guardrail was asked, and what it said."""

    kind: Literal["guardrail_decision"] = "guardrail_decision"
    guardrail: str
    allowed: bool
    detail: str = ""


class ApprovalRequired(ProgressEvent):
    """A call is held until a human decides. Surfaced so the queue is visible live."""

    kind: Literal["approval_required"] = "approval_required"
    call_id: str
    tool: str
    reason: str = ""


class UsageUpdated(ProgressEvent):
    """What the run has consumed so far.

    A long run attributable only at the end is unattributable while that still matters.
    """

    kind: Literal["usage_updated"] = "usage_updated"
    usage: Usage


class RunCompleted(ProgressEvent):
    """The run reached a good terminal state.

    Emitted once, last, and never for a response that was truncated in flight.
    """

    kind: Literal["run_completed"] = "run_completed"
    state: RunState
    usage: Usage


class RunFailed(ProgressEvent):
    """The run ended badly, naming the failure rather than the symptom."""

    kind: Literal["run_failed"] = "run_failed"
    state: RunState
    error: str
    detail: str = ""


class RunCancelled(ProgressEvent):
    """The run was stopped by its caller or by a deadline."""

    kind: Literal["run_cancelled"] = "run_cancelled"
    state: RunState
    reason: str = ""


Progress = Annotated[
    RunStarted
    | IterationStarted
    | AnswerDelta
    | StructuredDelta
    | ToolCallStarted
    | ToolCallFinished
    | ToolCallFailed
    | GuardrailDecision
    | ApprovalRequired
    | UsageUpdated
    | RunCompleted
    | RunFailed
    | RunCancelled,
    Field(discriminator="kind"),
]

_BY_KIND: dict[str, type[ProgressEvent]] = {
    variant.model_fields["kind"].default: variant
    for variant in (
        RunStarted,
        IterationStarted,
        AnswerDelta,
        StructuredDelta,
        ToolCallStarted,
        ToolCallFinished,
        ToolCallFailed,
        GuardrailDecision,
        ApprovalRequired,
        UsageUpdated,
        RunCompleted,
        RunFailed,
        RunCancelled,
    )
}


def decode_progress(payload: Mapping[str, object]) -> ProgressEvent | None:
    """Decode one event, or `None` where this version has never heard of it.

    Adding a variant is a minor release, so a consumer pinned to an older kit must skip
    what it does not know rather than fall over. A variant it *does* know and cannot parse
    is the opposite case and raises: a delta decoded by guesswork renders as an answer
    nobody wrote.

    Raises:
        ValueError: If a known variant's payload does not validate.

    Example:
        >>> decode_progress({"kind": "quantum_delta", "run_id": "r", "sequence": 0}) is None
        True
    """
    kind = payload.get("kind")
    variant = _BY_KIND.get(kind) if isinstance(kind, str) else None
    if variant is None:
        return None
    try:
        return variant.model_validate(dict(payload))
    except ValidationError as broken:
        raise ValueError(f"a {kind} event did not decode: {broken}") from broken


class SequenceCheck:
    """Tracks one run's numbering, so a consumer can see what it lost.

    Example:
        >>> check = SequenceCheck()
        >>> check.accept(AnswerDelta(run_id="r", sequence=0, text="hi"))
        True
        >>> check.accept(AnswerDelta(run_id="r", sequence=0, text="hi"))
        False
    """

    def __init__(self) -> None:
        self.missing = 0
        self._next = 0

    def accept(self, event: ProgressEvent) -> bool:
        """Whether `event` is the next one, counting anything skipped on the way.

        A late or duplicate event is rejected rather than reordered into place: putting a
        delta back where it belongs is how a consumer renders an answer nobody wrote.
        """
        if event.sequence < self._next:
            return False
        self.missing += event.sequence - self._next
        self._next = event.sequence + 1
        return True


@dataclass(frozen=True, slots=True)
class Provisional[OutputT: BaseModel]:
    """Structured content that has arrived so far, and is not a result.

    Half a JSON object parses into something that looks exactly like the declared output
    type, so a consumer holding one cannot tell by inspection whether acting on it is safe.
    It can tell by type: `Provisional[TripPlan]` is not a `TripPlan`, the checker refuses it
    everywhere one is required, and `snapshot` deliberately hands back a plain mapping
    rather than a validated model.

    Args:
        text: The structured text emitted so far, exactly as it arrived.
    """

    text: str = ""

    def snapshot(self) -> dict[str, object] | None:
        """What has arrived, if it is currently a whole JSON object; otherwise nothing.

        A half-arrived object reads as nothing rather than as a guess: filling in the
        missing half would be inventing content the model never sent.
        """
        try:
            parsed: object = json.loads(self.text)
        except ValueError:
            return None
        if not isinstance(parsed, dict):
            return None
        return {str(key): value for key, value in parsed.items()}


@dataclass(frozen=True, slots=True)
class Backpressure:
    """What a run does when its consumer cannot keep up.

    Args:
        high_water: How many events may wait unread before text deltas start merging.
        byte_budget: How many bytes of unread events may wait, for the same reason. Events
            differ in size by three orders of magnitude, so a count alone is not a bound.
        stall_seconds: How long a reader may hold an unread buffer before the run is
            cancelled. Measured from the last read, and only while something is waiting:
            a quiet run is not a stalled one.
    """

    high_water: int = 256
    byte_budget: int = 8 * 1024 * 1024
    stall_seconds: float = 30.0

    @classmethod
    def shared(cls, *, total_bytes: int, streams: int) -> Backpressure:
        """A per-run budget derived from what the process as a whole may hold.

        A per-run bound multiplied by however many runs are in flight is not a bound on the
        process, which is the number an operator is actually sizing. Adjust the rest with
        `dataclasses.replace`.
        """
        return cls(byte_budget=total_bytes // streams)


@dataclass(frozen=True, slots=True)
class Pressure:
    """How close a stream came to its limits, and what it did about it.

    Args:
        buffered: Events waiting to be read right now.
        peak: The most that ever waited at once.
        coalesced: Text deltas merged into another delta rather than kept separate.
        oversize: Events admitted despite exceeding the whole byte budget on their own.
        stalled: Whether the run was cancelled because its reader stopped reading.
    """

    buffered: int = 0
    peak: int = 0
    coalesced: int = 0
    oversize: int = 0
    stalled: bool = False


class _Sink:
    """Where the runtime posts events, and where the stream collects them.

    Bounded, and never blocking. A queue that blocks the run because a consumer is slow
    deadlocks the run whose own tool result feeds that consumer, so pressure is answered by
    merging text deltas rather than by making the run wait. Nothing that makes the run
    attributable — lifecycle, tool, approval, usage, terminal — is ever merged or dropped.
    """

    def __init__(
        self, run_id: str, clock: Clock, policy: Backpressure, stop: Callable[[str], None]
    ) -> None:
        self._run_id = run_id
        self._clock = clock
        self._policy = policy
        self._stop = stop
        self._buffer: deque[ProgressEvent] = deque()
        self._bytes = 0
        self._arrived = asyncio.Event()
        self._closed = False
        self._reading = False
        self._read_at = clock.now()
        self._next = 0
        self.structured = ""
        self.pressure = Pressure()
        self.stalled: str | None = None

    @property
    def emitted(self) -> int:
        """How many events have been numbered on this run."""
        return self._next

    def reading(self) -> None:
        """Say that a consumer has attached and is reading events as they arrive."""
        self._reading = True
        self._read_at = self._clock.now()

    def emit(self, event: ProgressEvent) -> None:
        """Number `event` and hand it to whoever is watching.

        Numbering happens whether or not anyone is reading: sequence numbers are the run's,
        not the reader's, so an await-only caller and a reading one see the same numbering.
        """
        numbered = self.number(event)
        if not self._reading:
            return
        self._admit(numbered)
        self._arrived.set()
        self._note_stall()

    def _admit(self, event: ProgressEvent) -> None:
        """Buffer `event`, merging it into the last delta where there is no room for it."""
        size = _weight(event)
        if self._under_pressure() and (merged := _merged(self._buffer, event)) is not None:
            self._bytes += merged
            self.pressure = replace(self.pressure, coalesced=self.pressure.coalesced + 1)
            return
        if size > self._policy.byte_budget:
            self.pressure = replace(self.pressure, oversize=self.pressure.oversize + 1)
        self._buffer.append(event)
        self._bytes += size
        self.pressure = replace(
            self.pressure,
            buffered=len(self._buffer),
            peak=max(self.pressure.peak, len(self._buffer)),
        )

    def _under_pressure(self) -> bool:
        return (
            len(self._buffer) >= self._policy.high_water or self._bytes >= self._policy.byte_budget
        )

    def _note_stall(self) -> None:
        """Record a reader that is holding an unread buffer for longer than it may."""
        if self.stalled is not None or not self._buffer:
            return
        if self._clock.now() - self._read_at > self._policy.stall_seconds:
            self.stalled = "the consumer stopped reading, so the run was not worth continuing"
            self.pressure = replace(self.pressure, stalled=True)
            self._stop(self.stalled)

    def close(self) -> None:
        """Say that the run is over and no further event will arrive."""
        self._closed = True
        self._arrived.set()

    async def next_event(self) -> ProgressEvent | None:
        """The next event, or `None` once the run is over."""
        while not self._buffer:
            if self._closed:
                return None
            self._arrived.clear()
            await self._arrived.wait()
        event = self._buffer.popleft()
        self._bytes -= _weight(event)
        self._read_at = self._clock.now()
        self.pressure = replace(self.pressure, buffered=len(self._buffer))
        return event

    def number(self, event: ProgressEvent) -> ProgressEvent:
        """Stamp identity, position and time onto an event built without them."""
        numbered = event.model_copy(
            update={"run_id": self._run_id, "sequence": self._next, "at": self._clock.now()}
        )
        self._next += 1
        if isinstance(numbered, StructuredDelta):
            self.structured += numbered.fragment
        return numbered


def _weight(event: ProgressEvent) -> int:
    """What holding `event` costs, near enough to bound a buffer by."""
    return len(event.model_dump_json())


def _merged(buffer: deque[ProgressEvent], event: ProgressEvent) -> int | None:
    """Fold `event` into the delta at the end of `buffer`, returning what that added.

    Only text and structured deltas merge, and only into a delta of their own kind: their
    content concatenates, so merging keeps every character a consumer would have rendered.
    `None` where nothing merges, which is everything a run is accounted for by.
    """
    if not buffer:
        return None
    last = buffer[-1]
    if isinstance(event, AnswerDelta) and isinstance(last, AnswerDelta):
        buffer[-1] = last.model_copy(
            update={"text": last.text + event.text, "coalesced": last.coalesced + 1}
        )
        return len(event.text)
    if isinstance(event, StructuredDelta) and isinstance(last, StructuredDelta):
        buffer[-1] = last.model_copy(
            update={"fragment": last.fragment + event.fragment, "coalesced": last.coalesced + 1}
        )
        return len(event.fragment)
    return None


class RunStream[OutputT: BaseModel]:
    """One run, watched while it happens.

    Iterating drives the run; the run itself is available once the stream is drained. Two
    objects rather than one because a terminal event is not a `Run` — a consumer rendering
    progress wants the event, and a caller recording what happened wants the record.

    Example:
        >>> stream = runner.stream(agent, "hello", tenant="acme")  # doctest: +SKIP
        >>> async for event in stream:  # doctest: +SKIP
        ...     print(event.kind)
        >>> stream.run.state  # doctest: +SKIP
        <RunState.COMPLETED: 'completed'>
    """

    def __init__(
        self,
        run_id: str,
        clock: Clock,
        drive: Callable[[], Awaitable[Run[OutputT]]],
        cancel: Callable[[str], None],
        backpressure: Backpressure | None = None,
    ) -> None:
        self._drive = drive
        self._cancel = cancel
        self._sink = _Sink(run_id, clock, backpressure or Backpressure(), cancel)
        self._task: asyncio.Task[Run[OutputT]] | None = None
        self._run: Run[OutputT] | None = None
        self._abandoned = False
        self._run_id = run_id

    @property
    def run_id(self) -> str:
        """Which run this is, before it has produced anything to read it from."""
        return self._run_id

    @property
    def run(self) -> Run[OutputT]:
        """The finished run, whatever state it reached.

        Raises:
            RuntimeError: If the stream has not been drained. A half-run reported as a run
                is the failure this whole surface exists to prevent.
        """
        if self._run is None:
            raise RuntimeError("the run is still running; drain the stream before reading it")
        return self._run

    @property
    def pressure(self) -> Pressure:
        """How close this stream is to its limits, readable while the run is still going."""
        return self._sink.pressure

    @property
    def provisional(self) -> Provisional[OutputT]:
        """The structured content that has arrived so far. Never a result."""
        return Provisional(text=self._sink.structured)

    def __await__(self) -> Generator[Any, None, Run[OutputT]]:
        """Drive the run to its end and give back the authoritative record.

        The await-only pattern, and the one a consumer that iterated then wants the result
        uses too: the run is driven once however many callers await it.
        """
        return self._result().__await__()

    async def __aenter__(self) -> RunStream[OutputT]:
        """Take the stream, guaranteeing it is closed however the block is left."""
        return self

    async def __aexit__(self, kind: object, value: BaseException | None, traceback: object) -> None:
        """Close the stream, without replacing what the consumer's own body raised."""
        if value is None:
            await self.aclose()
            return
        with contextlib.suppress(Exception):
            await self.aclose()

    async def aclose(self) -> None:
        """Stop a run nobody is reading any more, and wait for it to let go of what it holds.

        A consumer that stops reading has stopped paying attention, not stopped paying: a
        run left driving in the background still calls providers and still bills. So the
        run is cancelled through the same path a caller's own token uses, and the cancelled
        record stays readable on `run`.
        """
        task = self._task
        if task is None or not task.done():
            self._abandoned = True
            self._cancel("the consumer stopped reading the stream")
        if task is None:
            return
        self._run = await task

    async def __aiter__(self) -> AsyncGenerator[ProgressEvent]:
        """Drive the run, yielding every event, terminal one last.

        Raises:
            RuntimeError: If the stream has already been iterated. Events are consumed as
                they are read, so a second reader would see a truncated run rather than a
                replay of the first.
        """
        if self._task is not None:
            raise RuntimeError("this stream has already been consumed; a run is watched once")
        self._sink.reading()
        self._task = asyncio.create_task(self._watched())
        while (event := await self._sink.next_event()) is not None:
            yield event
        self._run = await self._task
        yield self._sink.number(terminal_for(self._run))

    async def _result(self) -> Run[OutputT]:
        """The run, or a refusal to call an interrupted one a result.

        Raises:
            StreamInterruptedError: If the stream was abandoned before the run reached a
                terminal event. What arrived is carried on the error rather than returned:
                partial content handed back as a result is a wrong answer that looks right.
        """
        if self._task is None:
            self._task = asyncio.create_task(self._watched())
        if self._run is None:
            self._run = await self._task
        if self._abandoned:
            raise StreamInterruptedError(
                "the stream was abandoned before the run finished, so the run was cancelled "
                "and has no result",
                partial=self._sink.structured,
                received=self._sink.emitted,
            )
        return self._run

    async def _watched(self) -> Run[OutputT]:
        """Run with this stream's sink in scope, and close it however the run ends."""
        token = WATCHING.set(self._sink)
        try:
            return await self._drive()
        finally:
            WATCHING.reset(token)
            self._sink.close()


def terminal_for(run: Run[Any]) -> ProgressEvent:
    """The one event that closes a stream, derived from the run's own terminal state."""
    detail = next(
        (event.detail or "" for event in reversed(run.events) if event.detail is not None), ""
    )
    if run.state is RunState.COMPLETED:
        return RunCompleted(state=run.state, usage=run.usage)
    if run.state is RunState.CANCELLED:
        return RunCancelled(state=run.state, reason=detail)
    error, _, rest = detail.partition(": ")
    named = error.endswith("Error") and rest != ""
    return RunFailed(
        state=run.state,
        error=error if named else run.state.value,
        detail=rest if named else detail,
    )


# The sink for the run in flight, or None where nobody is watching. A context variable
# rather than runner state: one runner drives many concurrent runs, and a task inherits a
# copy of the context it was created in, so each streamed run sees only its own sink.
WATCHING: ContextVar[_Sink | None] = ContextVar("adk_progress_sink", default=None)
