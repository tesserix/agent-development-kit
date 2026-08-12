"""In-memory implementations of the core protocols, for tests that must not touch a network.

These are the reference implementations the conformance suite is written against.
They are deliberately simple: a fake that acquires its own behaviour stops being a
control and becomes a second thing to debug.
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import json
import re
from collections import deque
from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING, Any

from tesserix_adk.core.budget import (
    BudgetDecision,
    BudgetLimits,
    BudgetScope,
    Consumed,
    ResolvedBudget,
)
from tesserix_adk.core.capabilities import Capability, ModelCapabilities
from tesserix_adk.core.errors import (
    BudgetExceededError,
    BudgetUnavailableError,
    ToolExecutionError,
)
from tesserix_adk.core.guards import GuardResult
from tesserix_adk.core.primitives import TextPart, Usage
from tesserix_adk.core.streaming import (
    ReasoningDelta,
    StreamEnd,
    TextDelta,
    ToolCallDelta,
    UsageDelta,
)
from tesserix_adk.runtime import ModelRequest, ModelResponse, ToolDeclaration

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable, Iterator, Mapping, Sequence

    from tesserix_adk.core.primitives import Message
    from tesserix_adk.core.streaming import StreamEvent

__all__ = [
    "CAPABLE",
    "BudgetExceededError",
    "FakeBudgetPolicy",
    "FakeClock",
    "FakeGuardrail",
    "FakeKeyValueStore",
    "FakeMeter",
    "FakeSecrets",
    "FakeTenantLedger",
    "FakeToolRegistry",
    "FakeTracer",
    "MetricPoint",
    "RecordedEvent",
    "ScriptedProvider",
    "SequentialIds",
    "StallingProvider",
    "ToolExecutionError",
    "estimate_tokens",
]

# A loop turn is free; a test that waits more than this is waiting for the wrong thing.
_SETTLE_TURNS = 100

_CHARS_PER_TOKEN = 4

# Structured output is off so that the prompt-side fallback is what a default test exercises.
CAPABLE = ModelCapabilities(
    tool_calling=True, parallel_tool_calls=True, vision=True, streaming=True
)


def estimate_tokens(messages: Sequence[Message]) -> int:
    """Estimate tokens by character count, for a provider with no tokeniser to call.

    Four characters to the token is wrong for every model and close enough for all of
    them; a provider that ships a tokeniser should use it instead.
    """
    return (
        sum(len(part.text) for m in messages for part in m.content if isinstance(part, TextPart))
        // _CHARS_PER_TOKEN
    )


class SequentialIds:
    """Ids a test can write down: `run_1`, `run_2`, and so on.

    Args:
        prefix: What each id starts with.
    """

    def __init__(self, prefix: str = "run") -> None:
        self._prefix = prefix
        self._issued = 0

    def __call__(self) -> str:
        """Return the next id in the sequence."""
        self._issued += 1
        return f"{self._prefix}_{self._issued}"


class FakeClock:
    """A clock that only moves when a test moves it.

    Args:
        start: The time it starts at.
        auto_advance: Whether `sleep` returns immediately, moving the clock by the
            slept-for amount. That is what a test for elapsed time wants. A test that
            races a sleep against work — a timeout, a grace window — wants the opposite,
            because a sleep that returns immediately wins every race. Pass False there,
            then `advance` past the sleeper to fire it.
    """

    def __init__(self, start: float = 0.0, *, auto_advance: bool = True) -> None:
        self._now = start
        self._auto_advance = auto_advance
        self._sleepers: list[tuple[float, asyncio.Event]] = []
        self.slept: list[float] = []

    def now(self) -> float:
        """Return the current fake time."""
        return self._now

    async def sleep(self, seconds: float) -> None:
        """Advance the clock by `seconds`, or suspend until a test advances past them."""
        self.slept.append(seconds)
        if self._auto_advance:
            self._now += seconds
            return
        sleeper = (self._now + seconds, asyncio.Event())
        self._sleepers.append(sleeper)
        try:
            await sleeper[1].wait()
        finally:
            with contextlib.suppress(ValueError):
                self._sleepers.remove(sleeper)

    def set(self, when: float) -> None:
        """Put the clock at `when`, forwards or back, for testing a clock that moved."""
        self._now = when
        self._wake_what_is_due()

    def advance(self, seconds: float) -> None:
        """Move the clock forward without recording a sleep, waking anything now due."""
        self._now += seconds
        self._wake_what_is_due()

    def _wake_what_is_due(self) -> None:
        for due, event in list(self._sleepers):
            if due <= self._now:
                event.set()

    async def wait_for_sleep(self, count: int = 1) -> None:
        """Yield until `count` sleeps have been started, so a test can advance past them.

        Yields to the event loop rather than waiting on the clock, so it is ordering that
        is being waited on and not time.

        Raises:
            AssertionError: If they have not started after `_SETTLE_TURNS` loop turns,
                which is a test waiting for something that is never going to happen.
        """
        for _ in range(_SETTLE_TURNS):
            if len(self.slept) >= count:
                return
            await asyncio.sleep(0)
        raise AssertionError(f"expected {count} sleeps, saw {len(self.slept)}: {self.slept}")


class FakeKeyValueStore:
    """A dictionary-backed `KeyValueStore`."""

    def __init__(self) -> None:
        self._items: dict[str, Any] = {}

    async def get(self, key: str) -> Any:
        """Return the stored value, or None."""
        return self._items.get(key)

    async def put(self, key: str, value: Any) -> None:
        """Store `value` under `key`."""
        self._items[key] = value

    async def delete(self, key: str) -> None:
        """Remove `key` if present."""
        self._items.pop(key, None)


@dataclass(frozen=True, slots=True)
class RecordedEvent:
    """A span or event captured by `FakeTracer`."""

    kind: str
    name: str
    attributes: dict[str, object]


class FakeTracer:
    """A tracer that records instead of exporting, and never raises."""

    def __init__(self) -> None:
        self.recorded: list[RecordedEvent] = []

    @contextlib.contextmanager
    def span(self, name: str, **attributes: object) -> Iterator[None]:
        """Record a span around the wrapped block."""
        self.recorded.append(RecordedEvent("span", name, attributes))
        yield

    def event(self, name: str, **attributes: object) -> None:
        """Record a point-in-time event."""
        self.recorded.append(RecordedEvent("event", name, attributes))

    def names(self) -> list[str]:
        """Return the recorded names in order, for concise assertions."""
        return [r.name for r in self.recorded]


@dataclass(frozen=True, slots=True)
class MetricPoint:
    """One counter increment captured by `FakeMeter`."""

    name: str
    value: float
    dimensions: dict[str, str]


class FakeMeter:
    """A meter that records instead of exporting, and never raises."""

    def __init__(self) -> None:
        self.points: list[MetricPoint] = []

    def count(self, name: str, value: float, **dimensions: str) -> None:
        """Record an increment."""
        self.points.append(MetricPoint(name, value, dimensions))

    def total(self, name: str, **dimensions: str) -> float:
        """Total the increments to `name`, narrowed to points carrying `dimensions`."""
        return sum(
            point.value
            for point in self.points
            if point.name == name and dimensions.items() <= point.dimensions.items()
        )


class FakeBudgetPolicy:
    """A counting budget with a hard token limit, for tests that need no ledger.

    Args:
        limit: Total input plus output tokens permitted across the lifetime of the policy.
    """

    def __init__(self, limit: int = 1_000_000) -> None:
        self.limit = limit
        self.reserved = 0
        self.spent = 0
        self.reservations: list[int] = []
        self.recorded: list[Usage] = []
        self.model_calls = 0
        self.tool_calls = 0
        self.iterations = 0
        self.peer_invocations = 0

    @property
    def resolved(self) -> ResolvedBudget:
        """The ceiling this fake enforces, stated in the kit's own vocabulary."""
        return ResolvedBudget(
            limits=BudgetLimits(max_input_tokens=self.limit),
            sources={"max_input_tokens": BudgetScope.RUN},
        )

    def limits(self) -> BudgetLimits:
        """What is left, as limits."""
        return BudgetLimits(max_input_tokens=max(self.limit - self.spent - self.reserved, 1))

    def child(self) -> FakeBudgetPolicy:
        """The same policy, so a child run spends what the parent has left."""
        return self

    async def reserve(self, estimate: int) -> None:
        """Reserve `estimate` tokens.

        Raises:
            BudgetExceededError: If the reservation would breach `limit`.
        """
        if self.spent + self.reserved + estimate > self.limit:
            raise BudgetExceededError(
                f"reserving {estimate} would exceed limit {self.limit} "
                f"(spent {self.spent}, reserved {self.reserved})",
                breached="max_input_tokens",
                scope=BudgetScope.RUN,
                limit=Decimal(self.limit),
                consumed=Decimal(self.spent),
                remaining=Decimal(max(self.limit - self.spent, 0)),
            )
        self.reserved += estimate
        self.reservations.append(estimate)

    async def record(
        self,
        usage: Usage,
        *,
        model_calls: int = 0,
        tool_calls: int = 0,
        iterations: int = 0,
        peer_invocations: int = 0,
    ) -> None:
        """Record consumption and release the outstanding reservation."""
        self.recorded.append(usage)
        self.spent += usage.input_tokens + usage.output_tokens
        self.model_calls += model_calls
        self.tool_calls += tool_calls
        self.iterations += iterations
        self.peer_invocations += peer_invocations
        self.reserved = 0

    def check(self) -> BudgetDecision:
        """Whether there is room left."""
        if self.spent > self.limit:
            return BudgetDecision(
                permitted=False,
                breached="max_input_tokens",
                scope=BudgetScope.RUN,
                limit=Decimal(self.limit),
                consumed=Decimal(self.spent),
            )
        return BudgetDecision(permitted=True)


class FakeTenantLedger:
    """A tenant ledger held in one process, for tests that need no shared store.

    Args:
        reachable: Whether the store answers. `False` is how a test exercises the
            fail-closed path without breaking a real one.
    """

    def __init__(self, reachable: bool = True) -> None:
        self.reachable = reachable
        self.totals: dict[tuple[str, str], Consumed] = {}

    async def total(self, tenant: str, window: str) -> Consumed:
        """What `tenant` has spent in `window`.

        Raises:
            BudgetUnavailableError: If this ledger was built unreachable.
        """
        self._answer_or_refuse(tenant)
        return self.totals.get((tenant, window), Consumed())

    async def consume(self, tenant: str, window: str, spent: Consumed) -> Consumed:
        """Add `spent` to the total and return it.

        Raises:
            BudgetUnavailableError: If this ledger was built unreachable.
        """
        self._answer_or_refuse(tenant)
        running = self.totals.get((tenant, window), Consumed()) + spent
        self.totals[(tenant, window)] = running
        return running

    def _answer_or_refuse(self, tenant: str) -> None:
        if not self.reachable:
            raise BudgetUnavailableError(f"the ledger for {tenant} is not answering")


class ScriptedProvider:
    """A provider that replays a fixed script, so a loop test needs no network.

    An entry that is an exception is raised rather than returned, which is how a test
    exercises a provider failure without a transport.

    Args:
        responses: What to return, in order.
        name: The provider name recorded on the run.
        capabilities: What this fake declares it can do. Defaults to `CAPABLE`, which
            declares everything except structured output — the case the prompt-side
            fallback has to be tested against.
    """

    def __init__(
        self,
        *responses: ModelResponse | BaseException,
        name: str = "scripted",
        capabilities: ModelCapabilities | None = None,
    ) -> None:
        self._responses = deque(responses)
        self._name = name
        self._capabilities = capabilities if capabilities is not None else CAPABLE
        self.requests: list[ModelRequest] = []

    @property
    def name(self) -> str:
        """The provider name."""
        return self._name

    @property
    def capabilities(self) -> ModelCapabilities:
        """What this fake declares."""
        return self._capabilities

    def count_tokens(self, messages: Sequence[Message]) -> int:
        """Count by characters, the estimate a provider without a tokeniser would give."""
        return estimate_tokens(messages)

    async def complete(self, request: ModelRequest) -> ModelResponse:
        """Return the next scripted response, or raise the next scripted exception.

        Raises:
            AssertionError: If the loop called more times than the script allows, which
                is a runaway loop and must fail the test rather than hang it.
        """
        self.requests.append(request)
        if not self._responses:
            raise AssertionError(
                f"{self._name} was called {len(self.requests)} times; the script has "
                f"{len(self.requests) - 1} responses"
            )
        nxt = self._responses.popleft()
        if isinstance(nxt, BaseException):
            raise nxt
        return nxt

    async def stream(self, request: ModelRequest) -> AsyncIterator[StreamEvent]:
        """Replay the next scripted response a piece at a time, ending with `StreamEnd`.

        The same script drives both paths, so a test asserting that the streamed and
        buffered views of a run agree is asserting about the runtime rather than about
        two fakes that were written to agree.

        Raises:
            CapabilityError: If this fake does not declare `streaming`.
        """
        self._capabilities.require(Capability.STREAMING, provider=self._name, model=request.model)
        return _replayed(await self.complete(request))


class StallingProvider:
    """A provider that answers a script and then stops answering at all.

    The provider a cancellation test needs: something in flight that will not finish on
    its own. `entered` says the stall has been reached, so a test can cancel at the point
    it means to rather than hoping.

    Args:
        responses: What to return before stalling, in order.
        name: The provider name recorded on the run.
        ignores_cancellation: How many cancellations to swallow before stopping. Above
            zero this is the provider that keeps streaming after the caller has gone —
            the case the kit must drop rather than wait for.
        capabilities: What this fake declares it can do.
    """

    def __init__(
        self,
        *responses: ModelResponse,
        name: str = "stalling",
        ignores_cancellation: int = 0,
        capabilities: ModelCapabilities | None = None,
    ) -> None:
        self._responses = deque(responses)
        self._name = name
        self._capabilities = capabilities if capabilities is not None else CAPABLE
        self._ignores = ignores_cancellation
        self._released = asyncio.Event()
        self.entered = asyncio.Event()
        self.calls = 0

    @property
    def name(self) -> str:
        """The provider name."""
        return self._name

    @property
    def capabilities(self) -> ModelCapabilities:
        """What this fake declares."""
        return self._capabilities

    def count_tokens(self, messages: Sequence[Message]) -> int:
        """Count by characters, the estimate a provider without a tokeniser would give."""
        return estimate_tokens(messages)

    def release(self) -> None:
        """Let a stalled call return, so an abandoned task can finish rather than linger."""
        self._released.set()

    async def complete(self, request: ModelRequest) -> ModelResponse:  # noqa: ARG002 — the script ignores it
        """Return the next scripted response, or stall until released or cancelled."""
        self.calls += 1
        if self._responses:
            return self._responses.popleft()
        self.entered.set()
        while True:
            try:
                await self._released.wait()
            except asyncio.CancelledError:
                if not self._ignores:
                    raise
                self._ignores -= 1
            else:
                break
        return ModelResponse(content="answered after all")

    async def stream(self, request: ModelRequest) -> AsyncIterator[StreamEvent]:
        """Stream the scripted answer, or stall mid-stream until released or cancelled."""
        return _replayed(await self.complete(request))


class FakeToolRegistry:
    """An in-memory `ToolRegistry` backed by plain callables.

    Args:
        tools: Callables by tool name. A callable that raises is how a test exercises a
            tool failure.
        declarations: Declarations by tool name, where a test cares about the schema.
    """

    def __init__(
        self,
        tools: Mapping[str, Callable[..., Any]] | None = None,
        declarations: Mapping[str, ToolDeclaration] | None = None,
    ) -> None:
        self._tools = dict(tools or {})
        self._declarations = dict(declarations or {})
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def declarations(self) -> tuple[ToolDeclaration, ...]:
        """Return declarations in registration order, which is the cacheable order."""
        return tuple(
            self._declarations.get(name, ToolDeclaration(name=name)) for name in self._tools
        )

    async def invoke(self, name: str, arguments: Any) -> Any:
        """Invoke tool `name`.

        Raises:
            ToolExecutionError: If no tool is registered under `name`.
        """
        self.calls.append((name, dict(arguments)))
        if name not in self._tools:
            raise ToolExecutionError(f"no tool named {name!r} is registered")
        result = self._tools[name](**arguments)
        if inspect.isawaitable(result):
            return await result
        return result


class FakeGuardrail:
    """A guardrail with a fixed verdict on both stages.

    Args:
        name: Its recorded name.
        allow: Whether it permits what it is shown.
        redacts: What it hands on instead, for the redaction path. Overrides `allow`.
        raises: An exception to raise instead of deciding, for the fail-closed path.
        code: The machine-readable reason it gives when it blocks or redacts.
    """

    def __init__(
        self,
        name: str = "fake",
        *,
        allow: bool = True,
        redacts: str | None = None,
        raises: Exception | None = None,
        code: str = "fake_refusal",
    ) -> None:
        self._name = name
        self._allow = allow
        self._redacts = redacts
        self._raises = raises
        self._code = code
        self.checked: list[Any] = []

    @property
    def name(self) -> str:
        """The guardrail name."""
        return self._name

    async def check_input(self, content: str) -> GuardResult:
        """Return the fixed verdict about what is going to the model."""
        return self._verdict(content)

    async def check_output(self, content: str) -> GuardResult:
        """Return the same fixed verdict about what is coming back."""
        return self._verdict(content)

    def _verdict(self, content: str) -> GuardResult:
        """One answer, whichever stage asked."""
        self.checked.append(content)
        if self._raises is not None:
            raise self._raises
        if self._redacts is not None:
            return GuardResult.redacted(self._redacts, code=self._code)
        if not self._allow:
            return GuardResult.blocked(code=self._code)
        return GuardResult.allow()


class FakeSecrets:
    """A `SecretProvider` backed by a mapping, so a test never touches the environment.

    Args:
        secrets: Values by variable name. A name that is absent answers `None`, which is
            how the unconfigured path is exercised.
    """

    def __init__(self, secrets: Mapping[str, str] | None = None) -> None:
        self._secrets = dict(secrets or {})
        self.asked: list[str] = []

    def secret(self, name: str) -> str | None:
        """Return the value recorded for `name`, or `None`."""
        self.asked.append(name)
        return self._secrets.get(name)


def _pieces(text: str) -> list[str]:
    """A scripted answer cut where a vendor would cut it — word by word, spaces kept."""
    return re.findall(r"\S+\s*", text)


async def _replayed(response: ModelResponse) -> AsyncIterator[StreamEvent]:
    """One already-decided response, told as the stream a vendor would have sent."""
    for piece in _pieces(response.content):
        yield TextDelta(text=piece)
    if response.reasoning:
        yield ReasoningDelta(text=response.reasoning)
    for index, call in enumerate(response.tool_calls):
        yield ToolCallDelta(
            index=index,
            id=call.id,
            name=call.name,
            arguments=json.dumps(call.arguments, sort_keys=True),
        )
    yield UsageDelta(usage=response.usage)
    yield StreamEnd(response=response)
