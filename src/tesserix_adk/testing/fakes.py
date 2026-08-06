"""In-memory implementations of the core protocols, for tests that must not touch a network.

These are the reference implementations the conformance suite is written against.
They are deliberately simple: a fake that acquires its own behaviour stops being a
control and becomes a second thing to debug.
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from tesserix_adk.core.errors import BudgetExceededError, ToolExecutionError
from tesserix_adk.runtime import ModelRequest, ModelResponse, ToolDeclaration

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping

__all__ = [
    "BudgetExceededError",
    "FakeBudgetPolicy",
    "FakeClock",
    "FakeGuardrail",
    "FakeMemoryStore",
    "FakeToolRegistry",
    "FakeTracer",
    "RecordedEvent",
    "ScriptedProvider",
    "StallingProvider",
    "ToolExecutionError",
]

# A loop turn is free; a test that waits more than this is waiting for the wrong thing.
_SETTLE_TURNS = 100


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

    def advance(self, seconds: float) -> None:
        """Move the clock forward without recording a sleep, waking anything now due."""
        self._now += seconds
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


class FakeMemoryStore:
    """A dictionary-backed `MemoryStore`."""

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


class FakeBudgetPolicy:
    """A counting budget with a hard limit.

    Args:
        limit: Total units permitted across the lifetime of the policy.
    """

    def __init__(self, limit: int = 1_000_000) -> None:
        self.limit = limit
        self.reserved = 0
        self.spent = 0
        self.reservations: list[int] = []

    async def reserve(self, estimate: int) -> None:
        """Reserve `estimate` units.

        Raises:
            BudgetExceededError: If the reservation would breach `limit`.
        """
        if self.spent + self.reserved + estimate > self.limit:
            raise BudgetExceededError(
                f"reserving {estimate} would exceed limit {self.limit} "
                f"(spent {self.spent}, reserved {self.reserved})"
            )
        self.reserved += estimate
        self.reservations.append(estimate)

    async def record(self, actual: int) -> None:
        """Record `actual` consumption and release the outstanding reservation."""
        self.spent += actual
        self.reserved = 0


class ScriptedProvider:
    """A provider that replays a fixed script, so a loop test needs no network.

    An entry that is an exception is raised rather than returned, which is how a test
    exercises a provider failure without a transport.

    Args:
        responses: What to return, in order.
        name: The provider name recorded on the run.
    """

    def __init__(self, *responses: ModelResponse | BaseException, name: str = "scripted") -> None:
        self._responses = deque(responses)
        self._name = name
        self.requests: list[ModelRequest] = []

    @property
    def name(self) -> str:
        """The provider name."""
        return self._name

    async def complete(self, request: Any) -> Any:
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

    async def stream(self, request: Any) -> Any:
        """Not scripted. Streaming has its own epic (#38)."""
        raise NotImplementedError("ScriptedProvider does not stream; see #38")


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
    """

    def __init__(
        self,
        *responses: ModelResponse,
        name: str = "stalling",
        ignores_cancellation: int = 0,
    ) -> None:
        self._responses = deque(responses)
        self._name = name
        self._ignores = ignores_cancellation
        self._released = asyncio.Event()
        self.entered = asyncio.Event()
        self.calls = 0

    @property
    def name(self) -> str:
        """The provider name."""
        return self._name

    def release(self) -> None:
        """Let a stalled call return, so an abandoned task can finish rather than linger."""
        self._released.set()

    async def complete(self, request: Any) -> Any:  # noqa: ARG002 — the script ignores it
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

    async def stream(self, request: Any) -> Any:
        """Not scripted. Streaming has its own epic (#38)."""
        raise NotImplementedError("StallingProvider does not stream; see #38")


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
    """A guardrail with a fixed verdict.

    Args:
        name: Its recorded name.
        allow: Whether it permits what it is shown.
        raises: An exception to raise instead of deciding, for the fail-closed path.
    """

    def __init__(
        self, name: str = "fake", *, allow: bool = True, raises: Exception | None = None
    ) -> None:
        self._name = name
        self._allow = allow
        self._raises = raises
        self.checked: list[Any] = []

    @property
    def name(self) -> str:
        """The guardrail name."""
        return self._name

    async def check(self, subject: Any) -> Any:
        """Return the fixed verdict, or raise the configured failure."""
        self.checked.append(subject)
        if self._raises is not None:
            raise self._raises
        return self._allow
