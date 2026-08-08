"""What an agent may call, and how much it may spend calling it, declared in one place.

A tool an agent may call is usually a filtered dictionary built where the agent is, and the
filter is enforced wherever the dispatch code happens to check. Two things follow: a tool
added for one agent becomes reachable by every agent sharing the process, and nothing bounds
a call that never returns — a hanging HTTP request inside a tool stalls the run until some
outer request timeout notices.

Here the allowlist is resolved once, into an `AgentToolView` that cannot widen: a tool
registered after the view was built is not in it, and a name outside it is refused before
anything is dispatched. Ceilings and widths are declared beside it, so what a call may cost
is configuration rather than a check somebody remembered to write.

Every name exported here is semver-governed: it appears in `docs/api-surface.txt`, so a
change to it shows up in a pull request's diff and follows `docs/versioning.md`.
"""

from __future__ import annotations

import asyncio
import contextlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from tesserix_adk.core.config import ConcurrencyConfig
from tesserix_adk.core.errors import (
    ConfigurationError,
    ToolDefinitionError,
    ToolNotFoundError,
    ToolNotPermittedError,
    ToolTimedOutError,
)
from tesserix_adk.core.provider import ToolDeclaration
from tesserix_adk.runtime.loop import SystemClock

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable, Iterable, Mapping, Sequence

    from tesserix_adk.core.protocols import Clock
    from tesserix_adk.runtime.blocking import WorkerPool
    from tesserix_adk.tools.decorator import Tool

__all__ = ["AgentToolView", "ToolCallSpan", "ToolRegistry"]

_ABANDON_AFTER = 30.0
"""How long a cancelled call is given to stop before it is left to finish alone."""


@dataclass(frozen=True, slots=True)
class ToolCallSpan:
    """One invocation, as it is recorded. Never the arguments and never the result.

    A tool's arguments are a booking reference, a customer id or a query someone typed, and
    a span is the one artefact that leaves the process for a backend nobody in this repo
    operates. What is worth exporting is what happened, not what it happened to.

    Args:
        tool: What was called, or what a refusal names.
        agent: Whose view called it, empty where the registry was called directly.
        permitted: The permission decision. Recorded on every call, so a call with no
            decision on its span is a bug rather than an omission.
        outcome: `ok`, `refused`, `not_found`, `timed_out` or `error`.
        duration_seconds: How long it took on the injected clock.
        failure: The exception class where one was raised, never its message.
        abandoned: Whether the call ignored cancellation and was left running.
    """

    tool: str
    agent: str = ""
    permitted: bool = True
    outcome: str = "ok"
    duration_seconds: float = 0.0
    failure: str = ""
    abandoned: bool = False


class ToolRegistry:
    """The tools a process holds, the ceilings on them, and the views agents call through.

    Registration is process-wide and views are per agent: registering a tool makes it
    available to be allowed, never allowed by default.

    Args:
        tools: What to register, in the order the model should be told about them.
        concurrency: How wide calls may run — a registry-wide width and tighter per-tool
            ones. The tightest declared bound decides.
        timeouts: Ceilings by tool name, overriding what the tool declared. A registry
            operator knows what the partner behind a tool is doing today; the author who
            wrote `@tool(timeout=...)` knew what it was doing in general.
        abandon_after: How long a timed-out call is given to honour cancellation before it
            is left running and its result discarded. Waiting indefinitely for a body that
            swallows `CancelledError` turns one stuck call into a stuck run.
        clock: Where elapsed time and ceilings come from. Injected so a test for a timeout
            does not spend the timeout.
        workers: Where synchronous tool bodies run. Without one they still leave the event
            loop, but nothing bounds how many run at once.

    Example:
        >>> import asyncio
        >>> from tesserix_adk.tools import tool
        >>> @tool(name="fare_for")
        ... def fare_for(leg: str) -> str:
        ...     '''Price a leg.'''
        ...     return f"{leg}: 40 EUR"
        >>> registry = ToolRegistry((fare_for,))
        >>> view = registry.view(allow=("fare_for",), agent="planner")
        >>> asyncio.run(view.invoke("fare_for", {"leg": "Osaka"}))
        'Osaka: 40 EUR'
    """

    def __init__(
        self,
        tools: Iterable[Tool[Any, Any]] = (),
        *,
        concurrency: ConcurrencyConfig | None = None,
        timeouts: Mapping[str, float] | None = None,
        abandon_after: float = _ABANDON_AFTER,
        clock: Clock | None = None,
        workers: WorkerPool | None = None,
    ) -> None:
        self._tools: dict[str, Tool[Any, Any]] = {}
        self._origins: dict[str, str] = {}
        self._concurrency = concurrency or ConcurrencyConfig()
        self._timeouts = dict(timeouts or {})
        self._abandon_after = abandon_after
        self._clock: Clock = clock or SystemClock()
        self._workers = workers
        self._observers: list[Callable[[ToolCallSpan], None]] = []
        self._lanes = _Lanes(self._concurrency)
        for registered in tools:
            self.register(registered)

    @property
    def names(self) -> tuple[str, ...]:
        """Every registered name, in registration order."""
        return tuple(self._tools)

    def register(self, tool: Tool[Any, Any], *, origin: str = "") -> None:
        """Add `tool`, refusing a name another tool already answers to.

        Registering the same tool twice is not a conflict — a module imported from two
        places is one tool — but two different tools under one name is: the model reaches
        whichever the registry happened to keep, and which one that is depends on import
        order.

        Args:
            tool: What to register.
            origin: Where it came from, named in the conflict if there is one. Defaults to
                the defining module and qualified name.

        Raises:
            ToolDefinitionError: If a different tool is already registered under the name.
        """
        came_from = origin or _origin_of(tool)
        held = self._tools.get(tool.name)
        if held is tool:
            return
        if held is not None:
            raise ToolDefinitionError(
                f"{tool.name} is already registered from {self._origins[tool.name]}, and "
                f"{came_from} claims the same name. The model would reach whichever the "
                f"registry happened to keep. Rename one, or namespace its source.",
                tool=tool.name,
            )
        self._tools[tool.name] = tool
        self._origins[tool.name] = came_from
        if not tool.parallel_safe:
            self._lanes.serialise(tool.name)

    def observe(self, observer: Callable[[ToolCallSpan], None]) -> None:
        """Call `observer` with a span for every invocation, permitted or refused."""
        self._observers.append(observer)

    def resolve(self, name: str) -> Tool[Any, Any]:
        """Return the tool registered under `name`.

        Raises:
            ToolNotFoundError: If nothing is registered under it.
        """
        found = self._tools.get(name)
        if found is None:
            raise ToolNotFoundError(name, known=self.names)
        return found

    def declarations(self) -> tuple[ToolDeclaration, ...]:
        """Every tool as the model is told about it, in registration order.

        Order is part of the cacheable prompt prefix, so it does not vary between calls.
        """
        return tuple(_declared(tool) for tool in self._tools.values())

    def timeout_for(self, name: str) -> float | None:
        """The ceiling this registry holds `name` to, or `None` where it declared none."""
        override = self._timeouts.get(name)
        if override is not None:
            return override
        return self._tools[name].timeout if name in self._tools else None

    def view(self, *, allow: Sequence[str], agent: str = "") -> AgentToolView:
        """Resolve an allowlist into the fixed set of tools one agent may call.

        Args:
            allow: The names that agent may call. Duplicates are the same permission
                twice, not two permissions.
            agent: Whose view it is, named in refusals and on spans.

        Raises:
            ToolNotFoundError: If the allowlist names a tool nobody registered. A
                misspelling silently granting nothing is a run that fails on its first
                turn, in production, for a reason nobody can see from the config.
            ConfigurationError: If the allowlist is empty. An agent with no tools is
                either a mistake or an agent that should not have been given a registry.
        """
        named = tuple(dict.fromkeys(allow))
        if not named:
            raise ConfigurationError(
                f"the allowlist for {agent or 'this agent'} is empty, so it could call "
                f"nothing. An agent that is meant to have no tools should be built "
                f"without a registry rather than with an empty view of one."
            )
        for name in named:
            self.resolve(name)
        return AgentToolView(self, named, agent)

    async def invoke(self, name: str, arguments: Any) -> Any:  # noqa: ANN401 — the tool's own
        """Call `name` with `arguments`, inside its ceiling and its lanes.

        Raises:
            ToolNotFoundError: If nothing is registered under `name`.
            ToolArgumentValidationError: If the arguments do not match the tool's schema.
            ToolTimedOutError: If the call outran the ceiling declared for it.
            Exception: Whatever the tool raised, unchanged.
        """
        return await self._called(name, arguments, agent="")

    async def _called(self, name: str, arguments: Any, *, agent: str) -> Any:  # noqa: ANN401
        """One invocation, recorded whatever it did."""
        started = self._clock.now()
        try:
            tool = self.resolve(name)
        except ToolNotFoundError:
            self._record(ToolCallSpan(tool=name, agent=agent, outcome="not_found"))
            raise
        try:
            async with self._lanes.held(name):
                result = await self._inside_its_ceiling(tool, arguments)
        except _Overran as overran:
            self._record(
                ToolCallSpan(
                    tool=name,
                    agent=agent,
                    outcome="timed_out",
                    duration_seconds=self._clock.now() - started,
                    failure="ToolTimedOutError",
                    abandoned=overran.abandoned,
                )
            )
            raise ToolTimedOutError(name, overran.seconds) from None
        except Exception as failure:
            self._record(
                ToolCallSpan(
                    tool=name,
                    agent=agent,
                    outcome="error",
                    duration_seconds=self._clock.now() - started,
                    failure=type(failure).__name__,
                )
            )
            raise
        self._record(
            ToolCallSpan(tool=name, agent=agent, duration_seconds=self._clock.now() - started)
        )
        return result

    async def _inside_its_ceiling(
        self,
        tool: Tool[Any, Any],
        arguments: Any,  # noqa: ANN401 — the tool's own
    ) -> Any:  # noqa: ANN401 — the tool's own
        """Hold the call to its ceiling, cancelling it for real and never waiting forever.

        Raises:
            _Overran: If the ceiling elapsed first, saying whether the body stopped.
        """
        ceiling = self.timeout_for(tool.name)
        dispatched = tool.invoke(arguments, workers=self._workers)
        if ceiling is None:
            return await dispatched
        work = asyncio.ensure_future(dispatched)
        if await self._first(work, ceiling):
            return work.result()
        work.cancel()
        stopped = await self._first(work, self._abandon_after)
        if not stopped:
            # A body that swallows cancellation is not going to stop because it is asked
            # again. Its late result is discarded rather than injected into a run that has
            # already been told the call failed.
            work.add_done_callback(_discard)
        raise _Overran(ceiling, abandoned=not stopped)

    async def _first(self, work: asyncio.Future[Any], seconds: float) -> bool:
        """Whether `work` finished before `seconds` elapsed on the injected clock."""
        timer = asyncio.ensure_future(self._clock.sleep(seconds))
        done, _ = await asyncio.wait([work, timer], return_when=asyncio.FIRST_COMPLETED)
        timer.cancel()
        await asyncio.gather(timer, return_exceptions=True)
        return work in done

    def _record(self, span: ToolCallSpan) -> None:
        """Hand the span to every observer. One that fails does not fail the call."""
        for observer in self._observers:
            with contextlib.suppress(Exception):
                observer(span)


@dataclass(frozen=True, slots=True)
class AgentToolView:
    """The fixed set of tools one agent may call, resolved once and never widened.

    Built by `ToolRegistry.view`. Frozen on purpose: an allowlist that can be appended to
    mid-run is an allowlist whose value at the moment of a call nobody can state.

    Args:
        registry: Where the tools and their ceilings live.
        names: What this agent may call, in registration order.
        agent: Whose view it is, named in refusals and on spans.
    """

    registry: ToolRegistry
    names: tuple[str, ...]
    agent: str = ""

    def declarations(self) -> tuple[ToolDeclaration, ...]:
        """The permitted tools as the model is told about them, in a stable order."""
        return tuple(_declared(self.registry.resolve(name)) for name in self.names)

    def resolve(self, name: str) -> Tool[Any, Any]:
        """The tool behind `name`, if this agent may call it.

        Raises:
            ToolNotPermittedError: If the allowlist does not name it.
            ToolNotFoundError: If nothing is registered under `name`.
        """
        if name not in self.names:
            raise ToolNotPermittedError(name, agent=self.agent, permitted=self.names)
        return self.registry.resolve(name)

    async def invoke(self, name: str, arguments: Any) -> Any:  # noqa: ANN401 — the tool's own
        """Call `name` on this agent's behalf, if this agent may.

        Raises:
            ToolNotPermittedError: If the allowlist does not name it. The tool is not
                called: an allowlist enforced after dispatch is a side effect that has
                already landed by the time the decision is recorded.
            ToolNotFoundError: If nothing is registered under `name`.
            ToolTimedOutError: If the call outran the ceiling declared for it.
            Exception: Whatever the tool raised, unchanged.
        """
        if name not in self.names:
            self.registry._record(
                ToolCallSpan(tool=name, agent=self.agent, permitted=False, outcome="refused")
            )
            raise ToolNotPermittedError(name, agent=self.agent, permitted=self.names)
        return await self.registry._called(name, arguments, agent=self.agent)


class _Lanes:
    """The semaphores a registry's calls stand in, rebuilt if the event loop changes.

    A registry built at import time and used by two `asyncio.run` calls would otherwise
    hold semaphores bound to a loop that has closed, and the second run fails on a lane
    rather than on anything it did.
    """

    def __init__(self, config: ConcurrencyConfig) -> None:
        self._config = config
        self._serial: set[str] = set()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._all = asyncio.Semaphore(config.max_concurrent_tools)
        self._per_tool: dict[str, asyncio.Semaphore] = {}

    def serialise(self, tool: str) -> None:
        """Hold `tool` to one call at a time, whatever width the config gave it."""
        self._serial.add(tool)
        self._loop = None

    @contextlib.asynccontextmanager
    async def held(self, tool: str) -> AsyncIterator[None]:
        """Hold the registry's lane and the tool's, taken in that order by every call."""
        self._for_this_loop()
        own = self._per_tool.get(tool)
        async with self._all:
            if own is None:
                yield
                return
            async with own:
                yield

    def _for_this_loop(self) -> None:
        running = asyncio.get_running_loop()
        if self._loop is running:
            return
        self._loop = running
        self._all = asyncio.Semaphore(self._config.max_concurrent_tools)
        self._per_tool = {
            name: asyncio.Semaphore(1 if name in self._serial else width)
            for name, width in self._config.per_tool.items()
        }
        self._per_tool.update(
            {name: asyncio.Semaphore(1) for name in self._serial if name not in self._per_tool}
        )


class _Overran(Exception):  # noqa: N818 — control flow, not an error the caller sees
    """A ceiling that elapsed, carrying whether the body stopped when it was asked to."""

    def __init__(self, seconds: float, *, abandoned: bool) -> None:
        self.seconds = seconds
        self.abandoned = abandoned
        super().__init__(f"the call outran its ceiling of {seconds:g}s")


def _declared(tool: Tool[Any, Any]) -> ToolDeclaration:
    """The tool as the model is told about it."""
    return ToolDeclaration(
        name=tool.name,
        description=tool.description,
        parameters=tool.parameters_schema,
        parallel_safe=tool.parallel_safe,
    )


def _origin_of(tool: Tool[Any, Any]) -> str:
    """Where a tool was defined, for naming both sides of a conflict."""
    function = tool.function
    module = getattr(function, "__module__", "?")
    return f"{module}.{getattr(function, '__qualname__', tool.name)}"


def _discard(work: asyncio.Future[Any]) -> None:
    """Consume an abandoned call's outcome so nothing is reported as never retrieved."""
    with contextlib.suppress(BaseException):
        work.result()
