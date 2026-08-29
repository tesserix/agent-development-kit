"""Crossing between the async core and the synchronous world, in both directions.

The kit is async all the way down, because a run is mostly waiting: on a provider, on a
tool, on a ledger. But consumers are not. Scripts, notebooks and synchronous services all
have to reach the same runtime, and the usual improvisations — `asyncio.run` inside a
request handler, a blocking client called from a coroutine, a second loop nested in the
first — fail as stalls and as 'this event loop is already running'.

So the two crossings are named. Going in, `drive` runs a coroutine from a caller with no
loop and refuses, by name, from one that has a loop already. Going out, a `WorkerPool`
takes a blocking body off the loop onto a bounded set of threads, and a `LoopMonitor`
catches the bodies that never asked. Identity crosses with the work: an `Ambient` carries
the run, the tenant and the caller's switch through a contextvar, copied per call so two
runs sharing a thread cannot read each other's.

Every name exported here is semver-governed: it appears in `docs/api-surface.txt`, so a
change to it shows up in a pull request's diff and follows `docs/versioning.md`.
"""

from __future__ import annotations

import asyncio
import contextvars
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, suppress
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from tesserix_adk.core.errors import EventLoopStalledError, RunningLoopError, WorkersBusyError

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Iterator, Mapping

    from tesserix_adk.core.protocols import BudgetPolicy
    from tesserix_adk.runtime.cancellation import CancellationToken

__all__ = [
    "Ambient",
    "LoopMonitor",
    "WorkerPool",
    "Workers",
    "carrying",
    "current_ambient",
    "drive",
]

_AMBIENT: contextvars.ContextVar[Ambient | None] = contextvars.ContextVar(
    "tesserix_adk_ambient", default=None
)


@dataclass(frozen=True, slots=True)
class Ambient:
    """Who the work in hand belongs to, carried rather than passed.

    A blocking body takes the arguments the model chose, so anything the model did not
    choose — the tenant, the run, the caller's switch — has to arrive another way or be
    dropped at the first hop onto a thread.

    Args:
        run_id: The run this work belongs to.
        tenant: The isolation boundary it must not be attributed outside of.
        user: The acting principal, where there is one.
        cancellation: The caller's switch, where the work can cooperate with it. A thread
            cannot be interrupted, so a long body is told and decides.
        idempotency_key: The key identifying the side effect in hand, where one was derived.
        scopes: Effective delegated authority. It has already been narrowed to the agent.
        trace: W3C propagation fields carried from the caller, never credentials.
        budget: The parent ledger shared with nested foreign work, never a fresh ceiling.
    """

    run_id: str
    tenant: str
    user: str | None = None
    cancellation: CancellationToken | None = None
    idempotency_key: str | None = None
    scopes: tuple[str, ...] = ()
    trace: Mapping[str, str] = field(default_factory=dict)
    budget: BudgetPolicy | None = None

    def raise_if_cancelled(self) -> None:
        """Raise if the run has been cancelled, otherwise do nothing.

        Raises:
            CancelledError: If the caller's switch has been flipped. Nothing is raised
                where there is no switch: absence is not consent to keep going, it is
                simply nothing to check.
        """
        if self.cancellation is not None:
            self.cancellation.raise_if_cancelled()


def current_ambient() -> Ambient | None:
    """Return the ambient of the work in hand, or `None` outside a run.

    `None` rather than an invented default: a tenant nobody set is the bug this exists to
    make visible, not a value to guess.
    """
    return _AMBIENT.get()


@contextmanager
def carrying(ambient: Ambient) -> Iterator[None]:
    """Bind `ambient` for the duration of the block, restoring what was there before.

    Restoring rather than clearing, because a delegated run has to give the parent's
    context back when it returns.
    """
    token = _AMBIENT.set(ambient)
    try:
        yield
    finally:
        _AMBIENT.reset(token)


@dataclass(frozen=True, slots=True)
class Workers:
    """How many blocking bodies may run at once, and how long one may wait for a turn.

    Args:
        size: Threads. Bounded on purpose: an unbounded pool turns a slow dependency into
            a thread per in-flight request and fails on memory rather than on the queue.
        queue_seconds: How long a body waits for a worker before it is refused. A refusal
            names the pressure; an unbounded wait hides it in latency.

    Raises:
        ValueError: If `size` is not positive or `queue_seconds` is negative.
    """

    size: int = 8
    queue_seconds: float = 30.0

    def __post_init__(self) -> None:
        """Refuse a pool that cannot run anything, or a wait that cannot be waited."""
        if self.size < 1:
            raise ValueError("a worker pool needs at least one worker")
        if self.queue_seconds < 0:
            raise ValueError("queue_seconds cannot be negative")


class WorkerPool:
    """Where a blocking body runs, so that it is not run on the event loop.

    Args:
        workers: The bound. Defaults to `Workers()`.

    A tool declares itself blocking by running its body here — `await pool.call("lookup",
    lambda: legacy_client.fetch(url))` — rather than by a flag the runtime has to trust.

    Example:
        >>> pool = WorkerPool(Workers(size=2))
        >>> pool.close()
    """

    def __init__(self, workers: Workers | None = None) -> None:
        self._workers = workers or Workers()
        self._pool = ThreadPoolExecutor(
            max_workers=self._workers.size, thread_name_prefix="adk-worker"
        )
        self._turns = asyncio.Semaphore(self._workers.size)
        self._closed = False

    def __enter__(self) -> WorkerPool:
        """Return the pool, so its threads have an owner and an end."""
        return self

    def __exit__(self, *_: object) -> None:
        """Close the pool. Bodies already running are left to finish."""
        self.close()

    def close(self) -> None:
        """Stop accepting work. Idempotent, and does not wait on bodies in flight."""
        self._closed = True
        self._pool.shutdown(wait=False)

    async def call[T](self, what: str, body: Callable[[], T]) -> T:
        """Run `body` on a worker and return what it returned.

        The caller's context is copied per call, so the ambient crosses the hop and
        nothing the body sets crosses back or reaches the next body on that thread.

        Cancelling the await stops the waiting, not the thread: a running body has no
        interrupt, so it is abandoned rather than claimed undone. A body that may run
        long should check `current_ambient().raise_if_cancelled()`.

        Raises:
            WorkersBusyError: If the pool is closed, or no worker came free within
                `queue_seconds`.
            Exception: Whatever `body` raised, unchanged.
        """
        if self._closed:
            raise WorkersBusyError(f"{what} was not run: the worker pool is closed")
        try:
            await asyncio.wait_for(self._turns.acquire(), timeout=self._workers.queue_seconds)
        except TimeoutError:
            raise WorkersBusyError(
                f"{what} queued {self._workers.queue_seconds:g}s for one of "
                f"{self._workers.size} workers and gave up rather than growing the pool"
            ) from None
        context = contextvars.copy_context()
        try:
            return await asyncio.get_running_loop().run_in_executor(
                self._pool, lambda: context.run(body)
            )
        finally:
            self._turns.release()


@dataclass(slots=True)
class _Beat:
    """When the loop last turned, as the only thing a blocked loop cannot fake."""

    last: float = 0.0


@dataclass(frozen=True, slots=True)
class LoopMonitor:
    """Catches work that blocked the event loop rather than awaiting on it.

    A blocking body is not a slow run: it is a slow process. Every other run sharing the
    loop pays, and the latency lands where nobody can attribute it. So the loop's own lag
    is measured while the work runs, and work that stopped it is failed by name.

    Args:
        stall_seconds: How far the loop may fall behind before the work is blamed.
        interval: How often the loop is asked whether it is still turning.
        monotonic: The clock. Injected for tests; the loop's lag is real time by nature.
    """

    stall_seconds: float = 1.0
    interval: float = 0.05
    monotonic: Callable[[], float] = field(default=time.monotonic)

    async def watching[T](self, what: str, work: Callable[[], Awaitable[T]]) -> T:
        """Await `work()`, failing it if the loop stopped turning while it ran.

        `work` is a factory rather than an awaitable so that a caller who never gets here
        — a cancelled step, a ceiling already spent — has nothing left un-awaited.

        Raises:
            EventLoopStalledError: If the loop fell more than `stall_seconds` behind.
                Raised only where the work itself succeeded: a body that failed on its
                own terms is reported on its own terms.
            Exception: Whatever `work` raised, unchanged.
        """
        beat = _Beat(self.monotonic())
        heartbeat = asyncio.ensure_future(self._pulse(beat))
        try:
            result = await work()
        finally:
            heartbeat.cancel()
            with suppress(asyncio.CancelledError):
                await heartbeat
        behind = self.monotonic() - beat.last - self.interval
        if behind > self.stall_seconds:
            raise EventLoopStalledError(what, behind)
        return result

    async def _pulse(self, beat: _Beat) -> None:
        """Tick until cancelled. A tick that is late is a loop that was held."""
        while True:
            await asyncio.sleep(self.interval)
            beat.last = self.monotonic()


def drive[T](work: Callable[[], Awaitable[T]], *, sync_name: str, async_name: str) -> T:
    """Run `work()` to completion from a caller that has no event loop.

    A loop of its own rather than `asyncio.run`, which clears the thread's event loop on
    exit and so would break whatever else had set one. `work` is a factory, so a refusal
    leaves no coroutine that was created and never awaited.

    Args:
        work: Builds the coroutine to run. Called only once it is safe to run it.
        sync_name: The helper being called, for the refusal.
        async_name: What to await instead, for the refusal.

    Raises:
        RunningLoopError: If there is already a loop running on this thread. Nesting a
            second one runs two schedulers over one set of tasks; blocking on the first
            deadlocks it against the work it is waiting for.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(work())
        finally:
            with suppress(Exception):
                loop.run_until_complete(loop.shutdown_asyncgens())
            loop.close()
    raise RunningLoopError(sync_name, async_name)
