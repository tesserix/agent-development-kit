"""What the agent called, under whose tenant, and what it did when the call failed.

Tool selection is asserted today by reading a transcript and forming an impression, which
catches the call that is missing and misses the call that ran under the wrong tenant. The
second is the one that matters: an agent acting with broader access than its caller is not
a wrong answer, it is a disclosure, and nothing in a log line makes it visible.

So the calls are recorded as they happen — name, validated arguments, the tenant and user
bound at the moment of dispatch, the idempotency key, and what came back — and the
assertions here read that record. Each failure names the first place the run diverged
rather than printing both sequences and leaving the diff to the reader.

`scoped_run` binds a real run context rather than a mock of one, so a test asserting tenant
propagation is asserting the mechanism production uses. A fixture that merely records the
tenant it was handed would pass for an implementation that never reads it.

Every name exported here is semver-governed: it appears in `docs/api-surface.txt`, so a
change to it shows up in a pull request's diff and follows `docs/versioning.md`.
"""

from __future__ import annotations

import asyncio
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from itertools import zip_longest
from typing import TYPE_CHECKING, Any, NoReturn

from tesserix_adk.core.hooks import ApprovalDecision, digest_of
from tesserix_adk.core.tenancy import TenantContext, tenant_here, tenant_scope
from tesserix_adk.core.tool_access import ToolAllowlist
from tesserix_adk.runtime.approvals import TIMEOUT_IDENTITY
from tesserix_adk.runtime.blocking import Ambient, carrying, current_ambient
from tesserix_adk.tools.context import ToolContext

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping, Sequence

    from tesserix_adk.core.hooks import ApprovalRecord
    from tesserix_adk.core.protocols import ToolRegistry
    from tesserix_adk.core.provider import ToolDeclaration

__all__ = [
    "ApprovalStub",
    "ConcurrencyProbe",
    "RecordedCall",
    "ScopedRun",
    "ToolSpy",
    "approving",
    "assert_context_propagated",
    "assert_idempotency_key_stable",
    "assert_no_tool_called",
    "assert_tool_called_once_with",
    "assert_tool_sequence",
    "denying",
    "failing_tool",
    "peak_concurrency",
    "scoped_run",
    "slow_tool",
]

_MISSING = object()


@dataclass(frozen=True, slots=True)
class RecordedCall:
    """One tool call as it happened, including the context it happened under.

    Args:
        name: The tool that was asked for, spelled the way it was asked for.
        arguments: The arguments it was called with.
        tenant: The tenant bound at dispatch, or None where nothing was bound — which is
            the defect this records rather than an absence to shrug at.
        user: The acting principal, where there was one.
        run_id: The run the call belonged to, where one was bound.
        idempotency_key: The key the call carried, so a retry can be proved not to have
            minted a second one.
        result: What the tool returned, or None where it raised.
        error: What it raised, or None where it returned.
        seconds: How long the call took, measured across the await.
    """

    name: str
    arguments: Mapping[str, Any]
    tenant: str | None = None
    user: str | None = None
    run_id: str | None = None
    idempotency_key: str | None = None
    result: Any = None
    error: BaseException | None = None
    seconds: float = 0.0


class ToolSpy:
    """A `ToolRegistry` that records every call made through it and dispatches unchanged.

    It wraps rather than replaces, so the registry under test keeps its own validation,
    allowlist and failure behaviour: a spy that answered calls itself would prove only that
    the spy works.

    Args:
        registry: The registry to record and dispatch to.

    Example:
        >>> import asyncio
        >>> from tesserix_adk.testing import FakeToolRegistry
        >>> spy = ToolSpy(FakeToolRegistry({"search": lambda **kw: kw}))
        >>> asyncio.run(spy.invoke("search", {"q": "renewals"}))
        {'q': 'renewals'}
        >>> assert_tool_sequence(spy, "search")
    """

    def __init__(self, registry: ToolRegistry) -> None:
        self._registry = registry
        self._calls: list[RecordedCall] = []

    @property
    def calls(self) -> tuple[RecordedCall, ...]:
        """Every call made through the spy, in the order they were dispatched."""
        return tuple(self._calls)

    def calls_to(self, name: str) -> tuple[RecordedCall, ...]:
        """Every recorded call to `name`, in order, so a test can count the attempts."""
        return tuple(call for call in self._calls if call.name == name)

    def declarations(self) -> tuple[ToolDeclaration, ...]:
        """Return the wrapped registry's declarations, unaltered and in its own order."""
        declared: tuple[ToolDeclaration, ...] = self._registry.declarations()
        return declared

    async def invoke(self, name: str, arguments: Any) -> Any:  # noqa: ANN401 — the registry protocol owns both types
        """Dispatch to the wrapped registry, recording the attempt either way.

        Raises:
            BaseException: Whatever the tool raised, unchanged. A recorded failure is
                still an attempt, so the record is written before the exception leaves —
                including for a cancellation, which is how a timed-out call stays visible.
        """
        ambient = current_ambient()
        tenant = tenant_here()
        started = time.perf_counter()
        result: Any = None
        error: BaseException | None = None
        try:
            result = await self._registry.invoke(name, arguments)
        except BaseException as raised:
            error = raised
            raise
        finally:
            self._calls.append(
                RecordedCall(
                    name=name,
                    arguments=dict(arguments),
                    tenant=ambient.tenant if ambient is not None else _tenant_of(tenant),
                    user=ambient.user if ambient is not None else _user_of(tenant),
                    run_id=ambient.run_id if ambient is not None else None,
                    idempotency_key=ambient.idempotency_key if ambient is not None else None,
                    result=result,
                    error=error,
                    seconds=time.perf_counter() - started,
                )
            )
        return result


def _tenant_of(context: TenantContext | None) -> str | None:
    return context.tenant if context is not None else None


def _user_of(context: TenantContext | None) -> str | None:
    return context.user if context is not None else None


@dataclass(frozen=True, slots=True)
class ScopedRun:
    """The run context a test's calls are made under.

    Args:
        tenant: The isolation boundary bound for the block.
        user: The acting principal, where there is one.
        run_id: The run identifier bound for the block.
        idempotency_key: The key tools called in the block will carry.
        context: The `ToolContext` a tool invoked in the block would be handed.
        allowlist: What the run may call, resolved from what the agent declares and what
            the caller's scopes cover.
    """

    tenant: str
    run_id: str
    context: ToolContext
    allowlist: ToolAllowlist
    user: str | None = None
    idempotency_key: str | None = None


@contextmanager
def scoped_run(
    *,
    tenant: str,
    user: str | None = None,
    run_id: str = "run-1",
    idempotency_key: str | None = None,
    scopes: Sequence[str] | None = None,
    declares: Sequence[str] | None = None,
) -> Iterator[ScopedRun]:
    """Bind a real run context for the block, and restore what was there afterwards.

    The tenant is bound through `tenant_scope` and the ambient through `carrying`, which is
    what the runtime itself does — so a tool, a store or a span inside the block reads the
    context the same way it would in production, and an implementation that never reads it
    fails the test rather than passing on a recorded value.

    Args:
        tenant: The isolation boundary to bind. Required: a fixture with a default tenant
            is a fixture that proves nothing about scoping.
        user: The acting principal, where the case has one.
        run_id: The run identifier to bind.
        idempotency_key: The key tools called in the block carry, so a retry can be
            asserted to have reused it.
        scopes: What the caller's token covers. Narrows the allowlist.
        declares: What the agent was built to call. Defaults to `scopes`, so the common
            case — a run whose scopes are exactly its tools — needs one argument.

    Yields:
        The bound run, including the allowlist and the tool context.

    Example:
        >>> with scoped_run(tenant="acme", scopes=("search",)) as run:
        ...     run.allowlist.permits("refund")
        False
    """
    declared = declares if declares is not None else (scopes or ())
    run = ScopedRun(
        tenant=tenant,
        user=user,
        run_id=run_id,
        idempotency_key=idempotency_key,
        context=ToolContext(
            run_id=run_id, tenant=tenant, user=user, idempotency_key=idempotency_key
        ),
        allowlist=ToolAllowlist.resolve(declared, caller=scopes, agent="agent-under-test"),
    )
    ambient = Ambient(run_id=run_id, tenant=tenant, user=user, idempotency_key=idempotency_key)
    with tenant_scope(TenantContext(tenant=tenant, user=user)), carrying(ambient):
        yield run


def assert_tool_sequence(spy: ToolSpy, *names: str) -> None:
    """Assert the spy recorded exactly `names`, in that order.

    Args:
        spy: The spy that watched the run.
        names: The tools expected, in the order they should have been called.

    Raises:
        AssertionError: Naming the first call that diverged — a call that never happened,
            one that happened and should not have, or the wrong tool in the right place.
    """
    called = [call.name for call in spy.calls]
    for position, (expected, actual) in enumerate(zip_longest(names, called), start=1):
        if expected == actual:
            continue
        if actual is None:
            raise AssertionError(f"call {position}: expected {expected!r}, nothing was called")
        if expected is None:
            raise AssertionError(f"call {position}: expected nothing, {actual!r} was called")
        raise AssertionError(f"call {position}: expected {expected!r}, {actual!r} was called")


def assert_tool_called_once_with(spy: ToolSpy, name: str, **arguments: object) -> None:
    """Assert `name` was called exactly once, with exactly `arguments`.

    Once rather than at-least-once on purpose: a second call to an effectful tool is a
    duplicated side effect, which is the defect worth catching rather than a detail.

    Args:
        spy: The spy that watched the run.
        name: The tool expected to have been called.
        arguments: The arguments it should have been called with, compared field by field.

    Raises:
        AssertionError: If it was never called, called more than once, or called with a
            differing argument — which is named rather than dumped alongside the whole
            payload.
    """
    calls = spy.calls_to(name)
    if not calls:
        ran = ", ".join(repr(call.name) for call in spy.calls) or "nothing"
        raise AssertionError(f"{name!r} was never called; what ran was {ran}")
    if len(calls) > 1:
        raise AssertionError(
            f"{name!r} was called {len(calls)} times, and a second call to a tool that acts "
            f"is a duplicated side effect"
        )
    actual = calls[0].arguments
    for key in sorted(set(arguments) | set(actual)):
        want = arguments.get(key, _MISSING)
        got = actual.get(key, _MISSING)
        if want != got:
            raise AssertionError(
                f"{name!r} was called with {key}={_shown(got)}, expected {_shown(want)}"
            )


def assert_no_tool_called(spy: ToolSpy, *names: str) -> None:
    """Assert nothing ran, or that none of `names` ran.

    Args:
        spy: The spy that watched the run.
        names: The tools that must not have been called. Empty means no tool at all.

    Raises:
        AssertionError: Quoting the call that should not have happened, arguments
            included, since which arguments it reached is the interesting half.
    """
    offending = [call for call in spy.calls if not names or call.name in names]
    if offending:
        first = offending[0]
        raise AssertionError(
            f"{first.name!r} was called with {dict(first.arguments)!r} and should not have been"
        )


def assert_idempotency_key_stable(spy: ToolSpy, name: str) -> None:
    """Assert repeated identical calls to `name` carried one stable idempotency key.

    Identical arguments are what makes two calls a retry rather than two pieces of work, so
    keys are compared within an argument digest and calls with differing arguments are free
    to differ.

    Args:
        spy: The spy that watched the run.
        name: The effectful tool whose retries are being asserted about.

    Raises:
        AssertionError: If the tool was never called, if a repeated call minted a second
            key, or if a repeated call carried no key at all — absence is not stability, it
            is a duplicate waiting for a retry.
    """
    calls = spy.calls_to(name)
    if not calls:
        raise AssertionError(f"{name!r} was never called, so there is no key to assert about")
    grouped: dict[str, list[RecordedCall]] = {}
    for call in calls:
        grouped.setdefault(digest_of(call.arguments), []).append(call)
    for repeated in grouped.values():
        if len(repeated) == 1:
            continue
        keys = {call.idempotency_key for call in repeated}
        if None in keys:
            raise AssertionError(
                f"{name!r} ran {len(repeated)} times with the same arguments and no "
                f"idempotency key, so a retry duplicates the side effect"
            )
        quoted = ", ".join(sorted(repr(key) for key in keys))
        if len(keys) > 1:
            raise AssertionError(
                f"{name!r} ran {len(repeated)} times with the same arguments under keys "
                f"{quoted}, so the retry is a second side effect rather than the same one"
            )


def assert_context_propagated(spy: ToolSpy, *, tenant: str, user: str | None = None) -> None:
    """Assert every recorded call ran under the given tenant, and user where one is given.

    Args:
        spy: The spy that watched the run.
        tenant: The tenant every call must have been dispatched under.
        user: The acting principal, where the case fixes one.

    Raises:
        AssertionError: If no tool ran at all — a run that called nothing proves nothing
            about propagation — or if a call escaped the context, which is named.
    """
    if not spy.calls:
        raise AssertionError(
            "no tool call was recorded, so nothing here proves the context reached one"
        )
    for call in spy.calls:
        if call.tenant != tenant or (user is not None and call.user != user):
            raise AssertionError(
                f"{call.name!r} ran under tenant={call.tenant!r} user={call.user!r}, "
                f"expected tenant={tenant!r} user={user!r}"
            )


class ApprovalStub:
    """An `ApprovalGate` that answers immediately and records what it was asked.

    Args:
        granted: The answer. `None` means nobody answered, which is a denial: a gate that
            hangs reports nothing and a gate that defaults to yes is not a gate.
        reason: Why, as the decision carries it.
        decided_by: Who decided, recorded on the decision.

    Example:
        >>> import asyncio
        >>> from tesserix_adk.core.hooks import ApprovalRecord
        >>> gate = ApprovalStub(granted=False, reason="too large for the desk")
        >>> record = ApprovalRecord.for_call(
        ...     run_id="run-1",
        ...     tenant="acme",
        ...     agent_name="booking",
        ...     tool_name="refund",
        ...     arguments={"amount": 10},
        ...     reason="effectful tool",
        ... )
        >>> asyncio.run(gate.request(record)).granted
        False
    """

    def __init__(
        self,
        *,
        granted: bool | None = True,
        reason: str = "",
        decided_by: str = "test:approver",
    ) -> None:
        self._granted = granted
        self._reason = reason
        self._decided_by = decided_by
        self.asked: list[ApprovalRecord] = []

    async def request(self, record: ApprovalRecord) -> ApprovalDecision:
        """Record the question and answer it, without waiting for anyone."""
        self.asked.append(record)
        if self._granted is None:
            return ApprovalDecision(
                record_id=record.id,
                granted=False,
                decided_by=TIMEOUT_IDENTITY,
                reason="nobody decided, and silence is not permission",
            )
        return ApprovalDecision(
            record_id=record.id,
            granted=self._granted,
            decided_by=self._decided_by,
            reason=self._reason,
        )


def approving(*, decided_by: str = "test:approver", reason: str = "") -> ApprovalStub:
    """A gate that grants what it is shown, for the approved half of a human-gate test."""
    return ApprovalStub(granted=True, decided_by=decided_by, reason=reason)


def denying(*, reason: str, decided_by: str = "test:approver") -> ApprovalStub:
    """A gate that refuses what it is shown, with the reason the decision carries."""
    return ApprovalStub(granted=False, decided_by=decided_by, reason=reason)


def failing_tool(error: BaseException) -> Callable[..., Any]:
    """A tool that raises `error` instead of running.

    Raising the caller's own error rather than a stand-in keeps the test asserting on the
    domain error the runtime is meant to surface.

    Example:
        >>> from tesserix_adk.core import ToolExecutionError
        >>> tool = failing_tool(ToolExecutionError("the ledger is closed"))
        >>> tool(id="1")
        Traceback (most recent call last):
        tesserix_adk.core.errors.ToolExecutionError: the ledger is closed
    """

    def _fail(**_arguments: object) -> NoReturn:
        raise error

    return _fail


def slow_tool(*, seconds: float) -> Callable[..., Any]:
    """A tool that takes `seconds`, so a test can prove its own timeout fires."""

    async def _slow(**_arguments: object) -> None:
        await asyncio.sleep(seconds)

    return _slow


@dataclass(slots=True)
class ConcurrencyProbe:
    """A tool that reports how many calls to it really overlapped.

    Asserting a concurrency cap by counting calls proves nothing about how many ran at
    once. This measures the overlap itself, so a cap that is configured but not applied
    fails the test.

    Args:
        peak: The most calls that were ever in flight together.
        inflight: How many are in flight now.
    """

    peak: int = 0
    inflight: int = 0
    _yields: int = field(default=2, repr=False)

    async def __call__(self, **_arguments: object) -> None:
        """Occupy a slot long enough for any overlapping call to be counted."""
        self.inflight += 1
        self.peak = max(self.peak, self.inflight)
        try:
            for _ in range(self._yields):
                await asyncio.sleep(0)
                self.peak = max(self.peak, self.inflight)
        finally:
            self.inflight -= 1


def peak_concurrency() -> ConcurrencyProbe:
    """A fresh probe, ready to be registered as a tool and asserted on afterwards."""
    return ConcurrencyProbe()


def _shown(value: object) -> str:
    return "absent" if value is _MISSING else repr(value)
