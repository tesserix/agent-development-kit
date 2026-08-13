"""Several branches at once, under one ceiling, aggregated by a rule that was declared.

`asyncio.gather` over a list of sub-agents is the shape everybody writes first, and it
loses the same three properties every time. Nothing caps how many branches are in flight,
so the fan-out's real concurrency is whatever the provider's rate limiter happens to
allow. Every branch spends from the run's ledger with no ordering between them, so a
ceiling that four branches respected individually is breached collectively. And the result
is a list: an aggregate built from three branches out of five is indistinguishable from
one built from all five, which is how a partial answer comes to be presented as a whole
one.

So a fan-out here is bounded, attributed and provenanced. `max_concurrency` is a number
somebody chose. Every branch runs through `Supervisor`, which means it holds the
intersection of its own scope and its caller's, spends against the one shared ledger, and
crosses back through the guardrail chain — none of that is re-implemented here. What this
module adds is the aggregation step: an `Aggregation` says what counts as an answer, and
an aggregate that cannot be formed under that rule is an `AggregationError` carrying which
branches contributed and why each of the others did not.

Results are in declared order regardless of completion order, so two runs of the same
fan-out aggregate identically. A branch cancelled mid-flight refuses the whole aggregate
rather than quietly leaving itself out.

Every name exported here is semver-governed: it appears in `docs/api-surface.txt`, so a
change to it shows up in a pull request's diff and follows `docs/versioning.md`. The
decisions behind these types are in `docs/parallel.md`.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any, ClassVar, overload

from tesserix_adk.core.errors import (
    AggregationError,
    ConfigurationError,
    DelegationError,
    DelegationLimitError,
)
from tesserix_adk.core.primitives import Usage

if TYPE_CHECKING:
    from collections.abc import Callable, Collection, Mapping, Sequence

    from tesserix_adk.core.budget import BudgetLimits
    from tesserix_adk.runtime.supervisor import Supervisor

__all__ = [
    "Aggregate",
    "Aggregation",
    "All",
    "Branch",
    "BranchOutcome",
    "BranchResult",
    "FirstSuccess",
    "Quorum",
    "Reduce",
    "fan_out",
]

_NOTHING = Usage(input_tokens=0, output_tokens=0)
_DRAINED = "budget: the shared ledger ran out before this branch started"


class BranchOutcome(StrEnum):
    """How one branch ended. Four states, because "not ok" is three different problems."""

    OK = "ok"
    FAILED = "failed"
    CANCELLED = "cancelled"
    BUDGET_EXHAUSTED = "budget_exhausted"


_OUTCOMES = {"budget": BranchOutcome.BUDGET_EXHAUSTED, "cancelled": BranchOutcome.CANCELLED}


@dataclass(frozen=True, slots=True, init=False)
class Branch:
    """One piece of work in a fan-out, named so that what it contributed can be said.

    Args:
        name: What this branch is called in the provenance. Unique within one fan-out.
        task: What is being asked, in the words the worker will see.
        needs: What doing it takes. Routed by the supervisor's roster.
        budget: What this branch alone may spend, deducted from the shared ledger. A
            branch that exhausts a slice of its own is that branch's problem; one that
            exhausts the shared ledger stops the branches that had not started.
        writes: The memory key this answer is destined for, claimed before the branch runs
            so a second branch writing it is refused rather than overwriting the first.

    Raises:
        ConfigurationError: If the branch is unnamed, or needs nothing. An unnamed branch
            cannot be attributed, and one that needs nothing routes to whichever worker
            happens to be first.

    Example:
        >>> Branch(name="fares", task="check fares", needs={"research"}).needs
        frozenset({'research'})
    """

    name: str
    task: str
    needs: frozenset[str]
    budget: BudgetLimits | None = None
    writes: str | None = None

    def __init__(
        self,
        *,
        name: str,
        task: str,
        needs: Collection[str],
        budget: BudgetLimits | None = None,
        writes: str | None = None,
    ) -> None:
        if not name:
            raise ConfigurationError(
                "an unnamed branch cannot be named in the provenance, so nobody could say "
                "afterwards what it contributed"
            )
        if not needs:
            raise ConfigurationError(
                f"branch {name!r} needs nothing, so it would route to whichever worker "
                f"happens to be first, which is not a routing decision anybody made"
            )
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "task", task)
        object.__setattr__(self, "needs", frozenset(needs))
        object.__setattr__(self, "budget", budget)
        object.__setattr__(self, "writes", writes)


@dataclass(frozen=True, slots=True)
class BranchResult:
    """What one branch gave back, whether or not it gave back an answer.

    Args:
        branch: Which branch this is.
        specialist: Who ran it, or empty where nothing was routed to.
        outcome: How it ended.
        data: What it handed back, having been through the supervisor's guardrails. Empty
            for anything but `OK`.
        usage: What it consumed, attributed to this branch rather than to the run.
        reason: Why there is no answer, as `reason: message`. Empty for `OK`.
    """

    branch: str
    specialist: str
    outcome: BranchOutcome
    data: str
    usage: Usage
    reason: str = ""

    @property
    def ok(self) -> bool:
        """Whether there is an answer to read."""
        return self.outcome is BranchOutcome.OK


@dataclass(frozen=True, slots=True)
class Aggregate[ValueT]:
    """What a fan-out added up to, and what it was added up from.

    Args:
        value: The aggregate, as the strategy formed it.
        strategy: Which rule formed it.
        results: Every branch, in declared order, answered or not.
        contributed: The branches inside `value`, in declared order.
        excluded: Why each of the others is not, by branch name.
        spent: What each branch consumed, by branch name.
        peak_in_flight: The most branches that were running at once, which is what the
            concurrency cap actually achieved rather than what it permitted.
    """

    value: ValueT
    strategy: str
    results: tuple[BranchResult, ...]
    contributed: tuple[str, ...]
    excluded: Mapping[str, str] = field(default_factory=dict)
    spent: Mapping[str, Usage] = field(default_factory=dict)
    peak_in_flight: int = 0

    @property
    def usage(self) -> Usage:
        """What the whole fan-out consumed, branches that were excluded included.

        Work that was paid for and then left out of the answer is exactly the spend that
        goes missing from a report, so it is totalled here rather than at the point the
        strategy decides.
        """
        total = _NOTHING
        for one in self.results:
            total = total + one.usage
        return total


class Aggregation[ValueT](ABC):
    """A rule for turning branches into one answer, and for refusing to.

    Two methods rather than one: `chosen` decides which branches are in and whether the
    aggregate can be formed at all, and `value` builds it from those. Keeping them apart
    is what lets a refusal carry the same provenance a success does.
    """

    __slots__ = ()

    name: ClassVar[str]

    @abstractmethod
    def chosen(self, results: Sequence[BranchResult]) -> tuple[tuple[BranchResult, ...], str]:
        """Which branches are in the aggregate, and why it cannot be formed if it cannot.

        Args:
            results: Every branch, in declared order.

        Returns:
            The contributing branches, and the refusal reason — empty where there is none.
        """

    @abstractmethod
    def value(self, chosen: Sequence[BranchResult]) -> ValueT:
        """Build the aggregate from the branches `chosen` picked."""


class All(Aggregation[tuple[str, ...]]):
    """Every branch, or none. The default, because a missing branch is usually a bug.

    Example:
        >>> All().name
        'all'
    """

    name: ClassVar[str] = "all"

    def chosen(self, results: Sequence[BranchResult]) -> tuple[tuple[BranchResult, ...], str]:
        """Everything, unless anything is missing, in which case nothing."""
        answered = tuple(one for one in results if one.ok)
        return answered, "" if len(answered) == len(results) else "failed"

    def value(self, chosen: Sequence[BranchResult]) -> tuple[str, ...]:
        """What every branch said, in declared order."""
        return tuple(one.data for one in chosen)


class FirstSuccess(Aggregation[str]):
    """The first branch that answered, in declared order rather than in finishing order.

    Whichever finished first is the obvious rule and the wrong one: it makes the answer
    depend on scheduling, so the same fan-out gives different answers on different days.

    Example:
        >>> FirstSuccess().name
        'first_success'
    """

    name: ClassVar[str] = "first_success"

    def chosen(self, results: Sequence[BranchResult]) -> tuple[tuple[BranchResult, ...], str]:
        """The first branch with an answer, or the refusal that nothing had one."""
        for one in results:
            if one.ok:
                return (one,), ""
        return (), "none"

    def value(self, chosen: Sequence[BranchResult]) -> str:
        """What that branch said."""
        return chosen[0].data


class Quorum(Aggregation[tuple[str, ...]]):
    """Every branch that answered, provided enough of them did.

    Args:
        needed: How many answers make an aggregate. A quorum nobody reached is a refusal,
            never the same answer with fewer branches behind it.

    Raises:
        ConfigurationError: If `needed` is below one, which no result could fail.

    Example:
        >>> Quorum(2).needed
        2
    """

    __slots__ = ("needed",)

    name: ClassVar[str] = "quorum"

    def __init__(self, needed: int) -> None:
        if needed < 1:
            raise ConfigurationError(
                "a quorum of at least one is what makes it a quorum; below that every "
                "fan-out succeeds, including the one where nothing answered"
            )
        self.needed = needed

    def chosen(self, results: Sequence[BranchResult]) -> tuple[tuple[BranchResult, ...], str]:
        """Everything that answered, if that is enough of them."""
        answered = tuple(one for one in results if one.ok)
        return answered, "" if len(answered) >= self.needed else "quorum"

    def value(self, chosen: Sequence[BranchResult]) -> tuple[str, ...]:
        """What each contributing branch said, in declared order."""
        return tuple(one.data for one in chosen)


class Reduce[ValueT](Aggregation[ValueT]):
    """A caller's own rule over the branches that answered.

    Args:
        reducer: What to build from them. Given the contributing `BranchResult`s in
            declared order rather than their strings, so it can attribute what it uses.

    Example:
        >>> Reduce(len).name
        'reduce'
    """

    __slots__ = ("_reducer",)

    name: ClassVar[str] = "reduce"

    def __init__(self, reducer: Callable[[Sequence[BranchResult]], ValueT]) -> None:
        self._reducer = reducer

    def chosen(self, results: Sequence[BranchResult]) -> tuple[tuple[BranchResult, ...], str]:
        """Everything that answered, provided something did."""
        answered = tuple(one for one in results if one.ok)
        return answered, "" if answered else "none"

    def value(self, chosen: Sequence[BranchResult]) -> ValueT:
        """Whatever the caller's rule makes of them."""
        return self._reducer(chosen)


@overload
async def fan_out(
    supervisor: Supervisor, branches: Sequence[Branch], *, max_concurrency: int = ...
) -> Aggregate[tuple[str, ...]]: ...


@overload
async def fan_out[ValueT](
    supervisor: Supervisor,
    branches: Sequence[Branch],
    *,
    into: Aggregation[ValueT],
    max_concurrency: int = ...,
) -> Aggregate[ValueT]: ...


async def fan_out(
    supervisor: Supervisor,
    branches: Sequence[Branch],
    *,
    into: Aggregation[Any] | None = None,
    max_concurrency: int = 8,
) -> Aggregate[Any]:
    """Run `branches` concurrently through `supervisor` and add them up under `into`.

    Args:
        supervisor: Who routes each branch, and whose scope, ledger and guardrails every
            branch is subject to. Fan-out adds no authority of its own.
        branches: What to run, in the order the results will be reported in.
        into: What counts as an answer. `All` by default, which fails closed.
        max_concurrency: How many branches may be in flight at once.

    Returns:
        The aggregate, carrying every branch, what each spent, and why anything excluded
        was excluded.

    Raises:
        AggregationError: If the aggregate cannot be formed under `into`, or if the
            fan-out was cancelled while branches were still running. A cancelled fan-out
            never aggregates what happened to have arrived: half a fan-out that reads like
            a whole one is the failure this module exists to prevent.
        ConfigurationError: If there are no branches, two answer to one name, or the
            concurrency cap is below one.
    """
    strategy: Aggregation[Any] = into if into is not None else All()
    _checked(branches, max_concurrency)
    results, peak = await _running(supervisor, branches, max_concurrency)
    excluded = {one.branch: one.reason for one in results if not one.ok}
    if supervisor.cancelled:
        raise _refusal(supervisor, strategy, len(results), _abandoned(results), "cancelled", ())
    chosen, refused = strategy.chosen(results)
    if refused:
        raise _refusal(
            supervisor,
            strategy,
            len(results),
            excluded,
            refused,
            tuple(one.branch for one in chosen),
        )
    return Aggregate(
        value=strategy.value(chosen),
        strategy=strategy.name,
        results=results,
        contributed=tuple(one.branch for one in chosen),
        excluded=excluded,
        spent={one.branch: one.usage for one in results},
        peak_in_flight=peak,
    )


def _abandoned(results: Sequence[BranchResult]) -> dict[str, str]:
    """Why nothing is in a cancelled aggregate, including the branches that did answer.

    A branch that finished before the stop reached it still has an answer, and leaving it
    off the provenance is how the refusal comes to look like it was about the other ones.
    """
    return {
        one.branch: one.reason or "cancelled: the fan-out was stopped before it was formed"
        for one in results
    }


def _checked(branches: Sequence[Branch], max_concurrency: int) -> None:
    """Refuse a fan-out nothing could come of, before anything runs."""
    if not branches:
        raise ConfigurationError(
            "a fan-out over no branches would aggregate to an empty answer that reads "
            "exactly like an answer nothing was found for"
        )
    names = [one.name for one in branches]
    duplicated = sorted({name for name in names if names.count(name) > 1})
    if duplicated:
        raise ConfigurationError(
            f"two branches answer to one name, so the provenance could not say which "
            f"contributed: {', '.join(duplicated)}"
        )
    if max_concurrency < 1:
        raise ConfigurationError("a fan-out runs at least one branch at a time")


async def _running(
    supervisor: Supervisor, branches: Sequence[Branch], max_concurrency: int
) -> tuple[tuple[BranchResult, ...], int]:
    """Run every branch under the cap, and report them in declared order."""
    gate = asyncio.Semaphore(max_concurrency)
    state = _InFlight()
    async with asyncio.TaskGroup() as group:
        tasks = [group.create_task(_bounded(supervisor, one, gate, state)) for one in branches]
    return tuple(task.result() for task in tasks), state.peak


@dataclass(slots=True)
class _InFlight:
    """How many branches are running, and how many ever were at once."""

    now: int = 0
    peak: int = 0
    drained: bool = False

    def entered(self) -> None:
        """One more branch is running."""
        self.now += 1
        self.peak = max(self.peak, self.now)

    def left(self) -> None:
        """One fewer."""
        self.now -= 1


async def _bounded(
    supervisor: Supervisor, branch: Branch, gate: asyncio.Semaphore, state: _InFlight
) -> BranchResult:
    """Wait for room, then run one branch — unless the shared ledger has already gone."""
    async with gate:
        if state.drained:
            return _refused(branch, "", _DRAINED, BranchOutcome.BUDGET_EXHAUSTED)
        if supervisor.cancelled:
            return _refused(
                branch, "", "cancelled: the fan-out was stopped", BranchOutcome.CANCELLED
            )
        state.entered()
        try:
            result = await _branch(supervisor, branch)
        finally:
            state.left()
    if result.outcome is BranchOutcome.BUDGET_EXHAUSTED and branch.budget is None:
        state.drained = True
    return result


async def _branch(supervisor: Supervisor, branch: Branch) -> BranchResult:
    """Hand one branch to the supervisor and read what came back as an outcome."""
    try:
        handed = await supervisor.delegate(
            branch.task, needs=branch.needs, budget=branch.budget, writes=branch.writes
        )
    except DelegationError as wiring:
        return _refused(branch, wiring.specialist, f"{wiring.reason}: {wiring}")
    except DelegationLimitError as capped:
        return _refused(branch, "", f"{capped.reason}: {capped}")
    if handed.error is None:
        return BranchResult(
            branch=branch.name,
            specialist=handed.specialist,
            outcome=BranchOutcome.OK,
            data=handed.data,
            usage=handed.usage,
        )
    return BranchResult(
        branch=branch.name,
        specialist=handed.specialist,
        outcome=_OUTCOMES.get(handed.error.reason, BranchOutcome.FAILED),
        data="",
        usage=handed.usage,
        reason=f"{handed.error.reason}: {handed.error}",
    )


def _refused(
    branch: Branch,
    specialist: str,
    reason: str,
    outcome: BranchOutcome = BranchOutcome.FAILED,
) -> BranchResult:
    """A branch that never produced an answer, and what it spent doing so: nothing."""
    return BranchResult(
        branch=branch.name,
        specialist=specialist,
        outcome=outcome,
        data="",
        usage=_NOTHING,
        reason=reason,
    )


def _refusal(
    supervisor: Supervisor,
    strategy: Aggregation[Any],
    branches: int,
    excluded: Mapping[str, str],
    reason: str,
    contributed: tuple[str, ...],
) -> AggregationError:
    """The refusal an unformable aggregate amounts to, carrying the same provenance."""
    return AggregationError(
        f"{strategy.name} could not be formed from {branches} branches: {reason}, with "
        f"{len(contributed)} contributing",
        strategy=strategy.name,
        reason=reason,
        contributed=contributed,
        excluded=excluded,
        tenant=supervisor.tenant,
    )
