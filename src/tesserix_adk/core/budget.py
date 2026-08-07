"""What a run may spend, in one vocabulary every scope can be stated in.

Spend control invented per product is spend control that disagrees with itself: one counts
iterations, one counts nothing, and the agent that looped overnight is discovered on the
invoice. `BudgetLimits` is the one way to say a ceiling, in every dimension a run can run
away in — money, tokens, calls, turns and wall-clock time.

Two rules make it enforceable rather than decorative. A limit nobody set resolves to a
conservative default, because forgetting is not a way to opt out; saying "no ceiling" takes
`BudgetLimits.unbounded()`, which is a sentence somebody wrote and a reviewer can see. And
where more than one scope applies, the tightest wins with its source recorded — a ceiling
nobody can attribute is a ceiling nobody can raise.
"""

from __future__ import annotations

from collections.abc import (  # noqa: TC003 — pydantic needs the runtime types
    Callable,
    Mapping,
)
from decimal import Decimal
from enum import StrEnum
from typing import TYPE_CHECKING, Protocol, Self, runtime_checkable

from pydantic import model_validator

from tesserix_adk.core.errors import (
    BudgetExceededError,
    BudgetUnavailableError,
    ConfigurationError,
)
from tesserix_adk.core.models import AdkModel
from tesserix_adk.core.primitives import Usage

if TYPE_CHECKING:
    from tesserix_adk.core.protocols import Clock

__all__ = [
    "BudgetDecision",
    "BudgetLimits",
    "BudgetScope",
    "Consumed",
    "LedgerFailure",
    "ResolvedBudget",
    "RunBudget",
    "ScopedLimits",
    "TenantLedger",
    "UnlimitedBudget",
    "most_restrictive",
]

# Enough for a working agent and nowhere near enough to be a surprise. A default nobody
# would notice is a default that stops nothing.
_CONSERVATIVE: dict[str, object] = {
    "max_cost": Decimal("1.00"),
    "max_input_tokens": 100_000,
    "max_output_tokens": 20_000,
    "max_model_calls": 20,
    "max_tool_calls": 40,
    "max_iterations": 10,
    "max_seconds": 300.0,
    "max_peer_invocations": 8,
}

# Shape, not spend: these bound how wide and how deep a run may go rather than how much of
# anything it has used, so nothing accumulates against them.
_SHAPE: dict[str, object] = {
    "max_parallel_tool_calls": 8,
    "max_delegation_depth": 4,
}

_DIMENSIONS = tuple(_CONSERVATIVE)
_CAPS = _DIMENSIONS + tuple(_SHAPE)


class BudgetScope(StrEnum):
    """Where a set of limits was attached, which is how the effective one is attributed.

    Args:
        RUN: This run alone.
        AGENT: Every run of one agent.
        TENANT: Everything one tenant does.
        TENANT_WINDOW: One tenant within a rolling window, such as an hour.
    """

    RUN = "run"
    AGENT = "agent"
    TENANT = "tenant"
    TENANT_WINDOW = "tenant_window"


class LedgerFailure(StrEnum):
    """What a run does when the shared ledger holding a tenant ceiling is unreachable.

    Args:
        REFUSE: Fail closed. Spending against a ceiling nobody can read is spending
            without a ceiling, and this is the default for that reason.
        PROCEED: Enforce only what is local and carry on. A deliberate choice, recorded on
            the run so the runs that took it can be found afterwards.
    """

    REFUSE = "refuse"
    PROCEED = "proceed"


class BudgetLimits(AdkModel):
    """A ceiling, in every dimension a run can run away in.

    Every field is optional to write and none is optional in effect: `filled()` replaces
    what was left unsaid with the conservative default. `unbounded()` is the one way to
    mean no ceiling at all, and it says so in the configuration rather than in an absence.

    Args:
        max_cost: Money, as a `Decimal`. Floats do not add up to an invoice.
        currency: ISO 4217 code the ceiling is stated in. Two scopes in different
            currencies are refused rather than converted at a rate nobody agreed to.
        max_input_tokens: Prompt tokens across the run, cache reads included.
        max_output_tokens: Generated tokens across the run.
        max_model_calls: Model calls, failed and retried attempts included — a retry
            storm is exactly the shape this bounds.
        max_tool_calls: Tool invocations across the run.
        max_iterations: Turns of the loop.
        max_seconds: Wall-clock time the run may occupy.
        max_peer_invocations: Runs this run's tree may start, counted across the whole
            tree rather than per hop — per-hop counting is how a tree of runs each stays
            under a cap they broke together.
        max_parallel_tool_calls: How many tool calls one model turn may ask for, and how
            many of them may be in flight at once. The turn is refused entire rather than
            trimmed, because half a fan-out is a set of side effects nobody chose.
        max_delegation_depth: How deep a chain of agents calling agents may go. A child
            cannot raise this: an agent narrows what it was given and never widens it.
        unlimited: No ceiling anywhere. The one way to mean it, and a contradiction
            alongside any stated ceiling, so it cannot be set by accident.
    """

    max_cost: Decimal | None = None
    currency: str = "USD"
    max_input_tokens: int | None = None
    max_output_tokens: int | None = None
    max_model_calls: int | None = None
    max_tool_calls: int | None = None
    max_iterations: int | None = None
    max_seconds: float | None = None
    max_peer_invocations: int | None = None
    max_parallel_tool_calls: int | None = None
    max_delegation_depth: int | None = None
    unlimited: bool = False

    @model_validator(mode="after")
    def _a_ceiling_of_zero_is_a_field_in_the_wrong_place(self) -> BudgetLimits:
        for name in _CAPS:
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ValueError(
                    f"{name} is {value}, which permits nothing at all; a run that may not "
                    f"do anything is a run nobody meant to configure"
                )
        if self.unlimited and any(getattr(self, name) is not None for name in _CAPS):
            raise ValueError(
                "limits are marked unlimited and also state a ceiling; one of the two is "
                "what somebody meant, and guessing which would be the wrong one"
            )
        return self

    @classmethod
    def conservative(cls) -> Self:
        """The documented default: bounded in every dimension, generous in none."""
        return cls(
            max_cost=Decimal("1.00"),
            max_input_tokens=100_000,
            max_output_tokens=20_000,
            max_model_calls=20,
            max_tool_calls=40,
            max_iterations=10,
            max_seconds=300.0,
            max_peer_invocations=8,
            max_parallel_tool_calls=8,
            max_delegation_depth=4,
        )

    @classmethod
    def unbounded(cls, currency: str = "USD") -> Self:
        """No ceiling anywhere. A deliberate sentence, recorded on the run that uses it."""
        return cls(currency=currency, unlimited=True)

    def filled(self) -> Self:
        """This, with every unsaid dimension replaced by the conservative default.

        Unbounded limits are returned untouched: growing a ceiling onto the one explicit
        way of saying there is none would make that sentence a lie.

        Example:
            >>> BudgetLimits(max_model_calls=2).filled().max_model_calls
            2
        """
        if self.unlimited:
            return self
        defaults = {**_CONSERVATIVE, **_SHAPE}
        unsaid = {name: defaults[name] for name in _CAPS if getattr(self, name) is None}
        return self.model_copy(update=unsaid)


class ScopedLimits(AdkModel):
    """One set of limits and where it was attached.

    Args:
        scope: Which boundary these apply to.
        limits: The ceiling at that boundary.
    """

    scope: BudgetScope
    limits: BudgetLimits


class ResolvedBudget(AdkModel):
    """The effective ceiling, and which scope each dimension of it came from.

    Args:
        limits: What is actually enforced.
        sources: Dimension name to the scope that won it. A dimension no scope stated is
            absent, because the conservative default came from nobody.
        unlimited_reason: Why this run has no ceiling, where it has none. Present exactly
            when the limits are unbounded, because a ceiling removed for a reason nobody
            wrote down is one nobody can review.
        on_ledger_failure: What happens when a shared ledger cannot be reached. `REFUSE`
            by default: spending against a ceiling nobody can read is spending without one.
    """

    limits: BudgetLimits
    sources: Mapping[str, BudgetScope] = {}
    unlimited_reason: str | None = None
    on_ledger_failure: LedgerFailure = LedgerFailure.REFUSE


def most_restrictive(*scoped: ScopedLimits) -> ResolvedBudget:
    """The tightest applicable ceiling in each dimension, with its source recorded.

    Nearness does not decide this. A tenant ceiling of 5.00 and a run ceiling of 1.00 make
    the run's 1.00 effective, and a run that asks for 5.00 under a tenant capped at 1.00
    gets 1.00 — otherwise the boundary that matters could be widened from inside it.

    Args:
        scoped: The limits that apply, in any order and at most one per scope.

    Returns:
        The effective limits, filled from the conservative defaults where no scope spoke,
        or unbounded where every applicable scope said so in as many words.

    Raises:
        ConfigurationError: If two scopes of one kind are given, or if two scopes cap money
            in different currencies. Both are configuration mistakes, and both are found
            here rather than mid-run.
    """
    _one_answer_per_scope(scoped)
    currency = _one_currency(scoped)
    unbounded = [one for one in scoped if one.limits.unlimited]
    if unbounded and len(unbounded) == len(scoped):
        return ResolvedBudget(
            limits=BudgetLimits.unbounded(),
            sources=dict.fromkeys(_CAPS, unbounded[0].scope),
        )
    winning: dict[str, object] = {}
    sources: dict[str, BudgetScope] = {}
    for name in _CAPS:
        stated = [(one.limits, getattr(one.limits, name), one.scope) for one in scoped]
        capped = [(value, scope) for _, value, scope in stated if value is not None]
        if not capped:
            continue
        tightest, scope = min(capped, key=lambda pair: pair[0])
        winning[name] = tightest
        sources[name] = scope
    return ResolvedBudget(
        limits=BudgetLimits.model_validate({"currency": currency, **winning}).filled(),
        sources=sources,
    )


def _one_answer_per_scope(scoped: tuple[ScopedLimits, ...]) -> None:
    seen: set[BudgetScope] = set()
    for one in scoped:
        if one.scope in seen:
            raise ConfigurationError(
                f"two sets of limits at {one.scope} scope: one boundary cannot have two "
                f"ceilings, and picking one of them here would be a guess"
            )
        seen.add(one.scope)


def _one_currency(scoped: tuple[ScopedLimits, ...]) -> str:
    priced = {one.limits.currency for one in scoped if one.limits.max_cost is not None}
    if len(priced) > 1:
        named = " and ".join(sorted(priced))
        raise ConfigurationError(
            f"limits cap spend in {named}: the kit will not convert between them, because "
            f"the rate it picked would be one nobody agreed to"
        )
    return next(iter(priced), "USD")


class Consumed(AdkModel):
    """What has been spent against a ceiling so far.

    Args:
        usage: Tokens and money, as the providers reported or the kit estimated them.
        model_calls: Model calls made, failed and retried attempts included.
        tool_calls: Tools invoked.
        iterations: Turns of the loop.
        seconds: Wall-clock time elapsed.
        peer_invocations: Runs started by this run's tree, counted across the tree.
    """

    usage: Usage = Usage(input_tokens=0, output_tokens=0)
    model_calls: int = 0
    tool_calls: int = 0
    iterations: int = 0
    seconds: float = 0.0
    peer_invocations: int = 0

    def __add__(self, other: Consumed) -> Consumed:
        """Total two records of spend."""
        return Consumed(
            usage=self.usage + other.usage,
            model_calls=self.model_calls + other.model_calls,
            tool_calls=self.tool_calls + other.tool_calls,
            iterations=self.iterations + other.iterations,
            seconds=self.seconds + other.seconds,
            peer_invocations=self.peer_invocations + other.peer_invocations,
        )

    def against(self, name: str) -> Decimal:
        """How much of dimension `name` has been used, in the units its limit counts."""
        if name == "max_cost":
            return self.usage.cost.total if self.usage.cost is not None else Decimal(0)
        return Decimal(str(_MEASURED[name](self)))


_MEASURED: dict[str, Callable[[Consumed], float | int]] = {
    "max_input_tokens": lambda spent: spent.usage.input_tokens,
    "max_output_tokens": lambda spent: spent.usage.output_tokens,
    "max_model_calls": lambda spent: spent.model_calls,
    "max_tool_calls": lambda spent: spent.tool_calls,
    "max_iterations": lambda spent: spent.iterations,
    "max_seconds": lambda spent: spent.seconds,
    "max_peer_invocations": lambda spent: spent.peer_invocations,
}


class BudgetDecision(AdkModel):
    """Whether there is room left, and where there is not.

    Args:
        permitted: Whether the run may continue.
        breached: The first limit with nothing left, by field name.
        scope: Where that limit was attached.
        limit: The ceiling, as a `Decimal` whatever it counts, so one type answers for
            money, tokens, calls and seconds alike.
        consumed: What has been spent against it.
        remaining: What is left. Never negative: a ceiling is not overdrawn, it is reached.
        priced: Whether every call so far had a price. A money ceiling checked against
            calls nobody could price is not enforcement, and this is how a caller knows.
    """

    permitted: bool = True
    breached: str | None = None
    scope: BudgetScope | None = None
    limit: Decimal | None = None
    consumed: Decimal = Decimal(0)
    remaining: Decimal = Decimal(0)
    priced: bool = True

    @property
    def overshoot(self) -> Decimal:
        """How far past the ceiling this went, which a reservation usually holds to zero.

        A call dearer than the estimate that reserved for it lands the run marginally
        over. That is a number whoever reconciles the invoice needs, and rounding it away
        is how a ceiling comes to mean something other than what it says.
        """
        if self.limit is None:
            return Decimal(0)
        return max(self.consumed - self.limit, Decimal(0))

    def as_error(self) -> BudgetExceededError:
        """The typed refusal this decision amounts to. Whether to raise it is the caller's."""
        return BudgetExceededError(
            f"{self.breached} is {self.limit} at {self.scope} scope and this run has "
            f"used {self.consumed}, leaving {self.remaining}",
            breached=self.breached or "",
            scope=self.scope,
            limit=self.limit,
            consumed=self.consumed,
            remaining=self.remaining,
        )


@runtime_checkable
class TenantLedger(Protocol):
    """Where a ceiling wider than one run is kept.

    A tenant ceiling only means anything if every run spending against it reads and writes
    the same total, which is a store outside the process. The kit states the shape and the
    failure mode; where that store lives is a deployment's decision.
    """

    async def total(self, tenant: str, window: str) -> Consumed:
        """What `tenant` has spent in `window` so far.

        Raises:
            BudgetUnavailableError: If the store cannot be reached. Never a zero total: a
                ledger that answers "nothing spent" when it is down hands out the whole
                ceiling again to every run that asks.
        """
        ...

    async def consume(self, tenant: str, window: str, spent: Consumed) -> Consumed:
        """Add `spent` to the total for `tenant` in `window` and return the new total.

        Adding and reading in one call is what makes concurrent runs safe: two runs that
        each read, then decide, then write, both see the same ceiling as free.

        Raises:
            BudgetUnavailableError: If the store cannot be reached.
        """
        ...


class RunBudget:
    """The ceiling for one run, enforced before the spend rather than reported after it.

    Reservations are provisional: `reserve` holds an estimate so a call cannot start that
    the ceiling could not cover, and `record` replaces the hold with what was actually
    consumed. A child run shares this ledger rather than opening its own — a sub-agent
    handed a fresh allowance is a way to spend one ceiling twice.

    Args:
        resolved: The effective limits and where each came from.
        clock: Source of time, for the wall-clock ceiling.
        spent: What has already been consumed, for a policy resuming a run.
        ledger: Where a tenant-wide total is kept, for ceilings this run does not own
            alone. Dimensions the resolution attributed to a tenant scope are checked
            against it; the rest stay local.
        tenant: Whose ledger to read. Required alongside `ledger`.
        window_seconds: The length of a rolling tenant window. The window is pinned when
            the run starts and does not move under it: a run beginning at 10:59 must not
            become a way to spend two hours of one hour's allowance.
        on_ledger_failure: What to do when the ledger is unreachable. Fails closed.
    """

    def __init__(
        self,
        resolved: ResolvedBudget,
        clock: Clock,
        spent: Consumed | None = None,
        ledger: TenantLedger | None = None,
        tenant: str | None = None,
        window_seconds: float | None = None,
        on_ledger_failure: LedgerFailure = LedgerFailure.REFUSE,
    ) -> None:
        if ledger is not None and not tenant:
            raise ConfigurationError(
                "a tenant ledger was given with no tenant to read it for, so the ceiling "
                "would be shared with nobody"
            )
        self._resolved = resolved.model_copy(
            update={"on_ledger_failure": LedgerFailure(on_ledger_failure)}
        )
        self._clock = clock
        self._started = clock.now()
        self._spent = spent or Consumed()
        self._held = 0
        self._unpriced = 0
        self.reservations: list[int] = []
        self._ledger = ledger
        self._tenant = tenant or ""
        self._window = _window_key(clock.now(), window_seconds)
        self._shared = frozenset(
            name
            for name, scope in resolved.sources.items()
            if scope in {BudgetScope.TENANT, BudgetScope.TENANT_WINDOW}
        )
        self._tenant_spent = Consumed()

    @property
    def resolved(self) -> ResolvedBudget:
        """The ceiling in force, with the scope each dimension came from."""
        return self._resolved

    @property
    def spent(self) -> Consumed:
        """What has been consumed so far, reservations excluded."""
        return self._spent

    def limits(self) -> BudgetLimits:
        """What is left, as limits — the allowance a child run starts from.

        For a shared ceiling this reflects the tenant total as this run last read it, which
        is at its first reservation. Reading the ledger is IO and this is not the place for
        it: a budget nobody has spent against yet has not asked.
        """
        if self._resolved.limits.unlimited:
            return self._resolved.limits
        left = {
            name: _left(self._ceiling(name), self._used(name), name)
            for name in _DIMENSIONS
            if self._ceiling(name) is not None
        }
        return self._resolved.limits.model_copy(update=left)

    def child(self) -> RunBudget:
        """A budget for a run this one called, sharing this ledger and this ceiling."""
        return _ChildBudget(self)

    async def reserve(self, estimate: int) -> None:
        """Hold `estimate` input tokens before a call.

        Raises:
            BudgetExceededError: If the reservation, or anything already spent, has passed
                a ceiling. Raised before the call, because a limit checked afterwards is a
                report rather than a limit.
        """
        await self._read_the_ledger()
        self._refuse_if_past(extra={"max_input_tokens": Decimal(estimate)})
        self._held += estimate
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
        """Record what was actually consumed, releasing the outstanding reservation.

        Args:
            usage: What the call consumed, carrying its cost where anything priced it.
            model_calls: Model calls this covers, failed attempts included.
            tool_calls: Tools invoked.
            iterations: Turns of the loop completed.
            peer_invocations: Runs this run's tree started.

        Raises:
            BudgetExceededError: If the run has now passed a ceiling.
            BudgetUnavailableError: If a shared ceiling applies and its ledger cannot be
                reached, unless the run was explicitly configured to proceed without it.
        """
        consumed = Consumed(
            usage=usage,
            model_calls=model_calls,
            tool_calls=tool_calls,
            iterations=iterations,
            peer_invocations=peer_invocations,
        )
        self._absorb(consumed)
        await self._write_the_ledger(consumed)
        self._refuse_if_past()

    def check(self) -> BudgetDecision:
        """Whether there is room left, and where there is not."""
        breach = self._breach()
        if breach is not None:
            return breach
        return BudgetDecision(permitted=True, priced=self._unpriced == 0)

    def _absorb(self, consumed: Consumed) -> None:
        if consumed.usage.cost is None and not consumed.usage._is_nothing:
            self._unpriced += 1
        self._spent = self._spent + consumed
        self._held = 0

    def _ceiling(self, name: str) -> Decimal | None:
        stated = getattr(self._resolved.limits, name)
        return None if stated is None else Decimal(str(stated))

    def _used(self, name: str) -> Decimal:
        if name == "max_seconds":
            return Decimal(str(self._clock.now() - self._started))
        used = (
            self._tenant_spent.against(name) if name in self._shared else self._spent.against(name)
        )
        return used + Decimal(self._held) if name == "max_input_tokens" else used

    async def _read_the_ledger(self) -> None:
        if self._ledger is None or not self._shared:
            return
        try:
            self._tenant_spent = await self._ledger.total(self._tenant, self._window)
        except BudgetUnavailableError:
            self._degrade_or_refuse()

    async def _write_the_ledger(self, consumed: Consumed) -> None:
        if self._ledger is None or not self._shared:
            return
        try:
            self._tenant_spent = await self._ledger.consume(self._tenant, self._window, consumed)
        except BudgetUnavailableError:
            self._degrade_or_refuse()

    def _degrade_or_refuse(self) -> None:
        """Fail closed unless somebody wrote down that this run may proceed without it."""
        if self._resolved.on_ledger_failure is LedgerFailure.REFUSE:
            raise BudgetUnavailableError(
                f"the ledger holding {self._tenant}'s ceiling could not be read, and a "
                f"ceiling nobody can read is not a ceiling"
            )
        self._shared = frozenset()

    def _breach(self, extra: Mapping[str, Decimal] | None = None) -> BudgetDecision | None:
        for name in _DIMENSIONS:
            ceiling = self._ceiling(name)
            if ceiling is None:
                continue
            used = self._used(name) + (extra or {}).get(name, Decimal(0))
            if used > ceiling:
                spent_now = self._used(name)
                return BudgetDecision(
                    permitted=False,
                    breached=name,
                    scope=self._resolved.sources.get(name),
                    limit=ceiling,
                    consumed=spent_now,
                    remaining=max(ceiling - spent_now, Decimal(0)),
                    priced=self._unpriced == 0,
                )
        return None

    def _refuse_if_past(self, extra: Mapping[str, Decimal] | None = None) -> None:
        breach = self._breach(extra)
        if breach is not None:
            raise breach.as_error()


class _ChildBudget(RunBudget):
    """A sub-agent's view of its parent's budget: the same ledger, never a fresh one."""

    def __init__(self, parent: RunBudget) -> None:
        super().__init__(parent.resolved, parent._clock, parent.spent)
        self._parent = parent

    def _absorb(self, consumed: Consumed) -> None:
        self._parent._absorb(consumed)
        self._spent = self._parent.spent
        self._held = 0


def _window_key(now: float, window_seconds: float | None) -> str:
    """The window a run belongs to, decided once at the start and not revisited."""
    if window_seconds is None:
        return "all"
    return str(int(now // window_seconds))


class UnlimitedBudget:
    """No ceiling at all, which is a thing somebody has to write down.

    The kit will not arrive here by omission. A runtime constructed without a policy gets
    the conservative defaults; removing them takes this class, a stated reason, and a
    reason that ends up on every run it governed.

    Args:
        reason: Why this workload has no ceiling. Recorded on the run.

    Raises:
        ValueError: If no reason is given.
    """

    def __init__(self, reason: str) -> None:
        if not reason.strip():
            raise ValueError(
                "an unlimited budget needs a stated reason: a ceiling removed for a reason "
                "nobody wrote down is one nobody can review"
            )
        self.reason = reason
        self.reservations: list[int] = []

    @property
    def resolved(self) -> ResolvedBudget:
        """Unbounded limits, carrying the reason they are unbounded."""
        return ResolvedBudget(limits=BudgetLimits.unbounded(), unlimited_reason=self.reason)

    def limits(self) -> BudgetLimits:
        """Unbounded, however much has been spent."""
        return BudgetLimits.unbounded()

    def child(self) -> UnlimitedBudget:
        """A child of no ceiling is no ceiling, for the same recorded reason."""
        return self

    async def reserve(self, estimate: int) -> None:
        """Permit the reservation, recording it so a test can still see what was asked."""
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
        """Accept the spend. Nothing is enforced, deliberately and on the record."""

    def check(self) -> BudgetDecision:
        """Always permitted."""
        return BudgetDecision(permitted=True)


def _left(ceiling: Decimal | None, used: Decimal, name: str) -> object:
    remaining = max((ceiling or Decimal(0)) - used, Decimal(0))
    if name == "max_cost":
        return remaining
    if name == "max_seconds":
        return float(remaining)
    return int(remaining)
