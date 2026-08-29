"""One agent handing work to another, under a scope and a ceiling it cannot widen.

Every product that hand-rolled this pattern reinvented a different set of safety
properties, and the ones they left out are always the same three. The worker held its own
tool allowlist rather than the intersection with its caller's, so delegating widened
access. It spent from a fresh allowance, so one ceiling paid for two runs. And its answer
was pasted into the caller's conversation bare, which makes whatever the worker read an
instruction channel into the caller.

So: a `Roster` declares who may be handed work and what each of them does; a task names
what it needs and is routed to a specialist that declared it; that specialist runs holding
the intersection of its own tools and its caller's, on a budget slice deducted from the
caller's ledger; and what comes back crosses into the supervisor as untrusted data, past
the guardrail chain, before it can influence anything.

Work that did not come back as an answer is a `DelegationError` the supervisor is handed
rather than one it has to catch — an agent that cannot delegate can often still answer,
and a refusal it can read is a refusal it can route around. A delegation declared `fatal`
raises instead, for the tasks where carrying on without the answer is worse.

Depth, fan-out and cycles are not this module's job: `Delegation` bounds the shape of the
run and refuses before anything starts. This module bounds what one hop may hold, spend
and say.

Every name exported here is semver-governed: it appears in `docs/api-surface.txt`, so a
change to it shows up in a pull request's diff and follows `docs/versioning.md`. The
decisions behind these types are in `docs/delegation.md`.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from pydantic import BaseModel

from tesserix_adk.core.errors import ConfigurationError, DelegationError, GuardrailError
from tesserix_adk.core.primitives import Usage
from tesserix_adk.core.run import Run, RunEvent, RunEventKind, RunGrant, RunState
from tesserix_adk.runtime.blocking import current_ambient
from tesserix_adk.runtime.cancellation import CancellationToken
from tesserix_adk.runtime.delegation import handed_back, narrowed_to

if TYPE_CHECKING:
    from collections.abc import Collection, Mapping, Sequence

    from tesserix_adk.core.agent import Agent
    from tesserix_adk.core.budget import BudgetLimits
    from tesserix_adk.core.guards import GuardrailPipeline
    from tesserix_adk.core.protocols import BudgetPolicy
    from tesserix_adk.core.run import RunContext
    from tesserix_adk.runtime.delegation import Delegation
    from tesserix_adk.runtime.loop import AgentRunner

__all__ = ["AddressableSubagent", "DelegationResult", "Roster", "Specialist", "Supervisor"]

_NOT_AN_ANSWER = {RunState.BUDGET_EXHAUSTED: "budget", RunState.CANCELLED: "cancelled"}
_NOTHING = Usage(input_tokens=0, output_tokens=0)


@runtime_checkable
class _Sliceable(Protocol):
    """A budget an allocation can be taken out of. Not every policy is one."""

    def sliced(self, limits: BudgetLimits) -> BudgetPolicy:
        """The allocation, deducted from this ledger."""
        ...


@runtime_checkable
class AddressableSubagent[AddressableOutputT: BaseModel](Protocol):
    """A non-native agent that a supervisor can address under the same run bounds.

    Implementations receive only the caller's explicit identity, narrowed tools, ambient
    scopes and trace, cancellation, lineage and shared budget. They return a normal kit
    :class:`Run`, so attribution and guarded hand-back remain identical to native workers.
    """

    name: str
    tools: tuple[str, ...]

    async def run(
        self,
        user_input: str,
        *,
        tenant: str,
        budget: BudgetPolicy,
        user: str | None = None,
        run_id: str | None = None,
        cancellation: CancellationToken | None = None,
        parent: RunContext | None = None,
        scopes: Collection[str] = (),
        trace: Mapping[str, str] | None = None,
        tools: Collection[str] | None = None,
    ) -> Run[AddressableOutputT]:
        """Execute one delegated run or raise a typed boundary refusal."""
        ...


@dataclass(frozen=True, slots=True)
class Specialist:
    """One agent a supervisor may hand work to, and what it declared it can do.

    Args:
        agent: The worker. What it may actually reach is this narrowed to its caller's
            scope, which is never wider and is usually narrower than what it declares.
        capabilities: The kinds of work it takes. A specialist that declared none could
            not be routed to, so an empty set is a mistake rather than a modest setting.
        budget: What it may spend on one task, where it is bounded more tightly than its
            caller. A task may state a slice of its own, which wins.

    Example:
        >>> from tesserix_adk.core import Agent
        >>> researcher = Agent(
        ...     name="researcher", instructions="Find things.", model="claude-sonnet-5",
        ...     free_text=True, tools=("search",),
        ... )
        >>> Specialist(agent=researcher, capabilities=frozenset({"research"})).name
        'researcher'
    """

    agent: Agent[Any] | AddressableSubagent[Any]
    capabilities: frozenset[str]
    budget: BudgetLimits | None = None

    def __post_init__(self) -> None:
        """Refuse a specialist nothing could route to, where it is declared."""
        if not self.capabilities:
            raise ConfigurationError(
                f"{self.agent.name!r} declares no capability, so no task could ever be routed to it"
            )

    @property
    def name(self) -> str:
        """What it is called, which is the name its spend is attributed to."""
        return self.agent.name


class Roster:
    """Who a supervisor may hand work to, and how a task finds one of them.

    Args:
        specialists: Who may be handed work, in the order they are considered.

    Raises:
        ConfigurationError: If the roster is empty, or two specialists answer to one name.
            The empty one is the dangerous case: a supervisor with nobody to delegate to
            does the work itself, at the wider access it delegates in order not to use.

    Example:
        >>> from tesserix_adk.core import Agent
        >>> clerk = Agent(
        ...     name="clerk", instructions="File things.", model="claude-sonnet-5",
        ...     free_text=True, tools=("file",),
        ... )
        >>> Roster((Specialist(agent=clerk, capabilities=frozenset({"filing"})),)).capabilities
        frozenset({'filing'})
    """

    __slots__ = ("_specialists",)

    def __init__(self, specialists: Sequence[Specialist]) -> None:
        if not specialists:
            raise ConfigurationError(
                "a roster with no worker on it would leave the supervisor doing the work "
                "itself, at the wider access it delegates in order not to use"
            )
        names = [one.name for one in specialists]
        duplicated = sorted({name for name in names if names.count(name) > 1})
        if duplicated:
            raise ConfigurationError(
                f"two workers answer to one name, so spend could be attributed to either: "
                f"{', '.join(duplicated)}"
            )
        self._specialists = tuple(specialists)

    @property
    def workers(self) -> tuple[Specialist, ...]:
        """Who is on it, in declaration order."""
        return self._specialists

    @property
    def capabilities(self) -> frozenset[str]:
        """Everything the roster can do between them."""
        return frozenset[str]().union(*(one.capabilities for one in self._specialists))

    def matching(self, needs: Collection[str]) -> Specialist | None:
        """Who a task that needs `needs` goes to, or `None` where nobody declared it.

        Args:
            needs: What the task takes. A specialist qualifies by declaring all of it.

        Returns:
            The narrowest qualified specialist — fewest capabilities, ties broken by
            declaration order. The specialist is asked before the generalist, so adding a
            broadly-capable worker does not quietly take work off the others.

        Raises:
            ConfigurationError: If the task needs nothing. A task with no requirement
                routes to whoever happens to be first, which is a routing decision nobody
                made.
        """
        wanted = frozenset(needs)
        if not wanted:
            raise ConfigurationError(
                "a task that needs nothing would route to whichever worker happens to be "
                "first, which is not a routing decision anybody made"
            )
        qualified = [one for one in self._specialists if wanted <= one.capabilities]
        if not qualified:
            return None
        return min(qualified, key=lambda one: len(one.capabilities))


@dataclass(frozen=True, slots=True)
class DelegationResult[OutputT: BaseModel]:
    """What one delegation gave back.

    Args:
        specialist: Who ran, which is the name its spend is attributed to.
        run: Its run, in a terminal state, carrying its events and its usage.
        data: What the supervisor may read, as an untrusted-data envelope that has been
            through the guardrail chain. Empty where a guard blocked it.
        error: Why there is no answer, where there is not one. A value rather than an
            exception: an agent that cannot delegate can often still answer.
    """

    specialist: str
    run: Run[OutputT]
    data: str
    error: DelegationError | None = None

    @property
    def answered(self) -> bool:
        """Whether there is an answer to read."""
        return self.error is None

    @property
    def usage(self) -> Usage:
        """What the work consumed, attributed to the specialist that did it."""
        return self.run.usage


class Supervisor:
    """One agent handing work to workers that can never hold or spend more than it does.

    Args:
        runner: What executes a delegated run.
        roster: Who may be handed work.
        agent: The supervising agent. Workers inherit its guardrails and its approval
            requirements, because a call is not cleared by being made one level down.
        delegation: Where this supervisor sits in the run and what it holds. Every
            delegation comes off it, so depth, fan-out and cycle ceilings are enforced
            before a worker starts.
        budget: The supervisor's ceiling. Every worker spends against it — one handed a
            fresh allowance is a way to spend a single ceiling twice.
        guardrails: What an answer passes through before the supervisor may read it. An
            empty pipeline is legal but is written as one, so a review can tell a
            deliberate opt-out from an omission.
        cancellation: The caller's switch. Flipping it stops the supervisor and every
            worker in flight.
    """

    __slots__ = (
        "_agent",
        "_budget",
        "_claims",
        "_delegation",
        "_events",
        "_guardrails",
        "_held",
        "_lock",
        "_roster",
        "_runner",
        "_spent",
        "_token",
    )

    def __init__(
        self,
        runner: AgentRunner,
        roster: Roster,
        *,
        agent: Agent[Any],
        delegation: Delegation,
        budget: BudgetPolicy,
        guardrails: GuardrailPipeline | None = None,
        cancellation: CancellationToken | None = None,
    ) -> None:
        self._runner = runner
        self._roster = roster
        self._agent = agent
        self._delegation = delegation
        self._budget = budget
        self._guardrails = guardrails
        self._token = cancellation or CancellationToken()
        self._held = frozenset(agent.tools) & delegation.scope.tools
        self._spent: dict[str, Usage] = {}
        self._claims: dict[str, str] = {}
        self._events: list[RunEvent] = []
        self._lock = asyncio.Lock()

    @property
    def held(self) -> frozenset[str]:
        """What this supervisor may reach, and the ceiling on what any worker may."""
        return self._held

    @property
    def spent(self) -> Mapping[str, Usage]:
        """What each specialist consumed, by name. Every run is here, answered or not."""
        return dict(self._spent)

    @property
    def claims(self) -> Mapping[str, str]:
        """Which specialist holds which memory key, so a second writer is visible."""
        return dict(self._claims)

    @property
    def events(self) -> tuple[RunEvent, ...]:
        """What was delegated and what was refused, in order."""
        return tuple(self._events)

    @property
    def tenant(self) -> str:
        """Whose run this is, which every refusal raised on its behalf is attributed to."""
        return self._delegation.context.tenant.tenant

    @property
    def cancelled(self) -> bool:
        """Whether the supervisor has been asked to stop."""
        return self._token.cancelled

    def cancel(self, reason: str = "the supervisor was cancelled") -> None:
        """Stop the supervisor and every worker in flight.

        A cancelled worker still reaches a terminal state and is still attributed what it
        spent: work that was paid for and then abandoned is the spend nobody finds.
        """
        self._token.cancel(reason)

    async def delegate(
        self,
        task: str,
        *,
        needs: Collection[str],
        budget: BudgetLimits | None = None,
        writes: str | None = None,
        fatal: bool = False,
        run_id: str | None = None,
    ) -> DelegationResult[Any]:
        """Hand `task` to the specialist that declared it can do it.

        Args:
            task: What is being asked, in the words the worker will see.
            needs: What doing it takes. Routed to the narrowest specialist that declared
                all of it.
            budget: What this task may spend, deducted from the supervisor's ledger.
                Defaults to the specialist's own slice, and to the supervisor's remainder
                where neither states one.
            writes: The memory key this answer is destined for, where it has one. Claimed
                before the worker runs, so a second worker writing it is refused rather
                than silently overwriting the first.
            fatal: That the supervisor cannot carry on without this answer, so a failure
                is raised rather than handed back.
            run_id: Identity for the delegated run, generated if absent.

        Returns:
            What came back, carrying either the checked data or the reason there is none.

        Raises:
            DelegationError: If nobody on the roster can do it, if the specialist holds
                nothing its caller also holds, if another specialist already claimed
                `writes`, or — where `fatal` — if the work did not come back as an answer.
                The first three are wiring mistakes: nothing ran, and nothing would on a
                retry.
            DelegationLimitError: If the run is already as deep, as wide or as long as its
                delegation ceilings allow.
            ConfigurationError: If a slice was asked for and the supervisor's budget is
                not one an allocation can be taken out of.
        """
        specialist = self._roster.matching(needs)
        if specialist is None:
            raise self._refusal(
                f"nothing on the roster does {', '.join(sorted(needs))}; between them the "
                f"roster does {', '.join(sorted(self._roster.capabilities))}",
                specialist="",
                reason="no_worker",
            )
        held = tuple(tool for tool in specialist.agent.tools if tool in self._held)
        if not held:
            raise self._refusal(
                f"{specialist.name!r} holds none of the tools its caller holds, so it "
                f"could not act on anything it was handed",
                specialist=specialist.name,
                reason="no_tools",
            )
        async with self._lock:
            self._claim(writes, specialist.name)
            self._delegation.to(specialist.name, tools=held)  # enforces the caller's fan-out limits
        caller = self._delegation.context
        parent = caller.model_copy(update={"grant": self._grant()})
        allowance = self._allowance(specialist, budget)
        if isinstance(specialist.agent, AddressableSubagent):
            ambient = current_ambient()
            run = await specialist.agent.run(
                task,
                tenant=caller.tenant.tenant,
                user=caller.tenant.user,
                run_id=run_id,
                cancellation=self._token,
                parent=parent,
                budget=allowance,
                scopes=ambient.scopes if ambient is not None else (),
                trace=ambient.trace if ambient is not None else {},
                tools=held,
            )
        else:
            run = await self._runner.run(
                narrowed_to(specialist.agent, held),
                task,
                tenant=caller.tenant.tenant,
                user=caller.tenant.user,
                run_id=run_id,
                cancellation=self._token,
                parent=parent,
                budget=allowance,
            )
        return await self._returned(specialist, run, fatal=fatal)

    async def _returned(
        self, specialist: Specialist, run: Run[Any], *, fatal: bool
    ) -> DelegationResult[Any]:
        """Check what came back, attribute what it cost, and say whether it is an answer."""
        async with self._lock:
            self._spent[specialist.name] = self._spent.get(specialist.name, _NOTHING) + run.usage
        data, error = await self._checked(specialist, run)
        self._events.append(
            RunEvent(
                kind=RunEventKind.DELEGATED if error is None else RunEventKind.DELEGATION_REFUSED,
                name=specialist.name,
                detail=None if error is None else f"{error.reason}: {error}",
                at=run.ended_at,
                usage=run.usage,
            )
        )
        if error is not None and fatal:
            raise error
        return DelegationResult(specialist=specialist.name, run=run, data=data, error=error)

    async def _checked(
        self, specialist: Specialist, run: Run[Any]
    ) -> tuple[str, DelegationError | None]:
        """Put what came back through the guardrails before anything may read it."""
        error = (
            None
            if run.state is RunState.COMPLETED
            else self._refusal(
                f"{specialist.name!r} did not answer: the run ended {run.state}",
                specialist=specialist.name,
                reason=_NOT_AN_ANSWER.get(run.state, "failed"),
                run_id=run.id,
            )
        )
        said = handed_back(run)
        if self._guardrails is None:
            return said, error
        try:
            return await self._guardrails.check_input(said), error
        except GuardrailError as blocked:
            return "", self._refusal(
                f"what {specialist.name!r} handed back was stopped by the supervisor's "
                f"guardrails: {blocked}",
                specialist=specialist.name,
                reason="blocked",
                run_id=run.id,
            )

    def _claim(self, writes: str | None, specialist: str) -> None:
        """Hold a memory key for one worker, so a second cannot overwrite it unseen."""
        if writes is None:
            return
        holder = self._claims.get(writes)
        if holder is not None and holder != specialist:
            raise self._refusal(
                f"{holder!r} already writes {writes!r}; two workers writing one key means "
                f"whichever finished last is the answer, which nobody chose",
                specialist=specialist,
                reason="conflict",
            )
        self._claims[writes] = specialist

    def _allowance(self, specialist: Specialist, asked: BudgetLimits | None) -> BudgetPolicy:
        """What a task may spend: its own slice, else the specialist's, else the remainder."""
        limits = asked if asked is not None else specialist.budget
        if limits is None:
            return self._budget.child()
        if not isinstance(self._budget, _Sliceable):
            raise ConfigurationError(
                f"{type(self._budget).__name__} cannot be sliced, so a per-worker "
                f"allowance would either be ignored or be a second ceiling nobody "
                f"deducts from"
            )
        return self._budget.sliced(limits)

    def _grant(self) -> RunGrant:
        """What the supervisor holds, in the form a worker inherits rather than declares."""
        return RunGrant(
            tools=tuple(sorted(self._held)),
            approval_required_tools=self._agent.approval_required_tools,
            guardrails=self._agent.guardrails,
        )

    def _refusal(
        self, message: str, *, specialist: str, reason: str, run_id: str | None = None
    ) -> DelegationError:
        """A refusal naming who it was for, why, and the path it happened on."""
        return DelegationError(
            message,
            specialist=specialist,
            reason=reason,
            path=(*self._delegation.path, specialist) if specialist else self._delegation.path,
            run_id=run_id,
            tenant=self._delegation.context.tenant.tenant,
        )
