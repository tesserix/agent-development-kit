"""How far one agent may hand work to another, and what the other is allowed to hold.

Multi-agent runs fail in two loud ways and one quiet one. They recurse until the budget is
gone; two agents pass a task back and forth forever; or — the quiet one — a sub-agent three
levels down ends up with the allowlist from its own configuration rather than the narrowed
scope of the caller it acts for. The third is a privilege escalation dressed as a default,
and it is the one nobody notices.

So the shape of a run is bounded and its scope only ever narrows. A child comes from its
parent (`parent.to(...)`) and from nowhere else: there is no constructor that produces one
holding a scope nobody narrowed, because escalation by omission is still escalation. What a
child asks for is intersected with what its parent holds, and asking for something the
parent never held is refused rather than granted.

Every refusal is typed, names the path it happened on, and is not retryable: the same call
refused for the same reason refuses again, so a parent that retries it is looping. A
refusal reaches the parent as something it can reason about — an agent that cannot delegate
can often still answer.

Every name exported here is semver-governed: it appears in `docs/api-surface.txt`, so a
change to it shows up in a pull request's diff and follows `docs/versioning.md`. The
decisions behind these types are in `docs/delegation.md`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from pydantic import BaseModel

from tesserix_adk.core.errors import (
    ConfigurationError,
    DelegationLimitError,
    ScopeEscalationError,
)
from tesserix_adk.core.primitives import TextPart
from tesserix_adk.core.run import RunContext, RunEventKind, RunState, TenantContext
from tesserix_adk.runtime.results import ToolResult

if TYPE_CHECKING:
    from collections.abc import Collection

    from tesserix_adk.core.protocols import Clock
    from tesserix_adk.core.run import Run

__all__ = ["Delegation", "DelegationLimits", "DelegationScope", "handed_back"]


@dataclass(frozen=True, slots=True)
class DelegationLimits:
    """How deep, how wide and how often one run may delegate.

    Configurable downward only: a child that declares roomier ceilings than its parent
    keeps its parent's, since a limit that can be raised from inside is not a limit.

    Args:
        max_depth: How many agents may sit between the root and the deepest worker.
        max_fan_out: How many children one agent may hand work to.
        max_delegations: How many delegations the whole run may make. Depth and fan-out
            bound one lineage each; this bounds their product, which is where a shallow,
            wide tree would otherwise escape both.

    Example:
        >>> DelegationLimits(max_depth=2).narrowed(DelegationLimits(max_depth=9)).max_depth
        2
    """

    max_depth: int = 3
    max_fan_out: int = 8
    max_delegations: int = 24

    def __post_init__(self) -> None:
        """Refuse ceilings that could never permit a delegation, where they are written."""
        for name in ("max_depth", "max_fan_out", "max_delegations"):
            if getattr(self, name) < 1:
                raise ConfigurationError(f"{name} below 1 would refuse every delegation")

    def narrowed(self, other: DelegationLimits) -> DelegationLimits:
        """The tighter of the two in every dimension.

        Args:
            other: What the child declared.

        Returns:
            Ceilings no roomier than either.
        """
        return DelegationLimits(
            max_depth=min(self.max_depth, other.max_depth),
            max_fan_out=min(self.max_fan_out, other.max_fan_out),
            max_delegations=min(self.max_delegations, other.max_delegations),
        )


@dataclass(frozen=True, slots=True)
class DelegationScope:
    """What an agent may reach, and until when.

    Args:
        tools: The tools it may call. A scope holding none could not act at all, so an
            empty allowlist is a mistake rather than a maximally safe setting.
        mutations: The mutation classes it may perform, where a deployment separates
            reading from writing.
        expires_at: When the scope stops being valid, on the clock the delegation reads.
            A grant that outlives its reason is a grant nobody reviewed.

    Example:
        >>> DelegationScope(tools=frozenset({"search"})).holds("search")
        True
    """

    tools: frozenset[str]
    mutations: frozenset[str] = frozenset()
    expires_at: float | None = None

    def __post_init__(self) -> None:
        """Refuse a scope that could not call anything, at the point it is written."""
        if not self.tools:
            raise ConfigurationError("a scope with no tool in it could never act")

    def holds(self, tool: str) -> bool:
        """Whether `tool` is in the allowlist."""
        return tool in self.tools


@dataclass(slots=True)
class _Ledger:
    """What one run has already spent on delegation. Shared by every node of its tree."""

    delegations: int = 0
    fan_out: dict[tuple[str, ...], int] = field(default_factory=dict)


class Delegation:
    """One agent's position in a run: how deep it sits, what it holds, what it may spend.

    Created by `Delegation.root` for the agent a run starts at, and by `to` for every
    agent below it. Never by its constructor — a delegation built by hand would carry a
    scope nobody narrowed and a depth nobody counted.

    Example:
        >>> scope = DelegationScope(tools=frozenset({"search", "summarise"}))
        >>> root = Delegation.root(run_id="run_1", tenant="acme", agent="supervisor", scope=scope)
        >>> child = root.to("researcher", tools={"search"})
        >>> child.path, sorted(child.scope.tools)
        (('supervisor', 'researcher'), ['search'])
    """

    __slots__ = ("_clock", "_context", "_ledger", "_limits", "_scope")

    _context: RunContext
    _scope: DelegationScope
    _limits: DelegationLimits
    _ledger: _Ledger
    _clock: Clock | None

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        """Not the way in.

        Raises:
            ConfigurationError: Always. Use `Delegation.root` for the agent a run starts
                at and `to` for everything below it.
        """
        raise ConfigurationError(
            "a delegation comes from Delegation.root() or from parent.to(), never from its "
            "constructor: one built by hand holds a scope nobody narrowed"
        )

    @classmethod
    def root(
        cls,
        *,
        run_id: str,
        tenant: str,
        agent: str,
        scope: DelegationScope,
        limits: DelegationLimits | None = None,
        user: str | None = None,
        clock: Clock | None = None,
    ) -> Delegation:
        """The agent a run starts at, holding everything the run was granted.

        Args:
            run_id: The run this tree belongs to. Every delegation below it shares it.
            tenant: The isolation boundary. No agent below can change it.
            agent: What the root agent is called. It is the first name on every path.
            scope: What the run was granted. Nothing below holds more.
            limits: The shape the run may take. Defaults are conservative and may only be
                tightened below.
            user: The acting principal, where there is one.
            clock: How an expiry is read. Required only when the scope declares one.

        Returns:
            The root delegation.

        Raises:
            ConfigurationError: If the scope expires but nothing can tell the time.
        """
        if scope.expires_at is not None and clock is None:
            raise ConfigurationError(
                "a scope that expires needs a clock to read it against, or the expiry is "
                "a comment rather than a ceiling"
            )
        context = RunContext(
            run_id=run_id,
            tenant=TenantContext(tenant=tenant, user=user),
            depth=0,
            path=(agent,),
        )
        return _built(cls, context, scope, limits or DelegationLimits(), _Ledger(), clock)

    @property
    def context(self) -> RunContext:
        """The run context this agent acts under — run, tenant, depth and path."""
        return self._context

    @property
    def scope(self) -> DelegationScope:
        """What this agent may reach."""
        return self._scope

    @property
    def limits(self) -> DelegationLimits:
        """The ceilings this agent delegates under."""
        return self._limits

    @property
    def depth(self) -> int:
        """How many agents sit between the root and this one."""
        return self._context.depth

    @property
    def path(self) -> tuple[str, ...]:
        """The agents this work came through, root first."""
        return self._context.path

    @property
    def delegations(self) -> int:
        """How many delegations the whole run has made, this agent's branch included."""
        return self._ledger.delegations

    def to(
        self,
        agent: str,
        *,
        tools: Collection[str] | None = None,
        mutations: Collection[str] | None = None,
        limits: DelegationLimits | None = None,
    ) -> Delegation:
        """Hand work to `agent`, under a scope no wider than this one.

        Args:
            agent: What the sub-agent is called.
            tools: What it needs. `None` passes this agent's allowlist through unchanged.
            mutations: The mutation classes it needs, treated the same way.
            limits: Ceilings it declares for itself. Applied only where they are tighter.

        Returns:
            The child delegation, one level deeper, sharing this run's ledger.

        Raises:
            DelegationLimitError: If the run is too deep, too wide, has delegated too
                often, would revisit an agent already on this path, or is acting on a
                scope that has expired. The child is not created and nothing is called.
            ScopeEscalationError: If it asked for a tool or mutation class this agent does
                not hold. Refused whether or not its own configuration would permit it.
        """
        path = (*self.path, agent)
        self._refuse_if_out_of_room(agent, path)
        scope = self._narrowed(tools, mutations, path)
        self._ledger.delegations += 1
        self._ledger.fan_out[self.path] = self._ledger.fan_out.get(self.path, 0) + 1
        context = RunContext(
            run_id=self._context.run_id,
            tenant=self._context.tenant,
            depth=self.depth + 1,
            path=path,
        )
        narrowed = self._limits if limits is None else self._limits.narrowed(limits)
        return _built(type(self), context, scope, narrowed, self._ledger, self._clock)

    def _refuse_if_out_of_room(self, agent: str, path: tuple[str, ...]) -> None:
        """Check every ceiling before anything is created, so a refusal costs nothing."""
        if self._expired():
            raise DelegationLimitError(
                f"the scope {'/'.join(self.path)} acts under expired before it delegated",
                reason="expired",
                path=path,
            )
        if self.depth + 1 > self._limits.max_depth:
            raise DelegationLimitError(
                f"delegating to {agent!r} would sit deeper than the run's {self._limits.max_depth}",
                reason="depth",
                path=path,
            )
        if agent in self.path:
            raise DelegationLimitError(
                f"{agent!r} is already working on this path, so handing it back would circle",
                reason="cycle",
                path=path,
            )
        if self._ledger.fan_out.get(self.path, 0) + 1 > self._limits.max_fan_out:
            raise DelegationLimitError(
                f"{self.path[-1]!r} may not hand work to more than "
                f"{self._limits.max_fan_out} sub-agents",
                reason="fan_out",
                path=path,
            )
        if self._ledger.delegations + 1 > self._limits.max_delegations:
            raise DelegationLimitError(
                f"the run has made its {self._limits.max_delegations} delegations",
                reason="run",
                path=path,
            )

    def _expired(self) -> bool:
        """Whether the grant this agent acts under has run out. Unreadable time fails closed."""
        if self._scope.expires_at is None:
            return False
        return self._clock is None or self._clock.now() >= self._scope.expires_at

    def _narrowed(
        self,
        tools: Collection[str] | None,
        mutations: Collection[str] | None,
        path: tuple[str, ...],
    ) -> DelegationScope:
        """Intersect what was asked for with what is held, refusing anything not held."""
        wanted_tools = self._scope.tools if tools is None else frozenset(tools)
        wanted_mutations = self._scope.mutations if mutations is None else frozenset(mutations)
        beyond = tuple(
            sorted((wanted_tools - self._scope.tools) | (wanted_mutations - self._scope.mutations))
        )
        if beyond:
            raise ScopeEscalationError(
                f"{path[-1]!r} asked for {', '.join(beyond)}, which {path[-2]!r} does not hold",
                requested=beyond,
                path=path,
            )
        return DelegationScope(
            tools=wanted_tools & self._scope.tools,
            mutations=wanted_mutations & self._scope.mutations,
            expires_at=self._scope.expires_at,
        )


def handed_back[OutputT: BaseModel](run: Run[OutputT]) -> str:
    """What a delegated run gives its caller, as data the caller's model may not obey.

    A sub-agent's answer is model output that read whatever the sub-agent read. Pasted
    into the caller's conversation bare, it is an instruction channel for whatever wrote
    it, so it crosses the boundary in the same envelope a tool result crosses in.

    Args:
        run: The finished child run.

    Returns:
        An untrusted-data envelope carrying the child's answer, or, where the child did
        not complete, why it stopped. A caller that cannot tell a refusal from an empty
        answer reads one as the other.

    Example:
        >>> from tesserix_adk.core import Run, RunState
        >>> stopped = Run(
        ...     id="run_1", tenant="acme", agent_name="researcher",
        ...     agent_version="1.0.0", model="claude-sonnet-5", state=RunState.FAILED,
        ... )
        >>> handed_back(stopped).splitlines()[0]
        '<untrusted-data source="delegated_agent">'
    """
    return ToolResult(
        tool=run.agent_name,
        text=_said_by(run),
        source="delegated_agent",
        tenant=run.tenant,
    ).rendered()


def _said_by[OutputT: BaseModel](run: Run[OutputT]) -> str:
    """The child's answer, or the reason there is not one."""
    if run.state is RunState.COMPLETED:
        return next(
            (
                "".join(part.text for part in message.content if isinstance(part, TextPart))
                for message in reversed(run.messages)
                if message.role == "assistant"
            ),
            "",
        )
    refusal = next(
        (event for event in reversed(run.events) if event.kind is RunEventKind.GUARDRAIL_REFUSAL),
        None,
    )
    if refusal is not None:
        return (
            f"{run.agent_name} was stopped by guardrail {refusal.name}: {refusal.detail}; "
            f"it produced no answer"
        )
    return f"{run.agent_name} did not finish: the run ended {run.state}"


def _built(
    cls: type[Delegation],
    context: RunContext,
    scope: DelegationScope,
    limits: DelegationLimits,
    ledger: _Ledger,
    clock: Clock | None,
) -> Delegation:
    """Assemble a delegation past the constructor the public API keeps shut."""
    made = object.__new__(cls)
    made._context = context
    made._scope = scope
    made._limits = limits
    made._ledger = ledger
    made._clock = clock
    return made
