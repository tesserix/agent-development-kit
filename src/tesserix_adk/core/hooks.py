"""Where policy attaches to a run, and what it is allowed to say.

Guardrails, spend checks and human approvals were bolted on at whatever call site a team
remembered, which is why an agent considered safe in one product was unsafe in another.
The points below are the loop's own, so a check declared once is enforced on every path.

A hook returns a decision, never a mutation: it is handed facts about the step and answers
with one of four words. It cannot reach the run, widen a tenant scope, disable another
hook or raise a cap, because it is never given anything that could.

Every name exported here is semver-governed: it appears in `docs/api-surface.txt`, so a
change to it shows up in a pull request's diff and follows `docs/versioning.md`.
"""

from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, model_validator

from tesserix_adk.core.errors import HookRegistrationError, ProtocolConformanceError
from tesserix_adk.core.protocols import verify_conformance
from tesserix_adk.core.run import RunState  # noqa: TC001 — pydantic resolves this annotation

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

__all__ = [
    "ApprovalDecision",
    "ApprovalGate",
    "ApprovalRecord",
    "Hook",
    "HookAction",
    "HookChain",
    "HookDecision",
    "HookPoint",
    "HookSubject",
    "resolve_hooks",
]

_FROZEN = ConfigDict(frozen=True, extra="forbid")


class HookPoint(StrEnum):
    """The seven places a run stops to ask.

    Each is a point where something is about to happen that cannot be taken back: a prompt
    assembled, a request sent, a tool dispatched, an answer accepted.
    """

    BEFORE_PROMPT_ASSEMBLY = "before_prompt_assembly"
    BEFORE_MODEL_CALL = "before_model_call"
    AFTER_MODEL_RESPONSE = "after_model_response"
    BEFORE_TOOL_DISPATCH = "before_tool_dispatch"
    AFTER_TOOL_RESULT = "after_tool_result"
    BEFORE_OUTPUT_VALIDATION = "before_output_validation"
    ON_TERMINAL = "on_terminal"


class HookAction(StrEnum):
    """What a hook may say, ordered from least to most restrictive.

    Four words and no fifth. A vocabulary a hook can extend is a vocabulary the loop
    cannot enforce.
    """

    CONTINUE = "continue"
    REWRITE = "rewrite"
    REQUIRE_APPROVAL = "require_approval"
    REFUSE = "refuse"


_RESTRICTIVENESS = {
    HookAction.CONTINUE: 0,
    HookAction.REWRITE: 1,
    HookAction.REQUIRE_APPROVAL: 2,
    HookAction.REFUSE: 3,
}


class HookDecision(BaseModel):
    """One hook's answer at one point.

    Args:
        action: What to do. See `HookAction`.
        reason: Why, recorded on the run. Required for anything but `CONTINUE`.
        replacement: The rewritten content, for `REWRITE` only.

    Example:
        >>> HookDecision.refuse("account number in the prompt").action
        <HookAction.REFUSE: 'refuse'>
    """

    model_config = _FROZEN

    action: HookAction = HookAction.CONTINUE
    reason: str = ""
    replacement: str | None = None

    @model_validator(mode="after")
    def _a_decision_carries_what_it_needs(self) -> HookDecision:
        if self.action is HookAction.REWRITE and self.replacement is None:
            raise ValueError("a rewrite must carry its replacement; there is nothing to swap in")
        if self.action is not HookAction.REWRITE and self.replacement is not None:
            raise ValueError(
                f"only a rewrite carries a replacement; {self.action} would discard it silently"
            )
        if self.action is not HookAction.CONTINUE and not self.reason:
            raise ValueError(
                f"{self.action} must say why: a run that stopped for an unrecorded reason "
                f"cannot be explained to whoever asked for it"
            )
        return self

    @property
    def restrictiveness(self) -> int:
        """How tight this answer is, so two hooks disagreeing resolve the same way twice."""
        return _RESTRICTIVENESS[self.action]

    @classmethod
    def proceed(cls) -> HookDecision:
        """Nothing to say."""
        return cls()

    @classmethod
    def rewrite(cls, replacement: str, *, reason: str = "rewritten") -> HookDecision:
        """Swap the content for `replacement` and carry on."""
        return cls(action=HookAction.REWRITE, replacement=replacement, reason=reason)

    @classmethod
    def refuse(cls, reason: str) -> HookDecision:
        """End the run here."""
        return cls(action=HookAction.REFUSE, reason=reason)

    @classmethod
    def require_approval(cls, reason: str) -> HookDecision:
        """Hold the step until a human decides."""
        return cls(action=HookAction.REQUIRE_APPROVAL, reason=reason)


class HookSubject(BaseModel):
    """What a hook is told about the step it is being asked about.

    Facts, not handles. There is no run, no config and no chain here, so a hook has
    nothing to widen a scope with even if it wanted to.

    Args:
        point: Where in the run this is.
        run_id: The run, for correlation with its audit record.
        tenant: The isolation boundary.
        user: The acting principal, where there is one.
        agent_name: Which agent is running.
        content: The text about to be used, where the point has one. What a rewrite
            replaces.
        tool_name: The tool about to be dispatched, or that just returned.
        tool_arguments: What it was called with.
        state: The terminal state, at `ON_TERMINAL` only.
    """

    model_config = _FROZEN

    point: HookPoint
    run_id: str = Field(min_length=1)
    tenant: str = Field(min_length=1)
    user: str | None = None
    agent_name: str = Field(min_length=1)
    content: str = ""
    tool_name: str | None = None
    tool_arguments: dict[str, Any] = Field(default_factory=dict)
    state: RunState | None = None


@runtime_checkable
class Hook(Protocol):
    """A policy attached to declared points in the loop.

    Hooks fail closed: one that raises or outruns its ceiling stops the run rather than
    being skipped, because a check that did not run is not a check that passed.
    """

    @property
    def name(self) -> str:
        """Stable identifier, recorded against every decision it makes."""
        ...

    @property
    def points(self) -> tuple[HookPoint, ...]:
        """Where it wants to be asked. Declared up front, so the chain is knowable."""
        ...

    async def on(self, subject: HookSubject) -> HookDecision:
        """Answer for `subject`."""
        ...


@runtime_checkable
class ApprovalGate(Protocol):
    """Where a run waits for a human to decide about a tool call."""

    async def request(self, record: ApprovalRecord) -> ApprovalDecision:
        """Return the decision for `record`, waiting for it if need be."""
        ...


class ApprovalRecord(BaseModel):
    """A tool call held for a human decision, in a form safe to put in a queue.

    An approval queue is a queryable store that outlives the run and is read by people who
    are not party to the case. It therefore carries a digest of the arguments and never
    the arguments: a reviewer can confirm the call they approved is the call that ran,
    without the account number living in a second system.

    Args:
        id: Identity of the request, echoed by the decision.
        run_id: The run waiting on it.
        tenant: The isolation boundary.
        agent_name: Which agent asked.
        tool_name: What it wants to call.
        arguments_digest: SHA-256 over the canonical arguments.
        reason: Why approval is required — the tool's declaration or a hook's decision.
        requested_at: Unix seconds.
    """

    model_config = _FROZEN

    id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    tenant: str = Field(min_length=1)
    agent_name: str = Field(min_length=1)
    tool_name: str = Field(min_length=1)
    arguments_digest: str = Field(min_length=64, max_length=64)
    reason: str = Field(min_length=1)
    requested_at: float = 0.0

    @classmethod
    def for_call(
        cls,
        *,
        run_id: str,
        tenant: str,
        agent_name: str,
        tool_name: str,
        arguments: Mapping[str, Any],
        reason: str,
        requested_at: float = 0.0,
        request_id: str | None = None,
    ) -> ApprovalRecord:
        """Build a record for one tool call, digesting its arguments canonically."""
        return cls(
            id=request_id or f"{run_id}:{tool_name}:{digest_of(arguments)[:12]}",
            run_id=run_id,
            tenant=tenant,
            agent_name=agent_name,
            tool_name=tool_name,
            arguments_digest=digest_of(arguments),
            reason=reason,
            requested_at=requested_at,
        )


class ApprovalDecision(BaseModel):
    """A human's answer about one held call.

    Args:
        record_id: Which request this answers. Checked, so an answer to another question
            cannot let a call through.
        granted: Whether it may proceed.
        decided_by: Who decided.
        decided_at: Unix seconds. Compared against the request's time-to-live, because an
            approval is permission at a moment rather than a standing licence.
        reason: Why, where the decider gave one.
    """

    model_config = _FROZEN

    record_id: str = Field(min_length=1)
    granted: bool
    decided_by: str = Field(min_length=1)
    decided_at: float = 0.0
    reason: str = ""


class HookChain:
    """The hooks a runner enforces, in declaration order.

    A chain is sealed when it is handed to a runner and is immutable from then on: the
    chain a run started with is the chain it is judged by, so a hook cannot add a
    permissive hook behind itself or drop the one that would have refused it.

    Args:
        hooks: The hooks, in the order they should be asked.

    Raises:
        HookRegistrationError: If a hook is missing a protocol member, declares no points,
            or shares a name with one already registered.
    """

    def __init__(self, hooks: Iterable[Hook] = ()) -> None:
        self._hooks: list[Hook] = []
        self._sealed = False
        for hook in hooks:
            self.register(hook)

    def register(self, hook: Hook) -> None:
        """Add `hook` to the end of the chain.

        Raises:
            HookRegistrationError: If the chain is sealed, or the hook is unusable.
        """
        if self._sealed:
            raise HookRegistrationError(
                f"chain is sealed; {getattr(hook, 'name', hook)!r} cannot be added once a "
                f"runner is enforcing it, or a run would be judged by a chain nobody declared"
            )
        try:
            verify_conformance(hook, Hook)
        except ProtocolConformanceError as missing:
            raise HookRegistrationError(str(missing)) from missing
        if not hook.points:
            raise HookRegistrationError(
                f"hook {hook.name!r} declares no hook points, so it would never be asked "
                f"anything while reading like a policy that is in force"
            )
        if hook.name in self.names:
            raise HookRegistrationError(
                f"hook {hook.name!r} registered twice; a name is how a decision is "
                f"attributed, and two owners of one name attribute nothing"
            )
        self._hooks.append(hook)

    def sealed(self) -> HookChain:
        """Seal the chain against further registration and return it.

        One-way and in place: the caller's reference is the runner's, so a hook holding on
        to the chain it was declared in finds it shut.
        """
        self._sealed = True
        return self

    def at(self, point: HookPoint) -> tuple[Hook, ...]:
        """Return the hooks declared for `point`, in declaration order."""
        return tuple(hook for hook in self._hooks if point in hook.points)

    @property
    def names(self) -> tuple[str, ...]:
        """Every registered hook name, in order."""
        return tuple(hook.name for hook in self._hooks)

    def __len__(self) -> int:
        """How many hooks are registered."""
        return len(self._hooks)


def resolve_hooks(decisions: Sequence[HookDecision] | Iterable[HookDecision]) -> HookDecision:
    """Return the decision that holds when several hooks answer at one point.

    The most restrictive wins, and ties go to the first declared. Two hooks disagreeing is
    not a coin to toss: the same chain must resolve the same way on every process, and the
    tighter answer is the one nobody has to justify afterwards.

    Example:
        >>> resolve_hooks([HookDecision.proceed(), HookDecision.refuse("no")]).action
        <HookAction.REFUSE: 'refuse'>
    """
    winner = HookDecision.proceed()
    for decision in decisions:
        if decision.restrictiveness > winner.restrictiveness:
            winner = decision
    return winner


def digest_of(arguments: Mapping[str, Any]) -> str:
    """SHA-256 over the arguments, key order independent, so equal calls digest equally."""
    canonical = json.dumps(dict(arguments), sort_keys=True, default=repr)
    return hashlib.sha256(canonical.encode()).hexdigest()
