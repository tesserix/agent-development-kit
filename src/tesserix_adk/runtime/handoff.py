"""A conversation changing hands, carrying a declared payload rather than a transcript.

The shortcut every staged flow reaches for — triage answers, then forwards the whole
conversation to a specialist — costs three things at once. It leaks context the target has
no business seeing, it pays for that context by the token on every turn, and it lets the
target infer permissions nobody granted it, because a transcript describes what the source
was allowed to do.

So a handoff is a contract. Each receiver declares the Pydantic model it accepts; a payload
that is not that model is refused before the target is invoked, with the fields it got
wrong. The target then runs holding the intersection of its own allowlist and the source's,
under the source's tenant and user, which are not parameters anybody could pass wrongly.

A person is a receiver like any other: a queue takes the same contract, so escalating to a
human desk is the same call with the same validation rather than a second path with its own
rules.

Every name exported here is semver-governed: it appears in `docs/api-surface.txt`, so a
change to it shows up in a pull request's diff and follows `docs/versioning.md`. The
decisions behind these types are in `docs/delegation.md`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, cast, runtime_checkable

from pydantic import BaseModel, ValidationError

from tesserix_adk.core.errors import ConfigurationError, HandoffContractError
from tesserix_adk.core.run import RunEvent, RunEventKind, RunGrant
from tesserix_adk.runtime.delegation import narrowed_to

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    from tesserix_adk.core.agent import Agent
    from tesserix_adk.core.guards import GuardrailPipeline
    from tesserix_adk.core.primitives import Message
    from tesserix_adk.core.run import Run
    from tesserix_adk.runtime.delegation import Delegation
    from tesserix_adk.runtime.loop import AgentRunner

__all__ = ["Handoff", "HandoffContract", "HandoffDesk", "HandoffQueue", "HandoffResult", "Receiver"]


@runtime_checkable
class HandoffQueue(Protocol):
    """Where a conversation waits for a person to pick it up."""

    async def receive(self, handoff: Handoff) -> None:
        """Take the handoff. Raising leaves the conversation with the source."""
        ...


@dataclass(frozen=True, slots=True)
class HandoffContract:
    """What one receiver accepts when a conversation is handed to it.

    Args:
        accepts: The model the payload has to be. Declared by the receiver rather than
            assembled by the source, so what crosses is the target's requirement and not
            whatever the source happened to have.

    Example:
        >>> from pydantic import BaseModel
        >>> class Ticket(BaseModel):
        ...     account: str
        >>> HandoffContract(accepts=Ticket).validated({"account": "ac_9"}).account
        'ac_9'
    """

    accepts: type[BaseModel]

    def validated(self, state: Mapping[str, Any] | BaseModel) -> BaseModel:
        """Return `state` as the declared model, or refuse it.

        Args:
            state: What is being handed over, as the model or as a mapping of it.

        Returns:
            The payload as the declared model.

        Raises:
            HandoffContractError: If it is not what the receiver declared it accepts. The
                fields it got wrong are on `violations`, since "invalid payload" tells the
                source nothing it can act on.
        """
        if isinstance(state, self.accepts):
            return state
        given = state.model_dump() if isinstance(state, BaseModel) else dict(state)
        try:
            return self.accepts.model_validate(given)
        except ValidationError as refused:
            raise HandoffContractError(
                f"the payload is not a {self.accepts.__name__}: {refused.error_count()} "
                f"field(s) the receiver declared were not satisfied",
                reason="contract",
                violations=tuple(str(one["loc"][0]) for one in refused.errors() if one["loc"]),
            ) from refused


@dataclass(frozen=True, slots=True)
class Receiver:
    """Somewhere a conversation may be handed to, and what it accepts.

    Args:
        contract: What it takes. Validated before it is handed anything.
        agent: The agent that takes over, for a receiver that is one.
        queue: Where it waits for a person, for a receiver that is a human desk.
        name: What it is called. Taken from `agent` where there is one, and required for a
            queue, since nothing else names it.

    Raises:
        ConfigurationError: If it is neither an agent nor a queue, if it is both, or if a
            queue has no name. Each would leave a target nothing could route to.
    """

    contract: HandoffContract
    agent: Agent[Any] | None = None
    queue: HandoffQueue | None = None
    name: str = ""

    def __post_init__(self) -> None:
        """Refuse a receiver nothing could hand to, where it is declared."""
        if self.agent is not None and self.queue is not None:
            raise ConfigurationError(
                "a receiver that is both an agent and a queue would hand the same "
                "conversation to two owners"
            )
        if self.agent is None and self.queue is None:
            raise ConfigurationError(
                "a receiver that is neither an agent nor a queue has nowhere to take the "
                "conversation"
            )
        if self.agent is not None:
            object.__setattr__(self, "name", self.agent.name)
        elif not self.name:
            raise ConfigurationError("a queue needs a name, because nothing else names it")


@dataclass(frozen=True, slots=True)
class Handoff:
    """The record of one conversation changing hands.

    Args:
        from_agent: Who handed it over.
        to_agent: Who took it.
        reason: Why, in the words the source gave. Recorded, so a chain of handoffs reads
            as a decision rather than as a sequence of jumps.
        state: The validated payload, as data.
        rendered: The payload as the target saw it, after redaction.
        scope: What the target actually held, which is never wider than the source.
        trace: The agents the conversation has passed through, root first.
        run_id: The run all of this belongs to. One conversation, one trace.
        tenant: The isolation boundary, which no handoff changes.
        user: The acting principal, which no handoff changes either.
    """

    from_agent: str
    to_agent: str
    reason: str
    state: Mapping[str, Any]
    rendered: str
    scope: tuple[str, ...]
    trace: tuple[str, ...]
    run_id: str
    tenant: str
    user: str | None = None


@dataclass(frozen=True, slots=True)
class HandoffResult:
    """What came of handing the conversation over.

    Args:
        handoff: What crossed, as it was recorded.
        run: The target's run, where the target was an agent. `None` for a queue: a
            person has not answered yet, and an empty run would say they had.
    """

    handoff: Handoff
    run: Run[Any] | None = None

    @property
    def queued(self) -> bool:
        """Whether it is waiting for a person rather than answered."""
        return self.run is None


class HandoffDesk:
    """Where one agent hands a live conversation to another, under a declared contract.

    Args:
        runner: Runs the target.
        receivers: Who may be handed to. Nobody else can be named.
        agent: The agent handing over. Its allowlist caps every target's.
        delegation: What the source holds and how far the conversation may travel. The
            tenant and user come from here, so a handoff cannot cross either.
        guardrails: What the payload passes through on its way out. Redaction happens
            here, before the payload reaches the target, the record or telemetry.

    Raises:
        ConfigurationError: If there is nobody to hand to, or two receivers answer to one
            name.
    """

    __slots__ = (
        "_agent",
        "_delegation",
        "_events",
        "_guardrails",
        "_handoffs",
        "_receivers",
        "_runner",
    )

    def __init__(
        self,
        runner: AgentRunner,
        receivers: Sequence[Receiver],
        *,
        agent: Agent[Any],
        delegation: Delegation,
        guardrails: GuardrailPipeline | None = None,
    ) -> None:
        if not receivers:
            raise ConfigurationError(
                "a desk with no receiver on it has nowhere to hand a conversation, so the "
                "source would carry on holding it at its own wider access"
            )
        names = [one.name for one in receivers]
        duplicated = sorted({name for name in names if names.count(name) > 1})
        if duplicated:
            raise ConfigurationError(
                f"two receivers answer to one name, so a handoff could go to either: "
                f"{', '.join(duplicated)}"
            )
        self._runner = runner
        self._receivers = {one.name: one for one in receivers}
        self._agent = agent
        self._delegation = delegation
        self._guardrails = guardrails
        self._handoffs: list[Handoff] = []
        self._events: list[RunEvent] = []

    @property
    def receivers(self) -> tuple[str, ...]:
        """Who this desk may hand to, in declaration order."""
        return tuple(self._receivers)

    @property
    def handoffs(self) -> tuple[Handoff, ...]:
        """What has changed hands, in order. A refused handoff is not one of them."""
        return tuple(self._handoffs)

    @property
    def events(self) -> tuple[RunEvent, ...]:
        """Every transfer and every refusal, in order, for the source run's record."""
        return tuple(self._events)

    async def hand_off(
        self,
        to: str,
        *,
        reason: str,
        state: Mapping[str, Any] | BaseModel,
        task: str,
        history: Iterable[Message] = (),
        memory: Iterable[str] = (),
        after: Run[Any] | None = None,
    ) -> HandoffResult:
        """Hand the conversation to `to`, carrying `state` and nothing else.

        Args:
            to: Who takes it. One of `receivers`.
            reason: Why it is being handed over. Required: a handoff nobody explained is
                unreviewable afterwards.
            state: What crosses, validated against the receiver's contract first.
            task: What the target is being asked, in the words it will see.
            history: The conversation so far, where continuity matters more than brevity.
                Nothing is forwarded unless it is passed here.
            memory: Recalled text that travels with the user rather than with the agent.
            after: The source's run, where it has one. It has to have settled: a handoff
                made with a tool call still in flight leaves the call owned by nobody.

        Returns:
            What crossed and, for an agent receiver, the run it produced.

        Raises:
            HandoffContractError: If nobody answers to `to`, if the payload is not what
                the receiver declared, if the receiver holds nothing the source also
                holds, or if the source run has not settled. Each is checked before the
                target is invoked, so nothing is written and nothing partially moves.
            ConfigurationError: If no reason was given.
            GuardrailError: If a guard stopped the payload. The conversation stays with
                the source.
            DelegationLimitError: If the conversation has already travelled as far as its
                delegation ceilings allow.
        """
        if not reason.strip():
            raise ConfigurationError(
                "a handoff needs a reason; a chain of them without one is unreviewable"
            )
        receiver = self._receivers.get(to)
        if receiver is None:
            raise self._refusal(
                f"nobody on this desk answers to {to!r}; it hands to {', '.join(self._receivers)}",
                target=to,
                reason="unknown_target",
            )
        if after is not None and not after.state.is_terminal:
            raise self._refusal(
                f"the conversation is still running in {after.state}, so handing it over "
                f"now would leave whatever is in flight owned by nobody",
                target=to,
                reason="in_flight",
            )
        payload = self._checked(receiver, state)
        held = self._held_by(receiver)
        rendered = await self._rendered(payload)
        self._delegation.to(receiver.name, tools=held or None)
        caller = self._delegation.context
        handoff = Handoff(
            from_agent=self._agent.name,
            to_agent=receiver.name,
            reason=reason,
            state=payload.model_dump(),
            rendered=rendered,
            scope=held,
            trace=(*caller.path, receiver.name),
            run_id=caller.run_id,
            tenant=caller.tenant.tenant,
            user=caller.tenant.user,
        )
        run = await self._taken(receiver, handoff, task, history=history, memory=memory)
        self._handoffs.append(handoff)
        self._events.append(
            RunEvent(
                kind=RunEventKind.HANDED_OFF,
                name=receiver.name,
                detail=reason,
                usage=run.usage if run is not None else None,
            )
        )
        return HandoffResult(handoff=handoff, run=run)

    async def _taken(
        self,
        receiver: Receiver,
        handoff: Handoff,
        task: str,
        *,
        history: Iterable[Message],
        memory: Iterable[str],
    ) -> Run[Any] | None:
        """Give it to whoever takes it: a run for an agent, a queue for a person."""
        queue = receiver.queue
        if queue is not None:
            await queue.receive(handoff)
            return None
        caller = self._delegation.context
        return await self._runner.run(
            narrowed_to(cast("Agent[Any]", receiver.agent), handoff.scope),
            task,
            tenant=handoff.tenant,
            user=handoff.user,
            history=history,
            memory=(*memory, handoff.rendered),
            parent=caller.model_copy(update={"grant": self._grant()}),
        )

    def _checked(self, receiver: Receiver, state: Mapping[str, Any] | BaseModel) -> BaseModel:
        """Validate the payload, and say who was handing what to whom when it failed."""
        try:
            return receiver.contract.validated(state)
        except HandoffContractError as refused:
            raise self._refusal(
                str(refused), target=receiver.name, reason="contract", violations=refused.violations
            ) from refused

    def _held_by(self, receiver: Receiver) -> tuple[str, ...]:
        """What the target holds: its own allowlist, capped by the source's."""
        if receiver.agent is None:
            return ()
        source = frozenset(self._agent.tools) & self._delegation.scope.tools
        held = tuple(tool for tool in receiver.agent.tools if tool in source)
        if not held:
            raise self._refusal(
                f"{receiver.name!r} holds none of the tools the agent handing to it holds, "
                f"so it could not act on the conversation",
                target=receiver.name,
                reason="no_tools",
            )
        return held

    async def _rendered(self, payload: BaseModel) -> str:
        """The payload as the target will see it, redacted before anything records it."""
        rendered = payload.model_dump_json()
        if self._guardrails is None:
            return rendered
        return await self._guardrails.check_output(rendered)

    def _grant(self) -> RunGrant:
        """What the source holds, in the form the target inherits rather than declares."""
        return RunGrant(
            tools=tuple(sorted(frozenset(self._agent.tools) & self._delegation.scope.tools)),
            approval_required_tools=self._agent.approval_required_tools,
            guardrails=self._agent.guardrails,
        )

    def _refusal(
        self,
        message: str,
        *,
        target: str,
        reason: str,
        violations: tuple[str, ...] = (),
    ) -> HandoffContractError:
        """A refusal naming the source, the target and the run it was refused on."""
        self._events.append(RunEvent(kind=RunEventKind.HANDOFF_REFUSED, name=target, detail=reason))
        return HandoffContractError(
            message,
            source=self._agent.name,
            target=target,
            reason=reason,
            violations=violations,
            path=(*self._delegation.path, target),
            run_id=self._delegation.context.run_id,
            tenant=self._delegation.context.tenant.tenant,
        )
