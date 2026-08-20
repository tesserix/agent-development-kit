"""Calling a peer as a typed operation rather than as a message to another service.

Delegation between agents is usually an HTTP call carrying a prompt and a service token:
the caller parses free text back out of the answer, the peer sees the platform rather than
the person, a failure spanning both agents is two unrelated traces, and whatever the peer
spent is invisible to the caller's ceiling.

A `PeerClient` makes it a call. The payload is held to the schema the peer published
before anything is sent, the answer is held to the schema it published for the answer, the
delegation carries the original principal with the caller's scope narrowed rather than the
caller's identity, the hop is a child span of the run, and what the peer reports spending
lands on the run's budget. Every failure is typed, and none of them returns a partial or
invented answer in place of the peer's.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Protocol, Self, runtime_checkable

from pydantic import Field

from tesserix_adk.a2a.delegation import DEFAULT_TTL_SECONDS, DelegationChain, PeerDelegator
from tesserix_adk.core import (
    AdkError,
    AdkModel,
    CapabilityError,
    CountSource,
    DelegationLimitError,
    Usage,
    idempotency_key,
    redact,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Mapping

    from tesserix_adk.a2a.card import AgentCard, AgentSkill
    from tesserix_adk.core import AgentIdentity, Clock, CredentialSource, RunBudget
    from tesserix_adk.runtime import CancellationToken

__all__ = [
    "DEFAULT_MAX_REPORTED_TOKENS",
    "DEFAULT_PEER_TIMEOUT_SECONDS",
    "PeerCall",
    "PeerClient",
    "PeerInvocationError",
    "PeerInvocationReason",
    "PeerProgress",
    "PeerReply",
    "PeerResult",
    "PeerTransport",
    "StreamingPeerTransport",
    "TraceCarrier",
    "UnsupportedSchemaError",
    "checkable",
    "conforms",
]

DEFAULT_PEER_TIMEOUT_SECONDS = 30.0
"""How long one peer call may take when neither the caller nor the card says otherwise."""

DEFAULT_MAX_REPORTED_TOKENS = 1_000_000
"""The most a peer may claim one call consumed before the figure stops being believed."""

_ANNOTATIONS = frozenset(
    {
        "$comment",
        "$defs",
        "$id",
        "$schema",
        "default",
        "deprecated",
        "description",
        "examples",
        "format",
        "readOnly",
        "title",
        "writeOnly",
    }
)
_CHECKED = frozenset(
    {
        "$ref",
        "additionalProperties",
        "anyOf",
        "const",
        "enum",
        "items",
        "maxItems",
        "maxLength",
        "maximum",
        "minItems",
        "minLength",
        "minimum",
        "properties",
        "required",
        "type",
    }
)
_DEFS = "#/$defs/"


class PeerInvocationReason(StrEnum):
    """Why a peer call was refused, in terms the caller can act on.

    Args:
        UNKNOWN_SKILL: The card does not publish it, so there is nothing to call.
        INPUT_SCHEMA: The payload is not what the peer said it takes.
        OUTPUT_SCHEMA: The answer is not what the peer said it returns.
        SCHEMA_UNSUPPORTED: The published schema uses a keyword the kit does not check.
        SCOPE_ESCALATION: The skill needs more than the calling run holds.
        TOO_LARGE: The payload is beyond what the peer said it accepts.
        UNAVAILABLE: The peer published itself as degraded.
        TRANSPORT: The call did not reach the peer, or the peer did not answer it.
        TIMED_OUT: The peer did not answer inside the deadline.
        CANCELLED: The calling run stopped waiting.
        BUDGET: The run's remaining ceiling could not cover the call.
    """

    UNKNOWN_SKILL = "unknown_skill"
    INPUT_SCHEMA = "input_schema"
    OUTPUT_SCHEMA = "output_schema"
    SCHEMA_UNSUPPORTED = "schema_unsupported"
    SCOPE_ESCALATION = "scope_escalation"
    TOO_LARGE = "too_large"
    UNAVAILABLE = "unavailable"
    TRANSPORT = "transport"
    TIMED_OUT = "timed_out"
    CANCELLED = "cancelled"
    BUDGET = "budget"


class PeerInvocationError(AdkError):
    """A peer call that did not produce an answer the caller may use.

    Args:
        message: What went wrong, in terms of the call that was attempted.
        peer: Which agent.
        skill: Which skill.
        reason: Which of the documented refusals this is.
        payload: What arrived, where anything did, for whoever has to explain it. Kept on
            the exception and never put in `details`, which reaches logs.
    """

    def __init__(
        self,
        message: str,
        *,
        peer: str,
        skill: str,
        reason: PeerInvocationReason,
        payload: str = "",
    ) -> None:
        self.peer = peer
        self.skill = skill
        self.reason = reason
        self.payload = payload
        super().__init__(message, details={"peer": peer, "skill": skill, "reason": str(reason)})


class UnsupportedSchemaError(AdkError):
    """A published schema uses a keyword `conforms` does not implement.

    Raised rather than ignored: a keyword that is skipped is a constraint the caller
    believes it checked.

    Args:
        message: Which keyword, and where in the schema.
        keyword: The keyword itself.
    """

    def __init__(self, message: str, *, keyword: str) -> None:
        self.keyword = keyword
        super().__init__(message, details={"keyword": keyword})


def conforms(value: object, schema: Mapping[str, Any]) -> str:
    """Whether `value` satisfies `schema`, as the empty string or the first violation.

    The subset is the one an agent card carries: types, `properties`, `required`,
    `additionalProperties`, `items`, `enum`, `const`, `anyOf`, local `$ref`, and the
    length, size and range bounds. Annotation keywords are ignored, as a JSON Schema
    validator ignores them; `format` is an annotation there and is one here too.

    Raises:
        UnsupportedSchemaError: If the schema uses anything else. Nothing is passed
            because the kit could not check it.

    Example:
        >>> conforms({"leg": "LHR"}, {"type": "object", "required": ["leg"]})
        ''
    """
    return _checked(value, schema, schema, "")


def checkable(schema: Mapping[str, Any], *, path: str = "") -> None:
    """Refuse a schema `conforms` could not hold a payload to, before anything relies on it.

    Walked whole, so a keyword buried in a nested property is found while a tool is being
    built rather than on the first call that happens to carry that field.

    Raises:
        UnsupportedSchemaError: The schema uses a keyword outside the subset.
    """
    _within_the_subset(schema, path)
    for keyword in ("additionalProperties", "items"):
        nested = schema.get(keyword)
        if isinstance(nested, dict):
            checkable(nested, path=_at(path, keyword))
    for keyword in ("$defs", "properties"):
        for name, nested in schema.get(keyword, {}).items():
            checkable(nested, path=_at(path, name))
    for index, branch in enumerate(schema.get("anyOf", ())):
        checkable(branch, path=f"{path}[{index}]")


class PeerProgress(AdkModel):
    """Something the peer said while it was still working.

    Args:
        note: What it is doing, as data for a human. Never an instruction to a model.
        fraction: How far through it says it is, from 0 to 1, where it says.
    """

    note: str = ""
    fraction: float | None = None


class PeerReply(AdkModel):
    """What a transport read off the wire, before anything here has believed it.

    Args:
        output: The peer's answer, unvalidated.
        usage: What the peer says it consumed, where it reports anything.
    """

    output: dict[str, Any] = Field(default_factory=dict)
    usage: Usage | None = None


class PeerResult(AdkModel):
    """A peer's answer, validated against the schema the peer published for it.

    Args:
        peer: Which agent answered.
        skill: Which skill.
        call_id: The identifier the call carried, so a peer's own logs join up.
        output: The answer, conforming to the declared output schema where there is one.
        usage: What the call consumed, as recorded against the calling run.
        usage_trusted: Whether the peer's figures were taken as reported. False where it
            reported nothing or reported more than the caller will believe.
        chain: The agents the work has passed through, this caller last.
    """

    peer: str
    skill: str
    call_id: str
    output: dict[str, Any] = Field(default_factory=dict)
    usage: Usage = Usage(input_tokens=0, output_tokens=0)
    usage_trusted: bool = True
    chain: tuple[str, ...] = ()

    def attributes(self) -> dict[str, str]:
        """What the calling run's span records: the call, never the answer."""
        return {
            "a2a.peer": self.peer,
            "a2a.skill": self.skill,
            "a2a.call": self.call_id,
            "a2a.chain": " -> ".join(self.chain),
            "a2a.usage_trusted": str(self.usage_trusted).lower(),
        }

    def redacted(self, *, tenant: str = "") -> dict[str, Any]:
        """The answer with identifiers replaced, for anything that stores or logs it."""
        return {name: _redacted(value, tenant) for name, value in self.output.items()}


@dataclass(frozen=True, slots=True)
class PeerCall:
    """One outbound call, as a transport receives it.

    Args:
        peer: Which agent is being called.
        skill: Which skill.
        payload: The arguments, already held to the peer's input schema.
        meta: The delegation and trace context, carrying no credential.
        headers: The transport headers, which do carry one. Never rendered.
        call_id: This call, stable across a replay of the same run.
        idempotency_key: What the peer deduplicates a retry on, where the skill has an
            effect. `None` where the peer declared the skill repeatable.
        deadline_seconds: How long the caller will wait.
    """

    peer: str
    skill: str
    payload: Mapping[str, Any]
    meta: Mapping[str, str]
    call_id: str
    deadline_seconds: float
    idempotency_key: str | None = None
    headers: Mapping[str, str] = field(default_factory=dict, repr=False)


@runtime_checkable
class TraceCarrier(Protocol):
    """Where the caller sits in a trace. `observability.W3CContext` satisfies it.

    Structural on purpose: the trace format is the observability package's business, and
    an agent calling a peer should not have to depend on it to stay in one trace.
    """

    def child(self, name: str, *, attempt: int = 1) -> Self:
        """The context for one operation inside this one."""

    def carried(self) -> dict[str, str]:
        """The headers to put on the hop."""


@runtime_checkable
class PeerTransport(Protocol):
    """How a call reaches the peer. Every transport the kit ships lives outside `a2a`."""

    async def invoke(self, call: PeerCall) -> PeerReply:
        """Send the call and return what came back, unvalidated."""

    async def cancel(self, call: PeerCall) -> None:
        """Tell the peer nobody is waiting. Best effort, and never raises at the caller."""


@runtime_checkable
class StreamingPeerTransport(PeerTransport, Protocol):
    """A transport that can also report progress while the peer works."""

    def stream(self, call: PeerCall) -> AsyncIterator[PeerProgress | PeerReply]:
        """Yield progress as it arrives, and the reply last."""


class PeerClient:
    """Calls one peer's skills, typed both ways and inside the caller's run.

    A client is bound to one resolved card. Resolution is `PeerDiscovery`'s job, and a
    client for a card that has since changed is replaced rather than patched: the schemas
    a call is held to must be the ones the peer published for that version.

    Args:
        card: The peer as it published itself.
        transport: How the call actually travels. The kit owns none of that.
        credentials: Where a per-call credential for the peer's audience comes from.
        identity: Who the calling run acts for and what it holds.
        run_id: The calling run, carried through as attribution.
        clock: Source of time, for the delegation's lifetime.
        budget: The run's ceiling. A call the remainder cannot cover is refused before it
            is made, and what the peer reports is spent from it afterwards.
        trace: Where the caller sits in the trace, so the peer joins it rather than
            starting a second one.
        chain: What the work has already passed through, for an onward hop. A peer already
            in it is refused as a cycle.
        agent_version: Which build of the caller is calling.
        timeout_seconds: The ceiling on one call. A deadline passed to `invoke` may only
            narrow it: a peer cannot be given longer than the run has left.
        ttl_seconds: How long each minted delegation is good for.
        max_reported_tokens: The most a peer may claim one call consumed before the
            figures are bounded and flagged.
    """

    __slots__ = (
        "_budget",
        "_calls",
        "_card",
        "_chain",
        "_delegator",
        "_identity",
        "_max_reported",
        "_run_id",
        "_skills",
        "_timeout",
        "_trace",
        "_transport",
        "_version",
    )

    def __init__(
        self,
        card: AgentCard,
        transport: PeerTransport,
        *,
        credentials: CredentialSource,
        identity: AgentIdentity,
        run_id: str,
        clock: Clock,
        budget: RunBudget | None = None,
        trace: TraceCarrier | None = None,
        chain: DelegationChain | None = None,
        agent_version: str = "1.0.0",
        timeout_seconds: float = DEFAULT_PEER_TIMEOUT_SECONDS,
        ttl_seconds: float = DEFAULT_TTL_SECONDS,
        max_reported_tokens: int = DEFAULT_MAX_REPORTED_TOKENS,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive; a call nobody waits for is none")
        self._card = card
        self._transport = transport
        self._delegator = PeerDelegator(credentials, clock=clock, ttl_seconds=ttl_seconds)
        self._identity = identity
        self._run_id = run_id
        self._budget = budget
        self._trace = trace
        self._chain = chain or DelegationChain()
        self._version = agent_version
        self._timeout = timeout_seconds
        self._max_reported = max_reported_tokens
        self._skills = {skill.name: skill for skill in card.skills}
        self._calls = 0

    @property
    def card(self) -> AgentCard:
        """The peer's card, as everything this client holds a call to."""
        return self._card

    @property
    def timeout_seconds(self) -> float:
        """The ceiling on one call, for a caller assembling something around this client."""
        return self._timeout

    async def invoke(
        self,
        skill: str,
        payload: Mapping[str, Any],
        *,
        idempotency_key: str | None = None,
        deadline_seconds: float | None = None,
        cancellation: CancellationToken | None = None,
    ) -> PeerResult:
        """Call one skill and return its validated answer.

        Args:
            skill: What to call, as the card names it.
            payload: The arguments, checked against the card before anything is sent.
            idempotency_key: What the peer should deduplicate on, where the caller is
                resuming work and knows the key the first attempt used.
            deadline_seconds: How long to wait, which may only narrow the client's own.
            cancellation: The run's switch. Flipping it stops the wait and tells the peer.

        Returns:
            The answer, held to the peer's declared output schema.

        Raises:
            PeerInvocationError: For every documented refusal, carrying the peer, the
                skill and what arrived. No partial or invented answer is returned instead.
            DelegationLimitError: If the hop would close a cycle or exceed the chain depth.
            AuthorisationError: If no credential could be minted for the peer.
            BudgetExceededError: If what the peer reported has taken the run past a ceiling.
        """
        declared = self._declared(skill)
        call = await self._prepared(declared, payload, idempotency_key, cancellation)
        reply = await self._answered(call, deadline_seconds, cancellation)
        return await self._accepted(call, declared, reply)

    async def stream(
        self,
        skill: str,
        payload: Mapping[str, Any],
        *,
        idempotency_key: str | None = None,
        cancellation: CancellationToken | None = None,
    ) -> AsyncIterator[PeerProgress | PeerResult]:
        """Call one skill, yielding the peer's progress and finally its validated answer.

        The last item is always a `PeerResult`; anything before it is a `PeerProgress`.
        The answer is held to the same schema as `invoke`'s, so a long-running call is not
        a way to receive something that was never checked.

        Args:
            skill: What to call.
            payload: The arguments, checked before anything is sent.
            idempotency_key: What the peer should deduplicate on.
            cancellation: The run's switch, checked between events.

        Raises:
            CapabilityError: If the transport does not stream.
            PeerInvocationError: As `invoke`, and where the run was cancelled mid-answer.
        """
        declared = self._declared(skill)
        streaming = getattr(self._transport, "stream", None)
        if streaming is None:
            raise CapabilityError(
                f"{type(self._transport).__name__} does not stream, so {skill} cannot be "
                f"followed while it runs",
                capability="a2a.stream",
            )
        call = await self._prepared(declared, payload, idempotency_key, cancellation)
        async for event in streaming(call):
            if cancellation is not None and cancellation.cancelled:
                await self._told_to_stop(call)
                raise self._refused(declared.name, PeerInvocationReason.CANCELLED)
            if isinstance(event, PeerProgress):
                yield event
            else:
                yield await self._accepted(call, declared, event)

    def _declared(self, skill: str) -> AgentSkill:
        published = self._skills.get(skill)
        if published is None:
            raise self._refused(skill, PeerInvocationReason.UNKNOWN_SKILL)
        return published

    async def _prepared(
        self,
        declared: AgentSkill,
        payload: Mapping[str, Any],
        key: str | None,
        cancellation: CancellationToken | None,
    ) -> PeerCall:
        """Everything that has to be true before the peer hears about the call at all."""
        if cancellation is not None and cancellation.cancelled:
            raise self._refused(declared.name, PeerInvocationReason.CANCELLED)
        if not self._card.available:
            raise self._refused(declared.name, PeerInvocationReason.UNAVAILABLE)
        body = json.dumps(payload, sort_keys=True, default=str).encode()
        ceiling = self._card.limits.max_payload_bytes if self._card.limits else len(body)
        if len(body) > ceiling:
            raise self._refused(declared.name, PeerInvocationReason.TOO_LARGE)
        self._holds(declared)
        self._not_a_loop()
        self._room(declared)
        violation = self._conforming(declared, payload, declared.input_schema)
        if violation:
            raise self._refused(declared.name, PeerInvocationReason.INPUT_SCHEMA, detail=violation)
        delegation = await self._delegator.delegate(
            identity=self._identity,
            peer=self._card,
            needs=declared.required_scopes or self._card.required_scopes or self._card.declared,
            run_id=self._run_id,
            chain=self._chain,
            agent_version=self._version,
        )
        self._calls += 1
        call_id = f"{self._run_id}:{self._card.agent}:{declared.name}:{self._calls}"
        return PeerCall(
            peer=self._card.agent,
            skill=declared.name,
            payload=dict(payload),
            meta={**delegation.meta(), **self._traced(declared.name)},
            headers=delegation.headers(),
            call_id=call_id,
            deadline_seconds=self._timeout,
            idempotency_key=self._deduplicated(declared, payload, key),
        )

    def _holds(self, declared: AgentSkill) -> None:
        """Refuse a skill that needs more than the run holds, rather than asking for it."""
        held = set(self._identity.effective)
        absent = [scope for scope in declared.required_scopes if scope not in held]
        if absent:
            raise self._refused(
                declared.name,
                PeerInvocationReason.SCOPE_ESCALATION,
                detail=f"it needs {absent[0]!r}, which this run does not hold",
            )

    def _not_a_loop(self) -> None:
        """Refuse a peer the work has already been through, the caller itself included.

        The chain records who passed the work on, so the peer is not in it yet; A calling
        B calling A is caught here rather than one hop later, where it would already have
        been made.

        Raises:
            DelegationLimitError: If the call would close a cycle.
        """
        travelled = (*self._chain.agents, self._identity.agent)
        if self._card.agent in travelled:
            raise DelegationLimitError(
                f"{self._card.agent} is already in {' -> '.join(travelled)}, "
                f"so calling it would be a cycle",
                reason="cycle",
                path=(*travelled, self._card.agent),
            )

    def _room(self, declared: AgentSkill) -> None:
        """Refuse before the call where the remaining ceiling could not cover it."""
        if self._budget is None:
            return
        if not self._budget.check().permitted:
            raise self._refused(
                declared.name, PeerInvocationReason.BUDGET, detail="the run has no budget left"
            )
        left = self._budget.limits().max_peer_invocations
        if left is not None and left < 1:
            raise self._refused(
                declared.name,
                PeerInvocationReason.BUDGET,
                detail="the run has used every peer call it was allowed",
            )

    def _conforming(self, declared: AgentSkill, value: object, schema: Mapping[str, Any]) -> str:
        try:
            return conforms(value, schema)
        except UnsupportedSchemaError as unchecked:
            raise self._refused(
                declared.name,
                PeerInvocationReason.SCHEMA_UNSUPPORTED,
                detail=str(unchecked),
            ) from unchecked

    def _deduplicated(
        self, declared: AgentSkill, payload: Mapping[str, Any], key: str | None
    ) -> str | None:
        if key is not None:
            return key
        if declared.idempotent:
            return None
        return idempotency_key(
            tenant=self._identity.principal.tenant,
            run_id=self._run_id,
            tool=f"{self._card.agent}/{declared.name}",
            arguments=payload,
        )

    def _traced(self, skill: str) -> dict[str, str]:
        if self._trace is None:
            return {}
        return dict(self._trace.child(f"a2a:{self._card.agent}/{skill}").carried())

    async def _answered(
        self,
        call: PeerCall,
        deadline_seconds: float | None,
        cancellation: CancellationToken | None,
    ) -> PeerReply:
        """Wait for the peer, no longer than the run has, and tell it when the wait ends."""
        limit = min(self._timeout, deadline_seconds) if deadline_seconds else self._timeout
        work = asyncio.ensure_future(self._transport.invoke(call))
        stop = asyncio.ensure_future(cancellation.wait()) if cancellation is not None else None
        waiting = {work} if stop is None else {work, stop}
        done, _ = await asyncio.wait(waiting, timeout=limit, return_when=asyncio.FIRST_COMPLETED)
        if stop is not None and stop not in done:
            await _dropped(stop)
        if work in done:
            try:
                return work.result()
            except Exception as failure:
                raise self._refused(
                    call.skill, PeerInvocationReason.TRANSPORT, detail=str(failure)
                ) from failure
        await _dropped(work)
        await self._told_to_stop(call)
        cancelled = cancellation is not None and cancellation.cancelled
        raise self._refused(
            call.skill,
            PeerInvocationReason.CANCELLED if cancelled else PeerInvocationReason.TIMED_OUT,
            detail=f"after {limit}s",
        )

    async def _told_to_stop(self, call: PeerCall) -> None:
        """Tell the peer nobody is waiting, so an abandoned call is not still billing."""
        with contextlib.suppress(Exception):
            await self._transport.cancel(call)

    async def _accepted(self, call: PeerCall, declared: AgentSkill, reply: PeerReply) -> PeerResult:
        """Hold the answer to what the peer published, then charge what it says it cost."""
        if declared.output_schema is not None:
            violation = self._conforming(declared, reply.output, declared.output_schema)
            if violation:
                raise self._refused(
                    declared.name,
                    PeerInvocationReason.OUTPUT_SCHEMA,
                    detail=violation,
                    payload=json.dumps(reply.output, sort_keys=True, default=str),
                )
        usage, trusted = _bounded(reply.usage, self._max_reported)
        if self._budget is not None:
            await self._budget.record(usage, peer_invocations=1)
        return PeerResult(
            peer=call.peer,
            skill=call.skill,
            call_id=call.call_id,
            output=dict(reply.output),
            usage=usage,
            usage_trusted=trusted,
            chain=(*self._chain.agents, self._identity.agent),
        )

    def _refused(
        self,
        skill: str,
        reason: PeerInvocationReason,
        *,
        detail: str = "",
        payload: str = "",
    ) -> PeerInvocationError:
        said = f": {detail}" if detail else ""
        return PeerInvocationError(
            f"{self._card.agent}/{skill} was not called ({reason}){said}"
            if reason is not PeerInvocationReason.OUTPUT_SCHEMA
            else f"{self._card.agent}/{skill} answered with something it does not publish{said}",
            peer=self._card.agent,
            skill=skill,
            reason=reason,
            payload=payload,
        )


async def _dropped(task: asyncio.Future[Any]) -> None:
    """Stop waiting on a task and swallow whatever it was going to say."""
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)


def _bounded(reported: Usage | None, ceiling: int) -> tuple[Usage, bool]:
    """What to charge the run, and whether the peer's own figures were believed."""
    if reported is None:
        return Usage(input_tokens=0, output_tokens=0, source=CountSource.HEURISTIC), False
    if reported.input_tokens <= ceiling and reported.output_tokens <= ceiling:
        return reported, True
    return (
        reported.model_copy(
            update={
                "input_tokens": min(reported.input_tokens, ceiling),
                "output_tokens": min(reported.output_tokens, ceiling),
                "source": CountSource.HEURISTIC,
            }
        ),
        False,
    )


def _redacted(value: object, tenant: str) -> Any:  # noqa: ANN401 — mirrors the JSON it is given
    if isinstance(value, str):
        return redact(value, tenant=tenant).text
    if isinstance(value, dict):
        return {name: _redacted(item, tenant) for name, item in value.items()}
    if isinstance(value, list):
        return [_redacted(item, tenant) for item in value]
    return value


def _checked(value: object, schema: Mapping[str, Any], root: Mapping[str, Any], path: str) -> str:
    """The first violation of `schema` in `value`, or the empty string."""
    _within_the_subset(schema, path)
    if "$ref" in schema:
        return _checked(value, _resolved(schema["$ref"], root), root, path)
    for check in (_typed, _enumerated, _one_of, _bounded_value, _shaped):
        violation = check(value, schema, root, path)
        if violation:
            return violation
    return ""


def _typed(value: object, schema: Mapping[str, Any], root: Mapping[str, Any], path: str) -> str:
    del root
    stated = schema.get("type")
    if stated is None:
        return ""
    names = (stated,) if isinstance(stated, str) else tuple(stated)
    if any(_is(value, name) for name in names):
        return ""
    return f"{path or 'the payload'}: expected {' or '.join(names)}"


def _enumerated(
    value: object, schema: Mapping[str, Any], root: Mapping[str, Any], path: str
) -> str:
    del root
    if "const" in schema and value != schema["const"]:
        return f"{path or 'the payload'}: expected {schema['const']!r}"
    allowed = schema.get("enum")
    if allowed is not None and value not in allowed:
        return f"{path or 'the payload'}: not one of the published values"
    return ""


def _one_of(value: object, schema: Mapping[str, Any], root: Mapping[str, Any], path: str) -> str:
    branches = schema.get("anyOf")
    if branches is None:
        return ""
    if any(not _checked(value, branch, root, path) for branch in branches):
        return ""
    return f"{path or 'the payload'}: matches none of the published alternatives"


def _bounded_value(
    value: object, schema: Mapping[str, Any], root: Mapping[str, Any], path: str
) -> str:
    del root
    where = path or "the payload"
    if isinstance(value, str):
        return _between(len(value), schema, "minLength", "maxLength", f"{where}: length")
    if isinstance(value, bool):
        return ""
    if isinstance(value, int | float):
        return _between(value, schema, "minimum", "maximum", where)
    if isinstance(value, list):
        return _between(len(value), schema, "minItems", "maxItems", f"{where}: items")
    return ""


def _shaped(value: object, schema: Mapping[str, Any], root: Mapping[str, Any], path: str) -> str:
    if isinstance(value, dict):
        return _object(value, schema, root, path)
    if isinstance(value, list):
        items = schema.get("items")
        if items is None:
            return ""
        for index, item in enumerate(value):
            violation = _checked(item, items, root, f"{path}[{index}]")
            if violation:
                return violation
    return ""


def _object(
    value: Mapping[str, Any], schema: Mapping[str, Any], root: Mapping[str, Any], path: str
) -> str:
    properties: Mapping[str, Any] = schema.get("properties", {})
    for name in schema.get("required", ()):
        if name not in value:
            return f"{_at(path, name)} is required and was not sent"
    if schema.get("additionalProperties") is False:
        unknown = sorted(set(value) - set(properties))
        if unknown:
            return f"{_at(path, unknown[0])} is not a property the peer publishes"
    for name, item in value.items():
        declared = properties.get(name)
        if declared is not None:
            violation = _checked(item, declared, root, _at(path, name))
            if violation:
                return violation
    return ""


def _between(value: float, schema: Mapping[str, Any], low: str, high: str, where: str) -> str:
    floor = schema.get(low)
    ceiling = schema.get(high)
    if floor is not None and value < floor:
        return f"{where} is below {floor}"
    if ceiling is not None and value > ceiling:
        return f"{where} is above {ceiling}"
    return ""


def _resolved(reference: str, root: Mapping[str, Any]) -> Mapping[str, Any]:
    """A local definition, or a refusal — a schema the kit cannot follow is unchecked."""
    if not reference.startswith(_DEFS):
        raise UnsupportedSchemaError(
            f"{reference!r} points outside the schema, which the kit does not fetch",
            keyword="$ref",
        )
    name = reference.removeprefix(_DEFS)
    definition = root.get("$defs", {}).get(name)
    if definition is None:
        raise UnsupportedSchemaError(
            f"{reference!r} names a definition the schema does not carry", keyword="$ref"
        )
    resolved: Mapping[str, Any] = definition
    return resolved


def _is(value: object, name: str) -> bool:
    """Whether `value` is of the named JSON type, with JSON's own idea of the types."""
    if name == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if name == "number":
        return isinstance(value, int | float) and not isinstance(value, bool)
    if name == "boolean":
        return isinstance(value, bool)
    if name == "string":
        return isinstance(value, str)
    if name == "null":
        return value is None
    if name == "object":
        return isinstance(value, dict)
    if name == "array":
        return isinstance(value, list)
    raise UnsupportedSchemaError(f"{name!r} is not a JSON type the kit knows", keyword="type")


def _within_the_subset(schema: Mapping[str, Any], path: str) -> None:
    """Refuse a keyword the kit does not check, rather than pass what it could not check."""
    unknown = set(schema) - _CHECKED - _ANNOTATIONS
    if unknown:
        keyword = sorted(unknown)[0]
        raise UnsupportedSchemaError(
            f"{keyword!r} at {path or 'the root'} is not a keyword the kit checks, so the "
            f"payload was not passed as checked",
            keyword=keyword,
        )


def _at(path: str, name: str) -> str:
    return f"{path}.{name}" if path else name
