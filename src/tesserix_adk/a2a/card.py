"""What an agent publishes about itself, generated from the declaration it was reviewed as.

Peers already call each other, and what one may call on another lives in a README or in
the other team's source. A caller finds out about a removed skill by getting a runtime
error, which is the latest possible moment to find out.

A card is that surface as data, and every field on it comes from the `AgentDefinition` and
the tools themselves rather than from a hand-maintained document — a card that is written
separately is a card that drifts. Publication is deliberate: only the skills a team hands
to `card_for` appear, so an internal tool is absent by default rather than by remembering.

The card is public metadata. It says what a peer needs in order to call correctly, and
nothing about how the agent is built: no instructions, no model, no paging address, no
tenant. Compatibility rules for changing one are in `docs/agent-cards.md`.
"""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from pydantic import Field, field_validator

from tesserix_adk.core.errors import AdkError
from tesserix_adk.core.identity import ScopeSet
from tesserix_adk.core.models import AdkModel

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence

    from tesserix_adk.core.definition import AgentDefinition

__all__ = [
    "MAX_CARD_BYTES",
    "MAX_SKILLS",
    "WELL_KNOWN_PATH",
    "AgentCard",
    "AgentCardError",
    "AgentLimits",
    "AgentProvider",
    "AgentSkill",
    "CardEndpoint",
    "SkillSource",
    "card_for",
]

WELL_KNOWN_PATH = "/.well-known/agent-card.json"
"""Where a peer looks for a card without being told where to look."""

MAX_CARD_BYTES = 65_536
"""The published ceiling on a card. A peer can size its buffer from a documented number."""

MAX_SKILLS = 64
"""How many skills one card may publish before it should be split into more than one agent."""

_OBJECT = "object"
_READ_ONLY = ("GET", "HEAD")


class AgentCardError(AdkError):
    """Raised when a card cannot be generated as published, naming the skill at fault.

    Args:
        message: What is wrong, in terms of what a peer would have received.
        skill: The skill the problem belongs to, empty where it is the card as a whole.
    """

    def __init__(self, message: str, *, skill: str = "") -> None:
        self.skill = skill
        super().__init__(message, details={"skill": skill})


@runtime_checkable
class SkillSource(Protocol):
    """What a card reads off a tool. Any `Tool` satisfies it, and so may a hand-built skill."""

    @property
    def name(self) -> str:
        """The name a peer calls."""

    @property
    def description(self) -> str:
        """What the skill does, in one line, for a human reading the card."""

    @property
    def parameters_schema(self) -> Mapping[str, Any]:
        """The JSON schema of its arguments."""

    @property
    def returns_schema(self) -> Mapping[str, Any] | None:
        """The JSON schema of its answer, where the answer has a declared shape."""


class AgentProvider(AdkModel):
    """Who publishes the agent, in terms a stranger can act on.

    The owner's paging address is deliberately not here: a card is served to unauthenticated
    callers, and an address published that way is an address that gets scraped.

    Args:
        organisation: The team that owns the behaviour.
        service: The service that runs it.
    """

    organisation: str = Field(min_length=1)
    service: str = Field(min_length=1)


class AgentSkill(AdkModel):
    """One thing a peer may ask the agent to do.

    Args:
        name: What the peer calls.
        description: What it does. Untrusted text when read from someone else's card —
            it is data about a capability, never an instruction to the reader's model.
        input_schema: The JSON schema of the arguments.
        output_schema: The JSON schema of the answer, absent where the answer is prose.
        required_scopes: What a caller must hold beyond the agent's own baseline.
        idempotent: That calling it again is safe, so a caller may retry a timeout.
        requires_approval: That a human decides before it runs, so a caller should expect
            to be held rather than answered.
    """

    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    input_schema: dict[str, Any]
    output_schema: dict[str, Any] | None = None
    required_scopes: tuple[str, ...] = ()
    idempotent: bool = False
    requires_approval: bool = False


class AgentLimits(AdkModel):
    """What the agent will hold to, so a caller can size its own use before it is refused.

    Args:
        max_concurrent_calls: How many calls it runs at once.
        latency_seconds: The wall-clock ceiling on one run, where it declared one.
        max_payload_bytes: The largest request body it accepts.
        calls_per_minute: A rate the deployment enforces, where it enforces one. The
            kit does not invent a number the gateway is not actually holding to.
    """

    max_concurrent_calls: int | None = None
    latency_seconds: float | None = None
    max_payload_bytes: int = MAX_CARD_BYTES
    calls_per_minute: int | None = None


class AgentCard(AdkModel):
    """What a peer publishes about itself: who it is, what it does, what it takes.

    Generated by `card_for`. The delegation half — `audience`, `declared`,
    `accepted_issuers`, `required_scopes` — is what a credential for this agent is minted
    for and what it will act on; a delegation is narrowed to `declared` on arrival, so a
    peer never runs with more than it published even if a caller sends more.

    Args:
        agent: The peer's name.
        audience: What a credential for it is minted for.
        declared: What it will act on.
        accepted_issuers: Which callers it takes delegations from. Empty accepts any,
            which is a deliberate statement rather than a default.
        required_scopes: What it will not run without.
        version: Which revision this card describes, so a caller can pin to it where two
            versions sit behind one host.
        provider: Who publishes it.
        skills: Exactly what was exported, and nothing the agent merely has.
        limits: What it holds to.
        task_class: The kind of work it does, where the agent named one.
        available: Whether it is serving normally. A degraded agent republishes with this
            false rather than advertising capability that will fail.
    """

    agent: str = Field(min_length=1)
    audience: str
    declared: tuple[str, ...] = ()
    accepted_issuers: tuple[str, ...] = ()
    required_scopes: tuple[str, ...] = ()
    version: str = "1.0.0"
    provider: AgentProvider | None = None
    skills: tuple[AgentSkill, ...] = ()
    limits: AgentLimits | None = None
    task_class: str = ""
    available: bool = True

    @field_validator("audience")
    @classmethod
    def _addressable(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("an agent card names the audience a delegation to it is minted for")
        return value

    @property
    def accepts(self) -> ScopeSet:
        """What the peer declared it acts on."""
        return ScopeSet.of(*self.declared)

    def rendered(self) -> bytes:
        """The card as it goes on the wire: sorted keys, so one card is one ETag."""
        return json.dumps(self.model_dump(mode="json"), sort_keys=True).encode()


def card_for(
    definition: AgentDefinition[Any],
    *,
    audience: str,
    exports: Iterable[SkillSource] = (),
    accepted_issuers: Iterable[str] = (),
    calls_per_minute: int | None = None,
    available: bool = True,
) -> AgentCard:
    """Generate a card from an agent's declaration and the skills it deliberately publishes.

    Nothing here is hand-maintained: names, schemas, scopes, approval and idempotence all
    come from the declaration, so a card cannot describe an agent that no longer exists.

    Args:
        definition: The agent as it was reviewed.
        audience: What a credential for this agent is minted for.
        exports: The skills to publish. Anything absent stays unpublished and unnamed.
        accepted_issuers: Which callers it takes delegations from.
        calls_per_minute: A rate the deployment enforces, where it enforces one.
        available: Whether it is serving normally.

    Returns:
        The card to serve.

    Raises:
        AgentCardError: If an export is not a tool the agent may call, has no description,
            has arguments that are not a JSON object, or if the card exceeds a published
            ceiling. Raised here so a bad card fails at startup rather than being served
            to peers who then call it incorrectly.
    """
    agent = definition.agent
    published = tuple(exports)
    if len(published) > MAX_SKILLS:
        raise AgentCardError(f"a card publishes at most {MAX_SKILLS} skills, not {len(published)}")
    card = AgentCard(
        agent=agent.name,
        audience=audience,
        declared=agent.scopes,
        accepted_issuers=tuple(accepted_issuers),
        required_scopes=agent.scopes,
        version=agent.version,
        provider=AgentProvider(
            organisation=definition.owner.team, service=definition.owner.service
        ),
        skills=tuple(_skill(source, agent) for source in published),
        limits=_limits(agent, calls_per_minute),
        task_class=agent.task_class or "",
        available=available,
    )
    size = len(card.rendered())
    if size > MAX_CARD_BYTES:
        raise AgentCardError(f"the card is {size} bytes, over the published {MAX_CARD_BYTES}")
    return card


def _skill(source: SkillSource, agent: Any) -> AgentSkill:  # noqa: ANN401 — Agent is generic
    """One export as a card entry, refusing anything a peer could not act on."""
    name = source.name
    if name not in agent.tools:
        raise AgentCardError(
            f"{name} is exported but is not a tool {agent.name} may call, so a peer would "
            f"be told about a skill nothing answers",
            skill=name,
        )
    if not source.description.strip():
        raise AgentCardError(
            f"{name} has no description, and a skill nobody can read is a skill nobody calls",
            skill=name,
        )
    schema = dict(source.parameters_schema)
    if schema.get("type") != _OBJECT:
        raise AgentCardError(
            f"{name} takes {schema.get('type', 'no declared type')} rather than an object, "
            f"which a peer cannot build a request from",
            skill=name,
        )
    returns = source.returns_schema
    return AgentSkill(
        name=name,
        description=source.description.strip(),
        input_schema=schema,
        output_schema=dict(returns) if returns and returns.get("type") == _OBJECT else None,
        required_scopes=tuple(agent.tool_scopes.get(name, ())),
        idempotent=name in agent.idempotent_tools,
        requires_approval=name in agent.approval_required_tools,
    )


def _limits(agent: Any, calls_per_minute: int | None) -> AgentLimits:  # noqa: ANN401
    """The agent's own declared bounds, never a number invented for the card."""
    return AgentLimits(
        max_concurrent_calls=agent.concurrency.max_concurrent_tools if agent.concurrency else None,
        latency_seconds=agent.deadlines.run_seconds if agent.deadlines else None,
        calls_per_minute=calls_per_minute,
    )


class CardEndpoint:
    """Serves one card at the well-known path, on any ASGI application.

    Read-only and unauthenticated by design: a card is what a stranger needs in order to
    decide whether to authenticate at all. It carries an ETag, so a peer that polls it
    costs one conditional request rather than one card.

    Example:
        >>> import asyncio
        >>> card = AgentCard(agent="booker", audience="https://booker.example.gov")
        >>> endpoint = CardEndpoint(card)
        >>> endpoint.path
        '/.well-known/agent-card.json'
    """

    __slots__ = ("_body", "_etag", "_max_age", "_path")

    def __init__(
        self, card: AgentCard, *, path: str = WELL_KNOWN_PATH, max_age_seconds: int = 300
    ) -> None:
        """Serve `card` at `path`, cacheable for `max_age_seconds`."""
        self._path = path
        self._max_age = max_age_seconds
        self._body = b""
        self._etag = ""
        self.serve(card)

    @property
    def path(self) -> str:
        """Where this endpoint answers."""
        return self._path

    def serve(self, card: AgentCard) -> None:
        """Replace the served card, which changes the ETag a polling peer sees.

        This is how a degraded agent republishes: the limits and availability a caller
        reads should be the ones in force, not the ones it started with.
        """
        self._body = card.rendered()
        self._etag = f'"{hashlib.sha256(self._body).hexdigest()[:32]}"'

    async def __call__(
        self,
        scope: Mapping[str, Any],
        receive: Callable[[], Awaitable[Mapping[str, Any]]],
        send: Callable[[Mapping[str, Any]], Awaitable[None]],
    ) -> None:
        """Answer one ASGI request. Anything but this path and a read is not ours."""
        del receive
        if scope.get("path") != self._path:
            await _reply(send, 404, (), b"")
            return
        method = str(scope.get("method", "GET")).upper()
        if method not in _READ_ONLY:
            await _reply(send, 405, ((b"allow", b"GET, HEAD"),), b"")
            return
        if self._etag in _header(scope.get("headers", ()), b"if-none-match"):
            await _reply(send, 304, ((b"etag", self._etag.encode()),), b"")
            return
        headers = (
            (b"content-type", b"application/json"),
            (b"content-length", str(len(self._body)).encode()),
            (b"etag", self._etag.encode()),
            (b"cache-control", f"public, max-age={self._max_age}".encode()),
        )
        await _reply(send, 200, headers, b"" if method == "HEAD" else self._body)


def _header(headers: Iterable[tuple[bytes, bytes]], name: bytes) -> str:
    """One request header, decoded, or the empty string."""
    return next((value.decode() for key, value in headers if key.lower() == name), "")


async def _reply(
    send: Callable[[Mapping[str, Any]], Awaitable[None]],
    status: int,
    headers: Sequence[tuple[bytes, bytes]],
    body: bytes,
) -> None:
    """Send one complete ASGI response."""
    await send({"type": "http.response.start", "status": status, "headers": list(headers)})
    await send({"type": "http.response.body", "body": body})
