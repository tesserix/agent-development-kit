"""Finding a peer that can do the work, without trusting the thing that named it.

Peer endpoints live in per-environment configuration today, so moving an agent is a
coordinated change across every caller, and "which agent can price a leg" is a question
nobody can ask. A registry answers it — but a registry is also a thing that can be wrong,
be compromised, or be down, and none of those may become a call to the wrong host.

So resolution is a pipeline of refusals. A card is validated before it is used at all and
rejected whole where it does not validate; a peer outside the caller's allowlist is not
returned; a pinned card that has moved is refused rather than followed; and an answer that
cannot be obtained is a typed failure rather than a guessed endpoint. What is cached is
what was already checked, and it is keyed by tenant, because a peer resolved for one
tenant is not an answer for another.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections import deque
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Protocol

from pydantic import model_validator

from tesserix_adk.a2a.card import AgentCard
from tesserix_adk.core.errors import AdkError
from tesserix_adk.core.models import AdkModel
from tesserix_adk.runtime.loop import SystemClock

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
    from typing import Self

    from tesserix_adk.core import Clock

__all__ = [
    "DEFAULT_DISCOVERY_TIMEOUT_SECONDS",
    "DEFAULT_NEGATIVE_TTL_SECONDS",
    "DEFAULT_PEER_TTL_SECONDS",
    "DEFAULT_STALE_SECONDS",
    "PeerDiscovery",
    "PeerDiscoveryError",
    "PeerDiscoveryReason",
    "PeerNeed",
    "PeerRejection",
    "PeerResolution",
    "RegistryFetch",
    "RegistryPeers",
    "StaticPeers",
    "fingerprint",
]

DEFAULT_PEER_TTL_SECONDS = 300.0
"""How long a resolved peer is reused before the registry is asked again."""

DEFAULT_NEGATIVE_TTL_SECONDS = 30.0
"""How long "nobody does this" is remembered, so a missing skill is not asked about hourly."""

DEFAULT_STALE_SECONDS = 600.0
"""How far past its TTL an answer may still be served while the registry is unreachable."""

DEFAULT_DISCOVERY_TIMEOUT_SECONDS = 2.0
"""Discovery's own ceiling, so resolution cannot spend the run's deadline."""

_REJECTIONS_KEPT = 32
_ANY_TENANT = ""


class PeerDiscoveryReason(StrEnum):
    """Why a peer could not be resolved. Every one of these fails closed."""

    NOT_FOUND = "not_found"
    NOT_PERMITTED = "not_permitted"
    INVALID_CARD = "invalid_card"
    FINGERPRINT_MISMATCH = "fingerprint_mismatch"
    UNAVAILABLE = "unavailable"
    TIMED_OUT = "timed_out"


class PeerDiscoveryError(AdkError):
    """Raised when no peer could be resolved, saying which of the refusals it was.

    Args:
        message: What was asked for and why it was not answered.
        reason: The refusal, for a caller deciding whether to retry or to stop.
        agent: The peer at fault, where one entry was.
    """

    def __init__(self, message: str, *, reason: PeerDiscoveryReason, agent: str = "") -> None:
        self.reason = reason
        self.agent = agent
        super().__init__(message, details={"reason": reason.value, "agent": agent})


class PeerNeed(AdkModel):
    """What a caller is looking for.

    Args:
        agent: A peer by name, where the caller already knows which one it wants.
        skill: A capability, where it does not.
        version: A constraint on the peer's version — `""` any, `"2"` that major, `"2.1"`
            that minor, `"2.1.0"` exactly, `">=2.1.0"` that or newer.
        tenant: Whose work this is. Resolution is scoped to it, and so is the cache.
    """

    agent: str = ""
    skill: str = ""
    version: str = ""
    tenant: str = ""

    @model_validator(mode="after")
    def _asks_for_something(self) -> Self:
        if not self.agent and not self.skill:
            raise ValueError(
                "a need names an agent or a skill: asking for any peer at all is asking to "
                "be given whichever one a registry happened to list first"
            )
        return self


class PeerResolution(AdkModel):
    """A peer that was found, and where the answer came from.

    Args:
        card: The validated card.
        source: What answered — the registry, or local configuration.
        digest: The card's fingerprint, for pinning it once it has been reviewed.
        resolved_at: When it was resolved, on the discovery client's clock.
        stale: That it was served past its TTL because the registry was unreachable.
    """

    card: AgentCard
    source: str
    digest: str
    resolved_at: float
    stale: bool = False

    def attributes(self) -> dict[str, str]:
        """What a span records about the choice, so a call can be attributed to it."""
        return {
            "a2a.peer.agent": self.card.agent,
            "a2a.peer.version": self.card.version,
            "a2a.peer.source": self.source,
            "a2a.peer.digest": self.digest,
            "a2a.peer.stale": str(self.stale).lower(),
        }


class PeerRejection(AdkModel):
    """One registry entry that was refused, kept so somebody can explain the refusal.

    Args:
        agent: The name the entry claimed, where it claimed one.
        reason: Which refusal it was.
        detail: What was wrong, in terms that name no credential and no message content.
    """

    agent: str
    reason: PeerDiscoveryReason
    detail: str


def fingerprint(card: AgentCard) -> str:
    """The digest of a card, for pinning a peer to the card that was reviewed."""
    return hashlib.sha256(card.rendered()).hexdigest()


class PeerDiscovery(Protocol):
    """How an agent finds a peer. Substitutable, so a test never needs a registry."""

    async def find(self, need: PeerNeed) -> PeerResolution:
        """Resolve one peer, or raise `PeerDiscoveryError` rather than guess."""
        ...

    def invalidate(self, need: PeerNeed) -> None:
        """Forget what was resolved for this need, after a call to it failed."""
        ...


type RegistryFetch = Callable[[PeerNeed], Awaitable[Sequence[Mapping[str, Any]]]]
"""Ask the registry for candidate cards. However the org's registry is actually reached."""


class StaticPeers:
    """Discovery from configuration, for a deployment with no registry and for tests.

    Holds the same selection and constraint rules as the registry-backed client, so moving
    from one to the other does not change which peer a need resolves to.
    """

    __slots__ = ("_cards",)

    def __init__(self, cards: Iterable[AgentCard]) -> None:
        """Serve `cards` and nothing else."""
        self._cards = tuple(cards)

    async def find(self, need: PeerNeed) -> PeerResolution:
        """Resolve one configured peer.

        Raises:
            PeerDiscoveryError: With reason `not_found` where nothing configured matches.
        """
        chosen = _chosen(self._cards, need)
        if chosen is None:
            raise PeerDiscoveryError(_nothing_matches(need), reason=PeerDiscoveryReason.NOT_FOUND)
        return PeerResolution(
            card=chosen, source="static", digest=fingerprint(chosen), resolved_at=0.0
        )

    def invalidate(self, need: PeerNeed) -> None:
        """Nothing to forget: configuration is the source, not a copy of one."""
        del need


class RegistryPeers:
    """Discovery through the org registry, cached, pinned and bounded.

    Example:
        >>> peers = RegistryPeers(fetch, permitted={"": ("booker",)})  # doctest: +SKIP
        >>> found = await peers.find(PeerNeed(skill="price_leg"))      # doctest: +SKIP
    """

    __slots__ = (
        "_clock",
        "_entries",
        "_fetch",
        "_negative_ttl",
        "_permitted",
        "_pinned",
        "_rejections",
        "_stale",
        "_timeout",
        "_ttl",
    )

    def __init__(
        self,
        fetch: RegistryFetch,
        *,
        clock: Clock | None = None,
        ttl_seconds: float = DEFAULT_PEER_TTL_SECONDS,
        negative_ttl_seconds: float = DEFAULT_NEGATIVE_TTL_SECONDS,
        stale_seconds: float = DEFAULT_STALE_SECONDS,
        timeout_seconds: float = DEFAULT_DISCOVERY_TIMEOUT_SECONDS,
        permitted: Mapping[str, Iterable[str]] | None = None,
        pinned: Mapping[str, str] | None = None,
    ) -> None:
        """Resolve through `fetch`, holding to the ceilings and the trust rules given here.

        Args:
            fetch: How the registry is reached.
            clock: Source of time. Injected so a test moves it rather than sleeping.
            ttl_seconds: How long a resolved peer is reused.
            negative_ttl_seconds: How long an unmatched need is remembered.
            stale_seconds: How far past the TTL an answer may be served during an outage.
            timeout_seconds: Discovery's own ceiling.
            permitted: Which peers each tenant may be given, keyed by tenant with `""` as
                the default for any tenant. `None` permits any peer the registry lists,
                which is a deliberate choice for a closed network rather than a default
                anyone should reach by omission.
            pinned: Expected card fingerprints by agent name. A peer whose card no longer
                matches is refused rather than followed to wherever it now points.
        """
        self._fetch = fetch
        self._clock: Clock = clock or SystemClock()
        self._ttl = ttl_seconds
        self._negative_ttl = negative_ttl_seconds
        self._stale = stale_seconds
        self._timeout = timeout_seconds
        self._permitted = None if permitted is None else {k: tuple(v) for k, v in permitted.items()}
        self._pinned = dict(pinned or {})
        self._entries: dict[tuple[str, str, str, str], _Entry] = {}
        self._rejections: deque[PeerRejection] = deque(maxlen=_REJECTIONS_KEPT)

    @property
    def rejections(self) -> tuple[PeerRejection, ...]:
        """The most recent registry entries this client refused, oldest first."""
        return tuple(self._rejections)

    async def find(self, need: PeerNeed) -> PeerResolution:
        """Resolve one peer, from cache where it is fresh and from the registry otherwise.

        Raises:
            PeerDiscoveryError: `not_found` where nothing matches, `not_permitted` where
                the only matches are peers this tenant may not be given,
                `fingerprint_mismatch` where a pinned card has changed, `invalid_card`
                where every candidate failed validation, `timed_out` where the registry
                outran discovery's own ceiling, and `unavailable` where it could not be
                reached and nothing cached is recent enough to stand behind.
        """
        now = self._clock.now()
        key = _key(need)
        held = self._entries.get(key)
        if held is not None and now < held.expires_at:
            return held.answer(stale=False)
        try:
            listed = await self._listed(need)
        except PeerDiscoveryError:
            if held is not None and now < held.expires_at + self._stale and held.resolved:
                return held.answer(stale=True)
            raise
        return self._decided(need, key, listed, now)

    def invalidate(self, need: PeerNeed) -> None:
        """Forget this need's answer, so the next call asks the registry again."""
        self._entries.pop(_key(need), None)

    async def _listed(self, need: PeerNeed) -> Sequence[Mapping[str, Any]]:
        """Ask the registry, bounded by discovery's own ceiling."""
        try:
            async with asyncio.timeout(self._timeout):
                return await self._fetch(need)
        except TimeoutError as elapsed:
            raise PeerDiscoveryError(
                f"the registry did not answer within {self._timeout}s, and discovery does "
                f"not spend the run's deadline waiting",
                reason=PeerDiscoveryReason.TIMED_OUT,
            ) from elapsed
        except Exception as unreachable:
            raise PeerDiscoveryError(
                "the registry could not be reached, and an endpoint nobody published is "
                "not an endpoint to call",
                reason=PeerDiscoveryReason.UNAVAILABLE,
            ) from unreachable

    def _decided(
        self,
        need: PeerNeed,
        key: tuple[str, str, str, str],
        listed: Iterable[Mapping[str, Any]],
        now: float,
    ) -> PeerResolution:
        """Validate, filter and choose, remembering the verdict either way."""
        cards = tuple(self._validated(listed))
        matching = tuple(card for card in cards if _matches(card, need))
        allowed = tuple(card for card in matching if self._may_be_given(card, need.tenant))
        chosen = _newest(allowed)
        if chosen is None:
            self._entries[key] = _Entry(None, now + self._negative_ttl)
            raise self._nothing(need, cards, matching)
        self._pin_checked(chosen)
        resolution = PeerResolution(
            card=chosen, source="registry", digest=fingerprint(chosen), resolved_at=now
        )
        self._entries[key] = _Entry(resolution, now + self._ttl)
        return resolution

    def _validated(self, listed: Iterable[Mapping[str, Any]]) -> Iterable[AgentCard]:
        """Every entry that is a card, recording the ones that are not.

        Validated as JSON because that is what a registry entry is: the kit's models are
        strict in Python, where a list is not a tuple, and coercing them instead would
        relax every other check on the way past.
        """
        for entry in listed:
            try:
                yield AgentCard.model_validate_json(json.dumps(entry))
            except (TypeError, ValueError) as invalid:
                self._rejections.append(
                    PeerRejection(
                        agent=str(entry.get("agent", "")),
                        reason=PeerDiscoveryReason.INVALID_CARD,
                        detail=type(invalid).__name__,
                    )
                )

    def _may_be_given(self, card: AgentCard, tenant: str) -> bool:
        """Whether this tenant may be handed this peer at all."""
        if self._permitted is None:
            return True
        return card.agent in self._permitted.get(tenant, self._permitted.get(_ANY_TENANT, ()))

    def _pin_checked(self, card: AgentCard) -> None:
        """Refuse a pinned peer whose card is no longer the one that was reviewed."""
        expected = self._pinned.get(card.agent)
        if expected is not None and expected != fingerprint(card):
            self._rejections.append(
                PeerRejection(
                    agent=card.agent,
                    reason=PeerDiscoveryReason.FINGERPRINT_MISMATCH,
                    detail="the registry entry no longer matches the pinned card",
                )
            )
            raise PeerDiscoveryError(
                f"{card.agent}'s card is not the one that was pinned, so the entry now "
                f"points somewhere nobody reviewed",
                reason=PeerDiscoveryReason.FINGERPRINT_MISMATCH,
                agent=card.agent,
            )

    def _nothing(
        self, need: PeerNeed, cards: tuple[AgentCard, ...], matching: tuple[AgentCard, ...]
    ) -> PeerDiscoveryError:
        """Which refusal this was, distinguishing "none" from "none you may have".

        A tenant is told that nothing it may use matched, never which peer it was refused:
        the existence of another tenant's peer is that tenant's business.
        """
        if matching:
            return PeerDiscoveryError(
                f"no peer this tenant may be given matches {_asked(need)}",
                reason=PeerDiscoveryReason.NOT_PERMITTED,
            )
        if not cards and self._rejections:
            return PeerDiscoveryError(
                "every entry the registry listed failed validation, and a card that does "
                "not validate is not a card to call",
                reason=PeerDiscoveryReason.INVALID_CARD,
            )
        return PeerDiscoveryError(_nothing_matches(need), reason=PeerDiscoveryReason.NOT_FOUND)


class _Entry:
    """One cached verdict: the answer, or the fact that there was none."""

    __slots__ = ("expires_at", "resolution")

    def __init__(self, resolution: PeerResolution | None, expires_at: float) -> None:
        self.resolution = resolution
        self.expires_at = expires_at

    @property
    def resolved(self) -> bool:
        """Whether this entry holds a peer, rather than a remembered absence."""
        return self.resolution is not None

    def answer(self, *, stale: bool) -> PeerResolution:
        """The cached resolution, or the remembered refusal raised again."""
        if self.resolution is None:
            raise PeerDiscoveryError(
                "no peer matched this need, and the answer is still being remembered",
                reason=PeerDiscoveryReason.NOT_FOUND,
            )
        return self.resolution.model_copy(update={"stale": stale})


def _key(need: PeerNeed) -> tuple[str, str, str, str]:
    """The cache key. Tenant is part of it: one tenant's answer is not another's."""
    return (need.tenant, need.agent, need.skill, need.version)


def _chosen(cards: Iterable[AgentCard], need: PeerNeed) -> AgentCard | None:
    """The one peer this need resolves to, or nothing."""
    return _newest(tuple(card for card in cards if _matches(card, need)))


def _newest(cards: Sequence[AgentCard]) -> AgentCard | None:
    """The newest candidate, ties broken by name so a run is reproducible."""
    if not cards:
        return None
    return min(cards, key=lambda card: (tuple(-part for part in _parts(card.version)), card.agent))


def _matches(card: AgentCard, need: PeerNeed) -> bool:
    """Whether one card answers the need, by name, by skill and by version."""
    if need.agent and card.agent != need.agent:
        return False
    if need.skill and not any(skill.name == need.skill for skill in card.skills):
        return False
    return _version_satisfies(card.version, need.version)


def _version_satisfies(version: str, constraint: str) -> bool:
    """Whether `version` satisfies `constraint`. The four forms are in docs/peer-discovery.md."""
    if not constraint:
        return True
    if constraint.startswith(">="):
        return _parts(version) >= _parts(constraint[2:].strip())
    wanted = _parts(constraint)
    return _parts(version)[: len(wanted)] == wanted


def _parts(version: str) -> tuple[int, ...]:
    """A version as comparable numbers. A part that is not a number sorts as zero."""
    return tuple(int(part) if part.isdigit() else 0 for part in version.split("."))


def _asked(need: PeerNeed) -> str:
    """What was asked for, for a message somebody has to act on."""
    said = [what for what in (need.agent, need.skill, need.version) if what]
    return " ".join(said)


def _nothing_matches(need: PeerNeed) -> str:
    """The refusal for a need nothing answers."""
    return (
        f"no peer matches {_asked(need)}, and an endpoint nobody published is not an "
        f"endpoint to call"
    )
