"""Calling a peer agent as the person who started the run, not as the service.

An agent that authenticates to its peer as itself hands that peer its own access and
loses every trace of whose work it is. A peer with broader reach then becomes the way to
do what the caller could not. So the caller passes a delegation: the original principal,
the chain it came through, and a strict subset of what the caller itself holds.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping  # noqa: TC003 — pydantic needs the runtime types
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, ClassVar, Self

from pydantic import Field, field_validator

from tesserix_adk.core import (
    AdkModel,
    AgentIdentity,
    AuthMethod,
    AuthorisationError,
    CallCredential,
    CredentialSource,
    DelegationLimitError,
    Principal,
    ScopeSet,
)

if TYPE_CHECKING:
    from tesserix_adk.core import Clock

__all__ = [
    "DEFAULT_TTL_SECONDS",
    "MAX_CHAIN_DEPTH",
    "AgentCard",
    "DelegationChain",
    "DelegationClaims",
    "DelegationHop",
    "PeerDelegation",
    "PeerDelegator",
    "PeerVerifier",
]

MAX_CHAIN_DEPTH = 8
"""How many agents one piece of work may pass through, so the chain stays bounded."""

DEFAULT_TTL_SECONDS = 300.0
"""How long a delegation is good for, absent a shorter one."""


class DelegationHop(AdkModel):
    """One agent in a chain, and what it passed on.

    Args:
        agent: Whose hop it is.
        agent_version: Which build of it.
        scopes: What it handed the next agent, for reading the chain after the fact.
    """

    agent: str = Field(min_length=1)
    agent_version: str = "1.0.0"
    scopes: tuple[str, ...] = ()

    def describe(self) -> str:
        """The hop as one identifier, safe to log."""
        return f"{self.agent}@{self.agent_version}"


class DelegationChain(AdkModel):
    """Which agents a piece of work passed through, outermost first."""

    hops: tuple[DelegationHop, ...] = ()

    @property
    def agents(self) -> tuple[str, ...]:
        """The agent names, outermost first."""
        return tuple(hop.agent for hop in self.hops)

    @property
    def depth(self) -> int:
        """How many hops deep the work already is."""
        return len(self.hops)

    def extended(self, hop: DelegationHop) -> Self:
        """The chain with one more hop on the end.

        Raises:
            DelegationLimitError: If the chain is already at `MAX_CHAIN_DEPTH`, or if the
                hop is one the work has been through before. A cycle is refused where it
                would be created rather than when it runs out of depth, so the refusal
                names the loop instead of a ceiling.
        """
        if hop.agent in self.agents:
            raise DelegationLimitError(
                f"{hop.agent} is already in the chain {self.describe()}, which would be a cycle",
                reason="cycle",
                path=(*self.agents, hop.agent),
            )
        if self.depth >= MAX_CHAIN_DEPTH:
            raise DelegationLimitError(
                f"a delegation chain may be {MAX_CHAIN_DEPTH} agents deep; "
                f"{self.describe()} is already there",
                reason="depth",
                path=(*self.agents, hop.agent),
            )
        return self.model_copy(update={"hops": (*self.hops, hop)})

    def describe(self) -> str:
        """The chain as one line, safe to log and to put on a span."""
        return " -> ".join(hop.agent for hop in self.hops)

    def encoded(self) -> str:
        """The chain as one metadata value, for a peer that is not this kit."""
        return ">".join(f"{hop.describe()}={'+'.join(hop.scopes)}" for hop in self.hops)

    @classmethod
    def decoded(cls, encoded: str) -> Self:
        """The chain a peer sent, or a refusal if it is not one.

        Raises:
            ValueError: If a hop is not `agent@version=scope+scope`.
        """
        hops = []
        for part in encoded.split(">") if encoded else ():
            named, _, scopes = part.partition("=")
            agent, _, version = named.partition("@")
            hops.append(
                DelegationHop(
                    agent=agent,
                    agent_version=version or "1.0.0",
                    scopes=tuple(name for name in scopes.split("+") if name),
                )
            )
        return cls(hops=tuple(hops))


class DelegationClaims(AdkModel):
    """What a peer is told, and the only thing it is allowed to act on.

    Identifiers only: no credential material travels here, because this is the part that
    reaches spans, logs and a peer's own audit trail.
    """

    META_PREFIX: ClassVar[str] = "tesserix/adk/delegation"

    issuer: str = Field(min_length=1)
    subject: str
    tenant: str
    audience: str = Field(min_length=1)
    scopes: frozenset[str]
    expires_at: float
    run_id: str
    chain: DelegationChain = DelegationChain()

    @field_validator("scopes")
    @classmethod
    def _carries_something(cls, value: frozenset[str]) -> frozenset[str]:
        if not value:
            raise ValueError("a delegation that carries no scope authorises nothing")
        return value

    def expired(self, now: float) -> bool:
        """Whether the delegation has lapsed by `now`."""
        return now >= self.expires_at

    def meta(self) -> dict[str, str]:
        """The documented wire contract, for a peer inside this kit or outside it."""
        stated = {
            "issuer": self.issuer,
            "subject": self.subject,
            "tenant": self.tenant,
            "audience": self.audience,
            "scopes": " ".join(sorted(self.scopes)),
            "expires": repr(self.expires_at),
            "run": self.run_id,
            "chain": self.chain.encoded(),
        }
        return {f"{self.META_PREFIX}/{key}": value for key, value in stated.items()}

    @classmethod
    def from_meta(cls, meta: Mapping[str, str]) -> Self:
        """The claims a peer sent.

        Raises:
            AuthorisationError: If the metadata is absent, incomplete or malformed. A peer
                that cannot read the delegation refuses the call; it does not carry on
                with whatever it could parse.
        """
        read = {
            key.removeprefix(f"{cls.META_PREFIX}/"): value
            for key, value in meta.items()
            if key.startswith(f"{cls.META_PREFIX}/")
        }
        try:
            return cls(
                issuer=read["issuer"],
                subject=read["subject"],
                tenant=read["tenant"],
                audience=read["audience"],
                scopes=frozenset(read["scopes"].split()),
                expires_at=float(read["expires"]),
                run_id=read["run"],
                chain=DelegationChain.decoded(read.get("chain", "")),
            )
        except (KeyError, ValueError) as unreadable:
            raise AuthorisationError(
                f"the delegation metadata could not be read: {unreadable}", where="a2a"
            ) from unreadable

    def attributes(self) -> dict[str, str]:
        """What a run span records: who it is for and where it came from, never the token."""
        return {
            "a2a.issuer": self.issuer,
            "a2a.subject": self.subject,
            "a2a.tenant": self.tenant,
            "a2a.run": self.run_id,
            "a2a.chain": self.chain.describe(),
            "a2a.scopes": " ".join(sorted(self.scopes)),
        }


class AgentCard(AdkModel):
    """What a peer publishes about the delegations it takes.

    Args:
        agent: The peer's name.
        audience: What a credential for it is minted for.
        declared: What it will act on. A delegation is narrowed to this on arrival, so a
            peer never runs with more than it published even if a caller sends more.
        accepted_issuers: Which callers it takes delegations from. Empty accepts any,
            which is a deliberate statement rather than a default.
        required_scopes: What it will not run without.
    """

    agent: str = Field(min_length=1)
    audience: str
    declared: tuple[str, ...] = ()
    accepted_issuers: tuple[str, ...] = ()
    required_scopes: tuple[str, ...] = ()

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


@dataclass(frozen=True, slots=True)
class PeerDelegation:
    """One outbound call to a peer: a credential to present, claims to send alongside."""

    claims: DelegationClaims
    peer: str
    credential: CallCredential = field(repr=False)

    def headers(self) -> dict[str, str]:
        """The transport headers, the only rendering that carries the credential."""
        return self.credential.headers()

    def meta(self) -> dict[str, str]:
        """The delegation metadata to send in the request, carrying no token."""
        return self.claims.meta()

    def span_attributes(self) -> dict[str, str]:
        """What the calling run's span records."""
        return {"a2a.peer": self.peer, **self.claims.attributes()}


class PeerDelegator:
    """Mints a delegation for one peer call, narrowed on the way out."""

    def __init__(
        self,
        credentials: CredentialSource,
        *,
        clock: Clock,
        ttl_seconds: float = DEFAULT_TTL_SECONDS,
    ) -> None:
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive; a lapsed delegation authorises nothing")
        self._credentials = credentials
        self._clock = clock
        self._ttl = ttl_seconds

    async def delegate(
        self,
        *,
        identity: AgentIdentity,
        peer: AgentCard,
        needs: Iterable[str],
        run_id: str,
        chain: DelegationChain | None = None,
        agent_version: str = "1.0.0",
    ) -> PeerDelegation:
        """The authority for one call to one peer.

        Args:
            identity: Who the calling run acts for and what it holds.
            peer: What the peer published about itself.
            needs: What this call wants the peer to be able to do.
            run_id: The calling run, carried through as attribution.
            chain: What the work has already passed through, for an onward hop.
            agent_version: Which build of the caller is calling.

        Returns:
            The delegation: a credential for the peer's audience and the claims to send.

        Raises:
            AuthorisationError: If narrowing left nothing, or if no credential could be
                minted. Nothing half-authorised is returned, and there is no fallback to
                the caller's own service identity.
            DelegationLimitError: If the hop would exceed the chain ceiling or close a cycle.
        """
        granted = frozenset(
            ScopeSet.of(*needs).narrowed_to(identity.effective).narrowed_to(peer.accepts)
        )
        if not granted:
            held = ", ".join(identity.effective) or "nothing"
            raise AuthorisationError(
                f"a delegation to {peer.agent} would carry nothing: it declares "
                f"{', '.join(peer.declared) or 'no scopes'} and the caller holds {held}",
                agent=identity.agent,
                subject=identity.principal.subject,
                where=f"a2a:{peer.agent}",
            )
        extended = (chain or DelegationChain()).extended(
            DelegationHop(
                agent=identity.agent, agent_version=agent_version, scopes=tuple(sorted(granted))
            )
        )
        try:
            credential = await self._credentials.for_tool(
                identity=identity,
                audience=peer.audience,
                needs=sorted(granted),
                run_id=run_id,
                agent_version=agent_version,
            )
        except AuthorisationError:
            raise
        except Exception as failure:
            raise AuthorisationError(
                f"no credential could be minted for {peer.agent}, so it was not called",
                agent=identity.agent,
                subject=identity.principal.subject,
                where=f"a2a:{peer.agent}",
            ) from failure
        return PeerDelegation(
            claims=DelegationClaims(
                issuer=identity.agent,
                subject=identity.principal.subject,
                tenant=identity.principal.tenant,
                audience=peer.audience,
                scopes=granted,
                expires_at=self._clock.now() + self._ttl,
                run_id=run_id,
                chain=extended,
            ),
            peer=peer.agent,
            credential=credential,
        )


@dataclass(frozen=True, slots=True)
class PeerVerifier:
    """Checks an arriving delegation against what this agent published about itself."""

    card: AgentCard
    clock: Clock

    def accept(self, claims: DelegationClaims) -> AgentIdentity:
        """The identity this run may act under, or a refusal.

        The narrowing happens twice on purpose: the caller cut the delegation to what it
        held, and this cuts it again to what the card declared. A caller that sends more
        than it should therefore still cannot widen anything here.

        Args:
            claims: What arrived with the call.

        Returns:
            The peer's identity for this run, acting for the original principal.

        Raises:
            AuthorisationError: If the delegation is for another audience, from an issuer
                this card does not take, already lapsed, or missing a scope the card
                requires. There is no path where the call proceeds on this agent's own
                service identity instead.
        """
        if claims.audience != self.card.audience:
            raise AuthorisationError(
                f"the delegation names audience {claims.audience!r}, and this agent is "
                f"{self.card.audience!r}",
                agent=self.card.agent,
                subject=claims.subject,
                where="a2a",
            )
        if self.card.accepted_issuers and claims.issuer not in self.card.accepted_issuers:
            raise AuthorisationError(
                f"{claims.issuer!r} is not an issuer {self.card.agent} accepts",
                agent=self.card.agent,
                subject=claims.subject,
                where="a2a",
            )
        now = self.clock.now()
        if claims.expired(now):
            raise AuthorisationError(
                f"the delegation from {claims.issuer} expired at {claims.expires_at}",
                agent=self.card.agent,
                subject=claims.subject,
                where="a2a",
            )
        effective = ScopeSet.of(*claims.scopes).narrowed_to(self.card.accepts)
        absent = effective.missing(self.card.required_scopes)
        if absent:
            raise AuthorisationError(
                f"{self.card.agent} does not run without {absent[0]!r}, which the "
                f"delegation from {claims.issuer} does not carry",
                scope=absent[0],
                agent=self.card.agent,
                subject=claims.subject,
                where="a2a",
            )
        identity = AgentIdentity.resolve(
            agent=self.card.agent,
            declared=self.card.declared,
            principal=Principal(
                subject=claims.subject,
                tenant=claims.tenant,
                scopes=frozenset(effective.names),
                method=AuthMethod.DELEGATED,
                expires_at=claims.expires_at,
            ),
            now=now,
        )
        return replace(identity, chain=claims.chain.agents)
