"""What a peer agent is told about who it is acting for, and what it may do about it."""

from __future__ import annotations

from collections.abc import Iterable  # noqa: TC003 — used at runtime by the doubles

import pytest

from tesserix_adk.a2a import (
    MAX_CHAIN_DEPTH,
    AgentCard,
    DelegationChain,
    DelegationClaims,
    DelegationHop,
    PeerDelegation,
    PeerDelegator,
    PeerVerifier,
)
from tesserix_adk.core import (
    AgentIdentity,
    AuthorisationError,
    CallCredential,
    DelegationLimitError,
    Principal,
)
from tesserix_adk.testing import FakeClock
from tesserix_adk.tools import CredentialBroker, CredentialRequest, ExchangedCredentials

pytestmark = pytest.mark.anyio

READ = "itinerary:read"
WRITE = "payments:write"
BOOK = "itinerary:book"
PEER = "payments-agent"
ISSUER = "desk"

CARD = AgentCard(
    agent=PEER,
    audience="https://payments.peer",
    declared=(READ, WRITE),
    accepted_issuers=(ISSUER,),
)


class _Exchange:
    """A token endpoint that mints one token per call."""

    def __init__(self) -> None:
        self.calls = 0

    async def exchange(self, request: CredentialRequest) -> tuple[str, float]:
        self.calls += 1
        return f"tok-{self.calls}-{request.audience}", 300.0


class _Failing:
    """A credential source that reads the request and then fails."""

    def __init__(self, failure: Exception) -> None:
        self.failure = failure
        self.seen: list[tuple[str, str, tuple[str, ...], str, str]] = []

    async def for_tool(
        self,
        *,
        identity: AgentIdentity,
        audience: str,
        needs: Iterable[str],
        run_id: str,
        agent_version: str = "1.0.0",
    ) -> CallCredential:
        self.seen.append((identity.agent, audience, tuple(needs), run_id, agent_version))
        raise self.failure


def _identity(*held: str, agent: str = ISSUER, tenant: str = "acme") -> AgentIdentity:
    return AgentIdentity.resolve(
        agent=agent,
        declared=(READ, WRITE, BOOK),
        principal=Principal(subject="ada", tenant=tenant, scopes=frozenset(held or (READ,))),
    )


def _delegator(clock: FakeClock | None = None, exchange: _Exchange | None = None) -> PeerDelegator:
    ticking = clock or FakeClock()
    return PeerDelegator(
        CredentialBroker(
            ExchangedCredentials(exchange or _Exchange(), clock=ticking), clock=ticking
        ),
        clock=ticking,
    )


async def _delegate(
    delegator: PeerDelegator, identity: AgentIdentity, *needs: str, run_id: str = "run_1"
) -> PeerDelegation:
    return await delegator.delegate(
        identity=identity, peer=CARD, needs=needs or (READ,), run_id=run_id
    )


class TestTheCard:
    def test_a_card_without_an_audience_is_not_addressable(self) -> None:
        with pytest.raises(ValueError, match="audience"):
            AgentCard(agent=PEER, audience=" ", declared=(READ,))

    def test_a_card_declaring_nothing_accepts_no_delegation(self) -> None:
        assert AgentCard(agent=PEER, audience="https://x.invalid").declared == ()

    def test_a_delegation_carrying_no_scope_is_not_a_delegation(self) -> None:
        with pytest.raises(ValueError, match="authorises nothing"):
            DelegationClaims(
                issuer=ISSUER,
                subject="ada",
                tenant="acme",
                audience=CARD.audience,
                scopes=frozenset(),
                expires_at=1.0,
                run_id="run_1",
            )

    def test_a_delegator_that_never_lapses_is_refused(self) -> None:
        with pytest.raises(ValueError, match="ttl_seconds"):
            PeerDelegator(_Failing(TimeoutError()), clock=FakeClock(), ttl_seconds=0.0)


class TestNarrowingOnTheCallingSide:
    async def test_a_peer_receives_no_more_than_the_caller_holds(self) -> None:
        """The peer declares payments:write; the caller does not hold it."""
        delegation = await _delegate(_delegator(), _identity(READ), READ, WRITE)
        assert delegation.claims.scopes == frozenset({READ})

    async def test_a_peer_receives_no_more_than_it_declared(self) -> None:
        delegation = await _delegate(_delegator(), _identity(READ, BOOK), READ, BOOK)
        assert delegation.claims.scopes == frozenset({READ})

    async def test_a_delegation_carrying_nothing_is_refused(self) -> None:
        with pytest.raises(AuthorisationError, match="nothing"):
            await _delegate(_delegator(), _identity(BOOK), BOOK)

    async def test_the_credential_is_minted_for_the_peers_audience(self) -> None:
        exchange = _Exchange()
        delegation = await _delegate(_delegator(exchange=exchange), _identity(READ))
        assert delegation.headers()["Authorization"] == "Bearer tok-1-https://payments.peer"


class TestWhatTravels:
    async def test_the_claims_name_the_original_principal(self) -> None:
        delegation = await _delegate(_delegator(), _identity(READ))
        claims = delegation.claims
        assert (claims.subject, claims.tenant, claims.issuer) == ("ada", "acme", ISSUER)
        assert claims.audience == CARD.audience

    async def test_the_chain_names_every_agent_it_passed_through(self) -> None:
        delegation = await _delegate(_delegator(), _identity(READ))
        assert delegation.claims.chain.agents == (ISSUER,)

    async def test_the_claims_expire_with_the_credential(self) -> None:
        clock = FakeClock()
        delegation = await _delegate(_delegator(clock), _identity(READ))
        assert delegation.claims.expires_at == pytest.approx(clock.now() + 300.0)

    async def test_the_span_attribution_holds_identifiers_and_no_token(self) -> None:
        delegation = await _delegate(_delegator(), _identity(READ))
        attributes = delegation.span_attributes()
        assert attributes["a2a.peer"] == PEER
        assert attributes["a2a.subject"] == "ada"
        assert attributes["a2a.chain"] == ISSUER
        assert "tok-1" not in str(attributes)

    async def test_the_wire_contract_carries_no_token_either(self) -> None:
        delegation = await _delegate(_delegator(), _identity(READ))
        assert "tok-1" not in str(delegation.meta())
        assert "tok-1" not in repr(delegation)

    async def test_what_a_non_kit_peer_reads_round_trips(self) -> None:
        delegation = await _delegate(_delegator(), _identity(READ))
        assert DelegationClaims.from_meta(delegation.meta()) == delegation.claims

    def test_metadata_a_peer_did_not_send_is_refused(self) -> None:
        with pytest.raises(AuthorisationError, match="delegation"):
            DelegationClaims.from_meta({"unrelated": "header"})

    def test_metadata_a_peer_garbled_is_refused_rather_than_guessed(self) -> None:
        claims = DelegationClaims(
            issuer=ISSUER,
            subject="ada",
            tenant="acme",
            audience=CARD.audience,
            scopes=frozenset({READ}),
            expires_at=100.0,
            run_id="run_1",
            chain=DelegationChain(hops=(DelegationHop(agent=ISSUER, scopes=(READ,)),)),
        )
        meta = {**claims.meta(), f"{DelegationClaims.META_PREFIX}/expires": "soon"}
        with pytest.raises(AuthorisationError, match="delegation"):
            DelegationClaims.from_meta(meta)


class TestTheChain:
    def test_a_chain_grows_by_one_hop_at_a_time(self) -> None:
        chain = DelegationChain().extended(DelegationHop(agent="a", scopes=(READ,)))
        assert chain.extended(DelegationHop(agent="b")).agents == ("a", "b")

    def test_a_chain_that_would_revisit_an_agent_is_a_cycle(self) -> None:
        chain = DelegationChain().extended(DelegationHop(agent="a"))
        with pytest.raises(DelegationLimitError, match="cycle") as refused:
            chain.extended(DelegationHop(agent="b")).extended(DelegationHop(agent="a"))
        assert refused.value.reason == "cycle"

    def test_a_chain_deeper_than_the_ceiling_is_refused(self) -> None:
        chain = DelegationChain()
        for index in range(MAX_CHAIN_DEPTH):
            chain = chain.extended(DelegationHop(agent=f"a{index}"))
        with pytest.raises(DelegationLimitError, match="agents deep") as refused:
            chain.extended(DelegationHop(agent="one-too-many"))
        assert refused.value.reason == "depth"

    def test_the_ceiling_bounds_what_deep_composition_costs_to_carry(self) -> None:
        chain = DelegationChain()
        for index in range(MAX_CHAIN_DEPTH):
            chain = chain.extended(DelegationHop(agent=f"agent-{index}", scopes=(READ, WRITE)))
        assert len(chain.describe()) < 512

    async def test_a_second_hop_records_both_agents(self) -> None:
        clock = FakeClock()
        delegator = _delegator(clock)
        first = await _delegate(delegator, _identity(READ))
        peer_identity = PeerVerifier(CARD, clock=clock).accept(first.claims)
        onward = AgentCard(agent="ledger", audience="https://ledger.peer", declared=(READ,))
        second = await delegator.delegate(
            identity=peer_identity,
            peer=onward,
            needs=(READ,),
            run_id="run_2",
            chain=first.claims.chain,
        )
        assert second.claims.chain.agents == (ISSUER, PEER)
        assert second.claims.subject == "ada"


class TestTheReceivingSide:
    def _verifier(self, clock: FakeClock | None = None) -> PeerVerifier:
        return PeerVerifier(CARD, clock=clock or FakeClock())

    async def test_the_peer_holds_the_intersection_and_no_more(self) -> None:
        clock = FakeClock()
        delegation = await _delegate(_delegator(clock), _identity(READ, WRITE), READ, WRITE)
        identity = PeerVerifier(CARD, clock=clock).accept(delegation.claims)
        assert identity.effective.names == frozenset({READ, WRITE})

    async def test_a_tool_the_peer_declares_but_was_not_delegated_is_refused(self) -> None:
        clock = FakeClock()
        delegation = await _delegate(_delegator(clock), _identity(READ), READ, WRITE)
        identity = PeerVerifier(CARD, clock=clock).accept(delegation.claims)
        with pytest.raises(AuthorisationError, match=WRITE):
            identity.check((WRITE,), where="payments")

    async def test_the_peer_acts_for_the_original_principal(self) -> None:
        clock = FakeClock()
        delegation = await _delegate(_delegator(clock), _identity(READ))
        identity = PeerVerifier(CARD, clock=clock).accept(delegation.claims)
        assert (identity.principal.subject, identity.principal.tenant) == ("ada", "acme")
        assert identity.chain == (ISSUER,)

    def test_an_expired_delegation_is_refused_before_anything_runs(self) -> None:
        clock = FakeClock()
        claims = DelegationClaims(
            issuer=ISSUER,
            subject="ada",
            tenant="acme",
            audience=CARD.audience,
            scopes=frozenset({READ}),
            expires_at=clock.now() - 1.0,
            run_id="run_1",
            chain=DelegationChain(hops=(DelegationHop(agent=ISSUER, scopes=(READ,)),)),
        )
        with pytest.raises(AuthorisationError, match="expired"):
            self._verifier(clock).accept(claims)

    async def test_an_issuer_the_card_does_not_accept_is_refused(self) -> None:
        clock = FakeClock()
        delegation = await _delegate(_delegator(clock), _identity(READ, agent="stranger"))
        with pytest.raises(AuthorisationError, match="stranger"):
            PeerVerifier(CARD, clock=clock).accept(delegation.claims)

    async def test_a_delegation_addressed_elsewhere_is_refused(self) -> None:
        clock = FakeClock()
        elsewhere = AgentCard(
            agent="other",
            audience="https://other.peer",
            declared=(READ,),
            accepted_issuers=(ISSUER,),
        )
        delegation = await _delegator(clock).delegate(
            identity=_identity(READ), peer=elsewhere, needs=(READ,), run_id="run_1"
        )
        with pytest.raises(AuthorisationError, match="audience"):
            PeerVerifier(CARD, clock=clock).accept(delegation.claims)

    async def test_a_card_accepting_any_issuer_says_so_explicitly(self) -> None:
        clock = FakeClock()
        open_card = AgentCard(agent=PEER, audience=CARD.audience, declared=(READ,))
        delegation = await _delegator(clock).delegate(
            identity=_identity(READ, agent="stranger"),
            peer=open_card,
            needs=(READ,),
            run_id="run_1",
        )
        assert PeerVerifier(open_card, clock=clock).accept(delegation.claims).agent == PEER

    async def test_a_delegation_missing_a_scope_the_card_requires_is_refused(self) -> None:
        clock = FakeClock()
        strict = CARD.model_copy(update={"required_scopes": (WRITE,)})
        delegation = await _delegator(clock).delegate(
            identity=_identity(READ), peer=strict, needs=(READ,), run_id="run_1"
        )
        with pytest.raises(AuthorisationError, match=WRITE):
            PeerVerifier(strict, clock=clock).accept(delegation.claims)

    def test_a_peer_never_falls_back_to_its_own_service_identity(self) -> None:
        """There is no unauthenticated path: acceptance either returns an identity or raises."""
        clock = FakeClock()
        claims = DelegationClaims(
            issuer="stranger",
            subject="ada",
            tenant="acme",
            audience=CARD.audience,
            scopes=frozenset({READ}),
            expires_at=clock.now() + 60.0,
            run_id="run_1",
            chain=DelegationChain(hops=(DelegationHop(agent="stranger"),)),
        )
        with pytest.raises(AuthorisationError):
            self._verifier(clock).accept(claims)


class TestFailures:
    async def test_a_credential_that_cannot_be_minted_stops_the_delegation(self) -> None:
        source = _Failing(TimeoutError("token endpoint down"))
        with pytest.raises(AuthorisationError, match=PEER):
            await PeerDelegator(source, clock=FakeClock()).delegate(
                identity=_identity(READ), peer=CARD, needs=(READ,), run_id="run_1"
            )
        assert source.seen == [(ISSUER, CARD.audience, (READ,), "run_1", "1.0.0")]

    async def test_a_refusal_from_the_credential_source_travels_unchanged(self) -> None:
        source = _Failing(AuthorisationError("the run may not reach payments"))
        with pytest.raises(AuthorisationError, match="may not reach"):
            await PeerDelegator(source, clock=FakeClock()).delegate(
                identity=_identity(READ), peer=CARD, needs=(READ,), run_id="run_1"
            )
