"""Finding a peer that can do the work, without trusting the thing that named it."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import pytest

from tesserix_adk.a2a import (
    AgentCard,
    AgentSkill,
    PeerDiscoveryError,
    PeerDiscoveryReason,
    PeerNeed,
    RegistryPeers,
    StaticPeers,
    fingerprint,
)
from tesserix_adk.testing import FakeClock

if TYPE_CHECKING:
    from collections.abc import Sequence


def _card(agent: str = "booker", *, version: str = "2.1.0", skill: str = "price_leg") -> AgentCard:
    """A card as a registry would hand it back."""
    return AgentCard(
        agent=agent,
        audience=f"https://{agent}.example.gov",
        version=version,
        skills=(
            AgentSkill(
                name=skill,
                description="Price one leg of an itinerary.",
                input_schema={"type": "object"},
            ),
        ),
    )


class _Registry:
    """A registry that answers from a list, fails on demand, and counts its calls."""

    def __init__(self, *entries: AgentCard) -> None:
        self.entries = [entry.model_dump(mode="json") for entry in entries]
        self.calls = 0
        self.fails: Exception | None = None

    async def __call__(self, need: PeerNeed) -> Sequence[dict[str, Any]]:
        del need
        self.calls += 1
        if self.fails is not None:
            raise self.fails
        return self.entries


def _peers(*entries: AgentCard, **options: Any) -> tuple[RegistryPeers, _Registry, FakeClock]:
    """A registry-backed discovery client over those entries, on a clock a test moves."""
    registry = _Registry(*entries)
    clock = FakeClock()
    return RegistryPeers(registry, clock=clock, **options), registry, clock


class TestFindingAPeerThatCanDoTheWork:
    async def test_a_peer_is_resolved_by_the_skill_it_publishes(self) -> None:
        peers, _, _ = _peers(_card())
        found = await peers.find(PeerNeed(skill="price_leg"))
        assert found.card.agent == "booker"

    async def test_a_peer_can_be_asked_for_by_name(self) -> None:
        peers, _, _ = _peers(_card("booker"), _card("planner", skill="plan"))
        found = await peers.find(PeerNeed(agent="planner"))
        assert found.card.agent == "planner"

    async def test_a_skill_nobody_publishes_is_not_invented(self) -> None:
        peers, _, _ = _peers(_card())
        with pytest.raises(PeerDiscoveryError) as refused:
            await peers.find(PeerNeed(skill="book_hotel"))
        assert refused.value.reason is PeerDiscoveryReason.NOT_FOUND

    async def test_a_need_that_names_neither_an_agent_nor_a_skill_is_not_a_need(self) -> None:
        with pytest.raises(ValueError, match="agent or a skill"):
            PeerNeed()

    async def test_the_resolution_records_where_the_answer_came_from(self) -> None:
        peers, _, _ = _peers(_card())
        found = await peers.find(PeerNeed(skill="price_leg"))
        assert found.attributes()["a2a.peer.source"] == "registry"
        assert found.attributes()["a2a.peer.agent"] == "booker"


class TestVersionConstraints:
    async def test_an_incompatible_version_is_excluded_rather_than_returned(self) -> None:
        peers, _, _ = _peers(_card(version="1.4.0"))
        with pytest.raises(PeerDiscoveryError) as refused:
            await peers.find(PeerNeed(skill="price_leg", version="2"))
        assert refused.value.reason is PeerDiscoveryReason.NOT_FOUND

    async def test_a_major_constraint_takes_any_minor_of_that_major(self) -> None:
        peers, _, _ = _peers(_card(version="2.4.1"))
        assert (await peers.find(PeerNeed(skill="price_leg", version="2"))).card.version == "2.4.1"

    async def test_a_minimum_constraint_takes_anything_at_or_above_it(self) -> None:
        peers, _, _ = _peers(_card(version="2.1.0"))
        found = await peers.find(PeerNeed(skill="price_leg", version=">=2.0.0"))
        assert found.card.version == "2.1.0"

    async def test_an_exact_constraint_takes_only_that_version(self) -> None:
        peers, _, _ = _peers(_card(version="2.1.0"))
        with pytest.raises(PeerDiscoveryError):
            await peers.find(PeerNeed(skill="price_leg", version="2.1.1"))


class TestChoosingBetweenPeers:
    async def test_the_newest_compatible_peer_wins_and_the_choice_is_reproducible(self) -> None:
        chosen = set()
        for _ in range(3):
            peers, _, _ = _peers(_card(version="2.1.0"), _card("planner", version="2.9.0"))
            chosen.add((await peers.find(PeerNeed(skill="price_leg"))).card.agent)
        assert chosen == {"planner"}

    async def test_peers_at_the_same_version_are_chosen_by_name_not_by_registry_order(self) -> None:
        peers, _, _ = _peers(_card("zephyr"), _card("booker"))
        assert (await peers.find(PeerNeed(skill="price_leg"))).card.agent == "booker"


class TestTheRegistryIsNotTrusted:
    async def test_a_card_that_does_not_validate_is_rejected_not_partially_used(self) -> None:
        peers, registry, _ = _peers()
        registry.entries = [{"agent": "booker", "skills": "everything"}]
        with pytest.raises(PeerDiscoveryError) as refused:
            await peers.find(PeerNeed(agent="booker"))
        assert refused.value.reason is PeerDiscoveryReason.INVALID_CARD

    async def test_a_rejected_card_is_recorded_for_whoever_has_to_explain_it(self) -> None:
        peers, registry, _ = _peers()
        registry.entries = [{"agent": "booker", "skills": "everything"}]
        with pytest.raises(PeerDiscoveryError):
            await peers.find(PeerNeed(agent="booker"))
        assert peers.rejections[-1].agent == "booker"

    async def test_a_peer_outside_the_allowlist_is_never_returned(self) -> None:
        peers, _, _ = _peers(_card("stranger"), permitted={"": ("booker",)})
        with pytest.raises(PeerDiscoveryError) as refused:
            await peers.find(PeerNeed(skill="price_leg"))
        assert refused.value.reason is PeerDiscoveryReason.NOT_PERMITTED

    async def test_an_entry_moved_to_another_host_is_refused_when_the_card_is_pinned(self) -> None:
        expected = fingerprint(_card())
        moved = _card().model_copy(update={"audience": "https://elsewhere.example.net"})
        peers, _, _ = _peers(moved, pinned={"booker": expected})
        with pytest.raises(PeerDiscoveryError) as refused:
            await peers.find(PeerNeed(agent="booker"))
        assert refused.value.reason is PeerDiscoveryReason.FINGERPRINT_MISMATCH

    async def test_a_pinned_card_that_has_not_moved_resolves(self) -> None:
        peers, _, _ = _peers(_card(), pinned={"booker": fingerprint(_card())})
        assert (await peers.find(PeerNeed(agent="booker"))).card.agent == "booker"


class TestOneTenantDoesNotSeeAnother:
    async def test_a_peer_permitted_for_another_tenant_is_not_returned(self) -> None:
        peers, _, _ = _peers(_card(), permitted={"acme": ("booker",), "globex": ()})
        with pytest.raises(PeerDiscoveryError) as refused:
            await peers.find(PeerNeed(skill="price_leg", tenant="globex"))
        assert refused.value.reason is PeerDiscoveryReason.NOT_PERMITTED

    async def test_one_tenant_s_answer_is_not_served_from_another_tenant_s_cache(self) -> None:
        peers, registry, _ = _peers(_card(), permitted={"acme": ("booker",), "globex": ()})
        await peers.find(PeerNeed(skill="price_leg", tenant="acme"))
        with pytest.raises(PeerDiscoveryError):
            await peers.find(PeerNeed(skill="price_leg", tenant="globex"))
        assert registry.calls == 2


class TestNotAskingTheRegistryTwice:
    async def test_a_resolved_peer_is_cached_for_its_ttl(self) -> None:
        peers, registry, clock = _peers(_card(), ttl_seconds=300)
        await peers.find(PeerNeed(skill="price_leg"))
        clock.advance(299)
        await peers.find(PeerNeed(skill="price_leg"))
        assert registry.calls == 1

    async def test_the_registry_is_asked_again_once_the_ttl_has_passed(self) -> None:
        peers, registry, clock = _peers(_card(), ttl_seconds=300)
        await peers.find(PeerNeed(skill="price_leg"))
        clock.advance(301)
        await peers.find(PeerNeed(skill="price_leg"))
        assert registry.calls == 2

    async def test_a_skill_nobody_has_is_not_asked_about_on_every_call(self) -> None:
        peers, registry, _ = _peers(_card(), negative_ttl_seconds=30)
        for _ in range(3):
            with pytest.raises(PeerDiscoveryError):
                await peers.find(PeerNeed(skill="book_hotel"))
        assert registry.calls == 1

    async def test_a_withdrawn_peer_can_be_dropped_from_the_cache(self) -> None:
        peers, registry, _ = _peers(_card())
        need = PeerNeed(skill="price_leg")
        await peers.find(need)
        peers.invalidate(need)
        await peers.find(need)
        assert registry.calls == 2


class TestARegistryOutage:
    async def test_a_recently_expired_answer_is_served_rather_than_failing_the_call(self) -> None:
        peers, registry, clock = _peers(_card(), ttl_seconds=300, stale_seconds=600)
        await peers.find(PeerNeed(skill="price_leg"))
        registry.fails = ConnectionError("registry is down")
        clock.advance(400)
        found = await peers.find(PeerNeed(skill="price_leg"))
        assert (found.card.agent, found.stale) == ("booker", True)

    async def test_an_answer_too_stale_to_stand_behind_fails_closed(self) -> None:
        peers, registry, clock = _peers(_card(), ttl_seconds=300, stale_seconds=600)
        await peers.find(PeerNeed(skill="price_leg"))
        registry.fails = ConnectionError("registry is down")
        clock.advance(1000)
        with pytest.raises(PeerDiscoveryError) as refused:
            await peers.find(PeerNeed(skill="price_leg"))
        assert refused.value.reason is PeerDiscoveryReason.UNAVAILABLE

    async def test_an_outage_with_nothing_cached_does_not_guess_an_endpoint(self) -> None:
        peers, registry, _ = _peers(_card())
        registry.fails = ConnectionError("registry is down")
        with pytest.raises(PeerDiscoveryError) as refused:
            await peers.find(PeerNeed(skill="price_leg"))
        assert refused.value.reason is PeerDiscoveryReason.UNAVAILABLE

    async def test_discovery_cannot_spend_the_run_s_deadline(self) -> None:
        async def never(need: PeerNeed) -> Sequence[dict[str, Any]]:
            del need
            await asyncio.sleep(10)
            return []

        peers = RegistryPeers(never, timeout_seconds=0.01)
        with pytest.raises(PeerDiscoveryError) as refused:
            await peers.find(PeerNeed(skill="price_leg"))
        assert refused.value.reason is PeerDiscoveryReason.TIMED_OUT


class TestDiscoveryWithoutARegistry:
    async def test_a_configured_peer_resolves_without_anything_on_the_network(self) -> None:
        peers = StaticPeers((_card(),))
        found = await peers.find(PeerNeed(skill="price_leg"))
        assert (found.card.agent, found.attributes()["a2a.peer.source"]) == ("booker", "static")

    async def test_a_peer_nobody_configured_is_not_resolved(self) -> None:
        with pytest.raises(PeerDiscoveryError) as refused:
            await StaticPeers((_card(),)).find(PeerNeed(agent="planner"))
        assert refused.value.reason is PeerDiscoveryReason.NOT_FOUND

    async def test_configuration_honours_the_same_version_constraints(self) -> None:
        peers = StaticPeers((_card(version="1.0.0"),))
        with pytest.raises(PeerDiscoveryError):
            await peers.find(PeerNeed(skill="price_leg", version="2"))

    def test_invalidating_a_configured_peer_changes_nothing(self) -> None:
        StaticPeers((_card(),)).invalidate(PeerNeed(agent="booker"))
