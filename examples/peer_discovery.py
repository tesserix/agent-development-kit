"""Resolving a peer through a registry that is cached, pinned, and sometimes down.

Run it with `uv run python examples/peer_discovery.py`.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from tesserix_adk.a2a import (
    AgentCard,
    AgentSkill,
    PeerDiscoveryError,
    PeerNeed,
    RegistryPeers,
    StaticPeers,
    fingerprint,
)
from tesserix_adk.testing import FakeClock

if TYPE_CHECKING:
    from collections.abc import Sequence


def card(agent: str, *, version: str, skill: str = "price_leg") -> AgentCard:
    """A card as a registry hands it back."""
    return AgentCard(
        agent=agent,
        audience=f"https://{agent}.example.gov",
        version=version,
        skills=(
            AgentSkill(name=skill, description="Price one leg.", input_schema={"type": "object"}),
        ),
    )


class Registry:
    """A registry that answers from a list and can be taken down."""

    def __init__(self, *entries: AgentCard) -> None:
        self.entries = [entry.model_dump(mode="json") for entry in entries]
        self.calls = 0
        self.down = False

    async def __call__(self, need: PeerNeed) -> Sequence[dict[str, Any]]:
        """Answer, or fail the way an unreachable registry does."""
        del need
        self.calls += 1
        if self.down:
            raise ConnectionError("registry is unreachable")
        return self.entries


async def main() -> None:
    """Resolve a peer, serve it from cache, survive an outage, and refuse a moved entry."""
    registry = Registry(card("booker", version="2.1.0"), card("planner", version="2.9.0"))
    clock = FakeClock()
    peers = RegistryPeers(registry, clock=clock, ttl_seconds=300, stale_seconds=600)

    found = await peers.find(PeerNeed(skill="price_leg"))
    print("resolved:", found.card.agent, found.card.version)  # noqa: T201
    print("provenance:", found.attributes()["a2a.peer.source"])  # noqa: T201

    await peers.find(PeerNeed(skill="price_leg"))
    print("registry asked:", registry.calls, "times")  # noqa: T201

    registry.down = True
    clock.advance(400)
    stale = await peers.find(PeerNeed(skill="price_leg"))
    print("during the outage:", stale.card.agent, "stale:", stale.stale)  # noqa: T201

    clock.advance(1000)
    try:
        await peers.find(PeerNeed(skill="price_leg"))
    except PeerDiscoveryError as refused:
        print("too stale to stand behind:", refused.reason)  # noqa: T201

    moved = card("booker", version="2.1.0").model_copy(
        update={"audience": "https://elsewhere.example.net"}
    )
    pinned = RegistryPeers(
        Registry(moved), pinned={"booker": fingerprint(card("booker", version="2.1.0"))}
    )
    try:
        await pinned.find(PeerNeed(agent="booker"))
    except PeerDiscoveryError as refused:
        print("entry moved:", refused.reason)  # noqa: T201

    configured = StaticPeers((card("booker", version="2.1.0"),))
    print("without a registry:", (await configured.find(PeerNeed(skill="price_leg"))).source)  # noqa: T201


if __name__ == "__main__":
    asyncio.run(main())
