"""An agent card generated from a declaration, served at the well-known path.

Run it with `uv run python examples/agent_card.py`.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from tesserix_adk.a2a import WELL_KNOWN_PATH, AgentCard, AgentCardError, CardEndpoint, card_for
from tesserix_adk.core import Agent, AgentDefinition, Owner
from tesserix_adk.core.config import ConcurrencyConfig
from tesserix_adk.tools import tool


@tool(name="price_leg")
def price_leg(leg: str) -> dict[str, Any]:
    """Price one leg of an itinerary."""
    return {"leg": leg, "eur": 40}


@tool(name="refund")
def refund(order: str) -> dict[str, Any]:
    """Refund an order once a human has said so."""
    return {"order": order}


@tool(name="ledger")
def ledger() -> dict[str, Any]:
    """The agent's own bookkeeping, never published."""
    return {"balance": 1}


@tool(name="seats")
def seats() -> dict[str, Any]:
    """A tool this agent was never given."""
    return {"seats": 2}


def _definition() -> AgentDefinition[Any]:
    """The agent as it was reviewed: two publishable skills and one it keeps."""
    return AgentDefinition(
        agent=Agent(
            name="booker",
            version="2.1.0",
            instructions="Book the trip.",
            task_class="planning",
            free_text=True,
            tools=("price_leg", "refund", "ledger"),
            idempotent_tools=("price_leg",),
            approval_required_tools=("refund",),
            scopes=("bookings:read", "payments:refund"),
            tool_scopes={"refund": ("payments:refund",)},
            concurrency=ConcurrencyConfig(max_concurrent_tools=4),
        ),
        owner=Owner(team="travel", contact="travel@example.gov", service="booker-api"),
        evaluation_suite="suites/booker.yaml",
    )


async def _fetch(
    endpoint: CardEndpoint, headers: list[tuple[bytes, bytes]]
) -> tuple[int, dict[str, str], bytes]:
    """Fetch the card as an ASGI server would, and return status, headers and body."""
    sent: list[dict[str, Any]] = []

    async def receive() -> dict[str, Any]:
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message: dict[str, Any]) -> None:
        sent.append(message)

    scope = {"type": "http", "method": "GET", "path": WELL_KNOWN_PATH, "headers": headers}
    await endpoint(scope, receive, send)
    return (
        int(sent[0]["status"]),
        {key.decode(): value.decode() for key, value in sent[0]["headers"]},
        sent[1]["body"],
    )


async def main() -> None:
    """Generate a card, serve it, and watch what a peer can and cannot learn."""
    definition = _definition()
    card = card_for(definition, audience="https://booker.example.gov", exports=(price_leg, refund))
    endpoint = CardEndpoint(card)

    status, headers, body = await _fetch(endpoint, [])
    served = AgentCard.model_validate_json(body)
    print("fetched:", status, WELL_KNOWN_PATH)  # noqa: T201
    print("published:", [skill.name for skill in served.skills])  # noqa: T201
    print("elevated:", {s.name: s.required_scopes for s in served.skills if s.required_scopes})  # noqa: T201
    print("ledger disclosed:", "ledger" in body.decode())  # noqa: T201

    conditional = [(b"if-none-match", headers["etag"].encode())]
    unchanged, _, _ = await _fetch(endpoint, conditional)
    print("polled again:", unchanged)  # noqa: T201

    endpoint.serve(card.model_copy(update={"available": False}))
    status, _, degraded = await _fetch(endpoint, conditional)
    print("degraded:", status, json.loads(degraded)["available"])  # noqa: T201

    try:
        card_for(definition, audience="https://booker.example.gov", exports=(price_leg, seats))
    except AgentCardError as refused:
        print("cannot publish:", refused.skill)  # noqa: T201


if __name__ == "__main__":
    asyncio.run(main())
