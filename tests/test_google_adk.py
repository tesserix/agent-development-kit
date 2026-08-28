"""Google Agent Development Kit consumes a Tesserix A2A endpoint."""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("a2a")
pytest.importorskip("google.adk")

from google.adk.agents.remote_a2a_agent import RemoteA2aAgent

from tesserix_adk.adapters import (
    A2AInterface,
    A2ASkill,
    a2a_card_for,
    google_adk_remote_agent,
)
from tesserix_adk.core import Agent, AgentDefinition, Owner


def card() -> Any:
    definition = AgentDefinition(
        agent=Agent(
            name="trip-planner",
            version="1.0.0",
            instructions="Plan trips.",
            model="test-model",
            free_text=True,
        ),
        owner=Owner(team="Travel", contact="travel@example.test", service="planner"),
        evaluation_suite="evals/travel.jsonl",
    )
    return a2a_card_for(
        definition,
        description="Plans trips.",
        provider_url="https://agents.example.test",
        interfaces=(
            A2AInterface(
                url="https://agents.example.test/a2a/trip-planner",
                protocol_binding="JSONRPC",
            ),
        ),
        skills=(
            A2ASkill(
                id="plan-trip",
                name="Plan a trip",
                description="Creates an itinerary.",
                tags=("travel",),
            ),
        ),
    )


def test_google_adk_accepts_the_official_tesserix_agent_card() -> None:
    with pytest.warns(UserWarning, match="EXPERIMENTAL"):
        remote = google_adk_remote_agent(
            name="tesserix_trip_planner",
            description="A Tesserix agent reached over official A2A.",
            agent_card=card(),
            timeout_seconds=30.0,
        )

    assert isinstance(remote, RemoteA2aAgent)
    assert remote.name == "tesserix_trip_planner"
    assert remote.description == "A Tesserix agent reached over official A2A."
