"""Official A2A 1.x interoperability, distinct from the Tesserix peer protocol."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

pytest.importorskip("a2a")

from google.protobuf.json_format import MessageToDict

from tesserix_adk.adapters import (
    A2ABearerSecurity,
    A2ACardError,
    A2AInterface,
    A2ARegistryError,
    A2ASkill,
    a2a_card_for,
    a2a_client_factory,
    a2a_client_from_registry,
)
from tesserix_adk.core import Agent, AgentDefinition, Owner, TaskClass

if TYPE_CHECKING:
    from a2a.client import ClientConfig
    from a2a.client.transports.base import ClientTransport
    from a2a.types import AgentCard


def definition(name: str = "trip-planner") -> AgentDefinition[Any]:
    return AgentDefinition(
        agent=Agent(
            name=name,
            version="2.1.0",
            instructions="Plan travel without exposing this private instruction.",
            task_class=TaskClass("planning"),
            free_text=True,
            scopes=("trips:read",),
        ),
        owner=Owner(
            team="Travel Platform",
            contact="travel-oncall@example.com",
            service="trip-planner-api",
        ),
        evaluation_suite="suites/trip-planner.yaml",
    )


def official_card(name: str = "trip-planner") -> AgentCard:
    return a2a_card_for(
        definition(name),
        description="Plans an itinerary from a traveller's constraints.",
        provider_url="https://tesserix.ai",
        documentation_url="https://docs.tesserix.ai/agents/trip-planner",
        interfaces=(
            A2AInterface(
                url="https://agents.example.com/a2a",
                protocol_binding="JSONRPC",
            ),
        ),
        skills=(
            A2ASkill(
                id="plan-trip",
                name="Plan a trip",
                description="Creates a day-by-day itinerary.",
                tags=("travel", "planning"),
                examples=("Plan three accessible days in Melbourne.",),
            ),
        ),
        streaming=True,
        security=A2ABearerSecurity(scopes=("trips:read",)),
    )


class TestOfficialAgentCard:
    def test_the_official_sdk_accepts_the_generated_card_and_serialises_v1_fields(self) -> None:
        rendered = MessageToDict(official_card(), preserving_proto_field_name=False)
        assert rendered["name"] == "trip-planner"
        assert rendered["version"] == "2.1.0"
        assert rendered["supportedInterfaces"] == [
            {
                "url": "https://agents.example.com/a2a",
                "protocolBinding": "JSONRPC",
                "protocolVersion": "1.0",
            }
        ]
        assert rendered["capabilities"] == {"streaming": True}
        assert rendered["defaultInputModes"] == ["text/plain"]
        assert rendered["defaultOutputModes"] == ["text/plain"]
        assert rendered["skills"][0]["id"] == "plan-trip"

    def test_bearer_auth_is_published_as_a_scheme_and_requirement(self) -> None:
        rendered = MessageToDict(official_card(), preserving_proto_field_name=False)
        assert rendered["securitySchemes"]["bearer"]["httpAuthSecurityScheme"] == {
            "description": "Bearer access token",
            "scheme": "bearer",
            "bearerFormat": "JWT",
        }
        assert rendered["securityRequirements"] == [
            {"schemes": {"bearer": {"list": ["trips:read"]}}}
        ]

    def test_private_definition_fields_never_enter_public_discovery_metadata(self) -> None:
        rendered = str(MessageToDict(official_card(), preserving_proto_field_name=False))
        assert "Plan travel without exposing" not in rendered
        assert "travel-oncall@example.com" not in rendered
        assert "suites/trip-planner.yaml" not in rendered

    def test_a_card_missing_required_skills_is_refused_before_publication(self) -> None:
        with pytest.raises(A2ACardError, match="skills"):
            a2a_card_for(
                definition(),
                description="Plans trips.",
                provider_url="https://tesserix.ai",
                interfaces=(
                    A2AInterface(
                        url="https://agents.example.com/a2a",
                        protocol_binding="JSONRPC",
                    ),
                ),
                skills=(),
            )


class TestCustomGatewayTransport:
    def test_a_custom_protocol_binding_is_registered_with_the_official_factory(self) -> None:
        card = a2a_card_for(
            definition(),
            description="Plans trips.",
            provider_url="https://tesserix.ai",
            interfaces=(
                A2AInterface(
                    url="https://gateway.example.com/agents/trip-planner",
                    protocol_binding="TESSERIX-GATEWAY",
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

        class TransportSelectedError(RuntimeError):
            pass

        def selected(_card: AgentCard, url: str, _config: ClientConfig) -> ClientTransport:
            raise TransportSelectedError(url)

        factory = a2a_client_factory(
            protocol_bindings=("TESSERIX-GATEWAY",),
            transports={"TESSERIX-GATEWAY": selected},
        )
        with pytest.raises(TransportSelectedError, match=r"gateway\.example\.com"):
            factory.create(card)


class TestCustomRegistry:
    async def test_a_registry_cannot_substitute_a_different_named_agent(self) -> None:
        class SwappedRegistry:
            async def resolve(self, name: str) -> AgentCard:
                del name
                return official_card("billing-agent")

        with pytest.raises(A2ARegistryError, match="billing-agent"):
            await a2a_client_from_registry(
                SwappedRegistry(),
                "trip-planner",
                factory=a2a_client_factory(),
            )
