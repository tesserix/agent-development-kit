"""What a peer can learn about an agent without reading its source, and what it cannot."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import pytest

from tesserix_adk.a2a import (
    MAX_CARD_BYTES,
    MAX_SKILLS,
    WELL_KNOWN_PATH,
    AgentCard,
    AgentCardError,
    AgentLimits,
    AgentProvider,
    CardEndpoint,
    card_for,
)
from tesserix_adk.cli import card_main
from tesserix_adk.core import Agent, AgentDefinition, Owner, TaskClass
from tesserix_adk.core.config import ConcurrencyConfig, DeadlineConfig
from tesserix_adk.tools import tool

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


@dataclass(frozen=True)
class _Declared:
    """A skill source shaped like a tool, for the shapes a real tool cannot take."""

    name: str
    description: str
    parameters_schema: dict[str, Any]
    returns_schema: dict[str, Any] | None = None


@tool(name="card_price_leg")
def price_leg(leg: str) -> dict[str, Any]:
    """Price one leg of an itinerary."""
    return {"leg": leg, "eur": 40}


@tool(name="card_refund")
def refund(order: str) -> dict[str, Any]:
    """Refund an order once a human has said so."""
    return {"order": order}


@tool(name="card_summarise")
def summarise(text: str) -> str:
    """Summarise a passage in prose."""
    return text[:10]


@tool(name="card_stranger")
def stranger() -> dict[str, Any]:
    """A tool this agent was never given."""
    return {}


@tool(name="card_internal_ledger")
def internal_ledger() -> dict[str, Any]:
    """Never published: the agent's own bookkeeping."""
    return {"balance": 1}


@tool(name="card_internal_seats")
def internal_seats() -> dict[str, Any]:
    """Never published: seat inventory belonging to another tenant's contract."""
    return {"seats": 2}


def _definition() -> AgentDefinition[Any]:
    """A booking agent with three published skills and two it keeps to itself."""
    agent = Agent(
        name="booker",
        version="2.1.0",
        instructions="Book the trip.",
        free_text=True,
        task_class=TaskClass("planning"),
        tools=(
            "card_price_leg",
            "card_refund",
            "card_summarise",
            "card_internal_ledger",
            "card_internal_seats",
        ),
        idempotent_tools=("card_price_leg", "card_summarise"),
        approval_required_tools=("card_refund",),
        scopes=("bookings:read", "bookings:write", "payments:refund"),
        tool_scopes={"card_refund": ("payments:refund",)},
        concurrency=ConcurrencyConfig(max_concurrent_tools=4),
        deadlines=DeadlineConfig(run_seconds=30.0),
    )
    return AgentDefinition(
        agent=agent,
        owner=Owner(team="travel", contact="travel@example.gov", service="booker-api"),
        evaluation_suite="suites/booker.yaml",
    )


def _card(**overrides: Any) -> AgentCard:
    """The card the booking agent publishes."""
    arguments: dict[str, Any] = {
        "audience": "https://booker.example.gov",
        "exports": (price_leg, refund, summarise),
    }
    arguments.update(overrides)
    return card_for(_definition(), **arguments)


async def _fetch(
    endpoint: CardEndpoint,
    *,
    path: str = WELL_KNOWN_PATH,
    method: str = "GET",
    headers: Sequence[tuple[bytes, bytes]] = (),
) -> tuple[int, dict[str, str], bytes]:
    """Drive the endpoint as an ASGI server would, and return status, headers and body."""
    sent: list[Mapping[str, Any]] = []

    async def receive() -> Mapping[str, Any]:
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message: Mapping[str, Any]) -> None:
        sent.append(message)

    scope = {"type": "http", "method": method, "path": path, "headers": list(headers)}
    await endpoint(scope, receive, send)
    start = sent[0]
    body = b"".join(message.get("body", b"") for message in sent[1:])
    rendered = {k.decode(): v.decode() for k, v in start["headers"]}
    return int(start["status"]), rendered, body


class TestWhatACardSays:
    def test_it_names_the_agent_and_the_version_a_caller_can_pin_to(self) -> None:
        card = _card()
        assert card.agent == "booker"
        assert card.version == "2.1.0"
        assert card.task_class == "planning"

    def test_it_names_the_team_that_publishes_it(self) -> None:
        assert _card().provider == AgentProvider(organisation="travel", service="booker-api")

    def test_it_carries_the_schemas_a_caller_needs_to_build_a_request(self) -> None:
        priced = next(skill for skill in _card().skills if skill.name == "card_price_leg")
        assert priced.input_schema["properties"]["leg"]["type"] == "string"
        assert priced.description == "Price one leg of an itinerary."

    def test_a_skill_answering_in_prose_publishes_no_output_schema(self) -> None:
        prose = next(skill for skill in _card().skills if skill.name == "card_summarise")
        assert prose.output_schema is None

    def test_it_says_which_skill_needs_more_authority_than_the_others(self) -> None:
        refunding = next(skill for skill in _card().skills if skill.name == "card_refund")
        assert refunding.required_scopes == ("payments:refund",)
        assert refunding.requires_approval is True

    def test_it_says_which_skills_are_safe_to_call_again(self) -> None:
        idempotent = {skill.name for skill in _card().skills if skill.idempotent}
        assert idempotent == {"card_price_leg", "card_summarise"}

    def test_the_limits_are_the_agent_s_declared_limits_not_a_guess(self) -> None:
        assert _card().limits == AgentLimits(
            max_concurrent_calls=4, latency_seconds=30.0, max_payload_bytes=MAX_CARD_BYTES
        )

    def test_the_card_matches_the_declaration_it_was_generated_from(self) -> None:
        definition = _definition()
        card = card_for(
            definition, audience="https://booker.example.gov", exports=(price_leg, refund)
        )
        assert {skill.name for skill in card.skills} <= set(definition.agent.tools)


class TestExportIsDeliberate:
    def test_an_unexported_tool_is_not_on_the_card(self) -> None:
        assert {skill.name for skill in _card().skills} == {
            "card_price_leg",
            "card_refund",
            "card_summarise",
        }

    def test_an_unexported_tool_is_not_named_anywhere_in_the_rendered_card(self) -> None:
        rendered = _card().rendered().decode()
        assert "internal_ledger" not in rendered
        assert "internal_seats" not in rendered

    def test_exporting_nothing_publishes_nothing(self) -> None:
        assert _card(exports=()).skills == ()

    def test_publishing_a_skill_the_agent_may_not_call_is_refused(self) -> None:
        with pytest.raises(AgentCardError, match="card_stranger") as refused:
            card_for(
                _definition(),
                audience="https://booker.example.gov",
                exports=(price_leg, stranger),
            )
        assert refused.value.skill == "card_stranger"


class TestASkillThatCannotBeDescribed:
    def test_a_skill_whose_input_is_not_an_object_names_itself_and_the_problem(self) -> None:
        unsayable = _Declared("card_price_leg", "Price a leg.", {"type": "array"})
        with pytest.raises(AgentCardError, match="card_price_leg") as refused:
            card_for(_definition(), audience="https://booker.example.gov", exports=(unsayable,))
        assert refused.value.skill == "card_price_leg"

    def test_a_skill_with_no_description_is_refused_rather_than_published_blank(self) -> None:
        with pytest.raises(AgentCardError, match="card_price_leg"):
            card_for(
                _definition(),
                audience="https://booker.example.gov",
                exports=(_Declared("card_price_leg", "  ", {"type": "object"}),),
            )


class TestCeilings:
    def test_more_skills_than_the_ceiling_is_refused_at_generation(self) -> None:
        many = tuple(
            _Declared("card_price_leg", "Price one leg.", {"type": "object"})
            for _ in range(MAX_SKILLS + 1)
        )
        with pytest.raises(AgentCardError, match="skills"):
            card_for(_definition(), audience="https://booker.example.gov", exports=many)

    def test_a_card_larger_than_the_ceiling_is_refused_at_generation(self) -> None:
        wordy = _Declared("card_price_leg", "Price one leg. " * 6000, {"type": "object"})
        with pytest.raises(AgentCardError, match="bytes"):
            card_for(_definition(), audience="https://booker.example.gov", exports=(wordy,))


class TestServingTheCard:
    async def test_a_peer_fetches_a_schema_valid_card_from_the_well_known_path(self) -> None:
        status, headers, body = await _fetch(CardEndpoint(_card()))
        assert status == 200
        assert headers["content-type"] == "application/json"
        served = AgentCard.model_validate_json(body)
        assert {skill.name for skill in served.skills} == {
            "card_price_leg",
            "card_refund",
            "card_summarise",
        }

    async def test_the_card_carries_an_etag_and_a_cache_lifetime(self) -> None:
        _, headers, _ = await _fetch(CardEndpoint(_card(), max_age_seconds=600))
        assert headers["etag"].startswith('"')
        assert headers["cache-control"] == "public, max-age=600"

    async def test_an_unchanged_card_is_answered_without_its_body(self) -> None:
        endpoint = CardEndpoint(_card())
        _, headers, _ = await _fetch(endpoint)
        status, _, body = await _fetch(
            endpoint, headers=[(b"if-none-match", headers["etag"].encode())]
        )
        assert status == 304
        assert body == b""

    async def test_a_head_answers_the_same_headers_and_no_body(self) -> None:
        endpoint = CardEndpoint(_card())
        _, expected, served = await _fetch(endpoint)
        status, headers, body = await _fetch(endpoint, method="HEAD")
        assert (status, body) == (200, b"")
        assert headers["etag"] == expected["etag"]
        assert headers["content-length"] == str(len(served))

    async def test_another_path_is_not_this_endpoint_s_to_answer(self) -> None:
        status, _, _ = await _fetch(CardEndpoint(_card()), path="/healthz")
        assert status == 404

    async def test_a_card_is_read_not_written(self) -> None:
        status, headers, _ = await _fetch(CardEndpoint(_card()), method="POST")
        assert status == 405
        assert headers["allow"] == "GET, HEAD"


class TestTwoVersionsBehindOneHost:
    async def test_each_endpoint_states_its_own_version(self) -> None:
        older = card_for(_definition(), audience="https://booker.example.gov", exports=(price_leg,))
        _, _, body = await _fetch(CardEndpoint(older))
        assert json.loads(body)["version"] == "2.1.0"


class TestADegradedAgent:
    async def test_the_card_can_be_replaced_with_one_that_admits_the_degradation(self) -> None:
        endpoint = CardEndpoint(_card())
        _, before, _ = await _fetch(endpoint)
        endpoint.serve(_card(available=False))
        _, after, body = await _fetch(endpoint)
        assert after["etag"] != before["etag"]
        assert json.loads(body)["available"] is False


class TestTheCardIsPublicMetadata:
    def test_it_does_not_publish_where_to_page_the_owning_team(self) -> None:
        assert "travel@example.gov" not in _card().rendered().decode()

    def test_it_does_not_publish_how_the_agent_is_built(self) -> None:
        published = json.loads(_card().rendered())
        assert "Book the trip" not in _card().rendered().decode()
        assert {"instructions", "model", "evaluation_suite", "metadata"} & set(published) == set()


class TestLintingBeforeDeploy:
    async def test_a_valid_card_is_rendered_for_review(self, capsys: Any) -> None:
        assert await card_main([], build=_card) == 0
        assert json.loads(capsys.readouterr().out)["agent"] == "booker"

    async def test_linting_reports_what_a_peer_would_receive(self, capsys: Any) -> None:
        assert await card_main(["--lint"], build=_card) == 0
        assert "3 skills" in capsys.readouterr().out

    async def test_a_card_that_cannot_be_generated_fails_the_lint(self, capsys: Any) -> None:
        def broken() -> AgentCard:
            raise AgentCardError("input schema is not an object", skill="card_price_leg")

        assert await card_main(["--lint"], build=broken) == 1
        assert "card_price_leg" in capsys.readouterr().out

    async def test_a_command_line_it_cannot_read_is_not_a_verdict_on_the_card(self) -> None:
        assert await card_main(["--publish"], build=_card) == 2


class TestTheCardAsADelegationContract:
    def test_it_still_says_what_a_credential_for_the_agent_is_minted_for(self) -> None:
        assert _card().accepts.names == frozenset(
            {"bookings:read", "bookings:write", "payments:refund"}
        )

    def test_a_card_naming_no_audience_is_refused(self) -> None:
        with pytest.raises(ValueError, match="audience"):
            AgentCard(agent="booker", audience="   ")


class TestMountingItSomewhereElse:
    async def test_a_deployment_may_serve_the_card_from_a_path_of_its_own(self) -> None:
        endpoint = CardEndpoint(_card(), path="/cards/booker.json")
        assert endpoint.path == "/cards/booker.json"
        status, _, _ = await _fetch(endpoint, path="/cards/booker.json")
        assert status == 200
