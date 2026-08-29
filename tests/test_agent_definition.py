"""What an agent is as a reviewable artifact, rather than as a construction call site.

An agent whose model policy, tool allowlist, limits and owner are scattered across the
places that build it is an agent nobody can review, diff or trace a past run back to.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from pydantic import BaseModel, ValidationError

from tesserix_adk.core import Agent, BudgetLimits, LoopConfig, ModelCapabilities, TypedAgent
from tesserix_adk.core.definition import AgentDefinition, Owner, TypedAgentDefinition
from tesserix_adk.core.errors import ConfigurationError
from tesserix_adk.observability import attributes_of, spend_of
from tesserix_adk.runtime import AgentRunner, ModelResponse
from tesserix_adk.testing import FakeClock, ScriptedProvider

CAPABLE = ModelCapabilities(tool_calling=True, context_window_tokens=200_000)


class Cited(BaseModel):
    """An answer that has to name its page."""

    answer: str
    page: int


class Bare(BaseModel):
    """The same answer with the citation dropped."""

    answer: str


class Query(BaseModel):
    """The application input included in the reviewed definition."""

    question: str


OWNER = Owner(team="search-platform", contact="search@example.gov", service="aequitas-search")


def agent(**overrides: Any) -> Agent[Any]:
    fields: dict[str, Any] = {
        "name": "clerk",
        "instructions": "Answer from the file. Cite the page.",
        "model": "llama-3.1-8b",
        "free_text": True,
    }
    return Agent(**(fields | overrides))


def define(**overrides: Any) -> AgentDefinition[Any]:
    fields: dict[str, Any] = {
        "agent": agent(),
        "owner": OWNER,
        "evaluation_suite": "suites/clerk.yaml",
    }
    return AgentDefinition(**(fields | overrides))


class TestWhatADefinitionMustSay:
    def test_an_agent_without_an_owner_is_refused(self) -> None:
        """An agent nobody answers for is an agent nobody fixes at three in the morning."""
        with pytest.raises(ValidationError, match="owner"):
            AgentDefinition(agent=agent(), evaluation_suite="suites/clerk.yaml")  # type: ignore[call-arg]

    def test_an_agent_nobody_evaluates_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="evaluation_suite"):
            AgentDefinition(agent=agent(), owner=OWNER)  # type: ignore[call-arg]

    def test_an_owner_needs_somewhere_to_send_the_page(self) -> None:
        with pytest.raises(ValidationError, match="contact"):
            Owner(team="search-platform", contact="search-platform", service="aequitas-search")

    def test_declaring_no_tools_means_no_tools(self) -> None:
        """The empty allowlist is the safe reading; 'all of them' is never inferred."""
        assert define().agent.tools == ()

    def test_the_input_schema_is_recorded_and_changes_the_revision(self) -> None:
        fields: dict[str, Any] = {
            "name": "clerk",
            "instructions": "Answer from the file. Cite the page.",
            "model": "llama-3.1-8b",
            "free_text": True,
        }
        prose = TypedAgentDefinition(
            agent=TypedAgent(input_type=str, **fields),
            owner=OWNER,
            evaluation_suite="suites/clerk.yaml",
        )
        typed = TypedAgentDefinition(
            agent=TypedAgent(input_type=Query, **fields),
            owner=OWNER,
            evaluation_suite="suites/clerk.yaml",
        )

        assert typed.input_schema == Query.model_json_schema()
        assert prose.input_schema == {"type": "string"}
        assert typed.revision != prose.revision

    def test_a_zero_execution_limit_is_refused_rather_than_silently_disabling_the_agent(
        self,
    ) -> None:
        with pytest.raises(ValidationError, match="permits nothing at all"):
            define(agent=agent(budget=BudgetLimits(max_model_calls=0)))

    def test_a_zero_loop_cap_is_refused_too(self) -> None:
        with pytest.raises(ValidationError, match="at least 1"):
            define(agent=agent(loop=LoopConfig(max_repeated_calls=0)))


class TestValidatingAgainstWhatIsRegistered:
    def test_a_tool_that_is_not_registered_is_refused_at_construction(self) -> None:
        """Discovering this at first execution in production is discovering it late."""
        with pytest.raises(ConfigurationError, match="ledger"):
            AgentDefinition.declared(
                agent=agent(tools=("search", "ledger")),
                owner=OWNER,
                evaluation_suite="suites/clerk.yaml",
                known_tools=("search",),
            )

    def test_the_error_names_the_field_and_what_is_available(self) -> None:
        with pytest.raises(ConfigurationError, match=r"tools.*available.*search"):
            AgentDefinition.declared(
                agent=agent(tools=("ledger",)),
                owner=OWNER,
                evaluation_suite="suites/clerk.yaml",
                known_tools=("search",),
            )

    def test_a_definition_whose_tools_all_exist_is_returned(self) -> None:
        declared = AgentDefinition.declared(
            agent=agent(tools=("search",)),
            owner=OWNER,
            evaluation_suite="suites/clerk.yaml",
            known_tools=("search", "ledger"),
        )
        assert declared.agent.tools == ("search",)

    def test_declaring_without_a_registry_checks_nothing_and_says_so(self) -> None:
        """A definition built before the registry exists is still a definition."""
        assert AgentDefinition.declared(
            agent=agent(tools=("search",)),
            owner=OWNER,
            evaluation_suite="suites/clerk.yaml",
        ).agent.tools == ("search",)


class TestTheRevision:
    def test_two_definitions_of_the_same_agent_are_the_same_revision(self) -> None:
        assert define().revision == define().revision

    def test_changing_the_instructions_produces_a_new_revision(self) -> None:
        """A revision is derived from the content, so an edit cannot pass as the old one."""
        assert define().revision != define(agent=agent(instructions="Say nothing.")).revision

    def test_changing_the_tool_allowlist_produces_a_new_revision(self) -> None:
        assert define().revision != define(agent=agent(tools=("search",))).revision

    def test_changing_the_owner_produces_a_new_revision(self) -> None:
        """Who answers for an agent is part of what was reviewed."""
        moved = OWNER.model_copy(update={"team": "records"})
        assert define().revision != define(owner=moved).revision

    def test_changing_the_answer_shape_produces_a_new_revision(self) -> None:
        """The answer schema is part of what was reviewed, not a detail of the caller."""
        one = define(agent=agent(free_text=False, output_type=Cited))
        two = define(agent=agent(free_text=False, output_type=Bare))
        assert one.revision != two.revision

    def test_two_versions_of_one_name_coexist_without_collision(self) -> None:
        one, two = define(), define(agent=agent(version="2.0.0"))
        assert one.key == "clerk@1.0.0"
        assert two.key == "clerk@2.0.0"
        assert one.revision != two.revision

    def test_a_definition_cannot_be_mutated_into_a_new_one(self) -> None:
        with pytest.raises(ValidationError):
            define().evaluation_suite = "suites/other.yaml"

    def test_a_revision_survives_a_round_trip_through_json(self) -> None:
        """Review, diff and storage all read the serialised form; it must mean the same."""
        original = define(agent=agent(tools=("search",)))
        restored = AgentDefinition[Any].model_validate_json(original.model_dump_json())
        assert restored.revision == original.revision

    def test_a_stored_definition_still_says_what_shape_it_answers_in(self) -> None:
        """The answer type is a class the store cannot hold; its schema is data it can."""
        stored = json.loads(
            define(agent=agent(free_text=False, output_type=Cited)).model_dump_json()
        )
        assert stored["output_schema"] == Cited.model_json_schema()

    def test_a_schema_that_was_stored_is_kept_rather_than_recomputed(self) -> None:
        """A reviewer reads back the shape that was agreed, not one derived again today."""
        agreed = {"type": "object", "properties": {"answer": {"type": "string"}}}
        assert define(output_schema=agreed).output_schema == agreed


class TestPinningItToTheRun:
    async def test_the_run_records_the_definition_that_produced_it(self) -> None:
        definition = define()
        run = await AgentRunner(
            provider=ScriptedProvider(
                ModelResponse(content="page 12."), name="scripted", capabilities=CAPABLE
            ),
            clock=FakeClock(),
        ).run(definition, "what does page 12 say?", tenant="acme")
        assert run.definition_revision == definition.revision
        assert run.agent_name == "clerk"
        assert run.agent_version == "1.0.0"

    async def test_every_span_carries_the_revision(self) -> None:
        """A past run has to name the exact definition, not a name that has moved on."""
        definition = define()
        run = await AgentRunner(
            provider=ScriptedProvider(
                ModelResponse(content="page 12."), name="scripted", capabilities=CAPABLE
            ),
            clock=FakeClock(),
        ).run(definition, "what does page 12 say?", tenant="acme")
        for record in spend_of(run):
            assert attributes_of(record)["adk.definition"] == definition.revision

    async def test_a_run_started_from_a_bare_agent_says_the_revision_is_unknown(self) -> None:
        run = await AgentRunner(
            provider=ScriptedProvider(
                ModelResponse(content="page 12."), name="scripted", capabilities=CAPABLE
            ),
            clock=FakeClock(),
        ).run(agent(), "what does page 12 say?", tenant="acme")
        assert run.definition_revision is None
        for record in spend_of(run):
            assert attributes_of(record)["adk.definition"] == "unknown"
