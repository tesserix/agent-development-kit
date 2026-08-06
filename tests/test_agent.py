"""An agent as a declaration, and the errors every layer raises.

`Agent` describes what an agent is; it does not run one. It holds no provider client and
no I/O, so the same declaration can be constructed in a test, serialised into a config
file and diffed in review.

Every error carries the run and tenant it happened in, because "ProviderTimeoutError" in
a log with neither is a fact nobody can act on.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel, ValidationError

from tesserix_adk.core import (
    AdkError,
    Agent,
    BudgetConfig,
    BudgetExceededError,
    CancelledError,
    CapabilityError,
    GuardrailViolationError,
    MaxIterationsError,
    ProviderError,
    ProviderTimeoutError,
    SchemaViolationError,
    ToolExecutionError,
)

ERRORS = [
    CapabilityError,
    ProviderError,
    ProviderTimeoutError,
    SchemaViolationError,
    ToolExecutionError,
    GuardrailViolationError,
    BudgetExceededError,
    CancelledError,
    MaxIterationsError,
]


class TripPlan(BaseModel):
    destination: str


class TestAgent:
    def test_an_agent_is_a_declaration(self) -> None:
        agent = Agent(name="planner", instructions="Plan trips.", model="claude-sonnet-5")
        assert agent.name == "planner"

    def test_an_agent_is_frozen(self) -> None:
        agent = Agent(name="planner", instructions="Plan trips.", model="claude-sonnet-5")
        with pytest.raises(ValidationError):
            agent.name = "other"

    def test_an_agent_round_trips_without_its_output_type(self) -> None:
        """A declaration is diffed in review and stored in config, so it must serialise."""
        agent = Agent(name="planner", instructions="Plan trips.", model="claude-sonnet-5")
        assert Agent.model_validate_json(agent.model_dump_json()) == agent

    def test_an_agent_may_declare_a_task_class_instead_of_a_model(self) -> None:
        """Naming the job rather than the model is what lets a router choose."""
        agent = Agent(name="planner", instructions="Plan trips.", task_class="reasoning")
        assert agent.task_class == "reasoning"

    def test_an_agent_with_neither_a_model_nor_a_task_class_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="task_class"):
            Agent(name="planner", instructions="Plan trips.")

    def test_an_agent_with_both_a_model_and_a_task_class_is_refused(self) -> None:
        """Two answers to which model runs is a routing decision made twice."""
        with pytest.raises(ValidationError, match="task_class"):
            Agent(
                name="planner",
                instructions="Plan trips.",
                model="claude-sonnet-5",
                task_class="reasoning",
            )

    def test_the_tool_allowlist_is_empty_by_default(self) -> None:
        """A tool a consumer did not name is a tool the agent may not call."""
        agent = Agent(name="planner", instructions="Plan trips.", model="claude-sonnet-5")
        assert agent.tools == ()

    def test_an_agent_carries_its_output_type(self) -> None:
        agent = Agent(
            name="planner",
            instructions="Plan trips.",
            model="claude-sonnet-5",
            output_type=TripPlan,
        )
        assert agent.output_type is TripPlan

    def test_a_duplicated_tool_name_is_refused(self) -> None:
        """Listing it twice means one of the two entries was meant to be something else."""
        with pytest.raises(ValidationError, match="search"):
            Agent(
                name="planner",
                instructions="Plan trips.",
                model="claude-sonnet-5",
                tools=("search", "search"),
            )

    def test_an_agent_carries_its_ceiling_and_its_guardrail_chain(self) -> None:
        """Both are part of the declaration, so a review sees what an agent may do."""
        agent = Agent(
            name="planner",
            instructions="Plan trips.",
            model="claude-sonnet-5",
            budget=BudgetConfig(max_tokens_per_run=1000),
            guardrails=("no_pii", "no_prompt_leak"),
        )
        assert agent.budget is not None
        assert agent.budget.max_tokens_per_run == 1000
        assert agent.guardrails == ("no_pii", "no_prompt_leak")

    def test_the_guardrail_chain_keeps_its_order(self) -> None:
        """Order is the contract: a redaction after an export check redacts nothing."""
        agent = Agent(
            name="planner",
            instructions="Plan trips.",
            model="claude-sonnet-5",
            guardrails=("redact", "export_check"),
        )
        assert list(agent.guardrails) == ["redact", "export_check"]

    def test_an_agent_holds_no_provider_client(self) -> None:
        """Construction-time config only: a declaration with a socket cannot be reused."""
        with pytest.raises(ValidationError):
            Agent(
                name="planner",
                instructions="Plan trips.",
                model="claude-sonnet-5",
                provider=object(),  # type: ignore[call-arg]
            )


class TestErrors:
    @pytest.mark.parametrize("error", ERRORS, ids=lambda e: e.__name__)
    def test_every_error_is_catchable_as_one_kind(self, error: type[AdkError]) -> None:
        """A consumer catches this kit's failures without catching its own bugs too."""
        assert issubclass(error, AdkError)

    @pytest.mark.parametrize("error", ERRORS, ids=lambda e: e.__name__)
    def test_every_error_records_where_it_happened(self, error: type[AdkError]) -> None:
        """A failure with no run and no tenant is a fact nobody can act on."""
        raised = error("it failed", run_id="run_1", tenant="acme")
        assert (raised.run_id, raised.tenant) == ("run_1", "acme")

    @pytest.mark.parametrize("error", ERRORS, ids=lambda e: e.__name__)
    def test_the_message_survives(self, error: type[AdkError]) -> None:
        assert "it failed" in str(error("it failed"))

    def test_where_it_happened_is_optional(self) -> None:
        """Configuration fails before there is a run; a required run_id would be a lie."""
        assert ProviderError("no api key").run_id is None

    def test_a_timeout_is_a_provider_failure(self) -> None:
        """`except ProviderError` must not miss the most common provider failure."""
        assert issubclass(ProviderTimeoutError, ProviderError)

    def test_an_error_carries_a_payload_for_debugging(self) -> None:
        raised = ProviderError("429", details={"status": "429", "provider": "anthropic"})
        assert raised.details["provider"] == "anthropic"

    def test_a_schema_violation_names_the_offending_output(self) -> None:
        raised = SchemaViolationError("not a TripPlan", details={"output": "{'dest': 1}"})
        assert "not a TripPlan" in str(raised)

    def test_the_fake_budget_raises_the_error_the_kit_defines(self) -> None:
        """A fake that raises a lookalike lets `except BudgetExceededError` pass in tests
        and fail in production."""
        from tesserix_adk import testing

        assert testing.BudgetExceededError is BudgetExceededError

    def test_the_repr_carries_the_context_a_log_line_needs(self) -> None:
        raised = ProviderError("429", run_id="run_1", tenant="acme")
        assert "run_1" in repr(raised)
        assert "acme" in repr(raised)
