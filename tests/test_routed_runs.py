"""A run that names the job, and the runner that turns that into a vendor call.

`test_routing.py` covers the table in isolation. This covers the seam: an agent declaring
`task_class` runs, the routing table decides which vendor answers it, and the run record
says which one did and why. The point of the whole feature is the first test here — the
table changed, no code did, and the call went somewhere else.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from tesserix_adk.core import (
    CHEAP,
    REASONING,
    Agent,
    Capability,
    ConfigurationError,
    ModelCapabilities,
    ModelSpec,
    NoEligibleModelError,
    Run,
    RunEventKind,
    RunState,
    TaskClass,
    Usage,
)
from tesserix_adk.models.routing import RoutingRule, RoutingTable, TableRouter
from tesserix_adk.runtime import AgentRunner, ModelResponse
from tesserix_adk.testing import FakeClock, ScriptedProvider

CAPABLE = ModelCapabilities(
    tool_calling=True, streaming=True, vision=True, context_window_tokens=200_000
)
BLIND = ModelCapabilities(tool_calling=True, streaming=True, context_window_tokens=200_000)


def spec(ref: str, capabilities: ModelCapabilities = CAPABLE) -> ModelSpec:
    provider, _, model = ref.partition(":")
    return ModelSpec(provider=provider, model=model, capabilities=capabilities)


def table(*rules: RoutingRule) -> RoutingTable:
    return RoutingTable(rules=rules)


def rule(task_class: TaskClass, *candidates: ModelSpec, **scope: str) -> RoutingRule:
    return RoutingRule(task_class=task_class, candidates=tuple(candidates), **scope)


def answer(text: str = "Kyoto, four nights.") -> ModelResponse:
    return ModelResponse(content=text, usage=Usage(input_tokens=10, output_tokens=5))


def vendors(*names: str) -> dict[str, ScriptedProvider]:
    return {name: ScriptedProvider(answer(), name=name, capabilities=CAPABLE) for name in names}


def routed(
    routing: RoutingTable,
    fleet: dict[str, ScriptedProvider],
    **overrides: object,
) -> AgentRunner:
    fields: dict[str, object] = {
        "provider": next(iter(fleet.values())),
        "providers": fleet,
        "router": TableRouter(routing),
        "clock": FakeClock(),
    }
    return AgentRunner(**{**fields, **overrides})  # type: ignore[arg-type]


def agent(**overrides: object) -> Agent:
    fields: dict[str, object] = {
        "name": "planner",
        "instructions": "Plan trips.",
        "free_text": True,
        "task_class": CHEAP,
    }
    return Agent(**{**fields, **overrides})  # type: ignore[arg-type]


async def start[OutputT: BaseModel](
    runner: AgentRunner, agent_: Agent[OutputT], tenant: str = "acme"
) -> Run[OutputT]:
    return await runner.run(agent_, "plan a trip", tenant=tenant, run_id="run_1")


class TestRetuningWithoutACodeChange:
    async def test_the_table_decides_which_vendor_answers_a_class(self) -> None:
        fleet = vendors("openai", "anthropic")
        run = await start(routed(table(rule(CHEAP, spec("openai:gpt-4o-mini"))), fleet), agent())
        assert run.state is RunState.COMPLETED
        assert [len(vendor.requests) for vendor in fleet.values()] == [1, 0]

    async def test_the_same_agent_on_a_new_table_calls_the_other_vendor(self) -> None:
        """The whole feature: the table changed, the agent did not."""
        fleet = vendors("openai", "anthropic")
        retuned = table(rule(CHEAP, spec("anthropic:claude-haiku-4-5")))
        await start(routed(retuned, fleet), agent())
        assert [len(vendor.requests) for vendor in fleet.values()] == [0, 1]

    async def test_the_chosen_model_is_what_the_request_asks_for(self) -> None:
        fleet = vendors("openai")
        await start(routed(table(rule(CHEAP, spec("openai:gpt-4o-mini"))), fleet), agent())
        assert fleet["openai"].requests[0].model == "gpt-4o-mini"

    async def test_an_agent_naming_a_model_outright_still_uses_the_default_provider(self) -> None:
        """Routing is opt-in; every existing runner keeps the one provider it was given."""
        fleet = vendors("openai", "anthropic")
        run = await start(
            routed(table(rule(CHEAP, spec("anthropic:claude-haiku-4-5"))), fleet),
            agent(task_class=None, model="gpt-4o-mini"),
        )
        assert run.model == "gpt-4o-mini"
        assert len(fleet["openai"].requests) == 1


class TestTheRunSaysWhichModelAnsweredIt:
    async def test_the_run_records_the_model_the_router_chose(self) -> None:
        fleet = vendors("openai")
        run = await start(routed(table(rule(CHEAP, spec("openai:gpt-4o-mini"))), fleet), agent())
        assert run.model == "gpt-4o-mini"

    async def test_the_decision_is_an_event_on_the_run(self) -> None:
        fleet = vendors("openai", "anthropic")
        run = await start(
            routed(
                table(rule(CHEAP, spec("openai:gpt-4o-mini"), spec("anthropic:claude-haiku-4-5"))),
                fleet,
            ),
            agent(),
        )
        routing = [event for event in run.events if event.kind is RunEventKind.MODEL_ROUTED]
        assert [event.name for event in routing] == ["openai:gpt-4o-mini"]
        assert "cheap ->" in (routing[0].detail or "")

    async def test_a_run_that_named_its_model_records_no_routing_event(self) -> None:
        """There was no decision to explain, and an event saying otherwise is a fiction."""
        fleet = vendors("openai")
        run = await start(
            routed(table(rule(CHEAP, spec("openai:gpt-4o-mini"))), fleet),
            agent(task_class=None, model="gpt-4o-mini"),
        )
        assert not [event for event in run.events if event.kind is RunEventKind.MODEL_ROUTED]


class TestWhenTheRunCannotBeRouted:
    async def test_a_class_the_table_does_not_route_fails_the_run_rather_than_guessing(
        self,
    ) -> None:
        fleet = vendors("openai")
        with pytest.raises(NoEligibleModelError, match="reasoning"):
            await start(
                routed(table(rule(CHEAP, spec("openai:gpt-4o-mini"))), fleet),
                agent(task_class=REASONING),
            )

    async def test_a_chosen_model_whose_vendor_was_not_wired_is_a_configuration_error(self) -> None:
        fleet = vendors("openai")
        with pytest.raises(ConfigurationError, match="anthropic"):
            await start(
                routed(table(rule(CHEAP, spec("anthropic:claude-haiku-4-5"))), fleet), agent()
            )

    async def test_an_agent_routed_by_class_without_a_router_is_still_refused(self) -> None:
        with pytest.raises(ConfigurationError, match="task_class"):
            await AgentRunner(provider=ScriptedProvider(answer()), clock=FakeClock()).run(
                agent(), "plan a trip", tenant="acme"
            )


class TestWhatTheAgentNeedsOfTheModel:
    async def test_an_agent_that_sends_images_is_not_routed_to_a_blind_model(self) -> None:
        fleet = vendors("openai")
        with pytest.raises(NoEligibleModelError, match="vision"):
            await start(
                routed(table(rule(CHEAP, spec("openai:gpt-4o-mini", BLIND))), fleet),
                agent(requires={Capability.VISION}),
            )

    async def test_an_agent_with_tools_needs_a_candidate_that_calls_them(self) -> None:
        fleet = vendors("openai")
        deaf = ModelCapabilities(streaming=True, context_window_tokens=8_000)
        with pytest.raises(NoEligibleModelError, match="tool_calling"):
            await start(
                routed(table(rule(CHEAP, spec("openai:gpt-4o-mini", deaf))), fleet),
                agent(requires={Capability.TOOL_CALLING}),
            )


class TestAReloadAppliesToTheNextRun:
    """A run resolves once, before the first call, and carries that model to the end. A
    reload halfway through would otherwise make one record describe two runs."""

    async def test_a_reload_leaves_a_finished_runs_record_alone(self) -> None:
        fleet = vendors("openai", "anthropic")
        runner = routed(table(rule(CHEAP, spec("openai:gpt-4o-mini"))), fleet)
        run = await start(runner, agent())
        runner.reload(TableRouter(table(rule(CHEAP, spec("anthropic:claude-haiku-4-5")))))
        assert run.model == "gpt-4o-mini"

    async def test_the_next_run_uses_the_reloaded_table(self) -> None:
        fleet = vendors("openai", "anthropic")
        runner = routed(table(rule(CHEAP, spec("openai:gpt-4o-mini"))), fleet)
        await start(runner, agent())
        runner.reload(TableRouter(table(rule(CHEAP, spec("anthropic:claude-haiku-4-5")))))
        assert (await start(runner, agent())).model == "claude-haiku-4-5"

    async def test_a_reload_needs_a_router_to_reload(self) -> None:
        with pytest.raises(ConfigurationError, match="no router"):
            AgentRunner(provider=ScriptedProvider(answer()), clock=FakeClock()).reload(
                TableRouter(table(rule(CHEAP, spec("openai:gpt-4o-mini"))))
            )


class TestOutputTypesAreUnaffected:
    async def test_a_structured_run_routes_the_same_way(self) -> None:
        class TripPlan(BaseModel):
            destination: str
            nights: int

        fleet = {
            "openai": ScriptedProvider(
                ModelResponse(content='{"destination": "Kyoto", "nights": 4}'),
                name="openai",
                capabilities=CAPABLE.declaring(structured_output=True),
            )
        }
        run = await start(
            routed(table(rule(CHEAP, spec("openai:gpt-4o-mini"))), fleet),
            Agent[TripPlan](
                name="planner",
                instructions="Plan trips.",
                task_class=CHEAP,
                output_type=TripPlan,
            ),
        )
        assert run.output == TripPlan(destination="Kyoto", nights=4)
