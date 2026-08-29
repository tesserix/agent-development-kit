"""Consumer-style inference shared by strict mypy and strict Pyright lanes."""

from __future__ import annotations

from typing import assert_type

from pydantic import BaseModel

from tesserix_adk import Agent, AgentRunner, TypedAgent, tool
from tesserix_adk.core import ConfigOverrides, KnownTaskClass, ProviderName, Run, load_typed_config
from tesserix_adk.runtime import RunStream


class TripRequest(BaseModel):
    """Typed input supplied by an application."""

    destination: str
    nights: int = 1


class TripPlan(BaseModel):
    """Typed output guaranteed by the declaration."""

    summary: str


@tool
def plan_tool(request: TripRequest, *, limit: int = 3) -> TripPlan:
    """Keep positional, model, keyword-only and default parameter types.

    Args:
        request: Trip to plan.
        limit: Maximum suggestions.
    """
    return TripPlan(summary=f"{request.destination}:{limit}")


@tool
def variadic_tool(*values: int) -> int:
    """Keep variadic element types even though runtime schema admission rejects this shape."""
    return sum(values)


async def agent_inference(runner: AgentRunner) -> None:
    """Prove the released string input and typed output contract remains intact."""
    agent: Agent[TripPlan] = Agent(
        name="planner",
        instructions="Plan the trip.",
        model="typed-provider",
        output_type=TripPlan,
    )
    run = await runner.run(agent, "Plan Kyoto", tenant="acme")
    assert_type(run, Run[TripPlan])
    assert_type(run.output, TripPlan | None)
    stream = runner.stream(agent, "Plan Kyoto", tenant="acme")
    assert_type(stream, RunStream[TripPlan])

    request = TripRequest(destination="Kyoto")
    assert_type(await plan_tool(request, limit=2), TripPlan)
    assert_type(await variadic_tool(1, 2, 3), int)

    await runner.run(agent, request, tenant="acme")  # type: ignore[arg-type]
    await plan_tool("wrong input")  # type: ignore[arg-type]
    await plan_tool(request, limit="three")  # type: ignore[arg-type]
    await variadic_tool(1, "two")  # type: ignore[arg-type]


async def typed_agent_inference(runner: AgentRunner) -> None:
    """Prove structured input is additive and keeps output inference."""
    agent: TypedAgent[TripRequest, TripPlan] = TypedAgent(
        name="typed-planner",
        instructions="Plan the trip.",
        model="typed-provider",
        input_type=TripRequest,
        output_type=TripPlan,
    )
    request = TripRequest(destination="Kyoto")
    run = await runner.run_typed(agent, request, tenant="acme")
    assert_type(run, Run[TripPlan])
    assert_type(run.output, TripPlan | None)
    stream = runner.stream_typed(agent, request, tenant="acme")
    assert_type(stream, RunStream[TripPlan])

    await runner.run_typed(agent, "wrong input", tenant="acme")  # type: ignore[misc]


def literal_and_config_inference() -> None:
    """Prove typos fail at the authoring line, while explicit extension remains possible."""
    task = known_task_class("reasoning")
    provider = known_provider("openrouter")
    assert_type(task, KnownTaskClass)
    assert_type(provider, ProviderName)
    wrong_task: KnownTaskClass = "reasning"  # type: ignore[assignment]
    wrong_provider: ProviderName = "open-router"  # type: ignore[assignment]
    del wrong_task, wrong_provider

    overrides: ConfigOverrides = {
        "provider": {"endpoint": "https://provider.example.invalid"},
        "budget": {"max_model_calls": 2},
    }
    assert load_typed_config(overrides, env={}, start=None).budget.max_model_calls == 2
    misspelled: ConfigOverrides = {  # type: ignore[typeddict-unknown-key]
        "provder": {  # pyright: ignore[reportAssignmentType]
            "endpoint": "https://provider.example.invalid"
        }
    }
    del misspelled


def known_task_class(value: KnownTaskClass) -> KnownTaskClass:
    """Keep the declared alias visible after literal narrowing."""
    return value


def known_provider(value: ProviderName) -> ProviderName:
    """Keep the declared alias visible after literal narrowing."""
    return value
