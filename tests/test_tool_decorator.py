"""A tool is one typed function: the schema the model reads is derived, never written.

A hand-written schema beside the function it describes is a second declaration of the same
shape, and the second one is the one nobody updates. The drift is silent — the model keeps
calling with the arguments the old dictionary advertised — so everything here asserts that
the schema comes from the signature and the docstring, and that a signature no schema can
describe is refused where it is declared rather than on the call that first sends it.
"""

from __future__ import annotations

import gc
import socket  # noqa: TC003 — annotates a signature resolved at runtime
import threading
from collections.abc import Awaitable  # noqa: TC003 — annotates a body evaluated at runtime
from typing import TYPE_CHECKING, Literal, cast

import jsonschema
import pytest
from pydantic import BaseModel

from tesserix_adk.core import (
    JSON_SCHEMA,
    CancelledError,
    ToolDefinitionError,
    ToolExecutionError,
)
from tesserix_adk.runtime import Ambient, CancellationToken, WorkerPool, Workers, carrying
from tesserix_adk.tools import ToolContext, tool

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable, Iterator


class Leg(BaseModel):
    """One hop of a journey.

    Args:
        origin: Where the traveller boards.
        nights: How long they stay at the other end.
    """

    origin: str
    nights: int = 1


class Branch(BaseModel):
    """A stop that may lead to further stops.

    Args:
        city: Where this branch stops.
        onward: Where it goes next.
    """

    city: str
    onward: list[Branch] = []


def context() -> ToolContext:
    """The context a runtime would inject, standing in for a real run."""
    return ToolContext(run_id="run-1", tenant="acme", user="ada")


class TestSchemaFromSignature:
    """What the model is told, taken from the annotations and the docstring."""

    def test_the_primary_scenario(self) -> None:
        @tool
        async def plan(leg: Leg, note: str | None = None, ctx: ToolContext | None = None) -> str:
            """Plan one leg of a trip.

            Args:
                leg: The hop to plan.
                note: Anything the traveller asked for.
                ctx: Injected by the runtime.
            """
            del ctx
            return f"{leg.origin} {note}"

        assert plan.name == "plan"
        assert plan.description == "Plan one leg of a trip."
        assert plan.is_async
        schema = plan.parameters_schema
        assert set(schema["properties"]) == {"leg", "note"}
        assert schema["required"] == ["leg"]
        assert schema["properties"]["leg"]["properties"]["origin"] == {
            "type": "string",
            "description": "Where the traveller boards.",
        }
        assert schema["properties"]["note"]["description"] == "Anything the traveller asked for."
        assert "$defs" not in schema
        assert "$ref" not in str(schema)
        jsonschema.Draft202012Validator.check_schema(schema)

    def test_the_result_is_described_too(self) -> None:
        @tool
        def fare(leg: str) -> Literal["cheap", "dear"]:
            """Price a leg.

            Args:
                leg: The hop to price.
            """
            del leg
            return "cheap"

        assert fare.returns_schema == {"enum": ["cheap", "dear"], "type": "string"}
        assert not fare.is_async

    def test_a_tool_returning_nothing_says_so_rather_than_leaving_it_open(self) -> None:
        @tool
        def note(text: str) -> None:
            """Write something down.

            Args:
                text: What to write.
            """
            del text

        assert note.returns_schema == {"type": "null"}

    def test_the_name_and_description_can_be_overridden(self) -> None:
        @tool(name="lookup_fare", description="What a leg costs today.")
        def fare(leg: str) -> str:
            """Price a leg.

            Args:
                leg: The hop to price.
            """
            return leg

        assert fare.name == "lookup_fare"
        assert fare.description == "What a leg costs today."
        assert fare.parameters_schema["properties"]["leg"]["description"] == "The hop to price."

    def test_a_result_nobody_annotated_is_left_undescribed_rather_than_guessed(self) -> None:
        @tool
        def shrug(text: str):  # type: ignore[no-untyped-def]
            """Answer without saying what comes back.

            Args:
                text: Anything at all.
            """
            return text

        assert shrug.returns_schema is None

    def test_the_summary_is_not_repeated_inside_the_parameters(self) -> None:
        @tool
        def fare(leg: str) -> str:
            """Price a leg.

            Args:
                leg: The hop to price.
            """
            return leg

        assert "description" not in fare.parameters_schema


class TestRefusedWhereItIsDeclared:
    """A signature no model can be told about fails at decoration, not on the first call."""

    def test_an_unannotated_parameter_names_itself(self) -> None:
        with pytest.raises(ToolDefinitionError) as refused:

            @tool
            def book(leg) -> str:  # type: ignore[no-untyped-def]
                """Book a leg."""
                return str(leg)

        assert refused.value.parameter == "leg"
        assert refused.value.tool == "book"

    def test_a_variadic_parameter_is_refused(self) -> None:
        with pytest.raises(ToolDefinitionError) as refused:

            @tool
            def book(**extras: str) -> str:
                """Book a leg."""
                return str(extras)

        assert refused.value.parameter == "extras"

    def test_a_type_no_schema_can_describe_is_refused(self) -> None:
        with pytest.raises(ToolDefinitionError) as refused:

            @tool
            def send(sock: socket.socket) -> str:
                """Send something."""
                return str(sock)

        assert refused.value.parameter == "sock"

    def test_a_generator_is_refused_because_a_tool_returns_one_result(self) -> None:
        with pytest.raises(ToolDefinitionError, match="generator"):

            @tool
            def legs(count: int) -> Iterator[int]:
                """Yield legs."""
                yield from range(count)

    def test_an_async_generator_is_refused_too(self) -> None:
        with pytest.raises(ToolDefinitionError, match="generator"):

            @tool
            async def legs(count: int) -> AsyncIterator[int]:
                """Yield legs."""
                for index in range(count):
                    yield index

    def test_two_contexts_are_one_too_many_to_inject(self) -> None:
        with pytest.raises(ToolDefinitionError, match="one context to give"):

            @tool
            def book(here: ToolContext, there: ToolContext) -> str:
                """Book a leg."""
                return here.tenant + there.tenant

    def test_an_annotation_nobody_can_resolve_is_refused(self) -> None:
        namespace: dict[str, object] = {}
        exec(  # noqa: S102 — a forward reference to a type that does not exist needs one
            'def book(leg: \'Nowhere\') -> str:\n    """Book a leg."""\n    return leg\n',
            namespace,
        )
        with pytest.raises(ToolDefinitionError, match="cannot be resolved"):
            tool(cast("Callable[..., str]", namespace["book"]))

    def test_something_that_is_not_callable_is_not_a_tool(self) -> None:
        with pytest.raises(ToolDefinitionError, match="not callable"):
            tool(cast("Callable[[], str]", "book"))

    def test_a_callable_with_no_name_of_its_own_has_to_be_given_one(self) -> None:
        class Priced:
            """A callable object, which has no `__name__` to borrow."""

            def __call__(self, leg: str) -> str:
                return leg

        with pytest.raises(ToolDefinitionError, match="Pass name="):
            tool(Priced())

        named = tool(name="priced_object")(Priced())
        assert named.name == "priced_object"

    def test_a_model_that_refers_to_itself_cannot_be_inlined(self) -> None:
        with pytest.raises(ToolDefinitionError, match="refers to itself"):

            @tool
            def walk(branch: Branch) -> str:
                """Walk a branch."""
                return branch.city

    def test_the_same_model_is_fine_where_references_are_kept(self) -> None:
        @tool(name="walk_tree", dialect=JSON_SCHEMA)
        def walk(branch: Branch) -> str:
            """Walk a branch.

            Args:
                branch: Where to start.
            """
            return branch.city

        assert "$defs" in walk.parameters_schema


class TestOneNameOneTool:
    """Two tools answering to one name is the model reaching whichever won the race."""

    def test_a_second_live_tool_cannot_take_the_name(self) -> None:
        @tool(name="held")
        def first(leg: str) -> str:
            """Do a thing."""
            return leg

        with pytest.raises(ToolDefinitionError, match="already the name of a live tool"):

            @tool(name="held")
            def second(leg: str) -> str:
                """Do another thing."""
                return leg

        assert first.name == "held"

    def test_releasing_the_first_hands_the_name_over(self) -> None:
        @tool(name="handed-over")
        def first(leg: str) -> str:
            """Do a thing."""
            return leg

        first.release()
        first.release()

        @tool(name="handed-over")
        def second(leg: str) -> str:
            """Do another thing."""
            return leg

        assert second.name == "handed-over"

    def test_a_tool_nobody_holds_any_more_gives_its_name_back(self) -> None:
        @tool(name="collected")
        def first(leg: str) -> str:
            """Do a thing."""
            return leg

        del first
        gc.collect()

        @tool(name="collected")
        def second(leg: str) -> str:
            """Do another thing."""
            return leg

        assert second.name == "collected"

    def test_redecorating_one_function_keeps_its_own_name(self) -> None:
        def fare(leg: str) -> str:
            """Price a leg."""
            return leg

        first = tool(fare)
        second = tool(fare)
        assert first.name == second.name == "fare"


class TestDescriptionFallback:
    """A docstring in some other shape costs descriptions, never the tool."""

    def test_no_docstring_falls_back_to_the_name(self) -> None:
        @tool
        def undocumented(leg: str) -> str:
            return leg

        assert undocumented.description == "undocumented"
        assert "description" not in undocumented.parameters_schema["properties"]["leg"]

    def test_an_unparseable_docstring_still_yields_a_schema(self) -> None:
        @tool(name="scribbled")
        def odd(leg: str) -> str:
            """something something

            leg -- the hop, in a shape nothing parses
            """
            return leg

        assert odd.description == "something something"
        assert odd.parameters_schema["properties"]["leg"] == {"type": "string"}

    def test_an_override_beats_a_docstring_nobody_can_read(self) -> None:
        @tool(name="explicit", description="What a leg costs.")
        def odd(leg: str) -> str:
            return leg

        assert odd.description == "What a leg costs."


class TestOnePathForSyncAndAsync:
    """Whichever the body is, the caller awaits it and the event loop keeps turning."""

    async def test_an_async_tool_is_awaited(self) -> None:
        @tool(name="await_async")
        async def fare(leg: str) -> str:
            """Price a leg."""
            return f"{leg}: 40"

        assert await fare.invoke({"leg": "Osaka"}) == "Osaka: 40"
        assert await fare(leg="Osaka") == "Osaka: 40"

    async def test_a_sync_body_runs_off_the_event_loop(self) -> None:
        @tool(name="off_the_loop")
        def fare(leg: str) -> str:
            """Price a leg."""
            return f"{leg} on {threading.current_thread().name}"

        answered = await fare.invoke({"leg": "Osaka"})
        assert threading.main_thread().name not in answered

    async def test_a_worker_pool_bounds_where_sync_bodies_run(self) -> None:
        @tool(name="pooled")
        def fare(leg: str) -> str:
            """Price a leg."""
            return f"{leg} on {threading.current_thread().name}"

        with WorkerPool(Workers(size=1)) as pool:
            answered = await fare.invoke({"leg": "Osaka"}, workers=pool)
        assert "adk-worker" in answered

    async def test_a_body_that_hands_back_an_awaitable_is_awaited_too(self) -> None:
        async def priced(leg: str) -> str:
            return f"{leg}: 40"

        @tool(name="deferred")
        def fare(leg: str) -> Awaitable[str]:
            """Price a leg, eventually."""
            return priced(leg)

        assert await fare.invoke({"leg": "Osaka"}) == "Osaka: 40"
        assert fare.returns_schema == {"type": "string"}

    async def test_what_the_body_raised_is_what_the_caller_sees(self) -> None:
        @tool(name="raises")
        def fare(leg: str) -> str:
            """Price a leg."""
            raise LookupError(leg)

        with pytest.raises(LookupError, match="Osaka"):
            await fare.invoke({"leg": "Osaka"})


class TestContextIsInjectedNeverChosen:
    """The model picks arguments; it does not pick the tenant."""

    async def test_the_context_is_filled_in_and_never_described(self) -> None:
        @tool(name="context_injected")
        async def fare(leg: str, ctx: ToolContext) -> str:
            """Price a leg.

            Args:
                leg: The hop to price.
                ctx: Injected by the runtime.
            """
            return f"{leg} for {ctx.tenant}"

        assert fare.context_parameter == "ctx"
        assert "ctx" not in fare.parameters_schema["properties"]
        assert await fare.invoke({"leg": "Osaka"}, context()) == "Osaka for acme"

    async def test_an_argument_named_like_the_context_cannot_displace_it(self) -> None:
        @tool(name="context_guarded")
        async def fare(leg: str, ctx: ToolContext) -> str:
            """Price a leg."""
            return f"{leg} for {ctx.tenant}"

        answered = await fare.invoke({"leg": "Osaka", "ctx": "attacker"}, context())
        assert answered == "Osaka for acme"

    async def test_the_ambient_run_supplies_a_context_nobody_passed(self) -> None:
        @tool(name="context_ambient")
        async def fare(leg: str, ctx: ToolContext) -> str:
            """Price a leg."""
            return f"{leg} for {ctx.tenant}"

        with carrying(Ambient(run_id="run-2", tenant="beta")):
            assert await fare.invoke({"leg": "Osaka"}) == "Osaka for beta"

    async def test_a_context_nobody_can_supply_is_refused_rather_than_guessed(self) -> None:
        @tool(name="context_missing")
        async def fare(leg: str, ctx: ToolContext) -> str:
            """Price a leg."""
            return f"{leg} for {ctx.tenant}"

        with pytest.raises(ToolExecutionError, match="guessed tenant"):
            await fare.invoke({"leg": "Osaka"})

    async def test_an_optional_context_outside_a_run_is_simply_absent(self) -> None:
        @tool(name="context_optional")
        async def fare(leg: str, ctx: ToolContext | None = None) -> str:
            """Price a leg."""
            return f"{leg} for {ctx.tenant if ctx else 'nobody'}"

        assert await fare.invoke({"leg": "Osaka"}) == "Osaka for nobody"

    def test_a_context_carries_the_run_switch(self) -> None:
        token = CancellationToken()
        holding = ToolContext(run_id="run-1", tenant="acme", cancellation=token)
        holding.raise_if_cancelled()
        context().raise_if_cancelled()
        token.cancel("the caller left")
        with pytest.raises(CancelledError, match="the caller left"):
            holding.raise_if_cancelled()

    def test_there_is_no_context_outside_a_run(self) -> None:
        assert ToolContext.current() is None
        with carrying(Ambient(run_id="run-3", tenant="gamma", user="ada")):
            current = ToolContext.current()
        assert current is not None
        assert (current.run_id, current.tenant, current.user) == ("run-3", "gamma", "ada")
