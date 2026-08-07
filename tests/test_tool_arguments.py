"""Tool arguments are model output, and model output is checked before it runs anything.

The failure this file exists to prevent is a hallucinated field name, a string where an
integer was declared, or an identifier nobody bounded, reaching a function body and
failing deep inside it — or worse, not failing, and being interpolated into a query. The
schema the model was shown is the schema its call is held to, and the body is entered with
the tool's own types or not at all.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from pydantic import BaseModel, SecretStr

from tesserix_adk.core import (
    Agent,
    ModelCapabilities,
    RepairConfig,
    Run,
    RunEventKind,
    RunState,
    ToolArgumentValidationError,
    ToolCall,
    ToolDefinitionError,
    Usage,
)
from tesserix_adk.runtime import AgentRunner, ModelResponse
from tesserix_adk.testing import FakeClock, FakeToolRegistry, ScriptedProvider
from tesserix_adk.tools import (
    LENIENT,
    STRICT,
    ArgumentPolicy,
    ToolArgumentValidator,
    ToolContext,
    tool,
)

CAPABLE = ModelCapabilities(tool_calling=True, context_window_tokens=200_000)

entered: list[dict[str, Any]] = []


class Leg(BaseModel):
    """One hop.

    Args:
        origin: Where it starts.
        nights: How long the traveller stays.
    """

    origin: str
    nights: int


@tool
def price_leg(leg: Leg, currency: str = "EUR", refundable: bool = False) -> str:
    """Price one hop of a journey.

    Args:
        leg: The hop to price.
        currency: What to price it in.
        refundable: Whether the fare can be given back.
    """
    entered.append({"leg": leg, "currency": currency, "refundable": refundable})
    return f"{leg.origin}: {leg.nights * 40} {currency}"


@tool
def stow(secret: SecretStr, label: str) -> str:  # noqa: ARG001
    """Keep something.

    Args:
        secret: What to keep.
        label: What to call it.
    """
    return f"{label}: kept"


@pytest.fixture(autouse=True)
def _clear() -> None:
    entered.clear()


def rejected(payload: object, **overrides: object) -> ToolArgumentValidationError:
    validator = ToolArgumentValidator(price_leg.function, tool="price_leg", **overrides)  # type: ignore[arg-type]
    with pytest.raises(ToolArgumentValidationError) as refusal:
        validator.validate(payload)
    return refusal.value


GOOD = {"leg": {"origin": "Osaka", "nights": 2}}


class TestTheBodyIsNeverEnteredWithWhatTheModelSent:
    async def test_a_call_the_schema_does_not_allow_never_reaches_the_function(self) -> None:
        bad = {"lg": {"origin": "Osaka", "nights": "two"}, "class": "first"}
        with pytest.raises(ToolArgumentValidationError):
            await price_leg.invoke(bad)
        assert entered == []

    async def test_every_failing_field_is_named_not_only_the_first(self) -> None:
        bad = {"lg": {"origin": "Osaka", "nights": "two"}, "class": "first"}
        with pytest.raises(ToolArgumentValidationError) as refusal:
            await price_leg.invoke(bad)
        assert set(refusal.value.paths) == {"class", "leg", "lg"}

    async def test_the_error_names_the_tool_that_refused_the_call(self) -> None:
        with pytest.raises(ToolArgumentValidationError) as refusal:
            await price_leg.invoke({})
        assert refusal.value.tool == "price_leg"

    async def test_a_valid_call_reaches_the_body(self) -> None:
        assert await price_leg.invoke(GOOD) == "Osaka: 80 EUR"

    async def test_the_body_is_entered_with_the_declared_type_not_a_dict(self) -> None:
        await price_leg.invoke(GOOD)
        assert isinstance(entered[0]["leg"], Leg)

    async def test_an_argument_the_model_left_out_takes_the_function_s_own_default(self) -> None:
        await price_leg.invoke(GOOD)
        assert entered[0]["currency"] == "EUR"

    async def test_a_typed_instance_a_caller_passed_is_accepted_too(self) -> None:
        assert await price_leg.invoke({"leg": Leg(origin="Kyoto", nights=1)}) == "Kyoto: 40 EUR"


class TestUnknownFieldsAreRefusedRatherThanDropped:
    def test_a_field_the_tool_does_not_declare_is_an_error(self) -> None:
        assert "class" in rejected({**GOOD, "class": "first"}).paths

    def test_the_problem_says_the_field_is_not_declared(self) -> None:
        problems = rejected({**GOOD, "class": "first"}).problems
        assert "not permitted" in problems["class"].lower()

    def test_a_missing_required_field_is_not_invented(self) -> None:
        assert rejected({"currency": "JPY"}).paths == ("leg",)


class TestCoercionIsStrictByDefault:
    def test_a_string_is_not_read_as_an_integer(self) -> None:
        assert rejected({"leg": {"origin": "Osaka", "nights": "2"}}).paths == ("leg.nights",)

    def test_a_truthy_string_is_not_read_as_a_boolean(self) -> None:
        assert rejected({**GOOD, "refundable": "yes"}).paths == ("refundable",)

    def test_a_number_is_not_read_as_a_string(self) -> None:
        assert rejected({**GOOD, "currency": 3}).paths == ("currency",)

    def test_the_default_policy_is_the_strict_one(self) -> None:
        assert ArgumentPolicy() == STRICT

    def test_a_registry_that_chose_leniency_gets_the_documented_coercions(self) -> None:
        validator = ToolArgumentValidator(price_leg.function, tool="price_leg", policy=LENIENT)
        assert validator.validate({"leg": {"origin": "Osaka", "nights": "2"}}).leg.nights == 2

    def test_leniency_still_refuses_a_field_the_tool_does_not_declare(self) -> None:
        assert "class" in rejected({**GOOD, "class": "first"}, policy=LENIENT).paths


class TestTheRawPayloadIsNormalisedBeforeItIsValidated:
    def test_a_provider_that_sends_the_arguments_as_a_json_string_is_understood(self) -> None:
        validator = ToolArgumentValidator(price_leg.function, tool="price_leg")
        assert validator.validate(json.dumps(GOOD)).leg.origin == "Osaka"

    def test_a_redundant_envelope_is_unwrapped_rather_than_special_cased(self) -> None:
        validator = ToolArgumentValidator(price_leg.function, tool="price_leg")
        assert validator.validate({"arguments": json.dumps(GOOD)}).leg.origin == "Osaka"

    def test_json_that_does_not_parse_is_the_same_typed_error(self) -> None:
        assert "json" in str(rejected("{not json")).lower()

    def test_json_that_is_not_an_object_is_refused(self) -> None:
        assert "object" in str(rejected("[1, 2]")).lower()

    def test_a_duplicated_key_is_refused_rather_than_silently_resolved(self) -> None:
        payload = '{"leg": {"origin": "Osaka", "nights": 2}, "leg": {"origin": "Kobe"}}'
        assert "duplicate" in str(rejected(payload)).lower()

    def test_a_payload_over_the_ceiling_is_refused_before_it_is_parsed(self) -> None:
        huge = json.dumps({"leg": {"origin": "x" * 5_000, "nights": 2}})
        assert "ceiling" in str(rejected(huge, policy=ArgumentPolicy(max_bytes=1_000))).lower()

    def test_the_ceiling_holds_for_a_mapping_a_provider_already_parsed(self) -> None:
        huge = {"leg": {"origin": "x" * 5_000, "nights": 2}}
        assert "ceiling" in str(rejected(huge, policy=ArgumentPolicy(max_bytes=1_000))).lower()


class TestWhatIsSaidAboutARejectedCall:
    def test_the_raw_payload_is_kept_on_the_error_for_a_debugger(self) -> None:
        assert rejected({"nights": "two"}).payload == {"nights": "two"}

    def test_a_rejected_value_is_never_in_the_message(self) -> None:
        validator = ToolArgumentValidator(stow.function, tool="stow")
        with pytest.raises(ToolArgumentValidationError) as refusal:
            validator.validate({"secret": "hunter2", "label": 3})
        assert "hunter2" not in str(refusal.value)

    def test_a_rejected_value_is_never_in_what_goes_back_to_the_model(self) -> None:
        validator = ToolArgumentValidator(stow.function, tool="stow")
        with pytest.raises(ToolArgumentValidationError) as refusal:
            validator.validate({"secret": 1, "label": "passport"})
        assert "passport" not in refusal.value.feedback()

    def test_the_feedback_names_the_tool_and_every_field_that_failed(self) -> None:
        feedback = rejected({"leg": {"origin": "Osaka", "nights": "2"}}).feedback()
        assert "price_leg" in feedback
        assert "leg.nights" in feedback

    def test_the_feedback_says_the_tool_never_ran(self) -> None:
        assert "did not run" in rejected({}).feedback()

    def test_a_payload_that_is_not_json_at_all_is_the_same_refusal(self) -> None:
        assert "not json" in str(rejected({"leg": object()})).lower()


class TestTheValidatorStandsAloneToo:
    def test_it_reports_the_name_the_refusal_is_made_under(self) -> None:
        assert ToolArgumentValidator(price_leg.function, tool="price_leg").tool == "price_leg"

    def test_it_takes_the_function_s_name_where_none_was_given(self) -> None:
        assert ToolArgumentValidator(price_leg.function).tool == "price_leg"

    def test_it_exposes_the_argument_model_it_built(self) -> None:
        validator = ToolArgumentValidator(price_leg.function, tool="price_leg")
        assert sorted(validator.model.model_fields) == ["currency", "leg", "refundable"]

    def test_it_exposes_the_policy_it_reads_by(self) -> None:
        assert ToolArgumentValidator(price_leg.function, policy=LENIENT).policy is LENIENT

    def test_a_parameter_with_no_annotation_is_refused_where_it_is_declared(self) -> None:
        def untyped(code) -> str:  # type: ignore[no-untyped-def]
            return code  # type: ignore[no-any-return]

        with pytest.raises(ToolDefinitionError, match="no type"):
            ToolArgumentValidator(untyped, tool="untyped")

    def test_a_variadic_parameter_is_not_a_field_the_model_may_fill(self) -> None:
        def spread(page: int, *rest: int, **extra: str) -> int:
            return page + sum(rest) + len(extra)

        assert list(ToolArgumentValidator(spread, tool="spread").model.model_fields) == ["page"]


class TestTheInjectedContextIsNotTheModelSToSend:
    async def test_a_context_parameter_is_not_a_field_the_model_may_fill(self) -> None:
        @tool
        def file_it(reference: str, ctx: ToolContext | None = None) -> str:
            """File something.

            Args:
                reference: What to file.
                ctx: The run.
            """
            return f"filed {reference} for {ctx.tenant if ctx else 'nobody'}"

        with pytest.raises(ToolArgumentValidationError) as refusal:
            await file_it.invoke({"reference": "itinerary", "ctx": "acme"})
        assert refusal.value.paths == ("ctx",)
        file_it.release()


def agent(**overrides: object) -> Agent[Any]:
    fields: dict[str, object] = {
        "name": "planner",
        "instructions": "Plan trips.",
        "free_text": True,
        "model": "scripted-1",
        "tools": ("lookup",),
    }
    return Agent(**{**fields, **overrides})  # type: ignore[arg-type]


def calling(**arguments: object) -> ModelResponse:
    return ModelResponse(
        content="",
        tool_calls=(ToolCall(id="call_1", name="lookup", arguments=arguments),),
        usage=Usage(input_tokens=10, output_tokens=5),
    )


def answer(text: str = "Kyoto, four nights.") -> ModelResponse:
    return ModelResponse(content=text, usage=Usage(input_tokens=10, output_tokens=5))


def looking_up(**arguments: object) -> str:
    if not isinstance(arguments.get("page"), int):
        raise ToolArgumentValidationError(
            "lookup refused the arguments it was sent",
            tool="lookup",
            call_id="call_1",
            paths=("page",),
            problems={"page": "Input should be a valid integer"},
            payload=arguments,
        )
    return "page one"


def runner(*responses: ModelResponse, **overrides: object) -> AgentRunner:
    fields: dict[str, object] = {
        "provider": ScriptedProvider(*responses, capabilities=CAPABLE),
        "clock": FakeClock(),
        "tools": FakeToolRegistry({"lookup": looking_up}),
    }
    return AgentRunner(**{**fields, **overrides})  # type: ignore[arg-type]


async def start(runner_: AgentRunner, agent_: Agent[Any]) -> Run[Any]:
    return await runner_.run(agent_, "look it up", tenant="acme", run_id="run_1")


def details(run: Run[Any], kind: RunEventKind) -> list[str]:
    return [record.detail or "" for record in run.events if record.kind is kind]


def sent(provider: ScriptedProvider, index: int) -> str:
    return "\n".join(
        part.text
        for message in provider.requests[index].messages
        for part in message.content
        if hasattr(part, "text")
    )


class TestARejectedCallIsRepairedOnceAndThenFailsClosed:
    async def test_the_model_is_given_the_field_that_failed(self) -> None:
        provider = ScriptedProvider(
            calling(page="one"), calling(page=1), answer(), capabilities=CAPABLE
        )
        await start(runner(provider=provider), agent(repair=RepairConfig()))
        assert "page" in sent(provider, 1)

    async def test_a_corrected_call_completes_the_run(self) -> None:
        run = await start(
            runner(calling(page="one"), calling(page=1), answer()), agent(repair=RepairConfig())
        )
        assert run.state is RunState.COMPLETED

    async def test_the_repair_is_counted_against_the_run_s_own_cap(self) -> None:
        run = await start(
            runner(calling(page="one"), calling(page=1), answer()), agent(repair=RepairConfig())
        )
        assert "1 of 2" in details(run, RunEventKind.REPAIR_REQUESTED)[0]

    async def test_a_model_that_repeats_the_rejected_call_fails_the_run(self) -> None:
        run = await start(
            runner(calling(page="one"), calling(page="one"), calling(page="one"), answer()),
            agent(repair=RepairConfig(max_attempts=1)),
        )
        assert run.state is RunState.FAILED

    async def test_the_failure_names_the_tool_and_the_field(self) -> None:
        run = await start(
            runner(calling(page="one"), calling(page="one"), answer()),
            agent(repair=RepairConfig(max_attempts=1)),
        )
        assert "lookup" in details(run, RunEventKind.TERMINATED)[0]
        assert "page" in details(run, RunEventKind.TERMINATED)[0]

    async def test_an_agent_that_declared_no_repair_fails_on_the_first_rejection(self) -> None:
        run = await start(runner(calling(page="one"), answer()), agent())
        assert run.state is RunState.FAILED

    async def test_a_rejected_call_is_not_retried_against_the_tool(self) -> None:
        registry = FakeToolRegistry({"lookup": looking_up})
        await start(
            runner(calling(page="one"), answer(), tools=registry),
            agent(idempotent_tools=("lookup",)),
        )
        assert len(registry.calls) == 1
