"""What a vendor sent back becomes a tool call, or it becomes a typed refusal.

The adapter is the last place a malformed call can be stopped for free. Past it, the
arguments reach a tool body, and a tool body that receives a field it did not ask for has
already been called.
"""

from __future__ import annotations

import pytest

from tesserix_adk.core import (
    Capability,
    CapabilityError,
    Message,
    ModelCapabilities,
    ModelRequest,
    ModelResponseError,
    TextPart,
    ToolCall,
    ToolDeclaration,
)
from tesserix_adk.core.errors import ToolArgumentValidationError
from tesserix_adk.models.providers import normalised_tool_calls

LOOKUP = ToolDeclaration(
    name="lookup",
    description="Look a city's forecast up.",
    parameters={
        "type": "object",
        "properties": {
            "city": {"type": "string"},
            "days": {"type": "integer"},
            "units": {"type": "string", "enum": ["c", "f"]},
        },
        "required": ["city"],
        "additionalProperties": False,
    },
)

PARALLEL = ModelCapabilities(tool_calling=True, parallel_tool_calls=True)
SERIAL = ModelCapabilities(tool_calling=True)


def request(*tools: ToolDeclaration) -> ModelRequest:
    return ModelRequest(
        model="m",
        messages=(Message(role="user", content=[TextPart(text="hi")]),),
        tools=tools,
    )


def call(**arguments: object) -> ToolCall:
    return ToolCall(id="call_1", name="lookup", arguments=dict(arguments))


class TestAWellFormedCallPassesThrough:
    def test_arguments_matching_the_schema_are_returned_unchanged(self) -> None:
        calls = normalised_tool_calls(
            (call(city="Pune", days=3, units="c"),),
            request=request(LOOKUP),
            capabilities=SERIAL,
            provider="anthropic",
        )
        assert calls[0].arguments == {"city": "Pune", "days": 3, "units": "c"}

    def test_an_optional_field_may_be_absent(self) -> None:
        assert normalised_tool_calls(
            (call(city="Pune"),), request=request(LOOKUP), capabilities=SERIAL, provider="openai"
        )[0].arguments == {"city": "Pune"}

    def test_a_tool_the_request_never_declared_is_passed_on_unvalidated(self) -> None:
        """There is no schema to check it against, and inventing one refuses valid calls."""
        assert (
            normalised_tool_calls(
                (call(city="Pune"),), request=request(), capabilities=SERIAL, provider="gemini"
            )[0].name
            == "lookup"
        )


class TestArgumentsAreCheckedBeforeTheToolRuns:
    def test_a_missing_required_field_is_refused_rather_than_filled(self) -> None:
        with pytest.raises(ToolArgumentValidationError) as refused:
            normalised_tool_calls(
                (call(days=3),), request=request(LOOKUP), capabilities=SERIAL, provider="anthropic"
            )
        assert refused.value.tool == "lookup"
        assert refused.value.call_id == "call_1"
        assert refused.value.paths == ("city",)
        assert refused.value.payload == {"days": 3}

    def test_a_field_of_the_wrong_type_is_refused_rather_than_coerced(self) -> None:
        """`"3"` read as `3` and `"false"` read as `True` are the same bug."""
        with pytest.raises(ToolArgumentValidationError) as refused:
            normalised_tool_calls(
                (call(city="Pune", days="3"),),
                request=request(LOOKUP),
                capabilities=SERIAL,
                provider="openai",
            )
        assert refused.value.paths == ("days",)

    def test_a_field_nobody_declared_is_refused(self) -> None:
        with pytest.raises(ToolArgumentValidationError) as refused:
            normalised_tool_calls(
                (call(city="Pune", colour="blue"),),
                request=request(LOOKUP),
                capabilities=SERIAL,
                provider="gemini",
            )
        assert refused.value.paths == ("colour",)

    def test_a_value_outside_the_declared_choices_is_refused(self) -> None:
        with pytest.raises(ToolArgumentValidationError) as refused:
            normalised_tool_calls(
                (call(city="Pune", units="kelvin"),),
                request=request(LOOKUP),
                capabilities=SERIAL,
                provider="anthropic",
            )
        assert refused.value.paths == ("units",)

    def test_every_failing_field_is_reported_at_once(self) -> None:
        """One field per round trip is how a three-field call takes three attempts."""
        with pytest.raises(ToolArgumentValidationError) as refused:
            normalised_tool_calls(
                (ToolCall(id="c", name="lookup", arguments={"days": "3", "colour": "blue"}),),
                request=request(LOOKUP),
                capabilities=SERIAL,
                provider="openai",
            )
        assert refused.value.paths == ("city", "colour", "days")

    def test_the_error_is_a_schema_violation_so_existing_handlers_still_catch_it(self) -> None:
        from tesserix_adk.core import SchemaViolationError

        assert issubclass(ToolArgumentValidationError, SchemaViolationError)


class TestAVendorMayNotExceedWhatTheModelDeclared:
    def test_parallel_calls_from_a_model_that_does_not_do_them_are_refused(self) -> None:
        with pytest.raises(CapabilityError) as refused:
            normalised_tool_calls(
                (
                    ToolCall(id="a", name="lookup", arguments={"city": "Pune"}),
                    ToolCall(id="b", name="lookup", arguments={"city": "Delhi"}),
                ),
                request=request(LOOKUP),
                capabilities=SERIAL,
                provider="anthropic",
            )
        assert refused.value.capability == Capability.PARALLEL_TOOL_CALLS.value

    def test_parallel_calls_are_kept_where_the_model_declares_them(self) -> None:
        calls = normalised_tool_calls(
            (
                ToolCall(id="a", name="lookup", arguments={"city": "Pune"}),
                ToolCall(id="b", name="lookup", arguments={"city": "Delhi"}),
            ),
            request=request(LOOKUP),
            capabilities=PARALLEL,
            provider="anthropic",
        )
        assert [one.id for one in calls] == ["a", "b"]

    def test_one_id_used_twice_in_a_turn_is_refused(self) -> None:
        """Results are matched back by id, so a repeated id answers the wrong call."""
        with pytest.raises(ModelResponseError) as refused:
            normalised_tool_calls(
                (
                    ToolCall(id="a", name="lookup", arguments={"city": "Pune"}),
                    ToolCall(id="a", name="lookup", arguments={"city": "Delhi"}),
                ),
                request=request(LOOKUP),
                capabilities=PARALLEL,
                provider="openai",
            )
        assert refused.value.provider == "openai"

    def test_no_calls_at_all_is_not_a_failure(self) -> None:
        assert (
            normalised_tool_calls(
                (), request=request(LOOKUP), capabilities=SERIAL, provider="gemini"
            )
            == ()
        )


ITINERARY = ToolDeclaration(
    name="itinerary",
    description="Plan several stops.",
    parameters={
        "type": "object",
        "properties": {
            "stops": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
            "notes": {"type": "array"},
            "cancelled": {"type": "null"},
            "since": {"type": "date-time"},
        },
        "required": ["stops"],
    },
)


def planned(**arguments: object) -> ToolCall:
    return ToolCall(id="call_1", name="itinerary", arguments=dict(arguments))


def _checked(call: ToolCall) -> None:
    normalised_tool_calls(
        (call,), request=request(ITINERARY), capabilities=SERIAL, provider="gemini"
    )


class TestNestedArgumentsAreCheckedToo:
    def test_a_bad_field_inside_a_list_is_named_by_its_position(self) -> None:
        """ "a field is missing" is not a fix; "stops[1].city is missing" is."""
        with pytest.raises(ToolArgumentValidationError) as refused:
            _checked(planned(stops=[{"city": "Pune"}, {"town": "Delhi"}]))
        assert refused.value.paths == ("stops[1].city",)

    def test_a_list_where_an_object_belongs_is_refused(self) -> None:
        with pytest.raises(ToolArgumentValidationError) as refused:
            _checked(planned(stops=[["Pune"]]))
        assert refused.value.problems == {"stops[0]": "expected object, got list"}

    def test_a_string_where_a_list_belongs_is_refused(self) -> None:
        with pytest.raises(ToolArgumentValidationError) as refused:
            _checked(planned(stops="Pune"))
        assert refused.value.problems == {"stops": "expected array, got str"}

    def test_a_list_with_no_declared_item_shape_takes_anything(self) -> None:
        _checked(planned(stops=[], notes=["by rail", 2]))

    def test_a_field_declared_null_may_not_hold_a_value(self) -> None:
        with pytest.raises(ToolArgumentValidationError) as refused:
            _checked(planned(stops=[], cancelled="yes"))
        assert refused.value.problems == {"cancelled": "expected null, got str"}

    def test_a_type_the_checker_does_not_know_is_left_alone(self) -> None:
        """Refusing what this cannot read would refuse calls the vendor got right."""
        _checked(planned(stops=[], since="2026-08-07T00:00:00Z"))
