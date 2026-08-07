"""One type, one schema: what the model is told matches what the code will parse.

A hand-written JSON Schema drifts from the type it claims to describe the first time a
field is renamed, and the drift surfaces in production as a payload the code refuses.
Everything here derives the schema from the annotation, so the two cannot disagree, and
the awkward constructs providers care about — unions, enums, bounds, recursion — are
exercised against a real validator rather than asserted by eye.
"""

from __future__ import annotations

import dataclasses
import json
import socket  # noqa: TC003 — annotates a signature resolved at runtime
from enum import StrEnum
from typing import Annotated, Any, Literal, TypedDict

import jsonschema
import pytest
from pydantic import BaseModel, ConfigDict, Field, ValidationError, create_model

from tesserix_adk.core import (
    INLINE_REFS,
    JSON_SCHEMA,
    STRICT_SUBSET,
    AdkModel,
    CapabilityError,
    SchemaGenerationError,
    schema_for,
    schema_hash,
)
from tesserix_adk.core.schema import InlineRefs, SchemaDialect, annotations_of


class Priority(StrEnum):
    """How urgently the trip has to happen."""

    low = "low"
    high = "high"


class Waypoint(AdkModel):
    """A stop on the way.

    Args:
        city: Where the traveller stops.
        nights: How long they stay, in nights.
    """

    city: str
    nights: int = 1


class Flight(AdkModel):
    """A leg flown."""

    kind: Literal["flight"]
    carrier: str


class Train(AdkModel):
    """A leg by rail."""

    kind: Literal["train"]
    operator: str


Leg = Annotated[Flight | Train, Field(discriminator="kind")]


class Itinerary(AdkModel):
    """A trip proposed for a traveller.

    Args:
        traveller: Who is going.
        priority: How urgent the trip is.
        stops: Every stop, in order.
        leg: How the traveller gets there.
        budget: Ceiling in whole currency units, where there is one.
    """

    traveller: str
    priority: Priority
    stops: list[Waypoint]
    leg: Leg
    budget: Annotated[int, Field(ge=0, le=10_000)] | None = None


class Node(AdkModel):
    """A tree that contains itself.

    Args:
        name: What this node is called.
        child: The node below it, where there is one.
    """

    name: str
    child: Node | None = None


class Booking(TypedDict):
    """A booking as a plain mapping.

    Args:
        reference: The airline's own record locator.
        seats: How many seats were held.
    """

    reference: str
    seats: int


@dataclasses.dataclass(frozen=True)
class Passenger:
    """Someone on the booking.

    Args:
        name: Their name as printed on the ticket.
        loyalty: Their frequent-flyer number, where they gave one.
    """

    name: str
    loyalty: str | None = None


def search_flights(origin: str, destination: str, nights: int = 1) -> str:
    """Find a flight between two cities.

    Args:
        origin: The IATA code flown from.
        destination: The IATA code flown to.
        nights: How long the traveller stays.
    """
    return f"{origin}-{destination}-{nights}"


class Open(BaseModel):
    """A plain model, which ignores fields it was not told about.

    Args:
        stops: Every stop, in order.
    """

    stops: list[Waypoint] = []


class Pair(AdkModel):
    """Two required fields, for dialects that rewrite one of them.

    Args:
        always: The rewritten one.
        never: The one left alone.
    """

    always: str
    never: str


@dataclasses.dataclass(frozen=True)
class _Rewriting:
    """A dialect that replaces one property with a boolean schema."""

    always: bool = True
    name: str = "rewriting"
    forbidden: frozenset[str] = frozenset()

    def adapt(self, schema: dict[str, Any]) -> dict[str, Any]:
        """Swap `always` for a bare `true` or `false`."""
        return schema | {"properties": schema["properties"] | {"always": self.always}}


VALID = {
    "traveller": "ada",
    "priority": "high",
    "stops": [{"city": "kyoto", "nights": 2}],
    "leg": {"kind": "flight", "carrier": "JL"},
    "budget": 500,
}


class TestWhatItGenerates:
    def test_a_model_becomes_an_object_of_its_fields(self) -> None:
        schema = schema_for(Waypoint)

        assert schema["type"] == "object"
        assert sorted(schema["properties"]) == ["city", "nights"]

    def test_a_field_without_a_default_is_required_and_one_with_a_default_is_not(self) -> None:
        schema = schema_for(Waypoint)

        assert schema["required"] == ["city"]

    def test_an_enum_lists_the_values_the_model_may_answer_with(self) -> None:
        schema = schema_for(Itinerary)

        assert sorted(_resolve(schema, schema["properties"]["priority"])["enum"]) == [
            "high",
            "low",
        ]

    def test_a_constrained_number_carries_its_bounds(self) -> None:
        """A bound the model is never told is a bound only the code enforces, too late."""
        budget = schema_for(Itinerary)["properties"]["budget"]
        bounded = next(member for member in budget["anyOf"] if member.get("type") == "integer")

        assert (bounded["minimum"], bounded["maximum"]) == (0, 10_000)

    def test_a_discriminated_union_keeps_its_discriminator(self) -> None:
        leg = schema_for(Itinerary)["properties"]["leg"]

        assert leg["discriminator"]["propertyName"] == "kind"

    def test_a_nested_model_is_defined_once_and_referenced(self) -> None:
        schema = schema_for(Itinerary)

        assert "Waypoint" in schema["$defs"]
        assert schema["properties"]["stops"]["items"]["$ref"] == "#/$defs/Waypoint"

    def test_a_typed_dict_generates_like_a_model(self) -> None:
        schema = schema_for(Booking)

        assert schema["type"] == "object"
        assert schema["required"] == ["reference", "seats"]

    def test_a_dataclass_generates_like_a_model(self) -> None:
        schema = schema_for(Passenger)

        assert schema["required"] == ["name"]

    def test_a_callable_becomes_a_schema_of_its_parameters(self) -> None:
        schema = schema_for(search_flights)

        assert sorted(schema["properties"]) == ["destination", "nights", "origin"]
        assert schema["required"] == ["destination", "origin"]

    def test_no_property_carries_a_title(self) -> None:
        """Pydantic titles every field; a provider reads none of them and a diff reads all."""
        assert "title" not in repr(schema_for(Itinerary))


class TestDescriptionsComeFromTheDocstring:
    def test_a_field_documented_in_the_class_docstring_is_described(self) -> None:
        schema = schema_for(Waypoint)

        assert schema["properties"]["city"]["description"] == "Where the traveller stops."

    def test_a_nested_model_is_documented_too(self) -> None:
        schema = schema_for(Itinerary)

        assert "nights" in schema["$defs"]["Waypoint"]["properties"]["nights"]["description"]

    def test_an_explicit_field_description_wins_over_the_docstring(self) -> None:
        class Explicit(AdkModel):
            """A model documented twice.

            Args:
                name: From the docstring.
            """

            name: str = Field(description="From the field.")

        assert schema_for(Explicit)["properties"]["name"]["description"] == "From the field."

    def test_a_parameter_documented_in_a_function_docstring_is_described(self) -> None:
        schema = schema_for(search_flights)

        assert schema["properties"]["origin"]["description"] == "The IATA code flown from."

    def test_a_callable_carries_its_summary_line(self) -> None:
        assert schema_for(search_flights)["description"] == "Find a flight between two cities."

    def test_a_description_spanning_two_lines_is_joined(self) -> None:
        class Wrapped(AdkModel):
            """A model whose docs wrap.

            Args:
                name: The first line
                    and the second.
            """

            name: str

        assert schema_for(Wrapped)["properties"]["name"]["description"] == (
            "The first line and the second."
        )

    def test_a_model_without_a_docstring_still_generates(self) -> None:
        class Bare(AdkModel):
            name: str

        assert "description" not in schema_for(Bare)["properties"]["name"]

    def test_a_docstring_with_no_args_section_omits_descriptions_rather_than_raising(self) -> None:
        class Prose(AdkModel):
            """Only prose here, and not a colon in sight."""

            name: str

        assert "description" not in schema_for(Prose)["properties"]["name"]

    def test_a_malformed_args_section_describes_what_it_can(self) -> None:
        """Docs are guidance; a typo in them is not a reason to refuse to run."""

        class Malformed(AdkModel):
            """A model documented badly.

            Args:
                this is not a field at all
                name: But this one is.
            """

            name: str

        assert schema_for(Malformed)["properties"]["name"]["description"] == "But this one is."

    def test_a_blank_line_inside_the_args_block_does_not_end_it(self) -> None:
        class Spaced(AdkModel):
            """A model documented with room to breathe.

            Args:
                first: The first one.

                second: The second one.
            """

            first: str
            second: str

        assert schema_for(Spaced)["properties"]["second"]["description"] == "The second one."

    def test_the_section_after_args_is_not_read_as_a_field(self) -> None:
        class Sectioned(AdkModel):
            """A model documented in full.

            Args:
                name: What it is called.

            Returns:
                Nothing at all.
            """

            name: str

        assert sorted(schema_for(Sectioned)["properties"]) == ["name"]

    def test_a_nested_dataclass_is_documented_from_its_own_docstring(self) -> None:
        class Trip(AdkModel):
            """A trip with someone on it.

            Args:
                passenger: Who is travelling.
            """

            passenger: Passenger

        passengers = schema_for(Trip)["$defs"]["Passenger"]["properties"]

        assert passengers["name"]["description"] == "Their name as printed on the ticket."

    def test_two_types_sharing_a_name_generate_without_borrowing_each_other_s_docs(self) -> None:
        """Pydantic renames a `$def` when names collide; a description is worth less than a lie."""

        both = create_model(
            "Both", first=(_first_leg(), ...), second=(_second_leg(), ...), __base__=AdkModel
        )

        schema = schema_for(both)

        assert len(schema["$defs"]) == 2
        assert all(len(definition["properties"]) == 1 for definition in schema["$defs"].values())

    def test_a_documented_name_that_is_not_a_field_is_ignored(self) -> None:
        class Stale(AdkModel):
            """A model documenting a field that was removed.

            Args:
                name: Still here.
                removed: Long gone.
            """

            name: str

        assert sorted(schema_for(Stale)["properties"]) == ["name"]


class TestTheSchemaAndTheTypeAgree:
    @pytest.mark.parametrize("dialect", [JSON_SCHEMA, STRICT_SUBSET], ids=lambda d: d.name)
    @pytest.mark.parametrize(
        ("payload", "accepted"),
        [
            (VALID, True),
            (VALID | {"budget": None}, True),
            (VALID | {"stops": []}, True),
            (VALID | {"leg": {"kind": "train", "operator": "JR"}}, True),
            (VALID | {"priority": "urgent"}, False),
            (VALID | {"budget": 50_000}, False),
            (VALID | {"budget": -1}, False),
            (VALID | {"leg": {"kind": "boat", "line": "P&O"}}, False),
            (VALID | {"stops": [{"nights": 2}]}, False),
            (VALID | {"traveller": 7}, False),
            ({key: value for key, value in VALID.items() if key != "traveller"}, False),
        ],
        ids=[
            "whole",
            "no-budget",
            "no-stops",
            "by-train",
            "bad-enum",
            "over-max",
            "under-min",
            "unknown-union-member",
            "missing-nested",
            "wrong-type",
            "cut",
        ],
    )
    def test_the_schema_accepts_exactly_what_the_type_accepts(
        self, payload: dict[str, Any], accepted: bool, dialect: SchemaDialect
    ) -> None:
        """A provider answers in JSON, so the two are compared on the same JSON."""
        assert _the_type_accepts(payload) is accepted
        assert _the_schema_accepts(payload, schema_for(Itinerary, dialect=dialect)) is accepted

    def test_the_emitted_schema_is_itself_a_valid_json_schema(self) -> None:
        schema = schema_for(Itinerary)

        jsonschema.Draft202012Validator.check_schema(schema)

    def test_an_inlined_schema_accepts_exactly_what_the_referenced_one_did(self) -> None:
        inlined = schema_for(Itinerary, dialect=INLINE_REFS)

        assert "$defs" not in inlined
        jsonschema.validate(VALID, inlined)


class TestTheHashIsStable:
    def test_generating_twice_gives_the_same_hash(self) -> None:
        assert schema_hash(schema_for(Itinerary)) == schema_hash(schema_for(Itinerary))

    def test_key_order_does_not_change_the_hash(self) -> None:
        schema = schema_for(Waypoint)
        shuffled = dict(reversed(list(schema.items())))

        assert schema_hash(shuffled) == schema_hash(schema)

    def test_renaming_a_field_changes_the_hash(self) -> None:
        """A cassette recorded against the old shape must miss, loudly, not replay."""

        class Before(AdkModel):
            city: str

        class After(AdkModel):
            town: str

        assert schema_hash(schema_for(Before)) != schema_hash(schema_for(After))

    def test_two_dialects_of_one_type_hash_differently(self) -> None:
        """The dialect is part of what the model was told, so it is part of the fingerprint."""
        assert schema_hash(schema_for(Open)) != schema_hash(schema_for(Open, dialect=STRICT_SUBSET))

    def test_the_hash_names_the_algorithm_that_produced_it(self) -> None:
        assert schema_hash(schema_for(Waypoint)).startswith("sha256:")


class TestProviderDialects:
    def test_the_strict_subset_closes_every_object(self) -> None:
        """A model that merely ignores unknown fields still emits a closed schema here."""
        schema = schema_for(Open, dialect=STRICT_SUBSET)

        assert schema["additionalProperties"] is False
        assert schema["$defs"]["Waypoint"]["additionalProperties"] is False

    def test_the_strict_subset_rejects_a_field_the_model_invented(self) -> None:
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(
                {"stops": [], "upgrade": True}, schema_for(Open, dialect=STRICT_SUBSET)
            )

    def test_a_dialect_cannot_loosen_a_required_field_past_the_guard(self) -> None:
        """`true` is a schema that accepts anything; a dialect may not smuggle one in."""
        with pytest.raises(SchemaGenerationError, match="always"):
            schema_for(Pair, dialect=_Rewriting(always=True))

    def test_a_dialect_may_narrow_a_field_to_nothing(self) -> None:
        """`false` accepts nothing, which is refusal rather than permission."""
        assert schema_for(Pair, dialect=_Rewriting(always=False))["properties"]["always"] is False

    def test_an_inlining_dialect_leaves_no_reference_behind(self) -> None:
        inlined = schema_for(Itinerary, dialect=INLINE_REFS)

        assert "$ref" not in repr(inlined)

    def test_a_recursive_type_under_an_inlining_dialect_is_refused_at_definition_time(self) -> None:
        """Inlining a cycle has no finite answer; a truncated one is a schema that lies."""
        with pytest.raises(CapabilityError, match="Node"):
            schema_for(Node, dialect=INLINE_REFS)

    def test_a_recursive_type_is_fine_where_references_are_allowed(self) -> None:
        assert schema_for(Node)["$defs"]["Node"]["properties"]["name"]["type"] == "string"

    def test_inlining_stops_at_the_depth_the_dialect_declares(self) -> None:
        with pytest.raises(CapabilityError, match="depth"):
            schema_for(Itinerary, dialect=InlineRefs(max_depth=0))

    def test_a_dialect_that_forbids_a_construct_says_so_rather_than_rewriting_it(self) -> None:
        """Dropping `anyOf` would ship a schema the provider takes and the code cannot."""

        @dataclasses.dataclass(frozen=True)
        class NoUnions:
            name: str = "no-unions"
            forbidden: frozenset[str] = frozenset({"anyOf"})

            def adapt(self, schema: dict[str, Any]) -> dict[str, Any]:
                return schema

        with pytest.raises(CapabilityError, match="anyOf"):
            schema_for(Itinerary, dialect=NoUnions())

    def test_a_dialect_is_named_so_a_failure_says_which_provider(self) -> None:
        with pytest.raises(CapabilityError, match=INLINE_REFS.name):
            schema_for(Node, dialect=INLINE_REFS)


class TestUnrepresentableTypesFailAtDefinitionTime:
    def test_an_unannotated_parameter_names_itself(self) -> None:
        def untyped(origin, destination: str) -> str:  # type: ignore[no-untyped-def]
            return f"{origin}{destination}"

        with pytest.raises(SchemaGenerationError, match="origin"):
            schema_for(untyped)

    def test_any_in_a_required_position_is_refused(self) -> None:
        """`{}` accepts everything, so the model may answer with anything at all."""

        class Loose(AdkModel):
            payload: Any

        with pytest.raises(SchemaGenerationError, match="payload"):
            schema_for(Loose)

    def test_a_documented_any_is_refused_too(self) -> None:
        """A description is not a constraint: the schema still accepts anything at all."""

        class Documented(AdkModel):
            """A model that describes what it does not constrain.

            Args:
                payload: Whatever the provider felt like sending.
            """

            payload: Any

        with pytest.raises(SchemaGenerationError, match="payload"):
            schema_for(Documented)

    def test_a_nested_any_is_refused_too(self) -> None:
        class Inner(AdkModel):
            payload: Any

        class Outer(AdkModel):
            inner: Inner

        with pytest.raises(SchemaGenerationError, match="payload"):
            schema_for(Outer)

    def test_a_type_that_cannot_be_described_names_the_field_and_the_type(self) -> None:
        class Opaque:
            pass

        class Holder(AdkModel):
            model_config = ConfigDict(arbitrary_types_allowed=True)

            thing: Opaque

        with pytest.raises(SchemaGenerationError, match="thing") as raised:
            schema_for(Holder)

        assert "Opaque" in str(raised.value)

    def test_variadic_parameters_are_refused_rather_than_ignored(self) -> None:
        def variadic(origin: str, **rest: str) -> str:
            return origin + "".join(rest)

        with pytest.raises(SchemaGenerationError, match="rest"):
            schema_for(variadic)

    def test_the_error_carries_the_offending_field_and_annotation(self) -> None:
        class Loose(AdkModel):
            payload: Any

        with pytest.raises(SchemaGenerationError) as raised:
            schema_for(Loose)

        assert raised.value.field == "payload"

    def test_a_generation_failure_is_not_worth_retrying(self) -> None:
        class Loose(AdkModel):
            payload: Any

        with pytest.raises(SchemaGenerationError) as raised:
            schema_for(Loose)

        assert raised.value.retryable is False

    def test_a_schema_past_the_size_limit_fails_rather_than_being_truncated(self) -> None:
        with pytest.raises(SchemaGenerationError, match="bytes"):
            schema_for(Itinerary, max_bytes=200)

    def test_a_schema_inside_the_size_limit_is_returned_whole(self) -> None:
        assert schema_for(Waypoint, max_bytes=10_000)["type"] == "object"

    def test_something_that_is_neither_a_type_nor_a_callable_is_refused(self) -> None:
        with pytest.raises(SchemaGenerationError, match="int"):
            schema_for(3)  # type: ignore[arg-type]


def _first_leg() -> type[AdkModel]:
    class Leg(AdkModel):
        """A leg of one kind.

        Args:
            carrier: Who flies it.
        """

        carrier: str

    return Leg


def _second_leg() -> type[AdkModel]:
    class Leg(AdkModel):
        """A leg of another kind.

        Args:
            operator: Who runs it.
        """

        operator: str

    return Leg


def _the_type_accepts(payload: dict[str, Any]) -> bool:
    try:
        Itinerary.model_validate_json(json.dumps(payload))
    except ValidationError:
        return False
    return True


def _the_schema_accepts(payload: dict[str, Any], schema: dict[str, Any]) -> bool:
    try:
        jsonschema.validate(payload, schema)
    except jsonschema.ValidationError:
        return False
    return True


def _resolve(schema: dict[str, Any], node: dict[str, Any]) -> dict[str, Any]:
    reference = node.get("$ref")
    if reference is None:
        return node
    return dict(schema["$defs"][reference.rsplit("/", 1)[-1]])


class TestWhatElseCanBeDescribed:
    """The targets a tool definition arrives with, beyond a model and a function."""

    def test_a_generic_alias_is_described_as_the_type_it_is(self) -> None:
        assert schema_for(list[str]) == {"items": {"type": "string"}, "type": "array"}

    def test_an_optional_is_a_type_too(self) -> None:
        assert schema_for(str | None) == {"anyOf": [{"type": "string"}, {"type": "null"}]}

    def test_a_builtin_does_not_lend_the_model_its_own_docstring(self) -> None:
        assert schema_for(str) == {"type": "string"}

    def test_an_injected_parameter_is_left_out_entirely(self) -> None:
        def book(city: str, connection: Waypoint) -> str:
            """Book a stay.

            Args:
                city: Where to stay.
                connection: Supplied by the caller, never by the model.
            """
            return city + connection.city

        schema = schema_for(book, exclude=("connection",))

        assert set(schema["properties"]) == {"city"}
        assert schema["required"] == ["city"]
        assert "Waypoint" not in str(schema)

    def test_a_callable_object_is_read_through_its_call(self) -> None:
        class Book:
            """A tool that holds a client."""

            def __call__(self, city: str) -> str:
                """Book a stay.

                Args:
                    city: Where to stay.
                """
                return city

        assert annotations_of(Book()) == {"city": str, "return": str}
        assert schema_for(Book())["properties"]["city"]["type"] == "string"

    def test_a_type_pydantic_refuses_names_the_parameter_that_carried_it(self) -> None:
        def send(sock: socket.socket) -> str:
            """Send something.

            Args:
                sock: The socket to send on.
            """
            return str(sock)

        with pytest.raises(SchemaGenerationError) as refused:
            schema_for(send)

        assert refused.value.field == "sock"


def test_the_type_still_validates_what_it_always_did() -> None:
    """The generator reads types; it must not change how any of them validate."""
    with pytest.raises(ValidationError):
        Waypoint(city=1)  # type: ignore[arg-type]
