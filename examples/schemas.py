"""Derive the schema from the type, so the model is told what the code will parse.

Four scenarios: a documented model becoming a schema whose descriptions came from its
docstring, one type in two provider dialects, a hash that moves the moment a field is
renamed, and a type that cannot be described failing where it is declared rather than in
production. Run it with `python examples/schemas.py`.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import Field

from tesserix_adk.core import (
    INLINE_REFS,
    STRICT_SUBSET,
    AdkModel,
    CapabilityError,
    SchemaGenerationError,
    schema_for,
    schema_hash,
)


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


class Itinerary(AdkModel):
    """A trip proposed for a traveller.

    Args:
        traveller: Who is going.
        stops: Every stop, in order.
        leg: How the traveller gets there.
        budget: Ceiling in whole currency units, where there is one.
    """

    traveller: str
    stops: list[Waypoint]
    leg: Annotated[Flight | Train, Field(discriminator="kind")]
    budget: Annotated[int, Field(ge=0, le=10_000)] | None = None


class Renamed(AdkModel):
    """The same trip after one field was renamed.

    Args:
        passenger: Who is going.
        stops: Every stop, in order.
        leg: How the traveller gets there.
        budget: Ceiling in whole currency units, where there is one.
    """

    passenger: str
    stops: list[Waypoint]
    leg: Annotated[Flight | Train, Field(discriminator="kind")]
    budget: Annotated[int, Field(ge=0, le=10_000)] | None = None


def documented() -> None:
    """The docstring is the description the model reads."""
    schema = schema_for(Waypoint)

    print("fields:  ", sorted(schema["properties"]))  # noqa: T201
    print("required:", schema["required"])  # noqa: T201
    print("city:    ", schema["properties"]["city"]["description"])  # noqa: T201


def dialects() -> None:
    """One type, two providers, neither of them the caller's problem."""
    closed = schema_for(Itinerary, dialect=STRICT_SUBSET)

    print("closed:  ", closed["additionalProperties"] is False)  # noqa: T201
    print("inlined: ", "$defs" not in schema_for(Itinerary, dialect=INLINE_REFS))  # noqa: T201


def versions() -> None:
    """A renamed field changes the hash, so a stale cassette misses instead of replaying."""
    before = schema_hash(schema_for(Itinerary))
    after = schema_hash(schema_for(Renamed))

    print("stable:  ", before == schema_hash(schema_for(Itinerary)))  # noqa: T201
    print("moved:   ", before != after)  # noqa: T201


def refusals() -> None:
    """Anything that cannot be described faithfully fails here, not in production."""

    class Loose(AdkModel):
        """A model with a field that describes nothing.

        Args:
            payload: Whatever the provider felt like sending.
        """

        payload: Any

    class Node(AdkModel):
        """A tree that contains itself.

        Args:
            name: What this node is called.
            child: The node below it, where there is one.
        """

        name: str
        child: Node | None = None

    try:
        schema_for(Loose)
    except SchemaGenerationError as refused:
        print("field:   ", refused.field, "-", refused.annotation)  # noqa: T201

    try:
        schema_for(Node, dialect=INLINE_REFS)
    except CapabilityError as refused:
        print("dialect: ", refused.details["dialect"], "-", refused.details["type"])  # noqa: T201


if __name__ == "__main__":
    documented()
    dialects()
    versions()
    refusals()
