"""What a model sent is held to the tool's signature before the body is entered.

Four scenarios: a call with a hallucinated field, a wrong type and a missing one refused
whole; the structured feedback that goes back to the model, without the values it sent;
the payload shapes different providers choose, all read the same way; and a registry that
chose the documented coercions instead. Run it with `python examples/tool_arguments.py`.
"""

from __future__ import annotations

import asyncio
import json

from tesserix_adk.core import AdkModel, ToolArgumentValidationError
from tesserix_adk.tools import LENIENT, ArgumentPolicy, ToolArgumentValidator, tool


class Leg(AdkModel):
    """A hop of a journey.

    Args:
        origin: Where the traveller boards.
        nights: How long they stay at the far end.
    """

    origin: str
    nights: int


@tool
async def book_leg(leg: Leg, seats: int) -> str:
    """Book one hop of a journey.

    Args:
        leg: The hop to book.
        seats: How many seats to hold.
    """
    return f"{leg.origin} x{seats}"


def refused() -> None:
    """One call, three things wrong with it, and a body that is never entered."""
    call = {"lg": {"origin": "Osaka", "nights": 2}, "seats": "two", "class": "first"}
    try:
        asyncio.run(book_leg.invoke(call))
    except ToolArgumentValidationError as rejected:
        print("fields:      ", rejected.paths)  # noqa: T201
        print("why:         ", rejected.problems["class"])  # noqa: T201
        print("feedback:    ", rejected.feedback().splitlines()[0])  # noqa: T201


def unquoted() -> None:
    """A rejected value may be a password, so the field is named and the value is not."""
    try:
        asyncio.run(book_leg.invoke({"leg": {"origin": "Osaka", "nights": 2}, "seats": "s3cret"}))
    except ToolArgumentValidationError as rejected:
        print("quoted:      ", "s3cret" in rejected.feedback())  # noqa: T201
        print("kept:        ", "s3cret" in json.dumps(rejected.payload))  # noqa: T201


def normalised() -> None:
    """Providers disagree about the envelope; the tool is not the place that finds out."""
    good = {"leg": {"origin": "Osaka", "nights": 2}, "seats": 2}
    for label, payload in (
        ("mapping", good),
        ("json text", json.dumps(good)),
        ("envelope", {"arguments": json.dumps(good)}),
    ):
        print(f"{label + ':':13}", asyncio.run(book_leg.invoke(payload)))  # noqa: T201

    strict = ToolArgumentValidator(book_leg.function, tool="book_leg")
    try:
        strict.validate({**good, "seats": "2"})
    except ToolArgumentValidationError as rejected:
        print("strict:      ", rejected.problems["seats"])  # noqa: T201


def lenient() -> None:
    """Leniency is a consumer's decision, made once, and it still refuses what is not declared."""
    reader = ToolArgumentValidator(book_leg.function, tool="book_leg", policy=LENIENT)
    good = {"leg": {"origin": "Osaka", "nights": "2"}, "seats": "2"}
    print("coerced:     ", reader.validate(good).seats)  # noqa: T201
    try:
        reader.validate({**good, "class": "first"})
    except ToolArgumentValidationError as rejected:
        print("still:       ", rejected.paths)  # noqa: T201

    tight = ArgumentPolicy(max_bytes=64)
    bounded = ToolArgumentValidator(book_leg.function, tool="book_leg", policy=tight)
    try:
        bounded.validate({"leg": {"origin": "x" * 500, "nights": 2}, "seats": 1})
    except ToolArgumentValidationError as rejected:
        print("ceiling:     ", str(rejected).split(", over")[1].strip())  # noqa: T201


if __name__ == "__main__":
    refused()
    unquoted()
    normalised()
    lenient()
