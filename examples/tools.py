"""One typed function is the whole tool: the schema the model reads comes from its signature.

Five scenarios: a documented function becoming a schema whose descriptions came from its
docstring, a signature no model could be told about failing at the line that declared it,
one name refused to a second live tool, a synchronous body awaited like an asynchronous one,
and a context the runtime injects that the model cannot reach. Run it with
`python examples/tools.py`.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Iterator  # noqa: TC003 — annotates an example signature

from tesserix_adk.core import (
    AdkModel,
    ToolArgumentValidationError,
    ToolDefinitionError,
    ToolExecutionError,
)
from tesserix_adk.tools import ToolContext, tool


class Leg(AdkModel):
    """A hop of a journey.

    Args:
        origin: Where the traveller boards.
        nights: How long they stay at the far end.
    """

    origin: str
    nights: int = 1


@tool
async def price_leg(leg: Leg, currency: str = "EUR") -> str:
    """Price one hop of a journey.

    Args:
        leg: The hop to price.
        currency: What to quote it in.
    """
    return f"{leg.origin}: {leg.nights * 40} {currency}"


def derived() -> None:
    """The schema, the description and the required set all come from the function."""
    print("name:        ", price_leg.name)  # noqa: T201
    print("description: ", price_leg.description)  # noqa: T201
    print("required:    ", price_leg.parameters_schema["required"])  # noqa: T201
    print("nested:      ", json.dumps(price_leg.parameters_schema["properties"]["leg"]))  # noqa: T201
    print("returns:     ", price_leg.returns_schema)  # noqa: T201
    # `invoke` is the model-facing path, so it reads the payload into the declared types.
    print("quote:       ", asyncio.run(price_leg.invoke({"leg": {"origin": "Osaka"}})))  # noqa: T201


def refusals() -> None:
    """Every one of these fails at the decorator, not on the call that first sends it."""
    try:

        @tool
        def unannotated(code) -> str:  # type: ignore[no-untyped-def]  # noqa: ANN001
            return code

    except ToolDefinitionError as refused:
        print("unannotated: ", refused.parameter, "-", refused.tool)  # noqa: T201

    try:

        @tool
        def streaming(count: int) -> Iterator[int]:
            yield count

    except ToolDefinitionError as refused:
        print("generator:   ", refused.tool)  # noqa: T201

    try:

        @tool(name="price_leg")
        def shadowing(leg: str) -> str:
            return leg

    except ToolDefinitionError as refused:
        print("shadowing:   ", refused.tool)  # noqa: T201


def uniform() -> None:
    """A synchronous body is awaited too, off the event loop rather than on it."""

    @tool
    def lookup(code: str) -> str:
        """Resolve an airport code."""
        return {"OSA": "Osaka"}.get(code, "unknown")

    print("sync:        ", lookup.is_async, asyncio.run(lookup.invoke({"code": "OSA"})))  # noqa: T201
    lookup.release()


def injected() -> None:
    """The context is filled by the caller and left out of what the model is told."""

    @tool
    async def archive(document: str, ctx: ToolContext) -> str:
        """File a document against the run's tenant."""
        ctx.raise_if_cancelled()
        return f"{document} filed for {ctx.tenant}"

    context = ToolContext(run_id="run-1", tenant="acme")
    print("described:   ", sorted(archive.parameters_schema["properties"]))  # noqa: T201
    print("filed:       ", asyncio.run(archive.invoke({"document": "itinerary"}, context)))  # noqa: T201

    forged = {"document": "itinerary", "ctx": {"run_id": "run-2", "tenant": "rival"}}
    try:
        asyncio.run(archive.invoke(forged, context))
    except ToolArgumentValidationError as refused:
        print("forged:      ", refused.paths)  # noqa: T201

    try:
        asyncio.run(archive.invoke({"document": "itinerary"}))
    except ToolExecutionError as refused:
        print("no run:      ", refused.details["parameter"])  # noqa: T201
    archive.release()


if __name__ == "__main__":
    derived()
    refusals()
    uniform()
    injected()
