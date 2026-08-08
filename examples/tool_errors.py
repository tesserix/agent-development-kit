"""What a tool failing means, said precisely enough for the run loop to act on it.

Four scenarios: the taxonomy's three answers; a library exception translated declaratively;
what an unmapped exception is assumed to be; and what a span records about each. Run it with
`python examples/tool_errors.py`.
"""

from __future__ import annotations

import asyncio
from contextlib import suppress

from tesserix_adk.testing import FakeClock
from tesserix_adk.tools import ToolCallSpan, ToolRegistry, tool
from tesserix_adk.tools.errors import (
    ToolErrorMap,
    ToolFailure,
    ToolRefusal,
    permanent,
    refusal,
    transient,
)


class SupplierUnavailableError(Exception):
    """What the supplier's client raises when it cannot reach the supplier."""


class SupplierSaidNoError(Exception):
    """What the supplier's client raises when the supplier answered and declined."""

    status_code = 409


SUPPLIER = ToolErrorMap(
    {
        SupplierUnavailableError: transient("supplier_unavailable", retry_after=2.0),
        ValueError: permanent("malformed_supplier_payload"),
    },
    statuses={409: refusal("booking_not_cancellable", "This fare is non-refundable.")},
)


@tool
async def cancel(booking: str) -> str:
    """Cancel a booking, where the supplier allows it.

    Args:
        booking: What to cancel.
    """
    raise SUPPLIER.classify(SupplierSaidNoError(f"{booking} is non-refundable"), tool="cancel")


@tool
async def book(leg: str) -> str:
    """Book a leg against a supplier that is having a bad day.

    Args:
        leg: What to book.
    """
    raise SUPPLIER.classify(SupplierUnavailableError(f"connection reset for {leg}"), tool="book")


async def main() -> None:
    """Run each scenario and print what the runtime is entitled to conclude."""
    for error in (
        ToolFailure("book", "supplier_unavailable", transient=True, retry_after=2.0),
        ToolFailure("book", "malformed_supplier_payload"),
        ToolRefusal("cancel", "booking_not_cancellable", "This fare is non-refundable."),
    ):
        print(f"{type(error).__name__} {error.code}: retryable={error.retryable}")  # noqa: T201

    translated = SUPPLIER.classify(SupplierUnavailableError("connection reset"), tool="book")
    print("translated:", translated.code, "retryable:", translated.retryable)  # noqa: T201

    unmapped = SUPPLIER.classify(KeyError("nobody classified this"), tool="book")
    print("unmapped is assumed permanent:", unmapped.code, unmapped.retryable)  # noqa: T201

    leaked = SUPPLIER.classify(
        SupplierUnavailableError("401 for token sk-live-abcdefghijklmnopqrst"), tool="book"
    )
    print("credential survived translation:", "sk-live-" in str(leaked))  # noqa: T201

    spans: list[ToolCallSpan] = []
    registry = ToolRegistry((cancel, book), clock=FakeClock())
    registry.observe(spans.append)
    for name, arguments in (("cancel", {"booking": "AB-1"}), ("book", {"leg": "Osaka"})):
        with suppress(ToolFailure, ToolRefusal):
            await registry.invoke(name, arguments)
    for span in spans:
        print(f"span {span.tool}: outcome={span.outcome} code={span.code}")  # noqa: T201

    cancel.release()
    book.release()


if __name__ == "__main__":
    asyncio.run(main())
