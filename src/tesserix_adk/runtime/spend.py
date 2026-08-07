"""Holding a stream to the same ceiling the rest of the run answers to.

A stream is the one place a ceiling is easy to lose: the tokens arrive after the call was
permitted, so a budget checked only at dispatch has already been passed by the time the
answer is half written. The wrapper here charges each running total as the vendor reports
it, and ends the stream with the typed error rather than stopping quietly — a consumer
that sees a stream simply end reads it as a finished answer.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from decimal import Decimal
from typing import TYPE_CHECKING

from tesserix_adk.core.primitives import Usage
from tesserix_adk.core.streaming import UsageDelta

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from tesserix_adk.core.cost import Cost
    from tesserix_adk.core.protocols import BudgetPolicy
    from tesserix_adk.core.streaming import StreamEvent

__all__ = ["budgeted_stream"]


async def budgeted_stream(
    events: AsyncIterator[StreamEvent], budget: BudgetPolicy
) -> AsyncIterator[StreamEvent]:
    """Pass `events` through, charging each reported total against `budget`.

    Vendors report usage as a running total that later events replace, so only the
    increment is charged and a stream reporting the same total twice is billed once.

    Args:
        events: What the provider is emitting.
        budget: The ceiling this stream answers to, usually the run's own.

    Yields:
        Every event, in arrival order, up to and including the one that fit.

    Raises:
        BudgetExceededError: When a reported total passes a ceiling. The event that
            carried it is not yielded and the stream is not resumed.
        BudgetUnavailableError: When a shared ceiling applies and its ledger cannot be
            reached, unless the run was configured to proceed without it.
    """
    charged = Usage(input_tokens=0, output_tokens=0)
    try:
        async for event in events:
            if isinstance(event, UsageDelta):
                await budget.record(_since(charged, event.usage))
                charged = event.usage
            yield event
    finally:
        # A stream abandoned mid-flight keeps the vendor sending, and the tokens still
        # arriving are charged to nobody.
        if isinstance(events, AsyncGenerator):
            await events.aclose()


_COUNTS = (
    "input_tokens",
    "output_tokens",
    "cached_tokens",
    "cache_write_tokens",
    "reasoning_tokens",
    "image_units",
)
_COMPONENTS = ("input", "output", "cache_read", "cache_write", "reasoning", "image")


def _since(charged: Usage, total: Usage) -> Usage:
    """The part of a running total that has not been charged yet."""
    counts = {name: max(getattr(total, name) - getattr(charged, name), 0) for name in _COUNTS}
    return Usage(**counts, cost=_cost_since(charged, total), source=total.source)


def _cost_since(charged: Usage, total: Usage) -> Cost | None:
    """The unbilled part of a running cost, component by component."""
    if total.cost is None:
        return None
    if charged.cost is None:
        return total.cost
    return total.cost.model_copy(
        update={
            name: max(getattr(total.cost, name) - getattr(charged.cost, name), Decimal(0))
            for name in _COMPONENTS
        }
    )
