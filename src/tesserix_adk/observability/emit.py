"""Exporting a run's spend: one span per metered step, and counters that are never sampled.

Nothing here is wired into the run loop. Emission reads a finished run, so a collector
outage, a slow exporter or a misconfigured redactor cannot reach into the run that produced
the numbers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel

from tesserix_adk.observability.attribution import (
    ATTRIBUTE_PREFIX,
    Step,
    attributes_of,
    spend_of,
)
from tesserix_adk.observability.metrics import CALLS, COST, TOKENS, Dimensions
from tesserix_adk.observability.redaction import Redactor

if TYPE_CHECKING:
    from collections.abc import Mapping

    from tesserix_adk.core import Run
    from tesserix_adk.core.protocols import Tracer
    from tesserix_adk.observability.attribution import SpendRecord
    from tesserix_adk.observability.metrics import Meter

__all__ = ["REDACTED_EVENT", "SPAN_NAMES", "record_spend"]

REDACTED_EVENT = "adk.redacted"

SPAN_NAMES = {Step.MODEL: "adk.model_call", Step.TOOL: "adk.tool_call"}


def record_spend[OutputT: BaseModel](
    run: Run[OutputT],
    *,
    tracer: Tracer | None = None,
    meter: Meter | None = None,
    dimensions: Dimensions | None = None,
    redactor: Redactor | None = None,
    extra: Mapping[str, str] | None = None,
    sampled: bool = True,
) -> tuple[SpendRecord, ...]:
    """Export what `run` spent, and return the records that were exported.

    Args:
        run: The finished run to read. Nothing on it is mutated.
        tracer: Where spans go. None exports no spans and still counts the spend.
        meter: Where counters go. None exports no counters.
        dimensions: Which values the counters are split by. Defaults to splitting by all
            of them, which is right for one tenant and wrong for a marketplace.
        redactor: What scrubs `extra` before export. Defaults to the built-in shapes.
        extra: Attributes a consumer wants on the spans, scrubbed before they leave.
        sampled: Whether this run's trace was kept. A sampled-away trace still counts its
            spend — losing the money with the trace is how a bill goes missing.

    Returns:
        The records exported, in run order. Empty for a run that spent nothing.
    """
    records = spend_of(run)
    if not records:
        return ()

    dimensions = dimensions or Dimensions()
    scrubbed, redaction = (redactor or Redactor()).scrub(extra or {})

    if tracer is not None and sampled:
        if redaction:
            tracer.event(
                REDACTED_EVENT,
                **{f"{ATTRIBUTE_PREFIX}redacted_keys": ", ".join(redaction.dropped)},
            )
        for record in records:
            with tracer.span(SPAN_NAMES[record.step], **(attributes_of(record) | scrubbed)):
                pass

    if meter is not None:
        for record in records:
            under = dimensions.of(record)
            meter.count(COST, float(record.cost.total), **under)
            meter.count(
                TOKENS, float(record.usage.input_tokens + record.usage.output_tokens), **under
            )
            meter.count(CALLS, 1.0, **under)
    return records
