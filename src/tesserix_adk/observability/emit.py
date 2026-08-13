"""Exporting a run's spend: one span per metered step, and counters that are never sampled.

Nothing here is wired into the run loop. Emission reads a finished run, so a collector
outage, a slow exporter or a misconfigured redactor cannot reach into the run that produced
the numbers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel

from tesserix_adk.core.errors import AttributionError
from tesserix_adk.observability.attribution import (
    ATTRIBUTE_PREFIX,
    UNKNOWN,
    Step,
    attributes_of,
    spend_of,
)
from tesserix_adk.observability.metrics import (
    CACHED_TOKENS,
    CALLS,
    COST,
    INPUT_TOKENS,
    TOKENS,
    Dimensions,
)
from tesserix_adk.observability.redaction import Redactor
from tesserix_adk.observability.trace import attributes_of_context

if TYPE_CHECKING:
    from collections.abc import Mapping

    from tesserix_adk.core import Run
    from tesserix_adk.core.protocols import Tracer
    from tesserix_adk.observability.attribution import SpendRecord
    from tesserix_adk.observability.metrics import Meter
    from tesserix_adk.observability.tree import Node, RunTree

__all__ = ["NODE_SPAN", "REDACTED_EVENT", "SPAN_NAMES", "record_spend", "record_tree"]

REDACTED_EVENT = "adk.redacted"

SPAN_NAMES = {Step.MODEL: "adk.model_call", Step.TOOL: "adk.tool_call"}

NODE_SPAN = "adk.participant"


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
            meter.count(INPUT_TOKENS, float(record.usage.input_tokens), **under)
            meter.count(CACHED_TOKENS, float(record.usage.cached_tokens), **under)
            meter.count(CALLS, 1.0, **under)
    return records


def record_tree(
    assembled: RunTree,
    *,
    tracer: Tracer | None = None,
    meter: Meter | None = None,
    dimensions: Dimensions | None = None,
    redactor: Redactor | None = None,
    extra: Mapping[str, str] | None = None,
    sampled: bool = True,
) -> tuple[Node, ...]:
    """Export a whole multi-agent run: one span per participant, counters per participant.

    Counters are emitted whatever the sampler did with the trace, and one per participant
    rather than one per run. That is what stops a wide fan-out losing its cost to sampling:
    the money never travels on a span in the first place.

    Args:
        assembled: The tree to export. Nothing on it is mutated.
        tracer: Where spans go. None exports no spans and still counts the spend.
        meter: Where counters go. None exports no counters.
        dimensions: Which values the counters are split by.
        redactor: What scrubs `extra` before export. Defaults to the built-in shapes.
        extra: Attributes a consumer wants on every participant's span — transferred
            context, a correlation id — scrubbed before they leave.
        sampled: Whether this trace was kept.

    Returns:
        The participants exported, in tree order.

    Raises:
        AttributionError: With `reason` `"no_tenant"`, where a participant has no tenant.
            An exported span nobody can attribute is worse than a missing one: it fills a
            bill with spend that has no owner, and the owner is unrecoverable once the run
            is gone. Nothing is exported — the refusal happens before the first span.
    """
    for one in assembled.nodes:
        if one.context.tenant in {"", UNKNOWN}:
            raise AttributionError(
                f"{one.context.run_id!r} has no tenant, so its spend would export as "
                f"unattributable and stay that way",
                reason="no_tenant",
                run_id=one.context.run_id,
            )

    dimensions = dimensions or Dimensions()
    scrubbed, redaction = (redactor or Redactor()).scrub(extra or {})
    currency = assembled.currency

    if tracer is not None and sampled:
        if redaction:
            tracer.event(
                REDACTED_EVENT,
                **{f"{ATTRIBUTE_PREFIX}redacted_keys": ", ".join(redaction.dropped)},
            )
        for one in assembled.nodes:
            with tracer.span(NODE_SPAN, **(_node_attributes(one, currency) | scrubbed)):
                pass

    if meter is not None:
        for one in assembled.nodes:
            stated = one.in_currency(currency)
            under = {
                "tenant": dimensions.bucket(one.context.tenant, dimensions.tenants),
                "agent": dimensions.bucket(one.context.agent or UNKNOWN, dimensions.agents),
                "pattern": str(one.context.pattern),
                "currency": currency,
                "attributed": "true" if stated is not None and one.reported else "false",
            }
            meter.count(COST, float(stated.total) if stated is not None else 0.0, **under)
            counted = one.usage
            meter.count(
                TOKENS,
                float(counted.input_tokens + counted.output_tokens) if counted else 0.0,
                **under,
            )
            meter.count(CALLS, 1.0, **under)
    return assembled.nodes


def _node_attributes(one: Node, currency: str) -> dict[str, str]:
    """What one participant's span says, including what it could not say."""
    stated = one.in_currency(currency)
    return attributes_of_context(one.context) | {
        f"{ATTRIBUTE_PREFIX}cost": str(stated.total) if stated is not None else UNKNOWN,
        f"{ATTRIBUTE_PREFIX}currency": currency,
        f"{ATTRIBUTE_PREFIX}input_tokens": str(one.usage.input_tokens) if one.usage else UNKNOWN,
        f"{ATTRIBUTE_PREFIX}output_tokens": str(one.usage.output_tokens) if one.usage else UNKNOWN,
        f"{ATTRIBUTE_PREFIX}latency_ms": f"{one.latency_ms:.3f}",
        f"{ATTRIBUTE_PREFIX}reported": "true" if one.reported else "false",
        f"{ATTRIBUTE_PREFIX}clock_skew": "true" if one.skewed else "false",
    }
