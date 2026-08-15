"""One set of names for agent telemetry, so one query works across products.

Every product names its fields differently — tenant is `tenant`, `tenant_id` or `org`
depending on which service exported the span — and the cost question ends up answered
twice, differently. This is the contract that stops that: a versioned set of names under
the `adk.` prefix, populated from what the run already knows rather than tagged by hand.

Nothing here estimates. A provider that reported no usage produces an attribute saying so
and a reason, never a number a dashboard would total as if it had been measured.
"""

from __future__ import annotations

from collections.abc import Mapping  # noqa: TC003 — pydantic needs the runtime type
from enum import StrEnum
from typing import Self

from pydantic import model_validator

from tesserix_adk.core import (
    AdkModel,
    AttributionError,
    Cost,
    CostConfidence,
    current_tenant,
)
from tesserix_adk.observability.attribution import UNKNOWN, Outcome

__all__ = [
    "CARDINALITY",
    "CONVENTION_VERSION",
    "MANDATORY",
    "MAX_VALUE_LENGTH",
    "RESERVED_PREFIX",
    "AttributeSet",
    "CacheStatus",
    "Cardinality",
    "Measured",
    "Unavailability",
    "conforms",
]

CONVENTION_VERSION = "1.0"
"""The version of this contract, carried on every span.

Within a major version the set only ever gains names, because a dashboard, an alert and a
saved query all break the day one is renamed or removed. A name that has to go is
deprecated through `tesserix_adk.core.deprecate` and kept alongside its replacement for a
full major version.
"""

RESERVED_PREFIX = "adk."
"""The namespace this convention owns. A product's own attributes go outside it."""

MAX_VALUE_LENGTH = 512
"""Longest attribute value exported. Backends drop longer ones without saying so."""

_TRUNCATED = "…(truncated)"


class Unavailability(StrEnum):
    """Why a number is absent. Absent with a reason is a fact; absent is a hole."""

    NOT_REPORTED = "not reported"
    NOT_PRICED = "not priced"
    NOT_MEASURED = "not measured"


class CacheStatus(StrEnum):
    """Whether a cache answered, across provider prompt caching and the kit's own caches.

    `NONE` and `MISS` are different numbers on a hit-rate dashboard: one deployment has no
    cache, the other has one that is not working.
    """

    HIT = "hit"
    PARTIAL = "partial"
    MISS = "miss"
    NONE = "none"


class Cardinality(StrEnum):
    """Whether an attribute is safe as a metric dimension.

    A metric split by user id is a time series per user, which is how a metrics backend
    falls over. High-cardinality attributes belong on spans, where they cost one write.
    """

    LOW = "low"
    HIGH = "high"


class Measured(AdkModel):
    """A number that was measured, or an explicit statement that it was not.

    Args:
        value: What was measured.
        unavailable: Why nothing was, when nothing was.
    """

    value: float | None = None
    unavailable: Unavailability | None = None

    @model_validator(mode="after")
    def _one_or_the_other(self) -> Self:
        if (self.value is None) == (self.unavailable is None):
            message = "a measurement is either a value or a reason there is none, not both"
            raise ValueError(message)
        return self

    @classmethod
    def of(cls, value: float) -> Self:
        """A measured number."""
        return cls(value=value)

    @classmethod
    def missing(cls, reason: Unavailability) -> Self:
        """No number, and why."""
        return cls(unavailable=reason)

    @property
    def known(self) -> bool:
        """Whether there is a number."""
        return self.value is not None

    def rendered(self) -> str:
        """The value as an attribute. Whole numbers keep no decimal point."""
        value = self.value or 0.0
        return str(int(value)) if value == int(value) else str(value)


def _named(field: str) -> str:
    return f"{RESERVED_PREFIX}{field}"


_LOW = (
    "convention",
    "tenant",
    "agent",
    "agent_version",
    "model",
    "provider",
    "prompt_version",
    "task_class",
    "outcome",
    "cache",
    "currency",
    "pricing_version",
    "cost.confidence",
    "cost.unavailable",
    "input_tokens.unavailable",
    "output_tokens.unavailable",
    "latency_seconds.unavailable",
)
_HIGH = (
    "user",
    "run_id",
    "input_tokens",
    "output_tokens",
    "latency_seconds",
    "cost",
    "iterations",
)

CARDINALITY: Mapping[str, Cardinality] = {
    **{_named(field): Cardinality.LOW for field in _LOW},
    **{_named(field): Cardinality.HIGH for field in _HIGH},
}
"""Whether each name may be a metric dimension. Consulted by `metric_dimensions`."""

MANDATORY = frozenset(
    _named(field)
    for field in (
        "convention",
        "tenant",
        "agent",
        "agent_version",
        "model",
        "provider",
        "outcome",
        "cache",
    )
)
"""The names every kit-emitted span carries. Without these no cross-product query works."""


def _truncated(value: str) -> str:
    """Cut a long value where a reader can see it was cut."""
    if len(value) <= MAX_VALUE_LENGTH:
        return value
    return value[: MAX_VALUE_LENGTH - len(_TRUNCATED)] + _TRUNCATED


class AttributeSet(AdkModel):
    """Everything one kit-emitted span says about the work it covers.

    Args:
        tenant: The isolation boundary the work belongs to.
        agent: Which agent ran.
        agent_version: Which version of it, so a change in cost or latency is attributable.
        model: The model that answered, not the one first asked for.
        provider: Whose API served it.
        run_id: The run, for joining a span back to what happened.
        outcome: Whether the step produced anything.
        cache: Whether a cache answered.
        input_tokens: Prompt tokens, or why they are not known.
        output_tokens: Generated tokens, or why they are not known.
        latency_seconds: Wall time, or why it was not measured.
        cost: What it came to. An unpriced model is `CostConfidence.UNKNOWN`, never zero.
        pricing_version: Which price list produced the cost, so a historical figure stays
            interpretable after a price change.
        user: The acting principal. High cardinality — spans only, never a metric split.
        prompt_version: Which assembled prompt design produced the call.
        task_class: The kind of work routing was asked for.
        iterations: How many times the loop went round.
        extra: A product's own attributes, which may not use the reserved namespace.
    """

    tenant: str
    agent: str
    agent_version: str
    model: str
    provider: str
    run_id: str
    outcome: Outcome
    cache: CacheStatus
    input_tokens: Measured
    output_tokens: Measured
    latency_seconds: Measured
    cost: Cost
    pricing_version: str
    user: str = UNKNOWN
    prompt_version: str = UNKNOWN
    task_class: str = UNKNOWN
    iterations: int = 0
    extra: Mapping[str, str] = {}

    @model_validator(mode="after")
    def _extras_stay_out_of_the_namespace(self) -> Self:
        squatted = sorted(name for name in self.extra if name.startswith(RESERVED_PREFIX))
        if squatted:
            message = f"{squatted} use the reserved {RESERVED_PREFIX!r} namespace"
            raise ValueError(message)
        return self

    @classmethod
    def here(cls, **fields: object) -> Self:
        """Build a set with the tenant and user taken from the bound context.

        Raises:
            MissingTenantContextError: Outside a tenant scope. A span attributed to no
                tenant is spend nobody chases.
        """
        context = current_tenant(where="telemetry attributes")
        return cls.model_validate(
            {"tenant": context.tenant, "user": context.user or UNKNOWN, **fields}
        )

    def rendered(self) -> dict[str, str]:
        """The attributes as an exporter receives them, values truncated where too long."""
        attributes: dict[str, str] = {
            _named("convention"): CONVENTION_VERSION,
            _named("tenant"): self.tenant,
            _named("agent"): self.agent,
            _named("agent_version"): self.agent_version,
            _named("model"): self.model,
            _named("provider"): self.provider,
            _named("run_id"): self.run_id,
            _named("user"): self.user,
            _named("prompt_version"): self.prompt_version,
            _named("task_class"): self.task_class,
            _named("outcome"): str(self.outcome),
            _named("cache"): str(self.cache),
            _named("iterations"): str(self.iterations),
        }
        for field in ("input_tokens", "output_tokens", "latency_seconds"):
            measured: Measured = getattr(self, field)
            if measured.known:
                attributes[_named(field)] = measured.rendered()
            else:
                attributes[f"{_named(field)}.unavailable"] = str(measured.unavailable)
        attributes.update(self._cost())
        return {name: _truncated(value) for name, value in attributes.items()} | dict(self.extra)

    def _cost(self) -> dict[str, str]:
        """Cost, currency and price list — or the reason there is no figure."""
        if self.cost.confidence is CostConfidence.UNKNOWN:
            return {f"{_named('cost')}.unavailable": str(Unavailability.NOT_PRICED)}
        return {
            _named("cost"): str(self.cost.total),
            _named("currency"): self.cost.currency,
            _named("pricing_version"): self.pricing_version,
            f"{_named('cost')}.confidence": str(self.cost.confidence),
        }

    def metric_dimensions(self) -> dict[str, str]:
        """The subset safe to split a counter by. High-cardinality names are left out."""
        return {
            name: value
            for name, value in self.rendered().items()
            if CARDINALITY.get(name) is Cardinality.LOW
        }


def conforms(attributes: Mapping[str, str]) -> None:
    """Check `attributes` against the convention.

    Args:
        attributes: What a span is about to carry.

    Raises:
        AttributionError: If a mandatory name is missing, or a name the convention has not
            defined uses the reserved namespace. Squatting a name is how the next version
            of the convention breaks somebody's dashboard.
    """
    missing = sorted(MANDATORY - set(attributes))
    squatted = sorted(
        name for name in attributes if name.startswith(RESERVED_PREFIX) and name not in CARDINALITY
    )
    if missing or squatted:
        problem = f"missing {missing}" if missing else f"undefined {squatted}"
        message = f"these attributes do not meet convention {CONVENTION_VERSION}: {problem}"
        raise AttributionError(message, reason="convention")
