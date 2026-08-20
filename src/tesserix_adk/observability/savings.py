"""Token savings reported as measured or as estimated, never as whichever reads better.

Input savings can be measured: the uncompressed and the compressed counts exist at the same
moment, so the difference is a fact. Output savings cannot. The system never observes the
response it would have received without shaping, so any figure for it is a counterfactual,
and reporting a counterfactual as a measurement is how a cost dashboard becomes fiction.

The honest treatment is to label the two differently and to make the estimated one
checkable: hold a small random slice of traffic out of shaping entirely and compare the two
arms. The holdout costs a few percent of the available savings and buys the only output
figure anyone can defend in a review.

Nothing here touches content. A record carries token counts, a run id, a tenant and an arm,
because an accounting record that carries text is a second copy of the conversation.
"""

from __future__ import annotations

from enum import StrEnum
from hashlib import blake2b
from math import sqrt
from typing import TYPE_CHECKING

from pydantic import Field, model_validator

from tesserix_adk.core.errors import ConfigurationError
from tesserix_adk.core.models import AdkModel

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

__all__ = [
    "MINIMUM_ARM",
    "Arm",
    "Basis",
    "Figure",
    "HoldoutPolicy",
    "SavingsReport",
    "ShapedRun",
    "account",
    "by_tenant",
]

MINIMUM_ARM = 30
"""How many runs an arm needs before a comparison between the arms says anything."""

_INTERVAL = 1.96
"""The 95% normal quantile. The arms are compared, not modelled."""


class Arm(StrEnum):
    """Which side of the experiment a run was on."""

    SHAPED = "shaped"
    HOLDOUT = "holdout"


class Basis(StrEnum):
    """How a figure came to exist, which is the part a reader has to see."""

    MEASURED = "measured"
    ESTIMATED = "estimated"
    INSUFFICIENT = "insufficient"


class HoldoutPolicy(AdkModel):
    """The slice of traffic that bypasses shaping so the rest can be checked against it.

    Args:
        fraction: The share held out, from `0.0` to `1.0`. Zero means no control exists,
            which the report says rather than hides.
        salt: Namespaces the assignment, so two independent experiments do not hold out the
            same runs and confound each other.
    """

    fraction: float = Field(default=0.0, ge=0.0, le=1.0)
    salt: str = "savings"

    @property
    def controlled(self) -> bool:
        """Whether any traffic is held out at all."""
        return self.fraction > 0.0

    def arm(self, run_id: str) -> Arm:
        """Which arm `run_id` belongs to.

        Assignment is a hash of the run id, so it is stable: a retried run stays in the arm
        it started in rather than being shaped on the second attempt and compared against
        itself.

        Example:
            >>> policy = HoldoutPolicy(fraction=0.5)
            >>> policy.arm("run-7") is policy.arm("run-7")
            True
        """
        if not self.controlled:
            return Arm.SHAPED
        digest = blake2b(f"{self.salt}:{run_id}".encode(), digest_size=8).digest()
        position = int.from_bytes(digest, "big") / float(1 << 64)
        return Arm.HOLDOUT if position < self.fraction else Arm.SHAPED


class ShapedRun(AdkModel):
    """One run's token accounting, and which arm it was in.

    Args:
        run_id: The run. Ids only, so the record cannot become a copy of the conversation.
        tenant: Whose traffic it was. Figures never cross this boundary.
        arm: Which arm it was assigned to, recorded even where shaping did nothing.
        input_tokens_before: What the prompt would have cost unshaped.
        input_tokens_after: What it cost as sent.
        output_tokens: What the response cost. There is no `before` for this one, which is
            the whole reason a holdout exists.
    """

    run_id: str = Field(min_length=1)
    tenant: str = Field(min_length=1)
    arm: Arm = Arm.SHAPED
    input_tokens_before: int = Field(ge=0)
    input_tokens_after: int = Field(ge=0)
    output_tokens: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def _shaping_does_not_add(self) -> ShapedRun:
        """Refuse a record where the prompt grew, which is a counting bug, not a saving."""
        if self.input_tokens_after > self.input_tokens_before:
            raise ConfigurationError("a shaped prompt cannot grow, so these counts disagree")
        return self

    @property
    def input_saved(self) -> int:
        """Tokens shaping kept out of the prompt. Zero for a holdout run, by definition."""
        if self.arm is Arm.HOLDOUT:
            return 0
        return self.input_tokens_before - self.input_tokens_after


class Figure(AdkModel):
    """A savings number and how it came to exist.

    Args:
        tokens: The figure itself.
        basis: Measured, estimated, or not supported by enough data to state.
        low: The bottom of the interval, where there is one.
        high: The top of it.
        reason: Why the basis is what it is, where it is not a plain measurement.
    """

    tokens: float = 0.0
    basis: Basis = Basis.MEASURED
    low: float | None = None
    high: float | None = None
    reason: str = ""

    def label(self) -> str:
        """What a dashboard prints beside the number, basis first.

        Example:
            >>> Figure(tokens=1200.0).label()
            '1,200 tokens (measured)'
        """
        interval = ""
        if self.low is not None and self.high is not None:
            interval = f" [{self.low:,.0f} to {self.high:,.0f}]"
        note = f" — {self.reason}" if self.reason else ""
        return f"{self.tokens:,.0f} tokens{interval} ({self.basis.value}){note}"


class SavingsReport(AdkModel):
    """What shaping saved over a window, with each figure labelled at the point of display.

    Args:
        runs: How many records are behind it.
        shaped: How many were shaped.
        holdout: How many bypassed shaping.
        input: The measured input saving.
        output: The output saving, which is an estimate or an admission that there is none.
    """

    runs: int = Field(default=0, ge=0)
    shaped: int = Field(default=0, ge=0)
    holdout: int = Field(default=0, ge=0)
    input: Figure = Figure()
    output: Figure = Figure()

    def summary(self) -> str:
        """One line carrying both figures and both bases."""
        return (
            f"{self.runs} runs ({self.shaped} shaped, {self.holdout} holdout): "
            f"input {self.input.label()}; output {self.output.label()}"
        )

    def as_dict(self) -> dict[str, object]:
        """The machine-readable form, with the basis travelling beside every number."""
        return {
            "runs": self.runs,
            "shaped": self.shaped,
            "holdout": self.holdout,
            "input": self.input.model_dump(mode="json"),
            "output": self.output.model_dump(mode="json"),
        }


def account(
    runs: Iterable[ShapedRun], *, policy: HoldoutPolicy, minimum: int = MINIMUM_ARM
) -> SavingsReport:
    """Total what shaping saved, measuring the input and comparing the arms for the output.

    Args:
        runs: The accounted runs. One record per run.
        policy: The holdout in force, which decides whether an output figure can exist.
        minimum: How many runs each arm needs before the comparison is reported at all.

    Returns:
        The report, with the input saving measured and the output saving either estimated
        from the holdout comparison or declared unsupported.

    Raises:
        ConfigurationError: A run appears twice, which would double-count its saving.
    """
    records = tuple(runs)
    ids = [record.run_id for record in records]
    if len(ids) != len(set(ids)):
        raise ConfigurationError("savings accounting takes one record per run")
    shaped = [record for record in records if record.arm is Arm.SHAPED]
    held = [record for record in records if record.arm is Arm.HOLDOUT]
    return SavingsReport(
        runs=len(records),
        shaped=len(shaped),
        holdout=len(held),
        input=Figure(tokens=float(sum(record.input_saved for record in records))),
        output=_output(shaped, held, policy=policy, minimum=minimum),
    )


def by_tenant(
    runs: Iterable[ShapedRun], *, policy: HoldoutPolicy, minimum: int = MINIMUM_ARM
) -> dict[str, SavingsReport]:
    """The same accounting, per tenant, with no figure drawing on another tenant's traffic.

    Args:
        runs: The accounted runs, of any number of tenants.
        policy: The holdout in force.
        minimum: How many runs each arm needs per tenant before its output figure exists.

    Returns:
        One report per tenant present, keyed by tenant.
    """
    grouped: dict[str, list[ShapedRun]] = {}
    for record in runs:
        grouped.setdefault(record.tenant, []).append(record)
    return {
        tenant: account(records, policy=policy, minimum=minimum)
        for tenant, records in grouped.items()
    }


def _output(
    shaped: Sequence[ShapedRun],
    held: Sequence[ShapedRun],
    *,
    policy: HoldoutPolicy,
    minimum: int,
) -> Figure:
    """The output figure, which is only ever an estimate or a statement that there is none."""
    if not policy.controlled:
        return Figure(
            tokens=0.0,
            basis=Basis.ESTIMATED,
            reason="no holdout is configured, so no output saving can be attributed to shaping",
        )
    if len(shaped) < minimum or len(held) < minimum:
        return Figure(
            tokens=0.0,
            basis=Basis.INSUFFICIENT,
            reason=(
                f"each arm needs {minimum} runs to compare; "
                f"{len(shaped)} shaped and {len(held)} holdout"
            ),
        )
    control = _mean(held)
    treated = _mean(shaped)
    per_run = control - treated
    error = sqrt(_variance(held) / len(held) + _variance(shaped) / len(shaped))
    margin = _INTERVAL * error * len(shaped)
    saved = per_run * len(shaped)
    return Figure(
        tokens=saved,
        basis=Basis.ESTIMATED,
        low=saved - margin,
        high=saved + margin,
        reason=f"from {len(held)} holdout runs against {len(shaped)} shaped",
    )


def _mean(records: Sequence[ShapedRun]) -> float:
    """Mean output tokens over an arm."""
    return sum(record.output_tokens for record in records) / len(records)


def _variance(records: Sequence[ShapedRun]) -> float:
    """Sample variance of an arm's output tokens, floored so no arm reports false certainty."""
    if len(records) < 2:
        return 1.0
    average = _mean(records)
    spread = sum((record.output_tokens - average) ** 2 for record in records)
    return max(spread / (len(records) - 1), 1.0)
