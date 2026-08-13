"""The whole multi-agent run as one tree, with the money attributed node by node.

A total taken from whichever process held the ledger is a total that quietly excludes every
participant that reported somewhere else. So this totals what was reported, names what was
not, and marks the figure a lower bound when anything is missing — an unreported worker's
spend is unknown, and treating unknown as zero is how a budget ceiling stops meaning
anything.

Nothing here converts money on its own authority. A peer reporting in another currency is
summed only through a rate somebody recorded; without one it is held as unattributed, which
is a smaller lie than a number that is true in neither currency.
"""

from __future__ import annotations

from decimal import Decimal  # noqa: TC003 — pydantic resolves field types at class creation
from typing import TYPE_CHECKING, Self

from pydantic import BaseModel, Field

from tesserix_adk.core import AdkModel, Cost, CostConfidence, Usage
from tesserix_adk.core.errors import AttributionError
from tesserix_adk.observability.attribution import (
    SpendRecord,
    Step,
    spend_of,
    totals_by,
    totals_of,
)
from tesserix_adk.observability.trace import TraceContext  # noqa: TC001 — pydantic needs it

if TYPE_CHECKING:
    from collections.abc import Sequence

    from tesserix_adk.core import Run
    from tesserix_adk.observability.attribution import Totals

__all__ = ["Node", "Rate", "RunTree", "TreeTotals", "node_of", "peer_node", "render", "tree"]


class Rate(AdkModel):
    """A conversion somebody recorded, so a converted figure can be argued with.

    Args:
        source: Where the rate came from — a provider, a treasury feed, a contract.
        recorded_at: When it was taken, as a unix timestamp. A rate without a time is a
            rate nobody can check against the day the spend happened.
        multiplier: What the peer's currency multiplies by to reach the tree's.
        of_currency: The currency being converted from.
        to_currency: The currency being converted to.
    """

    source: str = Field(min_length=1)
    recorded_at: float
    multiplier: Decimal = Field(gt=0)
    of_currency: str = Field(min_length=3)
    to_currency: str = Field(min_length=3)

    def applied(self, cost: Cost) -> Cost | None:
        """`cost` in the target currency, or None where this rate does not cover it."""
        if cost.currency != self.of_currency:
            return None
        return Cost(
            input=cost.input * self.multiplier,
            output=cost.output * self.multiplier,
            cache_read=cost.cache_read * self.multiplier,
            cache_write=cost.cache_write * self.multiplier,
            reasoning=cost.reasoning * self.multiplier,
            image=cost.image * self.multiplier,
            currency=self.to_currency,
            confidence=CostConfidence.ESTIMATED,
        )


class Node(AdkModel):
    """One participant in the run, and what it reported.

    Args:
        context: Where it sits in the trace.
        usage: What it consumed. None where it never reported — a crashed worker, or a
            peer that answered without an envelope.
        cost: What that came to. None where nothing was reported or nothing was priced.
        started_at: When it began, as a unix timestamp from its own clock.
        ended_at: When it ended, on the same clock.
        records: The metered steps behind it, for a local run. Empty for a peer, which
            reports a total rather than its steps.
        rate: The recorded conversion for a figure reported in another currency.
    """

    context: TraceContext
    usage: Usage | None = None
    cost: Cost | None = None
    started_at: float | None = None
    ended_at: float | None = None
    records: tuple[SpendRecord, ...] = ()
    rate: Rate | None = None

    @property
    def reported(self) -> bool:
        """Whether this participant said what it spent."""
        return self.usage is not None

    @property
    def skewed(self) -> bool:
        """Whether its clock disagreed with itself, ending before it started.

        Two processes do not share a clock, and a duration read across them can come out
        negative. That is a fact about the clocks; it is recorded rather than corrected.
        """
        return (
            self.started_at is not None
            and self.ended_at is not None
            and self.ended_at < self.started_at
        )

    @property
    def latency_ms(self) -> float:
        """How long it took, never negative and never guessed.

        Zero where either end is missing or the clock ran backwards, so a tree never shows
        a child that finished before it started.
        """
        if self.started_at is None or self.ended_at is None or self.skewed:
            return 0.0
        return (self.ended_at - self.started_at) * 1000

    def in_currency(self, currency: str) -> Cost | None:
        """This node's cost in `currency`, or None where it cannot be stated in one.

        Args:
            currency: The tree's currency, taken from its root.

        Returns:
            The cost as reported where it already matches, the converted cost where a rate
            covers it, and None where neither — which is how a foreign figure becomes
            unattributed rather than a naive sum.
        """
        if self.cost is None:
            return None
        if self.cost.currency == currency:
            return self.cost
        if self.rate is None or self.rate.to_currency != currency:
            return None
        return self.rate.applied(self.cost)


class TreeTotals(AdkModel):
    """What a whole multi-agent run came to, and what is missing from the figure.

    Args:
        cost: The attributed money, in the root's currency.
        input_tokens: Attributed tokens sent.
        output_tokens: Attributed tokens generated.
        nodes: How many participants there were.
        attributed: The run ids inside `cost`, in tree order.
        unattributed: The run ids outside it — unreported, or reported in a currency no
            recorded rate covers. Named, never folded into the total as zero.
        converted: The run ids that reached the total through a recorded rate.
    """

    cost: Cost
    input_tokens: int
    output_tokens: int
    nodes: int
    attributed: tuple[str, ...] = ()
    unattributed: tuple[str, ...] = ()
    converted: tuple[str, ...] = ()

    @property
    def lower_bound(self) -> bool:
        """Whether the total is a floor rather than the figure.

        A budget decision taken against a lower bound is a decision taken knowing it is
        one, which is the whole point of saying so.
        """
        return bool(self.unattributed)


class RunTree(AdkModel):
    """Every participant of one run, parents before children.

    Args:
        nodes: The participants, root first, each parent before its children.
    """

    nodes: tuple[Node, ...]

    @property
    def root(self) -> Node:
        """The run somebody asked for."""
        return self.nodes[0]

    @property
    def currency(self) -> str:
        """The currency the tree totals in, which is the root's."""
        return self.root.cost.currency if self.root.cost is not None else "USD"

    def node(self, run_id: str) -> Node | None:
        """The participant with this run id, or None."""
        return next((one for one in self.nodes if one.context.run_id == run_id), None)

    def children(self, run_id: str) -> tuple[Node, ...]:
        """Everything this participant called, in the order it was added."""
        return tuple(one for one in self.nodes if one.context.parent_run_id == run_id)

    def records(self) -> tuple[SpendRecord, ...]:
        """Every metered step of every local participant, for `totals_by`.

        A peer contributes no records: it reported a total, and inventing steps to make it
        look like a local run would put a shape on the bill that nobody measured.
        """
        return tuple(record for one in self.nodes for record in one.records)

    @property
    def totals(self) -> TreeTotals:
        """The run total, the participants inside it, and the ones that are not."""
        currency = self.currency
        cost, tokens_in, tokens_out = Cost.nothing(currency), 0, 0
        attributed: list[str] = []
        unattributed: list[str] = []
        converted: list[str] = []
        for one in self.nodes:
            stated = one.in_currency(currency)
            if one.usage is None or stated is None:
                unattributed.append(one.context.run_id)
                continue
            if one.cost is not None and one.cost.currency != currency:
                converted.append(one.context.run_id)
            cost = cost + stated
            tokens_in += one.usage.input_tokens
            tokens_out += one.usage.output_tokens
            attributed.append(one.context.run_id)
        return TreeTotals(
            cost=cost,
            input_tokens=tokens_in,
            output_tokens=tokens_out,
            nodes=len(self.nodes),
            attributed=tuple(attributed),
            unattributed=tuple(unattributed),
            converted=tuple(converted),
        )

    def totals_by(self, *by: str) -> dict[tuple[str, ...], Totals]:
        """Roll the local participants up by attribution dimension.

        Args:
            by: Dimension names as declared on `Attribution` — `agent`, `model`, and any
                other. Step is available through `records`.

        Returns:
            One `Totals` per distinct combination, as `attribution.totals_by` returns.
        """
        return totals_by(self.records(), *by)

    def by_step(self) -> dict[Step, Totals]:
        """What the model calls came to and what the tool calls came to."""
        grouped: dict[Step, list[SpendRecord]] = {}
        for record in self.records():
            grouped.setdefault(record.step, []).append(record)
        return {step: totals_of(group) for step, group in grouped.items()}

    @classmethod
    def of(cls, nodes: Sequence[Node]) -> Self:
        """Assemble a tree, refusing one that cannot be read as a tree.

        Args:
            nodes: The participants. The first is the root.

        Returns:
            The tree, in the order given.

        Raises:
            AttributionError: With `reason` `"empty"` (nothing to assemble), `"no_root"`
                (the first node has a parent), `"two_roots"` (a later node has none),
                `"duplicate"` (one run id twice) or `"orphan"` (a parent that is not in
                the tree). Every one of them would otherwise produce a tree that silently
                drops a participant, and a dropped participant is dropped spend.
        """
        if not nodes:
            raise _refused("a tree with no participants describes no run", reason="empty")
        seen: set[str] = set()
        for index, one in enumerate(nodes):
            context = one.context
            if index == 0 and context.parent_run_id is not None:
                raise _refused(
                    f"the first node {context.run_id!r} names a parent "
                    f"{context.parent_run_id!r}, so it is not the root",
                    reason="no_root",
                    context=context,
                )
            if index > 0 and context.parent_run_id is None:
                raise _refused(
                    f"{context.run_id!r} has no parent, so this is two trees rather than one run",
                    reason="two_roots",
                    context=context,
                )
            if context.run_id in seen:
                raise _refused(
                    f"{context.run_id!r} appears twice; one participant totalled twice is "
                    f"a bill nobody can reconcile",
                    reason="duplicate",
                    context=context,
                )
            if index > 0 and context.parent_run_id not in seen:
                raise _refused(
                    f"{context.run_id!r} was called by {context.parent_run_id!r}, which is "
                    f"not in the tree",
                    reason="orphan",
                    context=context,
                )
            seen.add(context.run_id)
        return cls(nodes=tuple(nodes))


def _refused(message: str, *, reason: str, context: TraceContext | None = None) -> AttributionError:
    return AttributionError(
        message,
        reason=reason,
        run_id=context.run_id if context else None,
        tenant=context.tenant if context else None,
    )


def tree(nodes: Sequence[Node]) -> RunTree:
    """Assemble `nodes` into one tree. See `RunTree.of`."""
    return RunTree.of(nodes)


def node_of[OutputT: BaseModel](run: Run[OutputT], context: TraceContext) -> Node:
    """A participant that ran here, read off its run.

    Args:
        run: The finished run. Nothing on it is mutated.
        context: Where it sits in the trace.

    Returns:
        A node carrying its metered steps, its usage and its own clock's timings.

    Example:
        >>> from tesserix_adk.core import Run
        >>> one = Run(id="r1", tenant="acme", agent_name="planner", agent_version="1", model="m")
        >>> node_of(one, TraceContext.root_of(one)).reported
        True
    """
    return Node(
        context=context,
        usage=run.usage,
        cost=run.usage.cost if run.usage.cost is not None else Cost.nothing(),
        started_at=run.started_at,
        ended_at=run.ended_at,
        records=spend_of(run),
    )


def peer_node(
    context: TraceContext,
    *,
    usage: Usage | None = None,
    cost: Cost | None = None,
    rate: Rate | None = None,
    started_at: float | None = None,
    ended_at: float | None = None,
) -> Node:
    """A participant that ran somewhere else, from what it reported back.

    Args:
        context: Where it sits in the trace, normally `Pattern.PEER`.
        usage: What it said it consumed. None — the default — is a peer that reported
            nothing, which is recorded as unattributed rather than as zero.
        cost: What it said that came to, in whatever currency it bills in.
        rate: The recorded conversion, where it bills in another currency than the root.
        started_at: When it began, on its own clock.
        ended_at: When it ended, on its own clock.

    Returns:
        A node with no records: a peer reports a total, not its steps.

    Example:
        >>> peer_node(TraceContext(root_run_id="r1", run_id="p1", parent_run_id="r1")).reported
        False
    """
    return Node(
        context=context,
        usage=usage,
        cost=cost,
        started_at=started_at,
        ended_at=ended_at,
        rate=rate,
    )


def render(assembled: RunTree) -> str:
    """The tree as text, one line per participant, with cost, tokens and latency.

    Args:
        assembled: The tree to draw.

    Returns:
        A block ending in a newline: the participants indented by depth, then a total that
        says plainly when it is a lower bound.
    """
    lines = [_line(one) for one in assembled.nodes]
    totals = assembled.totals
    summary = (
        f"total {totals.cost.quantised(6).total} {totals.cost.currency} over "
        f"{totals.nodes} participants, "
        f"{totals.input_tokens + totals.output_tokens} tokens"
    )
    if totals.lower_bound:
        summary += f" — lower bound: {', '.join(totals.unattributed)} reported nothing"
    if totals.converted:
        summary += f" — converted at a recorded rate: {', '.join(totals.converted)}"
    return "\n".join([*lines, summary]) + "\n"


def _line(one: Node) -> str:
    """One participant: who, where, what it cost and how long it took."""
    context = one.context
    where = str(context.pattern) + (f"/{context.branch}" if context.branch else "")
    spent = (
        "unreported"
        if one.usage is None or one.cost is None
        else (
            f"{one.cost.quantised(6).total} {one.cost.currency}, "
            f"{one.usage.input_tokens + one.usage.output_tokens} tokens"
        )
    )
    broken = "  [trace broken]" if context.broken else ""
    skewed = "  [clock skew]" if one.skewed else ""
    return (
        f"{'  ' * context.depth}{context.agent or 'unknown'} ({context.run_id})  "
        f"{where}  {spent}, {one.latency_ms:.0f} ms{broken}{skewed}"
    )
