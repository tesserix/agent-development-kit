"""W3C trace context across a peer hop, a queue and a workflow that suspends for a day.

A multi-agent durable flow otherwise produces one disconnected trace per participant and a
second set for the workflow, with nothing linking them: investigating a slow outcome means
correlating timestamps by hand across products. `W3CContext` is the standard header pair
that stops that, so a non-kit peer joins the same trace without knowing anything about the
kit.

Nothing here reads a clock or a random source. Identifiers come from the run id, so a
workflow replayed after a worker restart rebuilds exactly the trace the first attempt did
rather than a second one alongside it.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping  # noqa: TC003 — pydantic needs the runtime type
from enum import StrEnum
from typing import Self

from pydantic import Field, field_validator

from tesserix_adk.core import AdkModel
from tesserix_adk.observability.attribution import ATTRIBUTE_PREFIX

__all__ = [
    "CORRELATION",
    "MAX_STATE_ENTRIES",
    "MAX_STATE_LENGTH",
    "TRACEPARENT",
    "TRACESTATE",
    "VENDOR",
    "Link",
    "LinkKind",
    "W3CContext",
    "joined",
    "span_id_of",
    "trace_id_of",
]

TRACEPARENT = "traceparent"
TRACESTATE = "tracestate"
CORRELATION = "adk-correlation"
"""Header a caller sets so a hop that loses the trace still says who called."""

VENDOR = "adk"
MAX_STATE_ENTRIES = 32
MAX_STATE_LENGTH = 512
"""The spec's tracestate limits. A transport that truncates a header is a lost trace."""

_TRACE_ID_LENGTH = 32
_SPAN_ID_LENGTH = 16
_HEX = "0123456789abcdef"


def _digest(*parts: str, length: int) -> str:
    """A stable identifier from what the run already knows, never from a clock."""
    digest = hashlib.sha256("|".join(parts).encode()).hexdigest()[:length]
    return digest if set(digest) != {"0"} else "1" * length


def trace_id_of(run_id: str) -> str:
    """The trace id for a run, the same on every replay of it."""
    return _digest("trace", run_id, length=_TRACE_ID_LENGTH)


def span_id_of(run_id: str, name: str = "", attempt: int = 1) -> str:
    """The span id for one operation in a run, the same on every replay of it."""
    return _digest("span", run_id, name, str(attempt), length=_SPAN_ID_LENGTH)


class LinkKind(StrEnum):
    """Why two spans are related where parent-child would misstate the causality.

    Args:
        FOLLOWS_FROM: A continuation whose parent has already ended, such as a queue
            message delivered hours later.
        FAN_OUT: One branch of a parallel fan-out.
        CONTINUATION: Work resumed after a suspension, such as a human-gated workflow.
        BROKEN: A hop that arrived without a readable context, linked back by correlation
            id so the gap is visible rather than looking like an unrelated trace.
    """

    FOLLOWS_FROM = "follows_from"
    FAN_OUT = "fan_out"
    CONTINUATION = "continuation"
    BROKEN = "broken"


class Link(AdkModel):
    """A relationship to another span that is not this span's parent.

    Args:
        trace_id: The trace linked to.
        span_id: The span linked to.
        kind: Why they are related.
        attributes: Anything else worth carrying, such as the branch name.
    """

    trace_id: str
    span_id: str
    kind: LinkKind
    attributes: Mapping[str, str] = {}

    def rendered(self) -> dict[str, str]:
        """The link as attributes, for a backend with no first-class link type."""
        return {
            f"{ATTRIBUTE_PREFIX}link.trace_id": self.trace_id,
            f"{ATTRIBUTE_PREFIX}link.span_id": self.span_id,
            f"{ATTRIBUTE_PREFIX}link.kind": str(self.kind),
            **{_qualified(name): value for name, value in self.attributes.items()},
        }


class W3CContext(AdkModel):
    """Where this process sits in a trace, in the format every backend already speaks.

    Args:
        trace_id: The trace, shared by every participant in the causal chain.
        span_id: This process's current span.
        parent_span_id: The span that called it, where a hop was made.
        sampled: Whether the trace is being recorded. Carried rather than re-decided,
            because a peer deciding for itself is how a trace arrives truncated.
        state: Vendor entries travelling alongside, this vendor's first.

    Example:
        >>> W3CContext.rooted("run-1").carried()["traceparent"].endswith("-01")
        True
    """

    trace_id: str = Field(min_length=_TRACE_ID_LENGTH, max_length=_TRACE_ID_LENGTH)
    span_id: str = Field(min_length=_SPAN_ID_LENGTH, max_length=_SPAN_ID_LENGTH)
    parent_span_id: str | None = None
    sampled: bool = True
    state: Mapping[str, str] = {}

    @field_validator("trace_id", "span_id")
    @classmethod
    def _hexadecimal(cls, value: str) -> str:
        if set(value) - set(_HEX) or set(value) == {"0"}:
            message = "an identifier is lowercase hexadecimal and never all zeroes"
            raise ValueError(message)
        return value

    @classmethod
    def rooted(cls, run_id: str, *, sampled: bool = True) -> Self:
        """The context for a run this process started.

        Args:
            run_id: The run. The identifiers derive from it, so a replay rebuilds them.
            sampled: Whether to record the trace.
        """
        return cls(trace_id=trace_id_of(run_id), span_id=span_id_of(run_id), sampled=sampled)

    def child(self, name: str, *, attempt: int = 1) -> Self:
        """The context for an operation inside this one.

        Args:
            name: What the operation is, such as an activity name.
            attempt: Which try. A retry is a sibling span, never a reopened one.
        """
        return self.model_copy(
            update={
                "span_id": span_id_of(self.span_id, name, attempt),
                "parent_span_id": self.span_id,
            }
        )

    def link(self, kind: LinkKind, **attributes: str) -> Link:
        """A link back to this context, for a continuation this cannot parent."""
        return Link(trace_id=self.trace_id, span_id=self.span_id, kind=kind, attributes=attributes)

    def carried(self) -> dict[str, str]:
        """The headers to put on the hop, as any W3C-aware participant expects them."""
        flags = "01" if self.sampled else "00"
        return {
            TRACEPARENT: f"00-{self.trace_id}-{self.span_id}-{flags}",
            TRACESTATE: _rendered_state(self.state),
        }

    @classmethod
    def extracted(cls, headers: Mapping[str, str]) -> Self | None:
        """The context a hop carried, or None where nothing readable arrived.

        Args:
            headers: What arrived, read case-insensitively because a broker that
                normalises header case has not broken the trace.

        Returns:
            The caller's context, or None. None is not an error: an older service that
            never sent one is the case this has to survive.
        """
        parent = _header(headers, TRACEPARENT)
        fields = parent.split("-")
        if len(fields) < 4:
            return None
        _, trace_id, span_id, flags = fields[:4]
        if not _identifier(trace_id, _TRACE_ID_LENGTH) or not _identifier(span_id, _SPAN_ID_LENGTH):
            return None
        return cls(
            trace_id=trace_id,
            span_id=span_id,
            sampled=flags.endswith(("1", "3", "5", "7", "9", "b", "d", "f")),
            state=_parsed_state(_header(headers, TRACESTATE)),
        )


def joined(
    headers: Mapping[str, str], *, run_id: str, sampled: bool = True
) -> tuple[W3CContext, Link | None]:
    """Attach to the trace a hop carried, or start one and say the chain broke.

    Args:
        headers: What arrived with the message.
        run_id: The run this side is about to start.
        sampled: What to decide only where nothing arrived to be honoured.

    Returns:
        The context to run under, and a link back to the caller where the chain broke.
        A break is recorded rather than raised: refusing the work because its trace went
        missing costs the work as well as the trace.
    """
    upstream = W3CContext.extracted(headers)
    if upstream is not None:
        return upstream.child(run_id), None
    started = W3CContext.rooted(run_id, sampled=sampled)
    correlation = _header(headers, CORRELATION)
    if not correlation:
        return started, None
    return started, Link(
        trace_id=trace_id_of(correlation),
        span_id=span_id_of(correlation),
        kind=LinkKind.BROKEN,
        attributes={f"{ATTRIBUTE_PREFIX}correlation_id": correlation},
    )


def _qualified(name: str) -> str:
    """A link attribute under the link namespace, without prefixing an already-prefixed name."""
    return name if name.startswith(ATTRIBUTE_PREFIX) else f"{ATTRIBUTE_PREFIX}link.{name}"


def _header(headers: Mapping[str, str], name: str) -> str:
    """One header, case-insensitively."""
    return next((value for key, value in headers.items() if key.lower() == name), "")


def _identifier(value: str, length: int) -> bool:
    """Whether a field is an identifier the spec would accept."""
    return len(value) == length and not set(value) - set(_HEX) and set(value) != {"0"}


def _parsed_state(state: str) -> dict[str, str]:
    """The vendor entries in a tracestate, in the order they arrived."""
    entries = (entry.partition("=") for entry in state.split(",") if entry)
    return {key.strip(): value for key, _, value in entries if key.strip() and value}


def _rendered_state(state: Mapping[str, str]) -> str:
    """This vendor first, then as many of the rest as the spec's limits allow."""
    entries = [f"{VENDOR}={VENDOR}"] + [
        f"{key}={value}" for key, value in state.items() if key != VENDOR
    ]
    kept: list[str] = []
    for entry in entries[:MAX_STATE_ENTRIES]:
        if len(",".join([*kept, entry])) > MAX_STATE_LENGTH:
            break
        kept.append(entry)
    return ",".join(kept)
