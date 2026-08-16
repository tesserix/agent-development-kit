"""A run read without a collector, and a trace file safe to attach to a bug report.

Iterating locally, a developer either stands up a trace backend or debugs by print. The
telemetry is already there; what is missing is a way to read it. This renders the spans a
run emits as a tree — which tools ran in what order, what the guards decided, where the
time and the tokens went.

It reads the exported attribute set rather than one of its own, so a local view cannot
drift into disagreeing with what production shows. Saved files go through the export
redaction first, because a trace file is attached to bug reports and shared.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence  # noqa: TC003 — pydantic needs the runtime types
from typing import Self

from pydantic import Field, field_validator

from tesserix_adk.core import AdkModel
from tesserix_adk.observability.attribution import ATTRIBUTE_PREFIX
from tesserix_adk.observability.export import PendingSpan, RedactingSpanProcessor, RedactionPolicy
from tesserix_adk.observability.redaction import Redaction

__all__ = [
    "FILE_VERSION",
    "ORPHAN",
    "STOPPED",
    "RecordedSpan",
    "TraceFile",
    "TraceNode",
    "assembled",
    "machine_readable",
    "rendered",
]

FILE_VERSION = "adk-trace/1"
"""Format of a saved trace file. A file from a later version is refused, not guessed at."""

ORPHAN = "?"
STOPPED = "!"
"""Marks for a span whose parent never arrived and for the step a run stopped at."""

_INDENT = "  "
_FAILED = frozenset({"cancelled", "failed", "refused"})
_INTERESTING = (
    f"{ATTRIBUTE_PREFIX}attempt",
    f"{ATTRIBUTE_PREFIX}outcome",
    f"{ATTRIBUTE_PREFIX}error.type",
    f"{ATTRIBUTE_PREFIX}verdict",
    f"{ATTRIBUTE_PREFIX}budget",
    f"{ATTRIBUTE_PREFIX}input_tokens",
    f"{ATTRIBUTE_PREFIX}output_tokens",
    f"{ATTRIBUTE_PREFIX}cost",
)
"""What a developer reading a failure needs on the line, in the order they need it."""


class RecordedSpan(AdkModel):
    """One step of a run as it was exported.

    Args:
        span_id: This step.
        parent_span_id: The step that called it, where there was one.
        name: The span name, from the telemetry convention.
        started: Unix timestamp the step began.
        ended: Unix timestamp it finished.
        attributes: The exported attributes, unchanged from what the pipeline receives.
    """

    span_id: str = Field(min_length=1)
    parent_span_id: str | None = None
    name: str = Field(min_length=1)
    started: float = 0.0
    ended: float = 0.0
    attributes: Mapping[str, str] = {}

    @property
    def duration(self) -> float:
        """How long the step took, never negative where a clock stepped backwards."""
        return max(self.ended - self.started, 0.0)

    @property
    def failed(self) -> bool:
        """Whether this step is where a reader should look first."""
        outcome = self.attributes.get(f"{ATTRIBUTE_PREFIX}outcome", "")
        return outcome in _FAILED or f"{ATTRIBUTE_PREFIX}error.type" in self.attributes


class TraceNode(AdkModel):
    """A step and what it called.

    Args:
        span: The step itself.
        children: What it called, in the order those ran.
        orphaned: Whether its parent never arrived, so its position is a guess.
    """

    span: RecordedSpan
    children: tuple[TraceNode, ...] = ()
    orphaned: bool = False

    @property
    def failing(self) -> bool:
        """Whether this step or anything under it failed."""
        return self.span.failed or any(child.failing for child in self.children)


class TraceFile(AdkModel):
    """A trace saved to disk, redacted so it can be attached to a bug report.

    Args:
        version: The format that produced it.
        spans: The redacted spans.
        redaction: Which attributes were dropped, so a gap reads as a decision.
    """

    version: str = FILE_VERSION
    spans: tuple[RecordedSpan, ...] = ()
    redaction: Redaction = Redaction()

    @field_validator("version")
    @classmethod
    def _readable(cls, value: str) -> str:
        if value != FILE_VERSION:
            message = f"this file states version {value}, which this build cannot read"
            raise ValueError(message)
        return value

    @classmethod
    def of(cls, spans: Sequence[RecordedSpan], *, policy: RedactionPolicy | None = None) -> Self:
        """Redact `spans` and wrap them for sharing.

        Args:
            spans: What the run recorded.
            policy: What this deployment is willing to write to a file.

        Returns:
            A file whose contents have been through the same processor the export path
            uses, so sharing one cannot leak what exporting one would not.
        """
        processor = RedactingSpanProcessor(policy)
        dropped: list[str] = []
        redacted: list[RecordedSpan] = []
        for span in spans:
            exported = processor.process(
                PendingSpan(name=span.name, attributes=dict(span.attributes))
            )
            dropped.extend(exported.redaction.dropped)
            redacted.append(span.model_copy(update={"attributes": exported.attributes}))
        return cls(spans=tuple(redacted), redaction=Redaction(dropped=tuple(sorted(set(dropped)))))


def assembled(spans: Sequence[RecordedSpan]) -> tuple[TraceNode, ...]:
    """Build the forest a run's spans describe.

    Args:
        spans: The steps, in any order and possibly incomplete.

    Returns:
        The roots. A span whose parent never arrived becomes a marked root rather than
        being dropped, because the missing step is often the one being looked for.
    """
    ordered = sorted(spans, key=lambda span: (span.started, span.span_id))
    known = {span.span_id for span in ordered}
    children: dict[str, list[RecordedSpan]] = {}
    roots: list[tuple[RecordedSpan, bool]] = []
    for span in ordered:
        parent = span.parent_span_id
        if parent is None:
            roots.append((span, False))
        elif parent in known and parent != span.span_id:
            children.setdefault(parent, []).append(span)
        else:
            roots.append((span, True))
    reachable = _reachable({span.span_id for span, _ in roots}, children)
    for span in ordered:
        if span.span_id in reachable:
            continue
        roots.append((span, True))
        reachable |= _reachable({span.span_id}, children)
    return tuple(_node(span, children, orphaned, set()) for span, orphaned in roots)


def rendered(
    nodes: Sequence[TraceNode], *, depth: int | None = None, only: Sequence[str] = ()
) -> str:
    """Draw a trace for a person to read.

    Args:
        nodes: The roots to draw.
        depth: How many levels to show. Deeper steps are counted, not dropped silently.
        only: Span names to keep. A failing step is kept whatever this says, because a
            filtered view that looks clean is how a failure gets missed.

    Returns:
        The tree as text, one step per line.
    """
    if not nodes:
        return "no spans recorded\n"
    lines: list[str] = []
    for node in nodes:
        _draw(node, lines, level=0, depth=depth, only=tuple(only))
    return "".join(lines)


def machine_readable(nodes: Sequence[TraceNode]) -> str:
    """The same tree as JSON, for a test assertion rather than an eye."""
    return json.dumps([_document(node) for node in nodes])


def _reachable(roots: set[str], children: Mapping[str, list[RecordedSpan]]) -> set[str]:
    """Every span a root can reach, so a cycle is found rather than walked forever."""
    seen: set[str] = set()
    pending = list(roots)
    while pending:
        current = pending.pop()
        if current in seen:
            continue
        seen.add(current)
        pending.extend(child.span_id for child in children.get(current, ()))
    return seen


def _node(
    span: RecordedSpan,
    children: Mapping[str, list[RecordedSpan]],
    orphaned: bool,
    seen: set[str],
) -> TraceNode:
    """One node and its descendants, refusing to revisit a span a cycle points back to."""
    seen = seen | {span.span_id}
    below = tuple(
        _node(child, children, orphaned=False, seen=seen)
        for child in children.get(span.span_id, ())
        if child.span_id not in seen
    )
    return TraceNode(span=span, children=below, orphaned=orphaned)


def _draw(
    node: TraceNode, lines: list[str], *, level: int, depth: int | None, only: tuple[str, ...]
) -> None:
    """Append one node's line and its children's, honouring the filters."""
    if depth is not None and level >= depth:
        lines.append(f"{_INDENT * level}... {_counted(node)} hidden\n")
        return
    if _shown(node, only):
        lines.append(f"{_INDENT * level}{_line(node)}\n")
        level += 1
    for child in node.children:
        _draw(child, lines, level=level, depth=depth, only=only)


def _shown(node: TraceNode, only: tuple[str, ...]) -> bool:
    """Whether a filtered view keeps this step. A failure is never filtered away."""
    return not only or node.span.name in only or node.failing


def _counted(node: TraceNode) -> int:
    """How many steps a depth cap is hiding, including this one."""
    return 1 + sum(_counted(child) for child in node.children)


def _line(node: TraceNode) -> str:
    """One step: what it was, how long it took, and what it reported."""
    marks = f"{STOPPED} " if node.span.failed else f"{ORPHAN} " if node.orphaned else ""
    facts = " ".join(
        f"{name.removeprefix(ATTRIBUTE_PREFIX)}={node.span.attributes[name]}"
        for name in _INTERESTING
        if name in node.span.attributes
    )
    return f"{marks}{node.span.name} {node.span.duration:.3f}s {facts}".rstrip()


def _document(node: TraceNode) -> dict[str, object]:
    """One node as plain data, carrying the exported attributes unchanged."""
    return {
        "span_id": node.span.span_id,
        "name": node.span.name,
        "duration": node.span.duration,
        "orphaned": node.orphaned,
        "failed": node.span.failed,
        "attributes": dict(node.span.attributes),
        "children": [_document(child) for child in node.children],
    }
