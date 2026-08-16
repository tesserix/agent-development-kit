"""One trace across a peer hop, a broken hop, a retry and a day-long suspension.

Run it with `uv run python examples/trace_propagation.py`.
"""

from __future__ import annotations

from tesserix_adk.observability import (
    CORRELATION,
    TRACEPARENT,
    LinkKind,
    W3CContext,
    joined,
)

PEER = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"


def main() -> None:
    """What each participant sees of the same trace."""
    caller = W3CContext.rooted("run-1")
    print(f"caller: {caller.carried()[TRACEPARENT]}")  # noqa: T201

    callee, _ = joined(caller.carried(), run_id="run-2")
    print(f"callee joined the same trace: {callee.trace_id == caller.trace_id}")  # noqa: T201
    print(f"callee parents the calling span: {callee.parent_span_id == caller.span_id}")  # noqa: T201

    other_vendor = W3CContext.extracted({TRACEPARENT: PEER})
    print(f"\na non-kit peer is understood: {other_vendor is not None}")  # noqa: T201

    replayed = W3CContext.rooted("run-1")
    print(f"a replay rebuilds the trace: {replayed.trace_id == caller.trace_id}")  # noqa: T201
    first, second = caller.child("charge", attempt=1), caller.child("charge", attempt=2)
    print(f"a retry is a sibling span: {first.span_id != second.span_id}")  # noqa: T201

    resumed = caller.link(LinkKind.CONTINUATION, waited="approval")
    print(f"\nafter a day-long wait: {resumed.rendered()['adk.link.kind']}")  # noqa: T201

    orphan, broken = joined({TRACEPARENT: "nonsense", CORRELATION: "run-1"}, run_id="run-9")
    print(f"a broken hop still runs: {orphan.trace_id[:8]}...")  # noqa: T201
    print(f"and links back: {broken.rendered() if broken else None}")  # noqa: T201


if __name__ == "__main__":
    main()
