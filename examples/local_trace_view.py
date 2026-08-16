"""A run that timed out after two retries, read without a collector.

Run it with `uv run python examples/local_trace_view.py`.
"""

from __future__ import annotations

from tesserix_adk.observability import RecordedSpan, TraceFile, assembled, rendered


def _step(span_id: str, name: str, started: float, ended: float, **attributes: str) -> RecordedSpan:
    """One exported span, as the pipeline would receive it."""
    return RecordedSpan(
        span_id=span_id,
        parent_span_id=None if span_id == "root" else "root",
        name=name,
        started=started,
        ended=ended,
        attributes={"adk.tenant": "acme", **attributes},
    )


def recorded() -> tuple[RecordedSpan, ...]:
    """A refund run whose payment tool timed out on every attempt."""
    return (
        _step("root", "adk.run", 0.0, 9.4, **{"adk.outcome": "failed"}),
        _step("guard", "adk.guard", 0.0, 0.1, **{"adk.verdict": "allowed"}),
        _step("model", "adk.model", 0.1, 0.4, **{"adk.input_tokens": "812", "adk.cost": "0.0041"}),
        _step("try-1", "adk.tool", 0.4, 3.4, **{"adk.attempt": "1"}),
        _step("try-2", "adk.tool", 3.4, 6.4, **{"adk.attempt": "2"}),
        _step(
            "try-3",
            "adk.tool",
            6.4,
            9.4,
            **{"adk.attempt": "3", "adk.error.type": "ToolTimeout", "adk.outcome": "failed"},
        ),
        _step("leaky", "adk.memory", 9.4, 9.4, **{"http.authorization": "Bearer opaque"}),
    )


def main() -> None:
    """The whole run, a narrowed view, and the file that could be attached to a report."""
    print(rendered(assembled(recorded())))  # noqa: T201

    print("narrowed to model calls, failure kept anyway:")  # noqa: T201
    print(rendered(assembled(recorded()), only=("adk.model",)))  # noqa: T201

    shared = TraceFile.of(recorded())
    print(f"file version: {shared.version}")  # noqa: T201
    print(f"dropped before sharing: {list(shared.redaction.dropped)}")  # noqa: T201
    print(f"secret in the file: {'opaque' in shared.model_dump_json()}")  # noqa: T201


if __name__ == "__main__":
    main()
