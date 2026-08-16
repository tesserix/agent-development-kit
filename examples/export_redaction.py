"""A span carrying everything it should not, and what an exporter actually receives.

Run it with `uv run python examples/export_redaction.py`.
"""

from __future__ import annotations

import json

from pydantic import SecretStr

from tesserix_adk.observability import (
    PAYLOAD_ATTRIBUTES,
    PendingSpan,
    RedactingSpanProcessor,
    RedactionPolicy,
    SpanEvent,
)

CARD = "4111 1111 1111 1111"
TOKEN = "sk-live-0123456789abcdef"  # noqa: S105 — a synthetic shape, not a credential


def careless() -> PendingSpan:
    """A span an engineer instrumented in a hurry, on the way to a production incident."""
    return PendingSpan(
        name="adk.tool",
        attributes={
            "adk.tenant": "acme",
            "adk.prompt": f"refund the charge on {CARD} for ada@example.com",
            "adk.tool.arguments": json.dumps({"auth": {"authorization": "Bearer opaque"}}),
            "checkout.key": SecretStr(TOKEN),
            "http.authorization": "Bearer opaque",
        },
        events=(SpanEvent(name="tool.called", attributes={"arg": f"token {TOKEN}"}),),
        exception=f"ValueError: charge failed for {CARD}",
    )


def exploding(value: str) -> bool:
    """A detector whose model server is down, which is how a detector usually fails."""
    message = f"classifier unavailable for {len(value)} characters"
    raise RuntimeError(message)


def main() -> None:
    """The same span exported closed, opened up, and with the detector down."""
    exported = RedactingSpanProcessor().process(careless())
    print(f"prompt kept: {'adk.prompt' in exported.attributes}")  # noqa: T201
    print(f"correlate on instead: {exported.attributes['adk.prompt.ref']}")  # noqa: T201
    print(f"secret str: {exported.attributes['checkout.key']}")  # noqa: T201
    print(f"dropped outright: {exported.redaction.dropped}")  # noqa: T201
    print(f"event: {exported.events[0].attributes['arg']}")  # noqa: T201
    print(f"exception: {exported.exception}")  # noqa: T201

    opened = RedactingSpanProcessor(RedactionPolicy(payload_attributes=PAYLOAD_ATTRIBUTES))
    wide = opened.process(careless())
    print(f"\npayload capture on, still redacted: {wide.attributes['adk.prompt']}")  # noqa: T201
    print(f"nested argument: {wide.attributes['adk.tool.arguments']}")  # noqa: T201

    down = RedactingSpanProcessor(detector=exploding)
    degraded = down.process(careless())
    print(f"\ndetector down, failures={down.stats.failures}, span still exported")  # noqa: T201
    print(f"tenant survived: {degraded.attributes['adk.tenant']}")  # noqa: T201

    leaked = [seed for seed in (CARD, TOKEN) if seed in degraded.model_dump_json()]
    print(f"seeded values that escaped: {leaked}")  # noqa: T201


if __name__ == "__main__":
    main()
