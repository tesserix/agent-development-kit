"""A support message with too much in it, and the same transcript at two tenants.

Run it with `uv run python examples/pii_redaction.py`.
"""

from __future__ import annotations

import asyncio

from tesserix_adk.core import (
    ContentCategory,
    ContentSeverity,
    PIIKind,
    Thresholds,
    placeholder,
    redact,
)
from tesserix_adk.guardrails import ContentFilterGuard, PIIGuard
from tesserix_adk.memory.erasure import PIIRedactor

MESSAGE = (
    "I'm Ada, ada@example.test, booking ref X1234567. "
    "Charge card 4111 1111 1111 1111 and call me on +44 20 7946 0958."
)
TRANSCRIPT = "the customer says you should kill him"


async def main() -> None:
    """Redact a message, keep a booking reference, and split one transcript two ways."""
    guard = PIIGuard(tenant="acme")

    result = await guard.check_input(MESSAGE)
    print(f"verdict: {result.verdict.value} — {result.detail}")  # noqa: T201
    print(result.content)  # noqa: T201

    same = redact(MESSAGE, tenant="acme").text == redact(MESSAGE, tenant="acme").text
    other = redact(MESSAGE, tenant="acme").text == redact(MESSAGE, tenant="globex").text
    print(f"\nstable within a tenant: {same}; shared across tenants: {other}")  # noqa: T201

    default = redact(MESSAGE, tenant="acme").kinds
    raised = redact(MESSAGE, tenant="acme", threshold=0.9).kinds
    allowed = redact(MESSAGE, tenant="acme", allow=(r"X\d{7}",)).kinds
    print(f"\nat the default bar the booking ref is a passport: {default}")  # noqa: T201
    print(f"at 0.9 the ambiguous shapes go: {raised}")  # noqa: T201
    print(f"or the tenant says that one shape is not an identifier: {allowed}")  # noqa: T201

    stored, paths = PIIRedactor(tenant="acme").redact({"turns": [{"text": MESSAGE}]})
    print(f"\nstored with {paths} stood in for: {stored}")  # noqa: T201

    subject = placeholder(PIIKind.EMAIL, "ada@example.test", tenant="acme")
    print(f"the same traveller next week is still {subject}")  # noqa: T201

    facing = ContentFilterGuard(thresholds=Thresholds(default=ContentSeverity.MEDIUM))
    triage = ContentFilterGuard(
        thresholds=Thresholds(per_category={ContentCategory.VIOLENCE: ContentSeverity.HIGH})
    )
    print(f"\ncustomer-facing: {(await facing.check_output(TRANSCRIPT)).verdict.value}")  # noqa: T201
    print(f"triage: {(await triage.check_output(TRANSCRIPT)).verdict.value}")  # noqa: T201


if __name__ == "__main__":
    asyncio.run(main())
