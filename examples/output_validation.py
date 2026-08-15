"""A quote that parses but is not allowed, and the one re-ask it is given.

Run it with `uv run python examples/output_validation.py`.
"""

from __future__ import annotations

import asyncio

from pydantic import BaseModel

from tesserix_adk.core import Abstention, Bounded, Invariant, OneOf, OutputValidationError
from tesserix_adk.guardrails import PolicyGuard, SchemaGuard, validated
from tesserix_adk.runtime.structured import OutputContract


class Quote(BaseModel):
    """What the agent was asked for."""

    total: int
    currency: str = "AUD"
    citations: tuple[str, ...] = ()


CONTRACT = OutputContract.of(Quote)
POLICIES = (
    Bounded("total", minimum=10, maximum=100),
    OneOf("currency", ("AUD", "NZD")),
    Invariant[Quote]("quote_is_sourced", lambda quote: bool(quote.citations), "nothing behind it"),
)


async def main() -> None:
    """Reject prose, reject an out-of-band quote, repair once, and take an abstention."""
    schema = SchemaGuard(CONTRACT, abstention=True)
    policy = PolicyGuard(POLICIES)

    try:
        schema.parse("about ninety dollars, I think")
    except OutputValidationError as refused:
        print(f"prose: {refused.model} — attempts {refused.attempts}")  # noqa: T201

    try:
        policy.raise_for(Quote(total=4000, currency="XXX"))
    except OutputValidationError as refused:
        print(f"rules broken: {refused.policies}")  # noqa: T201
        print(f"the value itself is not in the error: {'4000' not in str(refused)}")  # noqa: T201

    asked: list[str] = []

    async def ask_again(correction: str) -> str:
        asked.append(correction)
        return '{"total": 90, "currency": "AUD", "citations": ["rate-card-7"]}'

    answer = await validated(
        '{"total": 4000, "currency": "XXX"}',
        schema=schema,
        policy=policy,
        reask=ask_again,
        attempts=3,
    )
    print(f"\nrepaired after {len(asked)} re-ask: {answer}")  # noqa: T201
    print(f"the correction supplies no value: {'90' not in asked[0]}")  # noqa: T201

    try:
        await validated(
            '{"total": 4000}',
            schema=schema,
            policy=policy,
            reask=lambda correction: _unchanged(correction, asked),
            attempts=2,
        )
    except OutputValidationError as refused:
        print(f"\nstill wrong at the cap: {refused.policies} after {refused.attempts}")  # noqa: T201

    said = await validated(
        '{"abstained": true, "reason": "no rate card for that route"}',
        schema=schema,
        policy=policy,
    )
    assert isinstance(said, Abstention)  # noqa: S101
    print(f"\nabstention is an answer: {said.reason}")  # noqa: T201


async def _unchanged(correction: str, asked: list[str]) -> str:
    """A model that keeps returning the same out-of-band quote."""
    asked.append(correction)
    return '{"total": 4000}'


if __name__ == "__main__":
    asyncio.run(main())
