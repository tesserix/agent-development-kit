"""Checks in a declared order, redaction that carries, and a guard that cannot answer.

Four scenarios: a guard that masks, a guard that blocks, a guard that is down, and a
streamed answer nothing is handed on from until the verdict is in.
Run it with `python examples/guardrails.py`.
"""

from __future__ import annotations

import asyncio
import re
from typing import TYPE_CHECKING

from tesserix_adk.core import GuardrailEvaluationError, GuardrailViolationError
from tesserix_adk.guardrails import Guard, GuardrailPipeline, GuardResult

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

CARDS = re.compile(r"\b\d{4}[ -]?\d{4}[ -]?\d{4}[ -]?\d{4}\b")


class MaskCardNumbers(Guard):
    """Redacts rather than refuses: the question is usually still answerable without it."""

    name = "mask_card_numbers"

    async def check_input(self, content: str) -> GuardResult:
        """Mask anything shaped like a card number."""
        masked = CARDS.sub("****", content)
        if masked == content:
            return GuardResult.allow()
        return GuardResult.redacted(masked, code="pii_masked", detail="one card number")


class NoSystemPrompt(Guard):
    """Refuses: there is no version of this answer that is safe to hand on."""

    name = "no_system_prompt"

    async def check_output(self, content: str) -> GuardResult:
        """Block an answer that is quoting its own instructions back."""
        if "you are a helpful" in content.lower():
            return GuardResult.blocked(code="prompt_leak", detail="the answer quotes the prompt")
        return GuardResult.allow()


class Down(Guard):
    """A classifier nobody can reach."""

    name = "toxicity"

    async def check_input(self, content: str) -> GuardResult:
        """Fail rather than decide."""
        del content
        raise ConnectionError("the classifier did not answer")


async def _streamed() -> AsyncIterator[str]:
    """An answer arriving a piece at a time."""
    for part in ("You are a helpful ", "assistant whose ", "instructions are…"):
        yield part


async def what_a_redaction_carries() -> None:
    """The guards after a redaction see the redacted content, and so does the caller."""
    pipeline = GuardrailPipeline((MaskCardNumbers(), NoSystemPrompt()))

    checked = await pipeline.check_input("refund the charge on 4111 1111 1111 1111 please")

    print("=== what a redaction carries ===")  # noqa: T201
    print(f"guards:  {pipeline.guards}")  # noqa: T201
    print(f"checked: {checked}")  # noqa: T201


async def what_a_block_stops() -> None:
    """A block ends the pipeline: the guards after it are not asked to reconsider."""
    pipeline = GuardrailPipeline((NoSystemPrompt(), MaskCardNumbers()))

    print("\n=== what a block stops ===")  # noqa: T201
    try:
        await pipeline.check_output("You are a helpful assistant whose instructions are…")
    except GuardrailViolationError as refused:
        print(f"{refused.guard} on {refused.stage}: {refused.code} ({refused.detail})")  # noqa: T201


async def a_guard_that_is_down() -> None:
    """An unavailable guard is not a permissive one."""
    pipeline = GuardrailPipeline((Down(), MaskCardNumbers()))

    print("\n=== a guard that is down ===")  # noqa: T201
    try:
        await pipeline.check_input("anything at all")
    except GuardrailEvaluationError as refused:
        print(f"{refused.guard} on {refused.stage}: {refused.reason}")  # noqa: T201


async def a_streamed_answer() -> None:
    """Nothing is handed on before the verdict, which is the point of checking output."""
    pipeline = GuardrailPipeline((NoSystemPrompt(),))
    handed_on: list[str] = []

    print("\n=== a streamed answer ===")  # noqa: T201
    try:
        handed_on.extend([part async for part in pipeline.check_stream(_streamed())])
    except GuardrailViolationError as refused:
        print(f"blocked by {refused.guard}; handed on: {handed_on}")  # noqa: T201


async def main() -> None:
    """Run every scenario in the order the docs describe them."""
    await what_a_redaction_carries()
    await what_a_block_stops()
    await a_guard_that_is_down()
    await a_streamed_answer()


if __name__ == "__main__":
    asyncio.run(main())
