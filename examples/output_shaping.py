"""Effort clamped where deliberation is not needed, and steering that does not rewrite.

Four calls in one run: a new question, a turn resuming a tool result, a turn recovering from
a failed one, and the final answer. Then the terseness suffix, applied twice.

Run it with `python examples/output_shaping.py`.
"""

from __future__ import annotations

from tesserix_adk.core import Message, ModelRequest, TextPart, ToolCall
from tesserix_adk.models import Effort, Shaping, errored, provider_effort

SYSTEM = Message(role="system", content=[TextPart(text="You answer from tool output only.")])
ASKED = Message(role="user", content=[TextPart(text="Which hosts are unreachable?")])
CALLED = Message(
    role="assistant",
    content=[TextPart(text="")],
    tool_calls=(ToolCall(id="c-1", name="hosts", arguments={}),),
)
RETURNED = Message(role="tool", tool_call_id="c-1", content=[TextPart(text="node-004")])


def main() -> None:
    """Shape four turns of one run, then steer the prompt without rewriting it."""
    shaping = Shaping(
        enabled=True,
        baseline=Effort.HIGH,
        resumption=Effort.LOW,
        terseness="Answer directly. No preamble.",
    )

    turns = {
        "new question": (ModelRequest(model="gpt-5", messages=(SYSTEM, ASKED)), False),
        "tool resumption": (
            ModelRequest(model="gpt-5", messages=(SYSTEM, ASKED, CALLED, RETURNED)),
            False,
        ),
        "error recovery": (
            ModelRequest(model="gpt-5", messages=(SYSTEM, ASKED, CALLED, errored(RETURNED))),
            False,
        ),
        "final answer": (
            ModelRequest(model="gpt-5", messages=(SYSTEM, ASKED, CALLED, RETURNED)),
            True,
        ),
    }
    for label, (request, final) in turns.items():
        shaped = shaping.shape(request, effort=Effort.HIGH, final=final)
        print(  # noqa: T201
            f"{label}: {shaped.requested} -> {shaped.effort} ({shaped.reason}); "
            f"openai sends {provider_effort(shaped.effort, provider='openai')}"
        )

    print(f"llama.cpp has no such parameter: {provider_effort(Effort.LOW, provider='llama_cpp')}")  # noqa: T201

    steered = shaping.steer(ModelRequest(model="gpt-5", messages=(SYSTEM, ASKED)))
    print(f"system prompt now: {[part.text for part in steered.messages[0].content]}")  # noqa: T201
    print(f"steering again changes nothing: {shaping.steer(steered) == steered}")  # noqa: T201


if __name__ == "__main__":
    main()
