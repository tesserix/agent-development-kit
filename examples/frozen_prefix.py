"""The boundary compression may not cross, and what happens when something crosses it.

Three turns: the first freezes a prompt, the second compresses only what arrived since, and
the third rewrites a frozen instruction block and is caught naming the layer that moved.

Run it with `python examples/frozen_prefix.py`.
"""

from __future__ import annotations

from tesserix_adk.core import Message, PrefixDriftError, TextPart
from tesserix_adk.memory import SECTION, FrozenPrefix, compressed

INSTRUCTIONS = Message(
    role="system",
    content=[TextPart(text="You answer from tool output only.")],
    metadata={SECTION: "instructions"},
)
ASKED = Message(role="user", content=[TextPart(text="Which hosts are unreachable?")])
ANSWERED = Message(role="assistant", content=[TextPart(text="node-004 and node-011.")])
FOLLOWED_UP = Message(role="user", content=[TextPart(text="Since when?")])


def main() -> None:
    """Freeze a turn, compress only below the boundary, then catch a rewritten prefix."""
    prefix = FrozenPrefix()
    turn_one = [INSTRUCTIONS, ASKED]

    prefix.verify(turn_one)
    boundary = prefix.advance(turn_one, turn=1)
    print(f"turn 1 frozen: {boundary.position} messages, {boundary.fingerprint[:12]}…")  # noqa: T201

    turn_two = [*turn_one, ANSWERED, FOLLOWED_UP]
    prefix.verify(turn_two)
    print(f"turn 2 eligible for compression: {prefix.compressible(turn_two)}")  # noqa: T201

    turn_two[2] = compressed(turn_two[2], by="prose")
    print(f"after compressing one of them: {prefix.compressible(turn_two)}")  # noqa: T201
    prefix.advance(turn_two, turn=2)

    turn_three = [*turn_two]
    turn_three[0] = INSTRUCTIONS.model_copy(
        update={"content": [TextPart(text="You answer briefly.")]}
    )
    try:
        prefix.verify(turn_three)
    except PrefixDriftError as drift:
        print(f"drift: {drift.layer!r} at position {drift.position} — {drift}")  # noqa: T201

    prefix.reset("the instruction block was redeployed")
    print(f"metrics: {prefix.boundary.attributes()}")  # noqa: T201


if __name__ == "__main__":
    main()
