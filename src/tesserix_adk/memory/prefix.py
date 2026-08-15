"""The line compression may not cross, so token savings and cache reuse do not cancel out.

Compression and prefix caching pull in opposite directions, and the conflict is silent.
Recompressing the whole context every turn produces slightly different bytes each time,
which invalidates the provider's cached prefix and forces a full prefill. On CPU inference
that trade is catastrophic: the compression saves perhaps a third of the tokens and the
cache miss costs tens of seconds of prefill on all of them.

So there is a boundary. Everything above it is frozen and byte-identical from turn to turn;
only content that arrived since is eligible for anything. Content is compressed once, on
arrival, and then becomes frozen history itself.

The boundary is per context. Nothing here is global, and a position never crosses a tenant.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

from tesserix_adk.core.errors import PrefixDriftError
from tesserix_adk.core.models import AdkModel

if TYPE_CHECKING:
    from collections.abc import Sequence

    from tesserix_adk.core.primitives import Message

__all__ = [
    "COMPRESSED",
    "POSITION_METRIC",
    "RESETS_METRIC",
    "SECTION",
    "FrozenPrefix",
    "PrefixBoundary",
    "compressed",
]

COMPRESSED = "adk.compressed"
"""Metadata key naming what compressed a message. Its presence means never again."""

SECTION = "adk.section"
"""Metadata key naming the layer of the prompt a message belongs to, for a drift report."""

POSITION_METRIC = "adk.frozen_prefix_position"
"""How many messages are frozen. Worth watching beside the cache hit ratio it protects."""

RESETS_METRIC = "adk.frozen_prefix_resets"
"""How many times the prefix was invalidated. Each one is a full prefill somebody paid for."""


def compressed(message: Message, *, by: str) -> Message:
    """Return `message` marked as already compressed, so no later turn compresses it again.

    Args:
        message: What was compressed.
        by: Which compressor did it, for a report that can name what shrank the content.

    Returns:
        A copy carrying the mark. The original is untouched, because the caller may still
        be holding the uncompressed version for a handle.

    Example:
        >>> from tesserix_adk.core import Message, TextPart
        >>> marked = compressed(Message(role="user", content=[TextPart(text="rows")]), by="x")
        >>> marked.metadata["adk.compressed"]
        'x'
    """
    return message.model_copy(update={"metadata": {**message.metadata, COMPRESSED: by}})


class PrefixBoundary(AdkModel):
    """Where the frozen region ends, and what it was when it was frozen.

    Args:
        position: How many leading messages are frozen. Everything from here on is live.
        fingerprint: The digest of the frozen region, so drift is provable rather than
            suspected.
        turn: The turn that last advanced it. An advance for a turn already recorded is
            ignored, so two concurrent turns cannot move it twice.
        resets: How many times it has gone back to zero. Never decremented: a reset is a
            full prefill somebody paid for, and the count is the record of that.
        reason: Why it last reset. Empty where it never has.
    """

    position: int = 0
    fingerprint: str = ""
    turn: int = 0
    resets: int = 0
    reason: str = ""

    def attributes(self) -> dict[str, int]:
        """The boundary as span or metric attributes. Positions and counts, never content."""
        return {POSITION_METRIC: self.position, RESETS_METRIC: self.resets}


class FrozenPrefix:
    """One context's boundary between cached history and content that just arrived.

    Advancing is monotonic within a turn and idempotent across concurrent ones. Verifying
    is the assertion that makes the whole thing enforceable: it fails naming the layer that
    moved, because a silent doubling of prefill latency should fail a build rather than
    reach production.
    """

    def __init__(self) -> None:
        self._boundary = PrefixBoundary()
        self._digests: tuple[str, ...] = ()
        self._layers: tuple[str, ...] = ()

    @property
    def boundary(self) -> PrefixBoundary:
        """Where the frozen region currently ends."""
        return self._boundary

    def advance(self, messages: Sequence[Message], *, turn: int) -> PrefixBoundary:
        """Freeze everything sent on `turn`, and return where that leaves the boundary.

        Args:
            messages: The prompt as it was sent, in order.
            turn: Which turn is completing. An advance for a turn at or before the one
                already recorded is ignored, so concurrent turns cannot double-advance.

        Returns:
            The boundary, advanced or unchanged.
        """
        if turn <= self._boundary.turn and self._boundary.position:
            return self._boundary
        self._digests = tuple(_digest(message) for message in messages)
        self._layers = tuple(_layer(message) for message in messages)
        self._boundary = self._boundary.model_copy(
            update={
                "position": len(messages),
                "fingerprint": _fingerprint(self._digests),
                "turn": turn,
            }
        )
        return self._boundary

    def live(self, messages: Sequence[Message]) -> tuple[Message, ...]:
        """The content that arrived since the boundary. Only this may be touched."""
        return tuple(messages[self._boundary.position :])

    def compressible(self, messages: Sequence[Message]) -> tuple[int, ...]:
        """The indices a compressor may be offered: live, and not already compressed.

        Args:
            messages: The prompt as it will be sent.

        Returns:
            Indices into `messages`, in order. Empty where nothing new has arrived.
        """
        start = self._boundary.position
        return tuple(
            start + offset
            for offset, message in enumerate(self.live(messages))
            if COMPRESSED not in message.metadata
        )

    def verify(self, messages: Sequence[Message]) -> None:
        """Assert that the frozen region of `messages` is what was frozen.

        Args:
            messages: The prompt about to be sent.

        Raises:
            PrefixDriftError: If any frozen message changed, moved or went missing. The
                error names the layer and the position, so the answer to "what invalidated
                the cache" is in the failure rather than in a latency graph a week later.
        """
        for position, (digest, layer) in enumerate(zip(self._digests, self._layers, strict=True)):
            if position >= len(messages):
                raise PrefixDriftError(
                    f"the frozen prefix lost {layer!r} at position {position}",
                    position=position,
                    layer=layer,
                )
            if _digest(messages[position]) != digest:
                raise PrefixDriftError(
                    f"{layer!r} at position {position} changed after it was frozen",
                    position=position,
                    layer=layer,
                )

    def reset(self, reason: str) -> PrefixBoundary:
        """Give up the frozen region, and record what cost the prefill.

        Args:
            reason: What invalidated it — an eviction, a redeployed system prompt, a
                changed tool declaration. Recorded rather than hidden.

        Returns:
            The boundary, back at nothing frozen, with the reset counted.
        """
        self._digests = ()
        self._layers = ()
        self._boundary = self._boundary.model_copy(
            update={
                "position": 0,
                "fingerprint": "",
                "resets": self._boundary.resets + 1,
                "reason": reason,
            }
        )
        return self._boundary


def _layer(message: Message) -> str:
    """What to call this message in a drift report: its section, or failing that its role."""
    return message.metadata.get(SECTION) or message.role


def _digest(message: Message) -> str:
    """The bytes of one message, as the provider would see them."""
    return hashlib.blake2b(
        message.model_dump_json(exclude={"metadata"}).encode("utf-8"), digest_size=16
    ).hexdigest()


def _fingerprint(digests: Sequence[str]) -> str:
    """One digest standing for the whole frozen region, order included."""
    if not digests:
        return ""
    return hashlib.blake2b("|".join(digests).encode("utf-8"), digest_size=16).hexdigest()
