"""A vector source that never reaches a network and answers the same vectors every run.

A retrieval test whose vectors move between runs asserts nothing, so these are derived from
the text itself: similar in nothing but identity, which is all a test of batching, caching
and refusal needs.
"""

from __future__ import annotations

import struct
from hashlib import blake2b
from typing import TYPE_CHECKING

from tesserix_adk.core.primitives import Usage
from tesserix_adk.rag.embedding import BatchVectors, EmbeddingModel

if TYPE_CHECKING:
    from collections.abc import Sequence

    from tesserix_adk.rag.embedding import Vector

__all__ = ["FakeEmbedder"]

DEFAULT_MODEL = EmbeddingModel(name="fake-embedder", version="1", dimension=8)


def deterministic_vector(text: str, width: int) -> Vector:
    """A vector for `text` that is the same in every process and every run.

    Built from ints rather than by reading the digest as floats, which would eventually
    produce a NaN, and a NaN does not compare equal to itself or to the vector a second
    run derives from the same text.
    """
    raw = bytearray()
    seed = text.encode()
    while len(raw) < 4 * width:
        seed = blake2b(seed, digest_size=64).digest()
        raw += seed
    return tuple(value / 2**32 for value in struct.unpack(f">{width}I", raw[: 4 * width]))


class FakeEmbedder:
    """A `VectorSource` that records what it was asked and can be told to fail.

    Args:
        model: What it claims to be, including the batch size it accepts.
        usage: What each call reports consuming, for testing cost attribution.
        failures: What to raise, one per call, until the script runs out. `None` in the
            script is a call that succeeds.
        width: The width it actually answers, where that must differ from the width it
            declares, which is what a model upgrade looks like from the index's side.
        short: Whether to answer one fewer vector than it was asked for.
    """

    def __init__(
        self,
        *,
        model: EmbeddingModel = DEFAULT_MODEL,
        usage: Usage | None = None,
        failures: Sequence[BaseException | None] = (),
        width: int | None = None,
        short: bool = False,
    ) -> None:
        self._model = model
        self._usage = usage or Usage(input_tokens=0, output_tokens=0)
        self._failures = list(failures)
        self._width = width if width is not None else model.dimension
        self._short = short
        self.calls = 0
        self.batches: list[tuple[str, ...]] = []
        self.in_flight = 0
        self.most_in_flight = 0

    @property
    def model(self) -> EmbeddingModel:
        """The model it claims to be."""
        return self._model

    async def vectors_for(self, texts: Sequence[str]) -> BatchVectors:
        """Answer a vector per text, or raise whatever the script says to raise.

        Raises:
            BaseException: Whatever the failure script holds for this call.
        """
        self.calls += 1
        self.batches.append(tuple(texts))
        self.in_flight += 1
        self.most_in_flight = max(self.most_in_flight, self.in_flight)
        try:
            if self._failures:
                failure = self._failures.pop(0)
                if failure is not None:
                    raise failure
            answering = texts[:-1] if self._short else texts
            return BatchVectors(
                vectors=tuple(deterministic_vector(text, self._width) for text in answering),
                usage=self._usage,
            )
        finally:
            self.in_flight -= 1
