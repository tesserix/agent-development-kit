"""Embedding and reranking on the CPU, without installing a native runtime to see it.

Four scenarios: the documented model set; a corrupt artefact refused before loading; a batch
embedded and then answered from the cache; and the same session shape reranking pairs.

Run it with `python examples/local_embeddings.py`. The session and tokenizer here are local
stand-ins — a real one comes from `load_session`, which needs `onnxruntime` installed
separately. See `docs/local-embeddings.md` for why it is not a dependency of the kit.
"""

from __future__ import annotations

import asyncio
import hashlib
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from tesserix_adk.core import ModelArtifactError
from tesserix_adk.models import (
    ONNX_MODELS,
    Encoded,
    OnnxCrossEncoder,
    OnnxEmbeddings,
    onnx_model,
    verify_artefact,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

MODEL = onnx_model("bge-small-en-v1.5")


class Words:
    """A tokenizer standing in for a real one: whitespace, then word lengths."""

    def encode(self, texts: Sequence[str], *, max_tokens: int) -> Encoded:
        """Encode every text, truncated to `max_tokens`."""
        ids = tuple(tuple(len(word) for word in text.split()[:max_tokens]) for text in texts)
        return Encoded(ids=ids, mask=tuple(tuple(1 for _ in row) for row in ids))


class Session:
    """A session standing in for a loaded ONNX model, counting what reaches it."""

    def __init__(self) -> None:
        self.calls = 0

    @property
    def device(self) -> str:
        """Where it runs."""
        return "cpu"

    def run(self, encoded: Encoded) -> Sequence[Sequence[float]]:
        """One row per encoded text."""
        self.calls += 1
        return [[float(sum(ids))] + [0.0] * (MODEL.dimension - 1) for ids in encoded.ids]


def the_model_set_states_its_cost() -> None:
    """Every shipped model says what it costs, at the batch it was measured with."""
    for model in ONNX_MODELS:
        budget = model.budget
        print(  # noqa: T201
            f"{model.name}: {budget.texts_per_second}/s "
            f"at batch {budget.batch}, {budget.threads} threads"
        )


def a_corrupt_artefact_is_refused() -> None:
    """At startup, for an operator, rather than at the first query, for a user."""
    with tempfile.TemporaryDirectory() as directory:
        (Path(directory) / MODEL.file).write_bytes(b"half a download")
        declared = MODEL.model_copy(update={"sha256": hashlib.sha256(b"the real one").hexdigest()})
        try:
            verify_artefact(declared, directory=Path(directory))
        except ModelArtifactError as refused:
            print(f"refused: {refused.reason}")  # noqa: T201


async def a_batch_then_the_cache() -> None:
    """The same text twice costs one forward pass."""
    session = Session()
    embeddings = OnnxEmbeddings(session, Words(), MODEL)

    await embeddings.embed(["a refund question", "a delivery question"], model=MODEL.name)
    await embeddings.embed(["a refund question", "a returns question"], model=MODEL.name)

    print(f"session calls: {session.calls}, cached last call: {embeddings.metrics.cached}")  # noqa: T201
    print(f"met the budget: {embeddings.metrics.meets_budget}")  # noqa: T201


def reranking_uses_the_same_shape() -> None:
    """Pairs in, one score each, in order."""
    encoder = OnnxCrossEncoder(Session(), Words(), onnx_model("bge-reranker-base"))

    scores = encoder.score([("refund", "our refund policy"), ("refund", "hi")])

    print(f"scores: {scores}")  # noqa: T201


async def main() -> None:
    """Run every scenario in order."""
    the_model_set_states_its_cost()
    a_corrupt_artefact_is_refused()
    await a_batch_then_the_cache()
    reranking_uses_the_same_shape()


if __name__ == "__main__":
    asyncio.run(main())
