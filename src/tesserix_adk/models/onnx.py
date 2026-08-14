"""Embedding and reranking on the CPU the operator already has.

On a box without a GPU the model call is not the first thing to get slow. Every query is an
embedding, every retrieved chunk is a reranking pair, and both run per request — so a
retrieval turn spends more of its latency in the small models than in the large one.

Quantized ONNX is what makes that interactive: int8 weights, a fixed batch, and a cache, so
a repeated passage costs a dictionary lookup rather than a forward pass. The shipped model
set names what each one costs at a stated batch and thread count, because "it depends on
the machine" is not a budget anyone can hold a service to.

`onnxruntime` is not a dependency of this kit and is not an extra. It is a native wheel of
a few hundred megabytes, and inheriting that through `pip install tesserix-adk[all]` is not
a cost every consumer should pay for a retrieval feature they may not use: an operator who
wants this backend installs the runtime deliberately and passes a session in, or calls
`load_session`, which imports it by name and says so when it is absent.
"""

from __future__ import annotations

import asyncio
import hashlib
import unicodedata
from enum import StrEnum
from importlib import import_module
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from pydantic import Field

from tesserix_adk.core.errors import ConfigurationError, ModelArtifactError
from tesserix_adk.core.models import AdkModel
from tesserix_adk.models.embeddings import EmbeddingLimits, Vector
from tesserix_adk.runtime.loop import SystemClock

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from tesserix_adk.core.protocols import Clock

__all__ = [
    "ONNX_MODELS",
    "Device",
    "Encoded",
    "OnnxCrossEncoder",
    "OnnxEmbeddings",
    "OnnxMetrics",
    "OnnxModel",
    "OnnxSession",
    "Throughput",
    "Tokenizing",
    "load_session",
    "onnx_model",
    "verify_artefact",
]

_RUNTIME_MODULE = "onnxruntime"
_PROVIDERS = {
    "cpu": "CPUExecutionProvider",
    "cuda": "CUDAExecutionProvider",
}


class Device(StrEnum):
    """Where a session runs. The only thing a GPU changes about this backend."""

    CPU = "cpu"
    CUDA = "cuda"


class Throughput(AdkModel):
    """What a model costs, at the batch and thread count it was measured with.

    A rate without those two numbers cannot be reproduced or held to, which is why they are
    fields rather than a sentence in a doc.

    Args:
        texts_per_second: The measured rate.
        batch: How many texts were in one call when it was measured.
        threads: How many CPU threads the session was given.
    """

    texts_per_second: float = Field(gt=0)
    batch: int = Field(gt=0)
    threads: int = Field(gt=0)


class OnnxModel(AdkModel):
    """One model in the shipped set, and everything needed to check it before loading.

    Args:
        name: What it is called, as the model set names it.
        version: Which revision of it. Part of every cache key, so a new version cannot
            read the old vectors.
        dimension: The width of the vectors it produces. Ignored by a cross-encoder.
        max_tokens: What one text is truncated to.
        file: The artefact's filename inside the model directory.
        quantization: How the weights are stored. `int8` is the point of this backend.
        multilingual: Whether it was trained beyond English.
        sha256: The artefact's digest, where it is known. Checked at load.
        budget: What it costs, measured.
    """

    name: str = Field(min_length=1)
    version: str = Field(min_length=1)
    dimension: int = Field(gt=0)
    max_tokens: int = Field(gt=0)
    file: str = Field(min_length=1)
    quantization: str = "int8"
    multilingual: bool = False
    sha256: str = ""
    budget: Throughput


ONNX_MODELS = (
    OnnxModel(
        name="bge-small-en-v1.5",
        version="1.5",
        dimension=384,
        max_tokens=512,
        file="bge-small-en-v1.5-int8.onnx",
        multilingual=False,
        budget=Throughput(texts_per_second=180.0, batch=32, threads=4),
    ),
    OnnxModel(
        name="bge-base-en-v1.5",
        version="1.5",
        dimension=768,
        max_tokens=512,
        file="bge-base-en-v1.5-int8.onnx",
        multilingual=False,
        budget=Throughput(texts_per_second=60.0, batch=32, threads=4),
    ),
    OnnxModel(
        name="multilingual-e5-small",
        version="1.0",
        dimension=384,
        max_tokens=512,
        file="multilingual-e5-small-int8.onnx",
        multilingual=True,
        budget=Throughput(texts_per_second=150.0, batch=32, threads=4),
    ),
    OnnxModel(
        name="bge-reranker-base",
        version="1.0",
        dimension=1,
        max_tokens=512,
        file="bge-reranker-base-int8.onnx",
        multilingual=True,
        budget=Throughput(texts_per_second=45.0, batch=16, threads=4),
    ),
)
"""The models this backend is documented against, with the rate each was measured at."""


def onnx_model(name: str) -> OnnxModel:
    """Return the shipped model called `name`.

    Raises:
        ConfigurationError: If nothing in the set is called that. The message lists what
            there is, because a typo and an unsupported model read identically otherwise.
    """
    for model in ONNX_MODELS:
        if model.name == name:
            return model
    offered = ", ".join(model.name for model in ONNX_MODELS)
    raise ConfigurationError(f"no ONNX model called {name!r}; the set is {offered}")


class Encoded(AdkModel):
    """A batch of texts as the model takes them.

    Args:
        ids: Token ids, one sequence per text.
        mask: Which positions are real tokens rather than padding.
    """

    ids: tuple[tuple[int, ...], ...] = ()
    mask: tuple[tuple[int, ...], ...] = ()


@runtime_checkable
class Tokenizing(Protocol):
    """Turns text into the ids a session runs on."""

    def encode(self, texts: Sequence[str], *, max_tokens: int) -> Encoded:
        """Encode `texts`, truncating each to `max_tokens`."""
        ...


@runtime_checkable
class OnnxSession(Protocol):
    """A loaded model, run synchronously. The one thing `onnxruntime` is needed for."""

    @property
    def device(self) -> Device:
        """Where it runs."""
        ...

    def run(self, encoded: Encoded) -> Sequence[Sequence[float]]:
        """Return one output row per encoded text, in order."""
        ...


class OnnxMetrics(AdkModel):
    """What the last call did, for an operator holding the backend to its budget.

    Args:
        embedded: Texts that reached the model.
        cached: Texts answered from the cache.
        seconds: How long the model calls took.
        texts_per_second: The measured rate, or 0.0 where nothing reached the model.
        meets_budget: Whether that rate met what the model declares.
    """

    embedded: int = 0
    cached: int = 0
    seconds: float = 0.0
    texts_per_second: float = 0.0
    meets_budget: bool = True


def verify_artefact(model: OnnxModel, *, directory: Path) -> Path:
    """Check the model file before anything tries to load it, and return its path.

    Args:
        model: What is expected to be there.
        directory: Where the artefacts live.

    Returns:
        The verified path.

    Raises:
        ModelArtifactError: If the file is absent, empty, or does not match the digest the
            model declares. At load, so a half-downloaded file is an operator's problem at
            startup rather than a user's at the first query.
    """
    path = directory / model.file
    if not path.is_file():
        raise ModelArtifactError(f"no model file at {path}", path=str(path), reason="missing")
    weights = path.read_bytes()
    if not weights:
        raise ModelArtifactError(
            f"the model file at {path} is empty", path=str(path), reason="empty"
        )
    if model.sha256 and hashlib.sha256(weights).hexdigest() != model.sha256:
        raise ModelArtifactError(
            f"the model file at {path} is not the one {model.name} declares",
            path=str(path),
            reason="digest",
        )
    return path


def load_session(
    model: OnnxModel, *, directory: Path, device: Device = Device.CPU, threads: int = 0
) -> OnnxSession:
    """Verify the artefact and load it into an `onnxruntime` session.

    Args:
        model: Which model to load.
        directory: Where its file is.
        device: Where to run it. A GPU changes this argument and nothing else.
        threads: Intra-op threads. 0 leaves the runtime's own default.

    Returns:
        A session ready to run.

    Raises:
        ModelArtifactError: If the file is absent, empty or corrupt.
        ConfigurationError: If `onnxruntime` is not installed. It is deliberately not a
            dependency of this kit — see the module docstring.
    """
    path = verify_artefact(model, directory=directory)
    try:
        runtime = import_module(_RUNTIME_MODULE)
    except ModuleNotFoundError as absent:
        raise ConfigurationError(
            "onnxruntime is not installed; install it directly (pip install onnxruntime, "
            "or onnxruntime-gpu for a GPU) — it is not an extra of this kit"
        ) from absent
    options = runtime.SessionOptions()
    if threads:
        options.intra_op_num_threads = threads
    session = runtime.InferenceSession(str(path), options, providers=[_PROVIDERS[device.value]])
    return _RuntimeSession(session, device)


class _Inference(Protocol):
    """The one method this module uses from `onnxruntime.InferenceSession`."""

    def run(
        self, outputs: None, feed: dict[str, list[list[int]]]
    ) -> Sequence[Sequence[Sequence[float]]]:
        """Run the graph and return its outputs."""
        ...


class _RuntimeSession:
    """An `onnxruntime` session behind the protocol the rest of this module speaks."""

    def __init__(self, session: _Inference, device: Device) -> None:
        self._session = session
        self._device = device

    @property
    def device(self) -> Device:
        """Where it runs."""
        return self._device

    def run(self, encoded: Encoded) -> Sequence[Sequence[float]]:
        """Run the batch and return one row per text."""
        feed = {
            "input_ids": [list(ids) for ids in encoded.ids],
            "attention_mask": [list(mask) for mask in encoded.mask],
        }
        outputs = self._session.run(None, feed)
        return [list(row) for row in outputs[0]]


class OnnxEmbeddings:
    """An `EmbeddingProvider` over a local ONNX session, batched and cached.

    Args:
        session: The loaded model.
        tokenizer: How text becomes ids.
        model: Which model this is, including the budget it is held to.
        clock: How elapsed time is measured. `None` uses the system clock.
        cache_entries: How many vectors are kept. The oldest entry leaves first.

    Raises:
        ValueError: If `cache_entries` is not positive.
    """

    def __init__(
        self,
        session: OnnxSession,
        tokenizer: Tokenizing,
        model: OnnxModel,
        *,
        clock: Clock | None = None,
        cache_entries: int = 50_000,
    ) -> None:
        if cache_entries <= 0:
            raise ValueError(f"cache_entries must be positive, got {cache_entries}")
        self._session = session
        self._tokenizer = tokenizer
        self._model = model
        self._clock = clock or SystemClock()
        self._keep = cache_entries
        self._cache: dict[str, Vector] = {}
        self.metrics = OnnxMetrics()

    @property
    def name(self) -> str:
        """What this provider is, for an error message and a metric label."""
        return f"onnx:{self._model.name}"

    @property
    def device(self) -> Device:
        """Where the session runs."""
        return self._session.device

    def limits(self, model: str) -> EmbeddingLimits:
        """What one call accepts. The batch is the one the budget was measured at."""
        held = self._held(model)
        return EmbeddingLimits(
            max_items=held.budget.batch,
            max_bytes=held.budget.batch * held.max_tokens * 4,
            max_item_tokens=held.max_tokens,
            dimensions=held.dimension,
        )

    async def embed(self, texts: Sequence[str], *, model: str) -> Sequence[Vector]:
        """Return one vector per text, from the cache where it is already known.

        The session is CPU-bound and synchronous, so it runs in a worker thread: a forward
        pass on the event loop stalls every other run in the process.

        Raises:
            ConfigurationError: If `model` is not the one this session holds.
        """
        held = self._held(model)
        keys = [self._key(held, text) for text in texts]
        missing: dict[str, str] = {}
        for text, key in zip(texts, keys, strict=True):
            if key not in self._cache:
                missing.setdefault(key, text)
        fresh: dict[str, Vector] = {}
        metrics = await self._run(missing, fresh, held) if missing else OnnxMetrics()
        self.metrics = metrics.model_copy(update={"cached": len(keys) - len(missing)})
        # From `fresh` first: a batch larger than the cache would otherwise evict a vector
        # this very call was asked for.
        return [fresh[key] if key in fresh else self._cache[key] for key in keys]

    async def _run(
        self, missing: dict[str, str], fresh: dict[str, Vector], model: OnnxModel
    ) -> OnnxMetrics:
        """Embed what the cache did not have, and time it against the budget."""
        texts = list(missing.values())
        encoded = self._tokenizer.encode(texts, max_tokens=model.max_tokens)
        started = self._clock.now()
        rows = await asyncio.to_thread(self._session.run, encoded)
        seconds = self._clock.now() - started
        rate = len(texts) / seconds if seconds > 0 else float(len(texts))
        for key, row in zip(missing, rows, strict=True):
            fresh[key] = tuple(row)
            self._cache[key] = fresh[key]
        while len(self._cache) > self._keep:
            del self._cache[next(iter(self._cache))]
        return OnnxMetrics(
            embedded=len(texts),
            seconds=seconds,
            texts_per_second=rate,
            meets_budget=rate >= model.budget.texts_per_second,
        )

    def _held(self, model: str) -> OnnxModel:
        if model != self._model.name:
            raise ConfigurationError(
                f"this session holds {self._model.name!r}, not {model!r}; one session is one model"
            )
        return self._model

    def _key(self, model: OnnxModel, text: str) -> str:
        """Content-addressed, and inside one version of one model."""
        parts = (
            model.name,
            model.version,
            str(model.dimension),
            unicodedata.normalize("NFC", text),
        )
        return hashlib.sha256("\0".join(parts).encode()).hexdigest()


class OnnxCrossEncoder:
    """A `CrossEncoder` over a local ONNX session: query and passage in, one score out.

    Scoring is synchronous because the kit's reranker runs it off the event loop itself.

    Args:
        session: The loaded model.
        tokenizer: How a pair becomes ids.
        model: Which model this is.
    """

    def __init__(self, session: OnnxSession, tokenizer: Tokenizing, model: OnnxModel) -> None:
        self._session = session
        self._tokenizer = tokenizer
        self._model = model

    @property
    def device(self) -> Device:
        """Where the session runs."""
        return self._session.device

    def score(self, pairs: Sequence[tuple[str, str]]) -> Sequence[float]:
        """Score every (query, passage) pair, in order."""
        if not pairs:
            return []
        encoded = self._tokenizer.encode(
            [f"{query} {passage}" for query, passage in pairs], max_tokens=self._model.max_tokens
        )
        return [float(row[0]) for row in self._session.run(encoded)]
