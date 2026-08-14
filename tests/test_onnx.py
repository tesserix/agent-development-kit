"""Embedding and reranking on a CPU, and the refusals that happen before the first query."""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

import pytest

from tesserix_adk.core import ConfigurationError, ModelArtifactError
from tesserix_adk.models import onnx
from tesserix_adk.models.onnx import (
    ONNX_MODELS,
    Device,
    Encoded,
    OnnxCrossEncoder,
    OnnxEmbeddings,
    OnnxModel,
    Throughput,
    load_session,
    onnx_model,
    verify_artefact,
)
from tesserix_adk.testing import FakeClock

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

pytestmark = pytest.mark.anyio

MODEL = OnnxModel(
    name="bge-small-en-v1.5",
    version="1.5",
    dimension=4,
    max_tokens=512,
    file="model.onnx",
    budget=Throughput(texts_per_second=200.0, batch=32, threads=4),
)


class Words:
    """A tokenizer that splits on whitespace, which is enough to test the plumbing."""

    def encode(self, texts: Sequence[str], *, max_tokens: int) -> Encoded:
        ids = tuple(tuple(len(word) for word in text.split()[:max_tokens]) for text in texts)
        return Encoded(ids=ids, mask=tuple(tuple(1 for _ in sequence) for sequence in ids))


class Session:
    """A session that answers from the token ids, and can be made to take time."""

    def __init__(self, *, device: Device = Device.CPU, seconds: float = 0.0) -> None:
        self._device = device
        self._seconds = seconds
        self.batches: list[int] = []
        self.clock: FakeClock | None = None

    @property
    def device(self) -> Device:
        return self._device

    def run(self, encoded: Encoded) -> Sequence[Sequence[float]]:
        self.batches.append(len(encoded.ids))
        if self.clock is not None:
            self.clock.advance(self._seconds)
        return [[float(sum(ids)), 1.0, 0.0, 0.0] for ids in encoded.ids]


class Inference:
    """What `onnxruntime.InferenceSession` looks like from this module's side."""

    def __init__(self) -> None:
        self.fed: dict[str, list[list[int]]] = {}

    def run(self, _: None, feed: dict[str, list[list[int]]]) -> list[list[list[float]]]:
        self.fed = feed
        return [[[float(id_) for id_ in ids] for ids in feed["input_ids"]]]


class Options:
    """`onnxruntime.SessionOptions`, with the one field this module sets."""

    intra_op_num_threads = 0


class Runtime:
    """The `onnxruntime` module, standing in for the wheel nobody has to install."""

    def __init__(self) -> None:
        self.inference = Inference()
        self.providers: list[str] = []
        self.threads = 0

    def SessionOptions(self) -> Options:  # noqa: N802
        return Options()

    def InferenceSession(  # noqa: N802
        self, path: str, options: Options, providers: list[str]
    ) -> Inference:
        self.providers = providers
        self.threads = options.intra_op_num_threads
        assert path.endswith(MODEL.file)
        return self.inference


def _absent(name: str) -> object:
    """An import of a module that is not installed."""
    raise ModuleNotFoundError(name)


def embedder(*, session: Session | None = None, clock: FakeClock | None = None) -> OnnxEmbeddings:
    """An embedder over the fake session, with the clock it measures itself by."""
    used = session or Session()
    used.clock = clock
    return OnnxEmbeddings(used, Words(), MODEL, clock=clock or FakeClock())


class TestTheDocumentedModelSet:
    """A model is chosen by name, and what it costs is written down beside it."""

    async def test_every_shipped_model_declares_its_budget(self) -> None:
        assert ONNX_MODELS
        assert all(model.budget.texts_per_second > 0 for model in ONNX_MODELS)
        assert all(model.dimension > 0 and model.max_tokens > 0 for model in ONNX_MODELS)

    async def test_a_multilingual_model_is_offered(self) -> None:
        assert any(model.multilingual for model in ONNX_MODELS)

    async def test_a_model_is_found_by_name(self) -> None:
        found = onnx_model(ONNX_MODELS[0].name)

        assert found == ONNX_MODELS[0]

    async def test_an_unknown_name_says_what_there_is(self) -> None:
        with pytest.raises(ConfigurationError, match=r"bge-small-en-v1\.5"):
            onnx_model("whatever-was-on-the-hub")


class TestLoadingRefusesEarly:
    """The failure scenario: at startup, not at the first query."""

    async def test_a_missing_file_is_named(self, tmp_path: Path) -> None:
        with pytest.raises(ModelArtifactError) as refused:
            verify_artefact(MODEL, directory=tmp_path)

        assert refused.value.reason == "missing"
        assert MODEL.file in str(refused.value)

    async def test_an_empty_file_is_not_a_model(self, tmp_path: Path) -> None:
        (tmp_path / MODEL.file).write_bytes(b"")

        with pytest.raises(ModelArtifactError) as refused:
            verify_artefact(MODEL, directory=tmp_path)

        assert refused.value.reason == "empty"

    async def test_a_corrupt_file_is_caught_by_its_digest(self, tmp_path: Path) -> None:
        (tmp_path / MODEL.file).write_bytes(b"not the weights")
        declared = MODEL.model_copy(update={"sha256": hashlib.sha256(b"the weights").hexdigest()})

        with pytest.raises(ModelArtifactError) as refused:
            verify_artefact(declared, directory=tmp_path)

        assert refused.value.reason == "digest"

    async def test_a_good_file_gives_back_its_path(self, tmp_path: Path) -> None:
        weights = tmp_path / MODEL.file
        weights.write_bytes(b"the weights")
        declared = MODEL.model_copy(update={"sha256": hashlib.sha256(b"the weights").hexdigest()})

        assert verify_artefact(declared, directory=tmp_path) == weights

    async def test_an_undeclared_digest_is_not_checked(self, tmp_path: Path) -> None:
        (tmp_path / MODEL.file).write_bytes(b"whatever this is")

        assert verify_artefact(MODEL, directory=tmp_path).exists()


class TestEmbeddingOnCpu:
    """The primary scenario: a batch, at the documented rate, cached."""

    async def test_a_batch_comes_back_in_order(self) -> None:
        vectors = await embedder().embed(["a bb", "ccc"], model=MODEL.name)

        assert len(vectors) == 2
        assert vectors[0][0] == 3.0
        assert vectors[1][0] == 3.0

    async def test_every_vector_is_the_declared_width(self) -> None:
        vectors = await embedder().embed(["one two three"], model=MODEL.name)

        assert all(len(vector) == MODEL.dimension for vector in vectors)

    async def test_the_whole_batch_is_one_session_call(self) -> None:
        session = Session()
        await embedder(session=session).embed([f"text {n}" for n in range(32)], model=MODEL.name)

        assert session.batches == [32]

    async def test_throughput_is_measured_against_the_budget(self) -> None:
        clock = FakeClock()
        session = Session(seconds=0.1)
        stage = embedder(session=session, clock=clock)

        await stage.embed([f"text {n}" for n in range(32)], model=MODEL.name)

        assert stage.metrics.texts_per_second == pytest.approx(320.0)
        assert stage.metrics.meets_budget is True

    async def test_a_slow_run_says_it_missed_the_budget(self) -> None:
        clock = FakeClock()
        stage = embedder(session=Session(seconds=4.0), clock=clock)

        await stage.embed([f"text {n}" for n in range(32)], model=MODEL.name)

        assert stage.metrics.meets_budget is False

    async def test_an_empty_batch_asks_nothing_of_the_model(self) -> None:
        session = Session()

        assert await embedder(session=session).embed([], model=MODEL.name) == []
        assert session.batches == []

    async def test_the_provider_names_the_model_it_holds(self) -> None:
        stage = embedder()

        assert MODEL.name in stage.name
        assert stage.limits(MODEL.name).dimensions == MODEL.dimension

    async def test_asking_for_another_model_is_refused(self) -> None:
        with pytest.raises(ConfigurationError, match="e5-base"):
            await embedder().embed(["a"], model="e5-base")


class TestTheCache:
    """Results are cached, and the cache belongs to one version of one model."""

    async def test_a_repeated_text_is_not_embedded_twice(self) -> None:
        session = Session()
        stage = embedder(session=session)

        first = await stage.embed(["the same text"], model=MODEL.name)
        second = await stage.embed(["the same text"], model=MODEL.name)

        assert first == second
        assert session.batches == [1]
        assert stage.metrics.cached == 1

    async def test_only_the_new_texts_reach_the_model(self) -> None:
        session = Session()
        stage = embedder(session=session)

        await stage.embed(["one", "two"], model=MODEL.name)
        await stage.embed(["two", "three"], model=MODEL.name)

        assert session.batches == [2, 1]

    async def test_a_new_model_version_cannot_read_the_old_entries(self) -> None:
        session = Session()
        stage = embedder(session=session)
        await stage.embed(["a text"], model=MODEL.name)

        moved = OnnxEmbeddings(
            session, Words(), MODEL.model_copy(update={"version": "1.6"}), clock=FakeClock()
        )
        await moved.embed(["a text"], model=MODEL.name)

        assert session.batches == [1, 1]

    async def test_multilingual_text_is_embedded_and_cached_like_any_other(self) -> None:
        session = Session()
        stage = embedder(session=session)

        first = await stage.embed(["退款政策 remboursement"], model=MODEL.name)
        second = await stage.embed(["退款政策 remboursement"], model=MODEL.name)

        assert first == second
        assert session.batches == [1]

    async def test_a_cache_that_holds_nothing_is_a_configuration_mistake(self) -> None:
        with pytest.raises(ValueError, match="cache_entries"):
            OnnxEmbeddings(Session(), Words(), MODEL, cache_entries=0)

    async def test_a_duplicate_inside_one_batch_is_embedded_once(self) -> None:
        session = Session()
        stage = embedder(session=session)

        vectors = await stage.embed(["same", "same"], model=MODEL.name)

        assert session.batches == [1]
        assert vectors[0] == vectors[1]

    async def test_the_cache_is_bounded(self) -> None:
        session = Session()
        stage = OnnxEmbeddings(session, Words(), MODEL, clock=FakeClock(), cache_entries=2)

        await stage.embed(["one", "two", "three"], model=MODEL.name)
        await stage.embed(["one"], model=MODEL.name)

        assert session.batches == [3, 1]


class TestLoadingTheRuntime:
    """`onnxruntime` is not a dependency of the kit, so its absence is a said thing."""

    async def test_a_missing_runtime_says_how_to_install_it(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (tmp_path / MODEL.file).write_bytes(b"the weights")
        monkeypatch.setattr(onnx, "import_module", _absent)

        with pytest.raises(ConfigurationError, match="not an extra"):
            load_session(MODEL, directory=tmp_path)

    async def test_a_session_is_loaded_with_the_asked_for_provider(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (tmp_path / MODEL.file).write_bytes(b"the weights")
        runtime = Runtime()
        monkeypatch.setattr(onnx, "import_module", lambda _: runtime)

        session = load_session(MODEL, directory=tmp_path, device=Device.CUDA, threads=4)

        assert session.device is Device.CUDA
        assert runtime.providers == ["CUDAExecutionProvider"]
        assert runtime.threads == 4

    async def test_the_runtime_default_thread_count_is_left_alone(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (tmp_path / MODEL.file).write_bytes(b"the weights")
        runtime = Runtime()
        monkeypatch.setattr(onnx, "import_module", lambda _: runtime)

        load_session(MODEL, directory=tmp_path)

        assert runtime.threads == 0

    async def test_the_loaded_session_feeds_ids_and_mask(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (tmp_path / MODEL.file).write_bytes(b"the weights")
        runtime = Runtime()
        monkeypatch.setattr(onnx, "import_module", lambda _: runtime)
        session = load_session(MODEL, directory=tmp_path)

        rows = session.run(Words().encode(["a bb"], max_tokens=8))

        assert rows == [[1.0, 2.0]]
        assert runtime.inference.fed == {"input_ids": [[1, 2]], "attention_mask": [[1, 1]]}


class TestTheSameInterfaceOnAGpu:
    """A GPU changes the device the session runs on and nothing else."""

    async def test_the_device_is_reported_not_assumed(self) -> None:
        stage = embedder(session=Session(device=Device.CUDA))

        assert stage.device is Device.CUDA

    async def test_the_results_do_not_depend_on_it(self) -> None:
        on_cpu = await embedder(session=Session()).embed(["a bb"], model=MODEL.name)
        on_gpu = await embedder(session=Session(device=Device.CUDA)).embed(
            ["a bb"], model=MODEL.name
        )

        assert on_cpu == on_gpu


class TestReranking:
    """The same session shape, scoring pairs instead of embedding texts."""

    async def test_every_pair_is_scored_in_order(self) -> None:
        encoder = OnnxCrossEncoder(Session(), Words(), MODEL)

        scores = encoder.score([("q", "a bb"), ("q", "ccc cccc")])

        assert len(scores) == 2
        assert scores[1] > scores[0]

    async def test_no_pairs_is_no_call(self) -> None:
        session = Session()

        assert OnnxCrossEncoder(session, Words(), MODEL).score([]) == []
        assert session.batches == []

    async def test_it_reports_its_device_too(self) -> None:
        encoder = OnnxCrossEncoder(Session(device=Device.CUDA), Words(), MODEL)

        assert encoder.device is Device.CUDA

    async def test_it_is_the_cross_encoder_the_kit_reranks_with(self) -> None:
        from tesserix_adk.rag import CrossEncoderReranker

        reranker = CrossEncoderReranker(OnnxCrossEncoder(Session(), Words(), MODEL))

        assert reranker.name == "cross-encoder"
