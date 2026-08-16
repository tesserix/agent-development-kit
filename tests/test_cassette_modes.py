"""Recording, replaying and refreshing provider traffic without a network."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import pytest

from tesserix_adk.core import (
    ConfigurationError,
    Message,
    ModelResponse,
    ProviderError,
    TextPart,
    ToolCall,
)
from tesserix_adk.core.streaming import StreamEnd, TextDelta, ToolCallDelta
from tesserix_adk.runtime import ModelRequest, fingerprint_of
from tesserix_adk.testing import (
    ENV_MODE,
    Cassette,
    CassetteHarness,
    CassetteLeakError,
    CassetteMismatchError,
    CassetteMode,
    MatchOn,
    RecordingProvider,
    ReplayingProvider,
    ScriptedProvider,
    mode_from_env,
)
from tesserix_adk.testing.pytest_plugin import cassette_dir as shipped_dir
from tesserix_adk.testing.pytest_plugin import cassettes

if TYPE_CHECKING:
    from collections.abc import Sequence

pytestmark = pytest.mark.anyio


@pytest.fixture
def cassette_dir(tmp_path: Path) -> Path:
    """Keep the fixture's cassettes out of the repository."""
    return tmp_path


def _request(text: str = "when is the next train", model: str = "gpt-5") -> ModelRequest:
    return ModelRequest(
        model=model, messages=(Message(role="user", content=[TextPart(text=text)]),)
    )


def _live(*texts: str) -> ScriptedProvider:
    return ScriptedProvider(*(ModelResponse(content=text) for text in texts), name="openai")


async def _recorded(*texts: str) -> Cassette:
    recorder = RecordingProvider(_live(*texts), provider="openai", version="1.2")
    for text in texts:
        await recorder.complete(_request(text))
    return recorder.cassette


class TestModes:
    def test_replay_is_the_default_so_a_suite_never_spends_by_accident(self) -> None:
        assert mode_from_env({}) is CassetteMode.REPLAY

    def test_the_mode_is_read_from_the_environment(self) -> None:
        assert mode_from_env({ENV_MODE: "refresh"}) is CassetteMode.REFRESH

    def test_the_mode_is_read_case_insensitively(self) -> None:
        assert mode_from_env({ENV_MODE: "RECORD"}) is CassetteMode.RECORD

    def test_a_mode_nobody_defined_is_refused_rather_than_guessed(self) -> None:
        with pytest.raises(ConfigurationError, match="live"):
            mode_from_env({ENV_MODE: "live"})


class TestMatchStrategy:
    def test_by_default_every_field_of_the_request_has_to_match(self) -> None:
        one = fingerprint_of(_request("a"))
        other = fingerprint_of(_request("b"))
        assert MatchOn().key(one) != MatchOn().key(other)

    def test_a_field_left_out_of_the_strategy_stops_separating_requests(self) -> None:
        """A test about tool wiring should not re-record because a word changed."""
        loose = MatchOn(messages=False)
        assert loose.key(fingerprint_of(_request("a"))) == loose.key(fingerprint_of(_request("b")))

    def test_the_model_still_separates_requests_under_a_loose_strategy(self) -> None:
        loose = MatchOn(messages=False)
        one = fingerprint_of(_request("a", model="gpt-5"))
        other = fingerprint_of(_request("a", model="claude"))
        assert loose.key(one) != loose.key(other)

    def test_a_strategy_that_matches_everything_is_refused(self) -> None:
        """A cassette that answers any request is a green test asserting nothing."""
        with pytest.raises(ValueError, match="at least one"):
            MatchOn(model=False, messages=False, tools=False, output_schema=False, hooks=False)


class TestAMissIsNeverPapered:
    async def test_an_unrecorded_request_raises_rather_than_calling_anything(self) -> None:
        replay = ReplayingProvider(await _recorded("recorded question"))
        with pytest.raises(CassetteMismatchError):
            await replay.complete(_request("a question nobody recorded"))

    async def test_the_error_names_the_field_that_moved(self) -> None:
        replay = ReplayingProvider(await _recorded("recorded question"))
        with pytest.raises(CassetteMismatchError, match="messages"):
            await replay.complete(_request("a question nobody recorded"))

    async def test_the_error_says_how_to_re_record(self) -> None:
        replay = ReplayingProvider(await _recorded("recorded question"))
        with pytest.raises(CassetteMismatchError, match="ADK_CASSETTE_MODE=refresh"):
            await replay.complete(_request("a question nobody recorded"))

    async def test_asking_more_times_than_the_recording_did_is_its_own_message(self) -> None:
        replay = ReplayingProvider(await _recorded("q"))
        await replay.complete(_request("q"))
        with pytest.raises(CassetteMismatchError, match="more times"):
            await replay.complete(_request("q"))


class TestStreaming:
    async def test_a_recorded_stream_replays_with_the_chunk_boundaries_it_had(self) -> None:
        """A consumer that renders per chunk behaves differently on different boundaries."""
        recorder = RecordingProvider(_live("one two three"), provider="openai")
        recorded = [event async for event in await recorder.stream(_request())]
        replay = ReplayingProvider(recorder.cassette)
        replayed = [event async for event in await replay.stream(_request())]
        assert _texts(replayed) == _texts(recorded)

    async def test_a_replayed_stream_ends_with_the_assembled_response(self) -> None:
        recorder = RecordingProvider(_live("hello there"), provider="openai")
        async for _ in await recorder.stream(_request()):
            pass
        replay = ReplayingProvider(recorder.cassette)
        events = [event async for event in await replay.stream(_request())]
        assert isinstance(events[-1], StreamEnd)
        assert events[-1].response.content == "hello there"

    async def test_a_recorded_tool_call_replays_as_a_tool_call_delta(self) -> None:
        answered = ModelResponse(tool_calls=(ToolCall(id="c1", name="timetable"),))
        recorder = RecordingProvider(ScriptedProvider(answered, name="openai"), provider="openai")
        async for _ in await recorder.stream(_request()):
            pass
        replay = ReplayingProvider(recorder.cassette)
        events = [event async for event in await replay.stream(_request())]
        assert any(isinstance(event, ToolCallDelta) for event in events)

    async def test_a_recorded_failure_is_raised_by_the_replayed_stream_too(self) -> None:
        """A recovery path tested only on the buffered call is a path half tested."""
        failing = ScriptedProvider(ProviderError("upstream is down", status=503), name="openai")
        recorder = RecordingProvider(failing, provider="openai")
        with pytest.raises(ProviderError):
            await recorder.complete(_request())
        replay = ReplayingProvider(recorder.cassette)
        with pytest.raises(ProviderError, match="upstream is down"):
            await replay.stream(_request())

    async def test_a_consumer_that_stops_part_way_still_records_what_arrived(self) -> None:
        """The cancellation point is what the test was about; losing it loses the test."""
        recorder = RecordingProvider(_live("one two three"), provider="openai")
        stream = await recorder.stream(_request())
        await anext(stream)
        await stream.aclose()  # type: ignore[attr-defined]
        assert recorder.cassette.interactions[0].chunks


class TestTheHarness:
    async def test_replay_mode_never_reaches_the_live_provider(self, tmp_path: Path) -> None:
        path = tmp_path / "trains.json"
        (await _recorded("q")).save(path)
        harness = CassetteHarness(path, mode=CassetteMode.REPLAY)
        assert isinstance(harness.provider(), ReplayingProvider)

    async def test_replay_mode_with_no_cassette_says_how_to_record_one(
        self, tmp_path: Path
    ) -> None:
        harness = CassetteHarness(tmp_path / "absent.json", mode=CassetteMode.REPLAY)
        with pytest.raises(CassetteMismatchError, match="ADK_CASSETTE_MODE=record"):
            harness.provider()

    async def test_record_mode_replays_a_cassette_that_already_exists(self, tmp_path: Path) -> None:
        """Recording over a cassette on every run is how a suite spends without meaning to."""
        path = tmp_path / "trains.json"
        (await _recorded("q")).save(path)
        harness = CassetteHarness(path, mode=CassetteMode.RECORD)
        assert isinstance(harness.provider(_live("q")), ReplayingProvider)

    async def test_record_mode_records_when_there_is_nothing_to_replay(
        self, tmp_path: Path
    ) -> None:
        harness = CassetteHarness(tmp_path / "new.json", mode=CassetteMode.RECORD)
        assert isinstance(harness.provider(_live("q")), RecordingProvider)

    async def test_refresh_mode_records_over_a_cassette_that_exists(self, tmp_path: Path) -> None:
        path = tmp_path / "trains.json"
        (await _recorded("old")).save(path)
        harness = CassetteHarness(path, mode=CassetteMode.REFRESH)
        assert isinstance(harness.provider(_live("new")), RecordingProvider)

    async def test_recording_with_no_live_provider_is_a_wiring_error(self, tmp_path: Path) -> None:
        harness = CassetteHarness(tmp_path / "new.json", mode=CassetteMode.REFRESH)
        with pytest.raises(ConfigurationError, match="live provider"):
            harness.provider()

    async def test_what_was_recorded_is_written_where_the_harness_was_pointed(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "new.json"
        harness = CassetteHarness(path, mode=CassetteMode.REFRESH)
        provider = harness.provider(_live("answer"))
        await provider.complete(_request())
        harness.save()
        assert Cassette.load(path).interactions

    async def test_a_replay_writes_nothing_back(self, tmp_path: Path) -> None:
        path = tmp_path / "trains.json"
        (await _recorded("q")).save(path)
        before = path.read_text()
        harness = CassetteHarness(path, mode=CassetteMode.REPLAY)
        harness.provider()
        harness.save()
        assert path.read_text() == before

    async def test_the_match_strategy_reaches_the_provider_the_harness_builds(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "trains.json"
        (await _recorded("recorded question")).save(path)
        harness = CassetteHarness(path, mode=CassetteMode.REPLAY, match=MatchOn(messages=False))
        replay = harness.provider()
        assert (await replay.complete(_request("a different question"))).content


class TestSharingOneFile:
    async def test_two_tests_replaying_one_cassette_do_not_consume_each_other_s_turns(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "trains.json"
        (await _recorded("q")).save(path)
        first = CassetteHarness(path, mode=CassetteMode.REPLAY).provider()
        second = CassetteHarness(path, mode=CassetteMode.REPLAY).provider()
        assert (await first.complete(_request("q"))).content
        assert (await second.complete(_request("q"))).content


class TestNothingSecretIsCommitted:
    async def test_a_recording_that_still_holds_a_credential_is_not_written(
        self, tmp_path: Path
    ) -> None:
        """A token in a committed file outlives every run that used it."""
        cassette = await _recorded("q")
        smuggled = cassette.model_copy(update={"provider_version": "sk-test-abcdefghijkl"})
        with pytest.raises(CassetteLeakError, match="credential"):
            smuggled.save(tmp_path / "leaky.json")

    async def test_a_secret_in_a_tool_argument_is_redacted_before_it_is_written(
        self, tmp_path: Path
    ) -> None:
        recorder = RecordingProvider(
            ScriptedProvider(
                ModelResponse(content="using key sk-test-abcdefghijkl"), name="openai"
            ),
            provider="openai",
        )
        await recorder.complete(_request())
        recorder.cassette.save(tmp_path / "clean.json")
        assert "sk-test-" not in (tmp_path / "clean.json").read_text()


class _Unmarked:
    """A test node carrying no cassette marker, which is the wiring mistake."""

    name = "test_without_a_marker"

    def get_closest_marker(self, name: str) -> None:  # noqa: ARG002 — the point is None
        return None


class TestTheFixture:
    @pytest.mark.cassette("greeting")
    def test_the_marker_names_the_file_the_harness_loads(self, cassettes: CassetteHarness) -> None:
        assert cassettes.path.name == "greeting.json"

    @pytest.mark.cassette("greeting")
    def test_the_fixture_replays_unless_the_environment_says_otherwise(
        self, cassettes: CassetteHarness
    ) -> None:
        assert cassettes.mode is CassetteMode.REPLAY

    def test_asking_for_the_fixture_without_a_marker_is_a_wiring_error(
        self, tmp_path: Path
    ) -> None:
        request = cast("pytest.FixtureRequest", SimpleNamespace(node=_Unmarked()))
        unwrapped = cassettes._get_wrapped_function()
        with pytest.raises(ConfigurationError, match=r"pytest\.mark\.cassette"):
            next(unwrapped(request, tmp_path))

    def test_cassettes_live_under_the_tests_tree_by_default(self) -> None:
        """Overridden above for this suite; the shipped default is what consumers get."""
        assert shipped_dir._get_wrapped_function()() == Path("tests/cassettes")


class TestRecordingIsTransparent:
    def test_the_recorder_counts_tokens_the_way_the_provider_it_wraps_does(self) -> None:
        live = _live("q")
        recorder = RecordingProvider(live, provider="openai")
        messages = (Message(role="user", content=[TextPart(text="when is the next train")]),)
        assert recorder.count_tokens(messages) == live.count_tokens(messages)


def _texts(events: Sequence[object]) -> list[str]:
    return [event.text for event in events if isinstance(event, TextDelta)]
