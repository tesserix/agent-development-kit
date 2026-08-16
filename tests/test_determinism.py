"""The same inputs produce the same run, and a recorded run replays without a network.

Behaviour that cannot be regression-tested is behaviour nobody can change safely. Every
source of ambient non-determinism the loop owns — the clock, the ids, the jitter, the order
of an assembled prompt — is injectable, and what the loop does not own is recorded.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import pytest

from tesserix_adk.core import (
    Agent,
    HookChain,
    HookDecision,
    HookPoint,
    HookSubject,
    Message,
    ModelCapabilities,
    ProviderError,
    RetryConfig,
    RunEventKind,
    RunState,
    TextPart,
    ToolCall,
    Usage,
)
from tesserix_adk.runtime import (
    AgentRunner,
    ModelRequest,
    ModelResponse,
    ToolDeclaration,
    fingerprint_of,
)
from tesserix_adk.testing import (
    CAPABLE,
    Cassette,
    CassetteMissError,
    CassetteVersionError,
    FakeClock,
    FakeToolRegistry,
    RecordingProvider,
    ReplayingProvider,
    ScriptedProvider,
    SequentialIds,
    assert_same_run,
    estimate_tokens,
)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

# Credential-shaped, credential to nothing: the fixture that proves redaction works.
FAKE_TOKEN = "sk-live-4eC39H"  # noqa: S105 — a fixture, not a credential; gitleaks:allow


def agent(**overrides: object) -> Agent:
    fields: dict[str, object] = {
        "name": "planner",
        "instructions": "Plan trips.",
        "free_text": True,
        "model": "claude-sonnet-5",
    }
    return Agent(**{**fields, **overrides})  # type: ignore[arg-type]


def answering(content: str = "Kyoto.") -> ModelResponse:
    return ModelResponse(content=content, usage=Usage(input_tokens=10, output_tokens=5))


def calling(tool: str = "search", **arguments: Any) -> ModelResponse:
    return ModelResponse(
        tool_calls=(ToolCall(id="call_1", name=tool, arguments=arguments),),
        usage=Usage(input_tokens=10, output_tokens=5),
    )


def tools() -> FakeToolRegistry:
    return FakeToolRegistry({"search": lambda **_: "a result"})


class FailingOnceProvider:
    """Fails the first call and answers the second, so a retry is on the recording."""

    name = "flaky"

    def __init__(self, response: ModelResponse) -> None:
        self._response = response
        self.calls = 0

    @property
    def capabilities(self) -> ModelCapabilities:
        return CAPABLE

    def count_tokens(self, messages: Sequence[Message]) -> int:
        return estimate_tokens(messages)

    async def complete(self, request: ModelRequest) -> ModelResponse:  # noqa: ARG002
        self.calls += 1
        if self.calls == 1:
            raise ProviderError("upstream hiccup", status=503)
        return self._response

    async def stream(self, request: ModelRequest) -> Any:
        raise NotImplementedError("see #150")


class Stamping:
    """A hook that reads wall-clock time, which is exactly what replay cannot reproduce."""

    name = "stamping"
    points = (HookPoint.BEFORE_PROMPT_ASSEMBLY,)

    async def on(self, subject: HookSubject) -> HookDecision:
        return HookDecision.rewrite(f"{subject.content} at {time.time_ns()}", reason="stamped")


def said(text: str) -> Message:
    return Message(role="user", content=[TextPart(text=text)])


def declaration(parameters: dict[str, Any] | None = None) -> ToolDeclaration:
    return ToolDeclaration(
        name="search", description="Search.", parameters=parameters or {"q": "string"}
    )


def request(**overrides: Any) -> ModelRequest:
    fields: dict[str, Any] = {
        "model": "claude-sonnet-5",
        "messages": (said("Trains to Kyoto?"),),
        "tools": (declaration(),),
    }
    return ModelRequest(**{**fields, **overrides})


class TestIdsAreInjectable:
    async def test_two_runs_given_the_same_ids_agree_on_them(self) -> None:
        """A uuid4 in the loop is a field no assertion can name."""
        runs = [
            await AgentRunner(
                provider=ScriptedProvider(answering()), ids=SequentialIds(), clock=FakeClock()
            ).run(agent(), "Trains to Kyoto?", tenant="acme")
            for _ in range(2)
        ]

        assert runs[0].id == runs[1].id

    async def test_an_explicit_run_id_still_wins(self) -> None:
        runner = AgentRunner(provider=ScriptedProvider(answering()), ids=SequentialIds())

        run = await runner.run(agent(), "Trains?", tenant="acme", run_id="run_declared")

        assert run.id == "run_declared"

    async def test_ids_are_unique_within_a_process(self) -> None:
        """Deterministic is not constant: two runs of one factory are still two runs."""
        ids = SequentialIds()
        runner = AgentRunner(provider=ScriptedProvider(answering(), answering()), ids=ids)

        first = await runner.run(agent(), "Trains?", tenant="acme")
        second = await runner.run(agent(), "Trains?", tenant="acme")

        assert first.id != second.id

    async def test_without_an_id_factory_a_run_still_gets_an_id(self) -> None:
        run = await AgentRunner(provider=ScriptedProvider(answering())).run(
            agent(), "Trains?", tenant="acme"
        )

        assert run.id


class TestTheFingerprint:
    def test_the_same_request_fingerprints_the_same_twice(self) -> None:
        assert fingerprint_of(request()).digest == fingerprint_of(request()).digest

    def test_argument_order_does_not_change_a_fingerprint(self) -> None:
        """A dict that iterates in a different order is the same request, not a new one."""
        one = request(tools=(declaration({"a": 1, "b": 2}),))
        two = request(tools=(declaration({"b": 2, "a": 1}),))

        assert fingerprint_of(one).digest == fingerprint_of(two).digest

    def test_a_changed_model_is_a_different_fingerprint(self) -> None:
        changed = fingerprint_of(request(model="claude-opus-5"))

        assert changed.digest != fingerprint_of(request()).digest
        assert changed.diff(fingerprint_of(request())) == ("model",)

    def test_a_changed_prompt_is_a_different_fingerprint(self) -> None:
        changed = fingerprint_of(request(messages=(said("Trains to Osaka?"),)))

        assert changed.diff(fingerprint_of(request())) == ("messages",)

    def test_a_changed_tool_schema_is_a_different_fingerprint(self) -> None:
        """The model was told a different thing about the tool, so it is a different call."""
        changed = fingerprint_of(request(tools=(declaration({"q": "string", "page": "int"}),)))

        assert changed.diff(fingerprint_of(request(tools=(declaration(),)))) == ("tools",)

    def test_a_changed_output_schema_is_a_different_fingerprint(self) -> None:
        changed = fingerprint_of(request(output_schema={"type": "object"}))

        assert changed.diff(fingerprint_of(request())) == ("output_schema",)

    def test_a_changed_hook_chain_is_a_different_fingerprint(self) -> None:
        """Hooks rewrite prompts, so a chain that changed can change what was asked."""
        changed = fingerprint_of(request(), hooks=("redactor",))

        assert changed.diff(fingerprint_of(request())) == ("hooks",)

    def test_every_diverging_field_is_named_at_once(self) -> None:
        changed = fingerprint_of(request(model="claude-opus-5"), hooks=("redactor",))

        assert changed.diff(fingerprint_of(request())) == ("hooks", "model")

    def test_two_identical_fingerprints_diverge_nowhere(self) -> None:
        assert fingerprint_of(request()).diff(fingerprint_of(request())) == ()


class TestRecording:
    async def test_a_recorded_run_can_be_replayed_without_a_provider(self, tmp_path: Path) -> None:
        recorder = RecordingProvider(ScriptedProvider(answering()), provider="scripted")
        await AgentRunner(provider=recorder, ids=SequentialIds()).run(
            agent(), "Trains to Kyoto?", tenant="acme"
        )
        recorder.cassette.save(tmp_path / "trains.json")

        replayed = await AgentRunner(
            provider=ReplayingProvider(Cassette.load(tmp_path / "trains.json")),
            ids=SequentialIds(),
        ).run(agent(), "Trains to Kyoto?", tenant="acme")

        assert replayed.state is RunState.COMPLETED
        assert replayed.messages[-1].content[0].text == "Kyoto."  # type: ignore[union-attr]

    async def test_a_recording_carries_what_it_was_recorded_against(self) -> None:
        """A cassette replayed on trust across an SDK upgrade is a green test that lies."""
        recorder = RecordingProvider(
            ScriptedProvider(answering()), provider="scripted", version="1.2.0"
        )
        await AgentRunner(provider=recorder).run(agent(), "Trains?", tenant="acme")

        assert recorder.cassette.provider == "scripted"
        assert recorder.cassette.provider_version == "1.2.0"

    async def test_secrets_never_reach_the_cassette(self, tmp_path: Path) -> None:
        """A cassette is a file people commit; a token in it outlives the run that used it."""
        leaking = calling("search", token=FAKE_TOKEN, q="kyoto")
        recorder = RecordingProvider(ScriptedProvider(leaking, answering()), provider="scripted")
        await AgentRunner(provider=recorder, tools=tools()).run(
            agent(tools=("search",)), "Trains?", tenant="acme"
        )
        path = tmp_path / "secret.json"
        recorder.cassette.save(path)

        assert FAKE_TOKEN not in path.read_text()
        assert "[REDACTED]" in path.read_text()

    async def test_a_recorded_failure_and_its_retry_both_replay(self) -> None:
        """A cassette of only the happy path proves the recovery nobody tested."""
        recorder = RecordingProvider(FailingOnceProvider(answering()), provider="scripted")
        retrying = RetryConfig(max_attempts=2, base_delay_seconds=0.0)
        await AgentRunner(provider=recorder, retry=retrying, clock=FakeClock()).run(
            agent(), "Trains?", tenant="acme"
        )

        replayed = await AgentRunner(
            provider=ReplayingProvider(recorder.cassette), retry=retrying, clock=FakeClock()
        ).run(agent(), "Trains?", tenant="acme")

        assert replayed.state is RunState.COMPLETED
        assert [event.kind for event in replayed.events].count(RunEventKind.ATTEMPT_FAILED) == 1


class TestReplayRefusesToGuess:
    async def test_a_changed_prompt_misses_and_says_which_field(self) -> None:
        recorder = RecordingProvider(ScriptedProvider(answering()), provider="scripted")
        await AgentRunner(provider=recorder).run(agent(), "Trains to Kyoto?", tenant="acme")

        run = await AgentRunner(provider=ReplayingProvider(recorder.cassette)).run(
            agent(), "Trains to Osaka?", tenant="acme"
        )

        assert run.state is RunState.FAILED
        assert "CassetteMismatchError" in (run.events[-1].detail or "")
        assert "messages" in (run.events[-1].detail or "")

    async def test_a_changed_model_misses_and_says_so(self) -> None:
        recorder = RecordingProvider(ScriptedProvider(answering()), provider="scripted")
        await AgentRunner(provider=recorder).run(agent(), "Trains?", tenant="acme")

        run = await AgentRunner(provider=ReplayingProvider(recorder.cassette)).run(
            agent(model="claude-opus-5"), "Trains?", tenant="acme"
        )

        assert "model" in (run.events[-1].detail or "")

    async def test_replay_never_falls_through_to_a_provider(self) -> None:
        """The point of a miss is that no network call happens instead."""
        recorder = RecordingProvider(ScriptedProvider(answering()), provider="scripted")
        await AgentRunner(provider=recorder).run(agent(), "Trains?", tenant="acme")
        replay = ReplayingProvider(recorder.cassette)

        run = await AgentRunner(provider=replay).run(
            agent(), "Something else entirely.", tenant="acme"
        )

        assert replay.served == []
        assert run.state is RunState.FAILED
        assert isinstance(CassetteMissError("x"), Exception)

    async def test_a_cassette_from_another_provider_version_is_refused(self) -> None:
        recorder = RecordingProvider(
            ScriptedProvider(answering()), provider="scripted", version="1.2.0"
        )
        await AgentRunner(provider=recorder).run(agent(), "Trains?", tenant="acme")

        with pytest.raises(CassetteVersionError, match=r"1\.2\.0"):
            ReplayingProvider(recorder.cassette, expect_version="2.0.0")


class TestTwoRunsUnderReplay:
    async def test_the_same_agent_replayed_twice_produces_the_same_run(self) -> None:
        recorder = RecordingProvider(ScriptedProvider(calling(), answering()), provider="scripted")
        await AgentRunner(provider=recorder, tools=tools()).run(
            agent(tools=("search",)), "Trains?", tenant="acme"
        )

        runs = [
            await AgentRunner(
                provider=ReplayingProvider(recorder.cassette),
                tools=tools(),
                ids=SequentialIds(),
                clock=FakeClock(),
            ).run(agent(tools=("search",)), "Trains?", tenant="acme")
            for _ in range(2)
        ]

        assert_same_run(runs[0], runs[1])

    async def test_a_hook_reading_the_wall_clock_is_caught(self) -> None:
        """A hook that reads time directly defeats replay, so the comparison must fail."""
        runs = [
            await AgentRunner(
                provider=ScriptedProvider(answering()),
                hooks=HookChain([Stamping()]),
                ids=SequentialIds(),
                clock=FakeClock(),
            ).run(agent(), "Trains?", tenant="acme")
            for _ in range(2)
        ]

        with pytest.raises(AssertionError, match="hook_rewrite"):
            assert_same_run(runs[0], runs[1])

    async def test_runs_differing_only_in_timing_are_the_same_run(self) -> None:
        """Timings are normalised: a slower machine is not a behaviour change."""
        recorder = RecordingProvider(ScriptedProvider(answering()), provider="scripted")
        await AgentRunner(provider=recorder).run(agent(), "Trains?", tenant="acme")

        first = await AgentRunner(
            provider=ReplayingProvider(recorder.cassette), ids=SequentialIds(), clock=FakeClock(0.0)
        ).run(agent(), "Trains?", tenant="acme")
        second = await AgentRunner(
            provider=ReplayingProvider(recorder.cassette),
            ids=SequentialIds(),
            clock=FakeClock(9_999.0),
        ).run(agent(), "Trains?", tenant="acme")

        assert_same_run(first, second)


class TestWhatACassetteRefusesToDo:
    def test_a_cassette_written_in_another_format_is_not_read(self, tmp_path: Path) -> None:
        path = tmp_path / "old.json"
        path.write_text('{"format": "0", "provider": "scripted", "interactions": []}')

        with pytest.raises(CassetteVersionError, match="format"):
            Cassette.load(path)

    async def test_a_cassette_from_another_provider_is_refused(self) -> None:
        recorder = RecordingProvider(ScriptedProvider(answering()), provider="scripted")
        await AgentRunner(provider=recorder).run(agent(), "Trains?", tenant="acme")

        with pytest.raises(CassetteVersionError, match="scripted"):
            ReplayingProvider(recorder.cassette, expect_provider="anthropic")

    async def test_asking_more_times_than_were_recorded_is_a_miss(self) -> None:
        recorder = RecordingProvider(ScriptedProvider(answering()), provider="scripted")
        await AgentRunner(provider=recorder).run(agent(), "Trains?", tenant="acme")
        replay = ReplayingProvider(recorder.cassette)
        await AgentRunner(provider=replay).run(agent(), "Trains?", tenant="acme")

        run = await AgentRunner(provider=replay).run(agent(), "Trains?", tenant="acme")

        assert "asked more times" in (run.events[-1].detail or "")

    async def test_an_empty_cassette_says_it_is_empty(self) -> None:
        run = await AgentRunner(provider=ReplayingProvider(Cassette(provider="scripted"))).run(
            agent(), "Trains?", tenant="acme"
        )

        assert "cassette is empty" in (run.events[-1].detail or "")

    async def test_a_secret_nested_in_an_argument_is_redacted(self, tmp_path: Path) -> None:
        nested = calling("search", filters={"api_key": FAKE_TOKEN}, page=2, deep=True)
        recorder = RecordingProvider(ScriptedProvider(nested, answering()), provider="scripted")
        await AgentRunner(provider=recorder, tools=tools()).run(
            agent(tools=("search",)), "Trains?", tenant="acme"
        )
        path = tmp_path / "nested.json"
        recorder.cassette.save(path)

        assert FAKE_TOKEN not in path.read_text()
        assert '"page": 2' in path.read_text()


class TestComparingRuns:
    async def test_two_different_runs_are_reported_with_the_field_that_moved(self) -> None:
        completed = await AgentRunner(provider=ScriptedProvider(answering())).run(
            agent(), "Trains?", tenant="acme", run_id="run_1"
        )
        failed = await AgentRunner(provider=ReplayingProvider(Cassette(provider="scripted"))).run(
            agent(), "Trains?", tenant="acme", run_id="run_1"
        )

        with pytest.raises(AssertionError, match="state"):
            assert_same_run(completed, failed)

    async def test_a_run_that_did_more_is_not_the_same_run(self) -> None:
        """Equal prefixes are not equal runs: the extra step is the whole finding."""
        plain = await AgentRunner(provider=ScriptedProvider(answering()), ids=SequentialIds()).run(
            agent(), "Trains?", tenant="acme"
        )
        with_tool = await AgentRunner(
            provider=ScriptedProvider(calling(), answering()),
            tools=tools(),
            ids=SequentialIds(),
        ).run(agent(tools=("search",)), "Trains?", tenant="acme")

        with pytest.raises(AssertionError, match=r"usage|diverge"):
            assert_same_run(plain, with_tool, message="tool run")

    async def test_a_longer_run_with_the_same_prefix_diverges_in_length(self) -> None:
        short = await AgentRunner(provider=ScriptedProvider(answering()), ids=SequentialIds()).run(
            agent(), "Trains?", tenant="acme"
        )
        clipped = short.model_copy(update={"events": short.events[:-1]})

        with pytest.raises(AssertionError, match="length"):
            assert_same_run(short, clipped)


class TestCanonicalOrdering:
    def test_a_list_inside_a_tool_schema_keeps_its_order(self) -> None:
        """Order in a list is meaning; order in a dict is not."""
        one = fingerprint_of(request(tools=(declaration({"enum": ["a", "b"]}),)))
        two = fingerprint_of(request(tools=(declaration({"enum": ["b", "a"]}),)))

        assert one.digest != two.digest
        assert (
            one.digest == fingerprint_of(request(tools=(declaration({"enum": ["a", "b"]}),))).digest
        )
