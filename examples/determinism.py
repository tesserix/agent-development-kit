"""Record a run once, then test it forever without a network.

Four scenarios: a run recorded and replayed, a replay that refuses to guess when the prompt
moves, a recorded failure whose retry replays too, and a cassette proving it holds no
secrets. Every one of them runs offline.

Run it with `python examples/determinism.py`.
"""

from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from tesserix_adk.core import (
    Agent,
    Message,
    ModelCapabilities,
    ProviderError,
    RetryConfig,
    Run,
    ToolCall,
    Usage,
)
from tesserix_adk.runtime import AgentRunner, ModelRequest, ModelResponse
from tesserix_adk.testing import (
    Cassette,
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

# Credential-shaped, credential to nothing: the fixture that proves redaction works.
FAKE_TOKEN = "sk-live-4eC39H"  # noqa: S105 — a fixture, not a credential; gitleaks:allow


class FlakyProvider:
    """Fails once with a 503, then answers — the shape a cassette must keep both halves of."""

    def __init__(self, response: ModelResponse) -> None:
        self._response = response
        self.calls = 0

    @property
    def name(self) -> str:
        """What this provider is called."""
        return "flaky"

    @property
    def capabilities(self) -> ModelCapabilities:
        """What it declares it can do; the kit checks this before it calls."""
        return ModelCapabilities(context_window_tokens=200_000)

    def count_tokens(self, messages: Sequence[Message]) -> int:
        """Estimated, since this example has no tokeniser to call."""
        return estimate_tokens(messages)

    async def complete(self, request: ModelRequest) -> ModelResponse:  # noqa: ARG002
        """Fail the first call, answer the second."""
        self.calls += 1
        if self.calls == 1:
            raise ProviderError("upstream hiccup", status=503)
        return self._response

    async def stream(self, request: ModelRequest) -> object:
        """Not streamed here."""
        raise NotImplementedError("see #150")


def agent(**overrides: object) -> Agent:
    """The same clerk throughout."""
    fields: dict[str, object] = {
        "name": "clerk",
        "instructions": "Answer from sources. Cite the page.",
        "free_text": True,
        "model": "claude-sonnet-5",
    }
    return Agent(**{**fields, **overrides})  # type: ignore[arg-type]


def answering(content: str = "The 09:12 to Kyoto.") -> ModelResponse:
    """A plain answer."""
    return ModelResponse(content=content, usage=Usage(input_tokens=10, output_tokens=5))


def searching(**arguments: object) -> ModelResponse:
    """A turn that calls a tool with whatever arguments it was handed."""
    return ModelResponse(
        tool_calls=(ToolCall(id="call_1", name="search", arguments=arguments),),
        usage=Usage(input_tokens=10, output_tokens=5),
    )


def tools() -> FakeToolRegistry:
    """One search tool."""
    return FakeToolRegistry({"search": lambda **_: "a timetable"})


def report(title: str, run: Run) -> None:
    """Print how the run ended and what it answered."""
    print(f"\n{title}")  # noqa: T201
    print(f"  state: {run.state}")  # noqa: T201
    if run.events and run.state.value == "failed":
        print(f"  reason: {run.events[-1].detail}")  # noqa: T201


async def a_recorded_run_replays_offline() -> None:
    """Record once against a provider, then run the same agent from the file."""
    recorder = RecordingProvider(ScriptedProvider(answering()), provider="scripted", version="1.0")
    await AgentRunner(provider=recorder, ids=SequentialIds()).run(
        agent(), "Trains to Kyoto?", tenant="acme"
    )

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "trains.json"
        recorder.cassette.save(path)
        runs = [
            await AgentRunner(
                provider=ReplayingProvider(Cassette.load(path)),
                ids=SequentialIds(),
                clock=FakeClock(),
            ).run(agent(), "Trains to Kyoto?", tenant="acme")
            for _ in range(2)
        ]

    report("a recorded run, replayed twice", runs[0])
    assert_same_run(runs[0], runs[1])
    print("  two replays produced the same run")  # noqa: T201


async def a_changed_prompt_misses_rather_than_guessing() -> None:
    """The nearest recording is not the recording. A replay says which field moved."""
    recorder = RecordingProvider(ScriptedProvider(answering()), provider="scripted")
    await AgentRunner(provider=recorder).run(agent(), "Trains to Kyoto?", tenant="acme")

    replay = ReplayingProvider(recorder.cassette)
    run = await AgentRunner(provider=replay).run(agent(), "Trains to Osaka?", tenant="acme")

    report("the prompt changed since recording", run)
    print(f"  provider calls served: {len(replay.served)}")  # noqa: T201


async def a_recorded_failure_replays_with_its_recovery() -> None:
    """A cassette of only the happy path proves the recovery nobody tested."""
    recorder = RecordingProvider(FlakyProvider(answering()), provider="flaky")
    retrying = RetryConfig(max_attempts=2, base_delay_seconds=0.0)
    await AgentRunner(provider=recorder, retry=retrying, clock=FakeClock()).run(
        agent(), "Trains to Kyoto?", tenant="acme"
    )

    run = await AgentRunner(
        provider=ReplayingProvider(recorder.cassette), retry=retrying, clock=FakeClock()
    ).run(agent(), "Trains to Kyoto?", tenant="acme")

    report("a recorded 503 and the retry after it", run)
    print(f"  events: {[event.kind.value for event in run.events][:4]}")  # noqa: T201


async def a_cassette_carries_no_secrets() -> None:
    """A cassette is a file people commit, and a token in one outlives the run that used it."""
    leaking = searching(q="kyoto", api_key=FAKE_TOKEN, page=2)
    recorder = RecordingProvider(ScriptedProvider(leaking, answering()), provider="scripted")
    await AgentRunner(provider=recorder, tools=tools()).run(
        agent(tools=("search",)), "Trains to Kyoto?", tenant="acme"
    )

    recorded = json.dumps(recorder.cassette.model_dump(mode="json"))
    print("\nwhat the cassette kept")  # noqa: T201
    print(f"  holds the secret: {FAKE_TOKEN in recorded}")  # noqa: T201
    print(f"  holds the prompt: {'Trains to Kyoto?' in recorded}")  # noqa: T201
    print(f"  fingerprint: {recorder.cassette.interactions[0].fingerprint.digest[:12]}…")  # noqa: T201


async def main() -> None:
    """Four runs, no network, one recording each."""
    await a_recorded_run_replays_offline()
    await a_changed_prompt_misses_rather_than_guessing()
    await a_recorded_failure_replays_with_its_recovery()
    await a_cassette_carries_no_secrets()


if __name__ == "__main__":
    asyncio.run(main())
