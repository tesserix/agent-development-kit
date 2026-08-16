"""Record a real run once, then test its behaviour offline for as long as it holds.

Live provider calls in CI are slow, costly and flaky, so teams stop running them and stop
noticing when a prompt change moves behaviour. A cassette keys each recorded exchange by
the fingerprint of the request that produced it, so a replay either serves the exchange
that was actually recorded or fails saying which field moved. There is no third path: a
replay that quietly reused the nearest response would be a green test asserting nothing.

A cassette holds digests of the request and never its content, and redacts secrets out of
what it does keep. It is a file people commit, and a token in a committed file outlives the
run that used it.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from pydantic import Field, model_validator

from tesserix_adk.core.errors import AdkError, ConfigurationError, ProviderError
from tesserix_adk.core.models import AdkModel
from tesserix_adk.core.streaming import StreamEnd, TextDelta, ToolCallDelta, UsageDelta
from tesserix_adk.runtime import (
    ModelResponse,
    RunFingerprint,
    canonical_digest,
    fingerprint_of,
)
from tesserix_adk.testing.fakes import CAPABLE, estimate_tokens

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterable, Sequence
    from pathlib import Path

    from tesserix_adk.core import Message, ModelCapabilities, ModelProvider, Run
    from tesserix_adk.core.streaming import StreamEvent
    from tesserix_adk.runtime import ModelRequest

__all__ = [
    "ENV_MODE",
    "Cassette",
    "CassetteHarness",
    "CassetteLeakError",
    "CassetteMismatchError",
    "CassetteMissError",
    "CassetteMode",
    "CassetteVersionError",
    "Interaction",
    "MatchOn",
    "RecordedError",
    "RecordingProvider",
    "ReplayingProvider",
    "assert_same_run",
    "mode_from_env",
    "redacted",
]


CASSETTE_FORMAT = "1"

REDACTED = "[REDACTED]"

_REFRESH = "Re-record it with ADK_CASSETTE_MODE=refresh"

_SECRET_KEYS = re.compile(
    r"(token|secret|password|passwd|api[_-]?key|authorization|auth|credential|cookie|session)",
    re.IGNORECASE,
)

# Shapes worth catching wherever they appear: a key named innocently still leaks a token.
_SECRET_VALUES = (
    re.compile(r"\b(sk|pk|rk)-[A-Za-z0-9-]{6,}\b"),
    re.compile(r"\bBearer\s+[A-Za-z0-9._-]{8,}\b", re.IGNORECASE),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{16,}\b"),
    re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]{2,}\b"),
    re.compile(r"\b(?:\d[ -]?){13,19}\b"),
)


class CassetteMissError(AdkError):
    """Raised when a replay is asked for an exchange the cassette does not hold.

    Names the fields on which the request diverged from the nearest recording, because a
    bare miss sends the reader off to diff two blobs by eye.
    """


class CassetteMismatchError(CassetteMissError):
    """Raised when the request a run issued is not the one that was recorded.

    Carries the diverging fields and the command that re-records, so the reader is not
    left to work out how to get back to green. A `CassetteMissError`, so a consumer that
    already catches the general case keeps working.
    """


class CassetteLeakError(AdkError):
    """Raised when a cassette about to be written still holds something secret.

    Refused at the write, not at review: a token in a committed file outlives every run
    that used it, and nobody reads a 4000-line JSON diff closely.
    """


class CassetteMode(StrEnum):
    """How a cassette-backed test treats the provider.

    `REPLAY` is the default everywhere, so a suite cannot spend money by forgetting to
    set something.
    """

    REPLAY = "replay"
    RECORD = "record"
    REFRESH = "refresh"


ENV_MODE = "ADK_CASSETTE_MODE"


def mode_from_env(environ: Mapping[str, str] | None = None) -> CassetteMode:
    """Read the cassette mode from the environment.

    Args:
        environ: Where to read it from. Defaults to the process environment.

    Raises:
        ConfigurationError: If the variable holds something no mode is named after. A
            typo silently meaning "replay" is a suite that never records again.
    """
    raw = (environ if environ is not None else os.environ).get(ENV_MODE, "").strip()
    if not raw:
        return CassetteMode.REPLAY
    try:
        return CassetteMode(raw.lower())
    except ValueError as err:
        named = ", ".join(mode.value for mode in CassetteMode)
        message = f"{ENV_MODE}={raw!r} names no cassette mode; it is one of {named}"
        raise ConfigurationError(message) from err


class MatchOn(AdkModel):
    """Which parts of a request have to agree for a recording to serve it.

    Every field is compared by default. Dropping one widens what a cassette answers: a
    test about tool wiring need not be re-recorded because a word in the prompt changed.

    Args:
        model: Whether the model name has to match.
        messages: Whether the assembled prompt has to match.
        tools: Whether the tool schemas offered have to match.
        output_schema: Whether the requested output schema has to match.
        hooks: Whether the hook chain has to match.
    """

    model: bool = True
    messages: bool = True
    tools: bool = True
    output_schema: bool = True
    hooks: bool = True

    @model_validator(mode="after")
    def _matches_something(self) -> MatchOn:
        """Refuse a strategy under which every request matches every recording."""
        if not self.fields:
            message = "a match strategy has to compare at least one field of the request"
            raise ValueError(message)
        return self

    @property
    def fields(self) -> tuple[str, ...]:
        """The fingerprint fields this strategy compares, in a stable order."""
        return tuple(name for name in _MATCHABLE if getattr(self, name))

    def key(self, fingerprint: RunFingerprint) -> str:
        """The key a recording is filed and found under."""
        return canonical_digest({name: getattr(fingerprint, name) for name in self.fields})


_MATCHABLE = ("model", "messages", "tools", "output_schema", "hooks")


class CassetteVersionError(AdkError):
    """Raised when a cassette was recorded against a different provider or version.

    Replaying across an SDK upgrade on trust is a green test that proves nothing about the
    code now shipping.
    """


class RecordedError(AdkModel):
    """A provider failure, kept so that the recovery replays as well as the success.

    Args:
        message: What the provider said, redacted.
        status: The HTTP status, where there was one.
        retry_after: Seconds the provider asked for.
    """

    message: str = ""
    status: int | None = None
    retry_after: float | None = None

    def raised(self) -> ProviderError:
        """Rebuild the error, so a replayed run retries exactly as the recorded one did."""
        return ProviderError(self.message, status=self.status, retry_after=self.retry_after)


class Interaction(AdkModel):
    """One request and what came back, keyed by the request's fingerprint.

    Args:
        fingerprint: The canonical summary of the request. Digests, never content.
        response: What the provider returned, redacted.
        error: What it raised instead, where it raised.
        chunks: The text deltas as they arrived, where the exchange was streamed. Kept
            because a consumer that renders per chunk behaves differently on different
            boundaries, and a replay that re-chunks tests the wrong thing.
    """

    fingerprint: RunFingerprint
    response: ModelResponse | None = None
    error: RecordedError | None = None
    chunks: tuple[str, ...] = ()


class Cassette(AdkModel):
    """Recorded exchanges plus what they were recorded against.

    Args:
        format: The cassette format, so an old file is refused rather than misread.
        provider: Which provider produced the recording.
        provider_version: Its version at recording time.
        interactions: The exchanges, in the order they happened.
    """

    format: str = CASSETTE_FORMAT
    provider: str = Field(min_length=1)
    provider_version: str = ""
    interactions: tuple[Interaction, ...] = ()

    def save(self, path: Path) -> None:
        """Write the cassette to `path` as indented JSON, so a diff is readable in review.

        Raises:
            CassetteLeakError: If anything credential-shaped survived redaction. The
                write is refused rather than left for review to catch.
        """
        written = json.dumps(self.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
        leaked = [pattern.pattern for pattern in _SECRET_VALUES if pattern.search(written)]
        if leaked:
            message = (
                f"refusing to write {path}: it still holds something credential-shaped "
                f"({len(leaked)} pattern(s) matched). Redact it at the source"
            )
            raise CassetteLeakError(message)
        path.write_text(written)

    @classmethod
    def load(cls, path: Path) -> Cassette:
        """Read a cassette from `path`.

        Raises:
            CassetteVersionError: If it was written in a format this kit does not read.
        """
        # Validated as JSON, not as a dict: a JSON array is a tuple, which strict Python
        # validation would refuse.
        loaded = cls.model_validate_json(path.read_text())
        if loaded.format != CASSETTE_FORMAT:
            raise CassetteVersionError(
                f"cassette at {path} is format {loaded.format!r}, and this kit reads "
                f"{CASSETTE_FORMAT!r}; re-record it rather than reading it on trust"
            )
        return loaded


class RecordingProvider:
    """Wraps a real provider and keeps a redacted copy of everything it answered.

    Args:
        inner: The provider actually being called.
        provider: Its name, stamped on the cassette.
        version: Its version, stamped on the cassette.
        hooks: The hook names in the chain, so the fingerprint covers them.
    """

    def __init__(
        self,
        inner: ModelProvider,
        *,
        provider: str,
        version: str = "",
        hooks: Iterable[str] = (),
    ) -> None:
        self._inner = inner
        self._hooks = tuple(hooks)
        self._interactions: list[Interaction] = []
        self._provider = provider
        self._version = version

    @property
    def name(self) -> str:
        """The provider being recorded, so a run attributes itself to what it really called."""
        return self._provider

    @property
    def capabilities(self) -> ModelCapabilities:
        """The wrapped provider's own record: recording must not widen what it can do."""
        declared: ModelCapabilities = self._inner.capabilities
        return declared

    def count_tokens(self, messages: Sequence[Message]) -> int:
        """The wrapped provider's own count."""
        counted: int = self._inner.count_tokens(messages)
        return counted

    @property
    def cassette(self) -> Cassette:
        """Everything recorded so far."""
        return Cassette(
            provider=self._provider,
            provider_version=self._version,
            interactions=tuple(self._interactions),
        )

    async def complete(self, request: ModelRequest) -> ModelResponse:
        """Call the wrapped provider and record what it did, failure included."""
        fingerprint = fingerprint_of(request, hooks=self._hooks)
        try:
            response: ModelResponse = await self._inner.complete(request)
        except ProviderError as failure:
            self._interactions.append(
                Interaction(
                    fingerprint=fingerprint,
                    error=RecordedError(
                        message=redacted(str(failure)),
                        status=failure.status,
                        retry_after=failure.retry_after,
                    ),
                )
            )
            raise
        self._interactions.append(
            Interaction(fingerprint=fingerprint, response=_scrubbed(response))
        )
        return response

    async def stream(self, request: ModelRequest) -> AsyncIterator[StreamEvent]:
        """Pass the wrapped provider's stream through, keeping its chunk boundaries.

        A consumer that stops part way still has what arrived recorded: the point it
        stopped at is usually what the test was about.
        """
        fingerprint = fingerprint_of(request, hooks=self._hooks)
        return self._recording(await self._inner.stream(request), fingerprint)

    async def _recording(
        self, events: AsyncIterator[StreamEvent], fingerprint: RunFingerprint
    ) -> AsyncIterator[StreamEvent]:
        chunks: list[str] = []
        answer: ModelResponse | None = None
        try:
            async for event in events:
                if isinstance(event, TextDelta):
                    chunks.append(redacted(event.text))
                if isinstance(event, StreamEnd):
                    answer = _scrubbed(event.response)
                yield event
        finally:
            self._interactions.append(
                Interaction(
                    fingerprint=fingerprint,
                    response=answer if answer is not None else _assembled(chunks),
                    chunks=tuple(chunks),
                )
            )


class ReplayingProvider:
    """Serves recorded exchanges and refuses to invent one.

    Args:
        cassette: The recording to serve.
        expect_provider: The provider the caller believes it is testing against.
        expect_version: The version it believes it is testing against.
        hooks: The hook names in the chain, so the fingerprint covers them.
        capabilities: What the recorded provider declared. A replay against a wider
            record passes a check the recording never passed.

    Raises:
        CassetteVersionError: If the cassette was recorded against a different provider or
            version than the caller declared.
    """

    def __init__(
        self,
        cassette: Cassette,
        *,
        expect_provider: str | None = None,
        expect_version: str | None = None,
        hooks: Iterable[str] = (),
        capabilities: ModelCapabilities | None = None,
        match: MatchOn | None = None,
    ) -> None:
        if expect_provider is not None and cassette.provider != expect_provider:
            raise CassetteVersionError(
                f"cassette was recorded against provider {cassette.provider!r}, not "
                f"{expect_provider!r}; re-record it rather than replaying it on trust"
            )
        if expect_version is not None and cassette.provider_version != expect_version:
            raise CassetteVersionError(
                f"cassette was recorded against {cassette.provider!r} "
                f"{cassette.provider_version!r}, not {expect_version!r}; re-record it "
                f"rather than replaying it on trust"
            )
        self._cassette = cassette
        self._hooks = tuple(hooks)
        self._capabilities = capabilities if capabilities is not None else CAPABLE
        self._match = match if match is not None else MatchOn()
        self.served: list[Interaction] = []

    @property
    def name(self) -> str:
        """The provider the cassette was recorded against, not a fiction of its own."""
        return self._cassette.provider

    @property
    def capabilities(self) -> ModelCapabilities:
        """What the recorded provider declared."""
        return self._capabilities

    def count_tokens(self, messages: Sequence[Message]) -> int:
        """Estimated, since the recorded provider's tokeniser is not on the cassette."""
        return estimate_tokens(messages)

    async def complete(self, request: ModelRequest) -> ModelResponse:
        """Serve the next recording for this request.

        Raises:
            CassetteMissError: If nothing was recorded for it. Nothing is called instead.
            ProviderError: If what was recorded for it was a failure.
        """
        fingerprint = fingerprint_of(request, hooks=self._hooks)
        interaction = self._next(fingerprint)
        self.served.append(interaction)
        if interaction.error is not None:
            raise interaction.error.raised()
        return interaction.response or ModelResponse()

    async def stream(self, request: ModelRequest) -> AsyncIterator[StreamEvent]:
        """Replay a recorded stream, chunk boundaries included.

        Raises:
            CassetteMismatchError: If nothing was recorded for the request.
            ProviderError: If what was recorded for it was a failure.
        """
        fingerprint = fingerprint_of(request, hooks=self._hooks)
        interaction = self._next(fingerprint)
        self.served.append(interaction)
        if interaction.error is not None:
            raise interaction.error.raised()
        return _served(interaction)

    def _next(self, fingerprint: RunFingerprint) -> Interaction:
        key = self._match.key(fingerprint)
        taken = sum(1 for served in self.served if self._match.key(served.fingerprint) == key)
        matching = [
            interaction
            for interaction in self._cassette.interactions
            if self._match.key(interaction.fingerprint) == key
        ]
        if len(matching) > taken:
            return matching[taken]
        raise CassetteMismatchError(self._miss(fingerprint, exhausted=bool(matching)))

    def _miss(self, fingerprint: RunFingerprint, *, exhausted: bool) -> str:
        if exhausted:
            return (
                "cassette has no further recording for this request; the run asked more "
                f"times than it did when recorded. {_REFRESH}"
            )
        nearest = min(
            (interaction.fingerprint for interaction in self._cassette.interactions),
            key=lambda recorded: len(fingerprint.diff(recorded)),
            default=None,
        )
        if nearest is None:
            return f"cassette is empty, so there is nothing to replay. {_REFRESH}"
        return (
            f"no recording for this request; it differs from the nearest one in "
            f"{', '.join(fingerprint.diff(nearest))}. {_REFRESH}"
        )


class CassetteHarness:
    """Picks the provider a test gets, from the mode and whether a cassette exists.

    Args:
        path: Where the cassette lives.
        mode: What to do about it. `REPLAY` never opens a socket; `RECORD` records only
            what is not on the cassette already, so a suite does not re-buy what it has;
            `REFRESH` records over it deliberately.
        hooks: The hook names in the chain, so the fingerprint covers them.
        match: Which parts of a request have to agree for a recording to serve it.
        provider: The provider name stamped on a new recording.
        version: The provider version stamped on a new recording.
        capabilities: What the recorded provider declared, for a replay.

    Example:
        >>> harness = CassetteHarness(path, mode=CassetteMode.REPLAY)   # doctest: +SKIP
        >>> runtime_provider = harness.provider()                       # doctest: +SKIP
    """

    def __init__(
        self,
        path: Path,
        *,
        mode: CassetteMode = CassetteMode.REPLAY,
        hooks: Iterable[str] = (),
        match: MatchOn | None = None,
        provider: str = "recorded",
        version: str = "",
        capabilities: ModelCapabilities | None = None,
    ) -> None:
        self.path = path
        self.mode = mode
        self._hooks = tuple(hooks)
        self._match = match
        self._provider = provider
        self._version = version
        self._capabilities = capabilities
        self._recorder: RecordingProvider | None = None

    def provider(self, live: ModelProvider | None = None) -> ModelProvider:
        """The provider this test should run against.

        Args:
            live: The real provider, needed only where this run will record.

        Raises:
            CassetteMismatchError: In replay mode with no cassette to replay, naming the
                command that records one. Falling through to a live call here is how a
                suite starts spending without anybody deciding to.
            ConfigurationError: If a recording mode was asked for without a live provider.
        """
        if self._recording():
            if live is None:
                message = (
                    f"{self.mode.value} mode needs a live provider to record from; "
                    f"pass one, or run in {CassetteMode.REPLAY.value} mode"
                )
                raise ConfigurationError(message)
            self._recorder = RecordingProvider(
                live, provider=self._provider, version=self._version, hooks=self._hooks
            )
            return self._recorder
        if not self.path.exists():
            message = (
                f"no cassette at {self.path} to replay. "
                f"Record one with {ENV_MODE}={CassetteMode.RECORD.value}"
            )
            raise CassetteMismatchError(message)
        return ReplayingProvider(
            Cassette.load(self.path),
            hooks=self._hooks,
            capabilities=self._capabilities,
            match=self._match,
        )

    def save(self) -> None:
        """Write what was recorded, if this run recorded anything.

        Raises:
            CassetteLeakError: If the recording still holds something credential-shaped.
        """
        if self._recorder is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._recorder.cassette.save(self.path)

    def _recording(self) -> bool:
        """Whether this run should call the real provider."""
        return self.mode is CassetteMode.REFRESH or (
            self.mode is CassetteMode.RECORD and not self.path.exists()
        )


def assert_same_run(first: Run, second: Run, *, message: str = "") -> None:
    """Assert two runs are the same run, timings aside.

    Compares state, ids, the event sequence with its names and details, tool calls and
    usage totals. Wall-clock instants are dropped: a slower machine is not a behaviour
    change, but a different sequence of events is.

    Raises:
        AssertionError: On the first field that diverges, naming it.
    """
    prefix = f"{message}: " if message else ""
    for field, left, right in (
        ("state", first.state, second.state),
        ("id", first.id, second.id),
        ("output", first.output, second.output),
        ("usage", first.usage, second.usage),
        ("tool_calls", first.tool_calls, second.tool_calls),
    ):
        if left != right:
            raise AssertionError(f"{prefix}runs diverge on {field}: {left!r} != {right!r}")
    for index, (one, two) in enumerate(zip(first.events, second.events, strict=False)):
        step = (one.kind, one.name, one.detail)
        other = (two.kind, two.name, two.detail)
        if step != other:
            raise AssertionError(
                f"{prefix}runs diverge at event {index} ({one.kind}): {step!r} != {other!r}"
            )
    if len(first.events) != len(second.events):
        raise AssertionError(
            f"{prefix}runs diverge in length: {len(first.events)} events != {len(second.events)}"
        )


def redacted(text: str) -> str:
    """Replace anything that looks like a credential or an identifier in `text`."""
    for pattern in _SECRET_VALUES:
        text = pattern.sub(REDACTED, text)
    return text


def _assembled(chunks: list[str]) -> ModelResponse:
    """What a stream that ended early amounts to, so the recording is not empty."""
    return ModelResponse(content="".join(chunks))


async def _served(interaction: Interaction) -> AsyncIterator[StreamEvent]:
    """A recorded exchange told again, with the boundaries it arrived on."""
    response = interaction.response or ModelResponse()
    for chunk in interaction.chunks:
        yield TextDelta(text=chunk)
    for index, call in enumerate(response.tool_calls):
        yield ToolCallDelta(
            index=index,
            id=call.id,
            name=call.name,
            arguments=json.dumps(call.arguments, sort_keys=True),
        )
    yield UsageDelta(usage=response.usage)
    yield StreamEnd(response=response)


def _scrubbed(response: ModelResponse) -> ModelResponse:
    return response.model_copy(
        update={
            "content": redacted(response.content),
            "tool_calls": tuple(
                call.model_copy(update={"arguments": _scrub(call.arguments)})
                for call in response.tool_calls
            ),
        }
    )


def _scrub(arguments: Mapping[str, Any]) -> dict[str, Any]:
    scrubbed: dict[str, Any] = {}
    for key, value in arguments.items():
        if _SECRET_KEYS.search(key):
            scrubbed[key] = REDACTED
        elif isinstance(value, Mapping):
            scrubbed[key] = _scrub(value)
        elif isinstance(value, str):
            scrubbed[key] = redacted(value)
        else:
            scrubbed[key] = value
    return scrubbed
