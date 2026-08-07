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
import re
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from pydantic import Field

from tesserix_adk.core.errors import AdkError, ProviderError
from tesserix_adk.core.models import AdkModel
from tesserix_adk.runtime import ModelResponse, RunFingerprint, fingerprint_of
from tesserix_adk.testing.fakes import CAPABLE, estimate_tokens

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterable, Sequence
    from pathlib import Path

    from tesserix_adk.core import Message, ModelCapabilities, ModelProvider, Run
    from tesserix_adk.core.streaming import StreamEvent
    from tesserix_adk.runtime import ModelRequest

__all__ = [
    "Cassette",
    "CassetteMissError",
    "CassetteVersionError",
    "Interaction",
    "RecordedError",
    "RecordingProvider",
    "ReplayingProvider",
    "assert_same_run",
    "redacted",
]


CASSETTE_FORMAT = "1"

REDACTED = "[REDACTED]"

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
    """

    fingerprint: RunFingerprint
    response: ModelResponse | None = None
    error: RecordedError | None = None


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
        """Write the cassette to `path` as indented JSON, so a diff is readable in review."""
        path.write_text(json.dumps(self.model_dump(mode="json"), indent=2, sort_keys=True) + "\n")

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
        """Not recorded yet — recorded streaming is #39."""
        raise NotImplementedError("RecordingProvider does not stream; see #39")


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
        """Not replayed yet — recorded streaming is #39."""
        raise NotImplementedError("ReplayingProvider does not stream; see #39")

    def _next(self, fingerprint: RunFingerprint) -> Interaction:
        taken = sum(1 for served in self.served if served.fingerprint.digest == fingerprint.digest)
        matching = [
            interaction
            for interaction in self._cassette.interactions
            if interaction.fingerprint.digest == fingerprint.digest
        ]
        if len(matching) > taken:
            return matching[taken]
        raise CassetteMissError(self._miss(fingerprint, exhausted=bool(matching)))

    def _miss(self, fingerprint: RunFingerprint, *, exhausted: bool) -> str:
        if exhausted:
            return (
                "cassette has no further recording for this request; the run asked more "
                "times than it did when recorded"
            )
        nearest = min(
            (interaction.fingerprint for interaction in self._cassette.interactions),
            key=lambda recorded: len(fingerprint.diff(recorded)),
            default=None,
        )
        if nearest is None:
            return "cassette is empty, so there is nothing to replay"
        return (
            f"no recording for this request; it differs from the nearest one in "
            f"{', '.join(fingerprint.diff(nearest))}"
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
