"""The transport the vendor adapters share, and the errors it translates them into.

The adapters speak HTTP directly rather than through each vendor's SDK. Three SDKs is
three dependency graphs, three release cadences and three object models to translate out
of again, in a kit whose claim is that a small agent stays a small dependency. What is
left is one `httpx` call per vendor, and `httpx` is already here.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from contextlib import aclosing
from typing import TYPE_CHECKING, Any, Self, cast

import httpx

from tesserix_adk.core.capabilities import ModelCapabilities
from tesserix_adk.core.errors import (
    ConfigurationError,
    ModelResponseError,
    ProviderError,
    ProviderTimeoutError,
    ProviderUnavailableError,
    StreamInterruptedError,
)
from tesserix_adk.core.primitives import TextPart
from tesserix_adk.core.streaming import StreamAccumulator, StreamEnd
from tesserix_adk.models.catalogue import model_card, priced
from tesserix_adk.models.credentials import Credential
from tesserix_adk.models.providers._normalise import normalised_tool_calls

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, AsyncIterator, Mapping, Sequence

    from tesserix_adk.core.primitives import Message, Usage
    from tesserix_adk.core.protocols import SecretProvider
    from tesserix_adk.core.provider import ModelRequest, ModelResponse, StopReason
    from tesserix_adk.core.streaming import StreamEvent
    from tesserix_adk.models.catalogue import ModelCard

__all__ = ["HttpProvider", "VendorStream"]

_CHARS_PER_TOKEN = 4
_BODY_IN_ERRORS = 500


class VendorStream(ABC):
    """One vendor's streaming state machine, translating frames into the kit's events.

    The state lives outside the events because every vendor sends the things a response
    needs — why it stopped, what it cost — outside the deltas themselves.
    """

    def __init__(self) -> None:
        self.stop_reason: StopReason | None = None
        self.failure: ProviderError | None = None
        self.text = ""
        self.frames = 0

    @abstractmethod
    def read(self, frame: Mapping[str, Any]) -> list[StreamEvent]:
        """Translate one server-sent frame into the kit's events, updating the state.

        An error the vendor reports mid-stream is recorded on `failure` rather than
        raised: the body is drained and the connection released first, and a generator
        abandoned part way through is a connection returned to the pool by a garbage
        collector.
        """


class HttpProvider:
    """What the vendor adapters have in common: a client, a key and an error map.

    Args:
        model: The model id to call, as the vendor spells it.
        capabilities: What that model does. Defaults to the catalogue's row for it. A
            model the catalogue does not know must be told, because the alternative is
            assuming a capability and finding out mid-run.
        secrets: Where the key comes from. Defaults to the environment, read per request
            so a rotation lands without a restart.
        api_key_variable: Which variable holds it. Defaults to the vendor's usual name.
        base_url: The endpoint, overridable for a gateway or a regional host.
        timeout: Seconds for one request.
        transport: An injected `httpx` transport. This is how the recorded tests run with
            no network, and how a consumer supplies its own proxy or retry layer.
    """

    provider_name = ""
    default_base_url = ""
    default_key_variable = ""

    def __init__(
        self,
        model: str,
        *,
        capabilities: ModelCapabilities | None = None,
        secrets: SecretProvider | None = None,
        api_key_variable: str | None = None,
        base_url: str | None = None,
        timeout: float = 60.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.model = model
        self._card = _card_for(self.provider_name, model)
        self._capabilities = capabilities or _declared(self._card)
        self._credential = Credential(
            api_key_variable or self.default_key_variable, secrets=secrets
        )
        self._client = httpx.AsyncClient(
            base_url=base_url or self.default_base_url,
            timeout=timeout,
            transport=transport,
        )

    @property
    def name(self) -> str:
        """The provider name, as a run records it and a `ModelRef` writes it."""
        return self.provider_name

    @property
    def capabilities(self) -> ModelCapabilities:
        """What this model declares. Read before each request rather than found out."""
        return self._capabilities

    def count_tokens(self, messages: Sequence[Message]) -> int:
        """Estimate the tokens in `messages` by character count.

        An exact count means a round trip to the vendor's own counter before every
        request, which doubles the calls to save none. This is what the declared window
        is checked against, and it is deliberately not a billing figure.
        """
        text = "".join(
            part.text
            for message in messages
            for part in message.content
            if isinstance(part, TextPart)
        )
        return len(text) // _CHARS_PER_TOKEN

    async def aclose(self) -> None:
        """Close the underlying connection pool."""
        await self._client.aclose()

    async def __aenter__(self) -> Self:
        """Return self, so a provider can own its pool for the length of a block."""
        return self

    async def __aexit__(self, *_: object) -> None:
        """Close the pool."""
        await self.aclose()

    async def _post(self, path: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        """POST `payload` and return the parsed body, in the kit's error vocabulary."""
        try:
            response = await self._client.post(path, json=dict(payload), headers=self._headers())
        except httpx.TimeoutException as timeout:
            raise ProviderTimeoutError(
                f"{self.provider_name} did not answer in time", details={"path": path}
            ) from timeout
        except httpx.TransportError as unreachable:
            raise ProviderUnavailableError(
                f"{self.provider_name} could not be reached: {unreachable}",
                details={"path": path},
            ) from unreachable
        _refuse_a_failure(response, self.provider_name, response.text)
        return _object(response, self.provider_name)

    async def _post_sse(
        self, path: str, payload: Mapping[str, Any]
    ) -> AsyncGenerator[dict[str, Any]]:
        """Yield each `data:` frame of a server-sent event stream, already parsed.

        Raises:
            StreamInterruptedError: If the connection drops or a frame is unreadable part
                way through. A truncated answer returned as a whole one is a wrong answer
                with nothing to show that it is wrong.
        """
        request = self._client.build_request(
            "POST", path, json=dict(payload), headers=self._headers()
        )
        try:
            response = await self._client.send(request, stream=True)
        except httpx.TimeoutException as timeout:
            raise ProviderTimeoutError(
                f"{self.provider_name} did not answer in time", details={"path": path}
            ) from timeout
        except httpx.TransportError as unreachable:
            raise ProviderUnavailableError(
                f"{self.provider_name} could not be reached: {unreachable}",
                details={"path": path},
            ) from unreachable
        try:
            if response.status_code >= httpx.codes.BAD_REQUEST:
                await response.aread()
                _refuse_a_failure(response, self.provider_name, response.text)
            async with aclosing(_frames(response, self.provider_name)) as frames:
                async for frame in frames:
                    yield frame
        finally:
            await response.aclose()

    async def _streamed(
        self,
        path: str,
        payload: Mapping[str, Any],
        *,
        request: ModelRequest,
        state: VendorStream,
    ) -> AsyncIterator[StreamEvent]:
        """Drive one vendor's stream through its translator and end on a `StreamEnd`.

        Raises:
            ProviderError: If the vendor sent an error frame.
            StreamInterruptedError: If the stream ended before the model said why it
                stopped, which is the shape a truncated answer arrives in.
        """
        accumulator = StreamAccumulator()
        async with aclosing(self._post_sse(path, payload)) as frames:
            async for frame in frames:
                for event in state.read(frame):
                    accumulator.feed(event)
                    yield event
        if state.failure is not None:
            raise state.failure
        if state.stop_reason is None:
            raise StreamInterruptedError(
                f"{self.provider_name} stopped streaming before the model said why it had stopped",
                partial=state.text,
                received=state.frames,
            )
        yield StreamEnd(response=self._settled(accumulator.finish(state.stop_reason), request))

    def _settled(self, response: ModelResponse, request: ModelRequest) -> ModelResponse:
        """Check the tool calls and price the usage, whichever path assembled the response.

        Raises:
            CapabilityError: If the model returned parallel calls it never declared.
            ToolArgumentValidationError: If a call's arguments contradict its schema.
        """
        return response.model_copy(
            update={
                "tool_calls": normalised_tool_calls(
                    response.tool_calls,
                    request=request,
                    capabilities=self._capabilities,
                    provider=self.provider_name,
                ),
                "usage": self._priced(response.usage),
            }
        )

    def _headers(self) -> dict[str, str]:
        """Return the request headers, resolving the key at the moment it is used."""
        raise NotImplementedError

    def _priced(self, usage: Usage) -> Usage:
        """Fill in the cost from the catalogue, where the catalogue records one."""
        return usage if self._card is None else priced(usage, self._card)

    def _output_cap(self) -> int:
        """The output ceiling, for the vendors that require one be sent on every request."""
        return self._capabilities.max_output_tokens or _DEFAULT_MAX_OUTPUT


_DEFAULT_MAX_OUTPUT = 4096
_NOT_THERE = frozenset({502, 503, 504})


def _card_for(provider: str, model: str) -> ModelCard | None:
    try:
        return model_card(f"{provider}:{model}")
    except ConfigurationError:
        return None


def _declared(card: ModelCard | None) -> ModelCapabilities:
    return ModelCapabilities() if card is None else card.capabilities


def _refuse_a_failure(response: httpx.Response, provider: str, body: str) -> None:
    if response.status_code < httpx.codes.BAD_REQUEST:
        return
    # A gateway status is nothing to do with the request: something upstream is not there
    # yet. Its own type, so a caller waits rather than rewriting a request that was fine.
    failure = ProviderUnavailableError if response.status_code in _NOT_THERE else ProviderError
    raise failure(
        f"{provider} answered {response.status_code}",
        status=response.status_code,
        retry_after=_retry_after(response),
        details={"body": body[:_BODY_IN_ERRORS], "request_id": _request_id(response)},
    )


def _retry_after(response: httpx.Response) -> float | None:
    """Believe the vendor's own wait over any backoff the kit would compute."""
    asked = response.headers.get("retry-after")
    try:
        return float(asked) if asked else None
    except ValueError:
        return None


def _request_id(response: httpx.Response) -> str:
    return response.headers.get("request-id") or response.headers.get("x-request-id") or ""


def _object(response: httpx.Response, provider: str) -> dict[str, Any]:
    try:
        parsed = response.json()
    except ValueError as unreadable:
        raise ModelResponseError(
            f"{provider} answered with a body that is not JSON",
            payload=response.text[:_BODY_IN_ERRORS],
            provider=provider,
            request_id=_request_id(response),
        ) from unreadable
    if not isinstance(parsed, dict):
        raise ModelResponseError(
            f"{provider} answered with {type(parsed).__name__}, not an object",
            payload=parsed,
            provider=provider,
            request_id=_request_id(response),
        )
    return parsed


async def _frames(response: httpx.Response, provider: str) -> AsyncGenerator[dict[str, Any]]:
    """Split the raw stream into `data:` frames, already parsed.

    An unreadable frame is held rather than raised where it is found. Raising there
    abandons the iteration, and abandoning it leaves the transport's own generators to be
    finalised by the garbage collector after the event loop they belong to has closed. The
    rest of the body is drained first — an unreadable frame ends the answer either way —
    and the failure is raised once the stream has finished properly.
    """
    seen = 0
    buffered = b""
    unreadable: StreamInterruptedError | None = None
    try:
        async with aclosing(cast("AsyncGenerator[bytes]", response.aiter_bytes())) as chunks:
            async for chunk in chunks:
                *lines, buffered = (buffered + chunk).split(b"\n")
                if unreadable is not None:
                    continue
                for line in lines:
                    payload = _data(line)
                    if payload:
                        seen += 1
                        try:
                            frame = _frame(payload, provider=provider, seen=seen)
                        except StreamInterruptedError as broken:
                            unreadable = broken
                            break
                        yield frame
    except httpx.HTTPError as broken:
        raise StreamInterruptedError(
            f"{provider} stopped streaming after {seen} events: {broken}", received=seen
        ) from broken
    if unreadable is not None:
        raise unreadable
    if payload := _data(buffered):
        yield _frame(payload, provider=provider, seen=seen + 1)


def _data(line: bytes) -> str:
    """The payload of one `data:` line, or empty for anything that carries none."""
    decoded = line.decode("utf-8", errors="replace").strip()
    if not decoded.startswith("data:"):
        return ""
    payload = decoded[len("data:") :].strip()
    return "" if payload == "[DONE]" else payload


def _frame(payload: str, *, provider: str, seen: int) -> dict[str, Any]:
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError as broken:
        raise StreamInterruptedError(
            f"{provider} sent a stream frame that is not JSON: {broken}", received=seen
        ) from broken
    if not isinstance(parsed, dict):
        raise StreamInterruptedError(
            f"{provider} sent a stream frame that is not an object", received=seen
        )
    return parsed
