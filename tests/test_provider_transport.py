"""The transport every vendor adapter shares, at the points where the network misbehaves.

A vendor-shaped answer is the easy case and each adapter's own suite covers it. This
covers the rest: a connection that never answers, a body that is not JSON, a stream that
stops halfway. Each has to arrive as one of the kit's errors, because a caller that has to
read httpx exceptions is a caller the provider layer did not abstract anything for.

`AnthropicProvider` stands in for all three: the code under test is `_http`, and picking
one adapter to reach it beats asserting the same thing three times.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import httpx
import pytest

from tesserix_adk.core import (
    Message,
    ModelRequest,
    ModelResponseError,
    ProviderError,
    ProviderTimeoutError,
    StreamInterruptedError,
    TextPart,
)
from tesserix_adk.models.providers import PHASE_DEFAULTS, AnthropicProvider
from tesserix_adk.testing import FakeSecrets

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable

MODEL = "claude-sonnet-4-5"


def provider(
    handler: Callable[[httpx.Request], httpx.Response],
    *,
    timeout: float = PHASE_DEFAULTS.read,
    connect_timeout: float = PHASE_DEFAULTS.connect,
) -> AnthropicProvider:
    return AnthropicProvider(
        MODEL,
        secrets=FakeSecrets({"ANTHROPIC_API_KEY": "test-key"}),
        transport=httpx.MockTransport(handler),
        timeout=timeout,
        connect_timeout=connect_timeout,
    )


def raising(failure: Exception) -> Callable[[httpx.Request], httpx.Response]:
    def handler(_: httpx.Request) -> httpx.Response:
        raise failure

    return handler


def answering(response: httpx.Response) -> Callable[[httpx.Request], httpx.Response]:
    return lambda _: response


def asked() -> ModelRequest:
    return ModelRequest(
        model=MODEL, messages=(Message(role="user", content=[TextPart(text="did it rain")]),)
    )


def streaming(*chunks: bytes, then: Exception | None = None) -> httpx.Response:
    async def body() -> AsyncIterator[bytes]:
        for chunk in chunks:
            yield chunk
        if then is not None:
            raise then

    return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=body())


class TestWhenTheCallNeverLands:
    async def test_a_timeout_is_a_provider_timeout(self) -> None:
        with pytest.raises(ProviderTimeoutError, match="did not answer in time"):
            await provider(raising(httpx.TimeoutException("slow"))).complete(asked())

    async def test_an_unreachable_host_is_a_provider_error(self) -> None:
        with pytest.raises(ProviderError, match="could not be reached"):
            await provider(raising(httpx.ConnectError("no route"))).complete(asked())

    async def test_a_timeout_opening_a_stream_is_a_provider_timeout(self) -> None:
        events = await provider(raising(httpx.TimeoutException("slow"))).stream(asked())
        with pytest.raises(ProviderTimeoutError, match="did not answer in time"):
            [event async for event in events]

    async def test_an_unreachable_host_on_a_stream_is_a_provider_error(self) -> None:
        events = await provider(raising(httpx.ConnectError("no route"))).stream(asked())
        with pytest.raises(ProviderError, match="could not be reached"):
            [event async for event in events]


class TestWhichWaitRanOut:
    """A host that will not accept a connection and a model that is thinking slowly are
    the same exception in httpx and different operational events. The phase is carried so
    an operator reads one graph rather than one counter."""

    async def test_a_connect_timeout_names_the_connect_phase(self) -> None:
        with pytest.raises(ProviderTimeoutError) as refused:
            await provider(raising(httpx.ConnectTimeout("no answer"))).complete(asked())
        assert refused.value.details["phase"] == "connect"

    async def test_a_read_timeout_names_the_read_phase(self) -> None:
        with pytest.raises(ProviderTimeoutError) as refused:
            await provider(raising(httpx.ReadTimeout("still thinking"))).complete(asked())
        assert refused.value.details["phase"] == "read"

    async def test_a_timeout_that_names_no_phase_is_still_a_timeout(self) -> None:
        with pytest.raises(ProviderTimeoutError) as refused:
            await provider(raising(httpx.TimeoutException("slow"))).complete(asked())
        assert refused.value.details["phase"] == "request"

    async def test_the_phase_survives_the_stream_path_too(self) -> None:
        events = await provider(raising(httpx.ConnectTimeout("no answer"))).stream(asked())
        with pytest.raises(ProviderTimeoutError) as refused:
            [event async for event in events]
        assert refused.value.details["phase"] == "connect"


class TestHowLongEachPhaseIsGiven:
    """Connecting and generating are given separate budgets. One number for both means
    either a dead host is waited on for a minute or a long answer is cut off."""

    async def test_connecting_is_given_far_less_than_generating_by_default(self) -> None:
        async with provider(answering(httpx.Response(200))) as default:
            assert default.timeouts.connect < default.timeouts.read

    async def test_each_phase_can_be_set_on_its_own(self) -> None:
        async with provider(answering(httpx.Response(200)), connect_timeout=2.5) as set_up:
            assert set_up.timeouts.connect == 2.5
            assert set_up.timeouts.read == PHASE_DEFAULTS.read

    async def test_the_overall_timeout_still_moves_the_generating_budget(self) -> None:
        async with provider(answering(httpx.Response(200)), timeout=5.0) as set_up:
            assert set_up.timeouts.read == 5.0
            assert set_up.timeouts.connect == PHASE_DEFAULTS.connect


class TestWhenTheAnswerIsNotAnAnswer:
    async def test_a_body_that_is_not_json_is_refused(self) -> None:
        """An HTML error page from a proxy is not a completion, and never was."""
        failing = provider(answering(httpx.Response(200, text="<html>gateway</html>")))
        with pytest.raises(ModelResponseError, match="not JSON"):
            await failing.complete(asked())

    async def test_a_json_body_that_is_not_an_object_is_refused(self) -> None:
        failing = provider(answering(httpx.Response(200, json=["nope"])))
        with pytest.raises(ModelResponseError, match="not an object"):
            await failing.complete(asked())

    async def test_the_raw_body_travels_with_the_error(self) -> None:
        """A refusal that drops the payload leaves nothing to debug the vendor with."""
        failing = provider(answering(httpx.Response(200, text="<html>gateway</html>")))
        with pytest.raises(ModelResponseError) as refused:
            await failing.complete(asked())
        assert "gateway" in str(refused.value.payload)


class TestWhenTheVendorRefuses:
    async def test_a_status_error_opening_a_stream_is_read_before_the_stream(self) -> None:
        """The failure body arrives on the streaming response, and has to be read for it."""
        failing = provider(answering(httpx.Response(429, json={"error": {"message": "slow"}})))
        events = await failing.stream(asked())
        with pytest.raises(ProviderError) as refused:
            [event async for event in events]
        assert refused.value.status == 429
        assert "slow" not in str(refused.value.details)

    async def test_the_vendors_own_wait_is_believed(self) -> None:
        failing = provider(answering(httpx.Response(429, headers={"retry-after": "30"}, json={})))
        with pytest.raises(ProviderError) as refused:
            await failing.complete(asked())
        assert refused.value.retry_after == 30.0

    async def test_a_retry_after_the_vendor_did_not_write_as_a_number_is_ignored(self) -> None:
        """A malformed header is no wait at all, rather than a crash on the error path."""
        failing = provider(answering(httpx.Response(429, headers={"retry-after": "soon"}, json={})))
        with pytest.raises(ProviderError) as refused:
            await failing.complete(asked())
        assert refused.value.retry_after is None


class TestWhenTheStreamStops:
    async def test_a_dropped_connection_mid_stream_is_an_interruption(self) -> None:
        dropping = provider(
            answering(
                streaming(
                    b'data: {"type":"content_block_delta","index":0,'
                    b'"delta":{"type":"text_delta","text":"it "}}\n\n',
                    then=httpx.ReadError("connection reset"),
                )
            )
        )
        events = await dropping.stream(asked())
        with pytest.raises(StreamInterruptedError, match="stopped streaming after 1 events"):
            [event async for event in events]

    async def test_a_frame_that_is_not_json_is_an_interruption(self) -> None:
        broken = provider(answering(streaming(b"data: {not json\n\n")))
        events = await broken.stream(asked())
        with pytest.raises(StreamInterruptedError, match="not JSON"):
            [event async for event in events]

    async def test_a_frame_that_is_not_an_object_is_an_interruption(self) -> None:
        broken = provider(answering(streaming(b"data: [1, 2]\n\n")))
        events = await broken.stream(asked())
        with pytest.raises(StreamInterruptedError, match="not an object"):
            [event async for event in events]

    async def test_a_done_marker_and_blank_lines_are_not_frames(self) -> None:
        """They carry nothing, and counting them would misreport how much arrived."""
        stalled = provider(answering(streaming(b"\n", b"data: [DONE]\n\n")))
        events = await stalled.stream(asked())
        with pytest.raises(StreamInterruptedError) as interrupted:
            [event async for event in events]
        assert interrupted.value.received == 0

    async def test_a_last_frame_with_no_trailing_newline_is_still_a_frame(self) -> None:
        """Not every server ends the body with one, and dropping it drops the answer."""
        model = provider(
            answering(
                streaming(
                    b'data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},'
                    b'"usage":{"output_tokens":4}}'
                )
            )
        )
        events = [event async for event in await model.stream(asked())]
        assert [type(event).__name__ for event in events][-1] == "StreamEnd"

    async def test_the_rest_of_the_body_is_drained_after_an_unreadable_frame(self) -> None:
        """Abandoning the read there leaves the connection to a garbage collector."""
        broken = provider(
            answering(
                streaming(
                    b"data: {not json}\n\n",
                    b'data: {"type":"message_delta","delta":{"stop_reason":"end_turn"}}\n\n',
                )
            )
        )
        events = await broken.stream(asked())
        with pytest.raises(StreamInterruptedError, match="not JSON"):
            [event async for event in events]
