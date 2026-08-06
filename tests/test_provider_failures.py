"""What a vendor failure arrives as, once the adapter has finished with it.

Every vendor signals the same five or six events differently: a rate limit is a 429 with
`rate_limit_error` at one, `rate_limit_exceeded` at another and `RESOURCE_EXHAUSTED` at a
third. A consumer that has to know which is a consumer writing three error handlers, so
this asserts the one taxonomy each of them lands in, and that the retry decision follows
from the type rather than from a string match on the body.

It also asserts what is *not* carried: a 400 body echoes the request that caused it, and a
raw body copied into an exception is prompt content in a log.
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from tesserix_adk.core import (
    AdkError,
    AuthenticationError,
    ContentFilteredError,
    ContextWindowExceededError,
    InvalidRequestError,
    Message,
    ModelRequest,
    ProviderError,
    ProviderTimeoutError,
    ProviderUnavailableError,
    RateLimitError,
    RetryConfig,
    TextPart,
)
from tesserix_adk.models.providers import AnthropicProvider, GeminiProvider, OpenAIProvider
from tesserix_adk.runtime import RetryPlan
from tesserix_adk.testing import FakeSecrets

Vendor = AnthropicProvider | OpenAIProvider | GeminiProvider

ANTHROPIC_MODEL = "claude-sonnet-4-5"
OPENAI_MODEL = "gpt-4o"
GEMINI_MODEL = "gemini-2.0-flash"


def anthropic(status: int, body: Any, **headers: str) -> AnthropicProvider:
    return AnthropicProvider(
        ANTHROPIC_MODEL,
        secrets=FakeSecrets({"ANTHROPIC_API_KEY": "test-key"}),
        transport=_answering(status, body, headers),
    )


def openai(status: int, body: Any, **headers: str) -> OpenAIProvider:
    return OpenAIProvider(
        OPENAI_MODEL,
        secrets=FakeSecrets({"OPENAI_API_KEY": "test-key"}),
        transport=_answering(status, body, headers),
    )


def gemini(status: int, body: Any, **headers: str) -> GeminiProvider:
    return GeminiProvider(
        GEMINI_MODEL,
        secrets=FakeSecrets({"GEMINI_API_KEY": "test-key"}),
        transport=_answering(status, body, headers),
    )


def _answering(status: int, body: Any, headers: dict[str, str]) -> httpx.MockTransport:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json=body, headers=headers)

    return httpx.MockTransport(handler)


def asked(model: str) -> ModelRequest:
    return ModelRequest(
        model=model, messages=(Message(role="user", content=[TextPart(text="did it rain")]),)
    )


async def refused[F: AdkError](provider: Vendor, kind: type[F]) -> F:
    """Call `provider` and return the failure it raised, asserting the type first."""
    async with provider:
        with pytest.raises(kind) as raised:
            await provider.complete(asked(provider.model))
    return raised.value


def _error(**fields: Any) -> dict[str, Any]:
    return {"error": fields}


class TestARateLimitIsARateLimitWhoeverSentIt:
    async def test_anthropic_signals_one_with_its_own_error_type(self) -> None:
        failure = await refused(
            anthropic(429, _error(type="rate_limit_error", message="slow down")), RateLimitError
        )
        assert failure.retryable
        assert failure.provider == "anthropic"

    async def test_openai_signals_the_same_thing_with_a_different_word(self) -> None:
        failure = await refused(
            openai(429, _error(type="requests", code="rate_limit_exceeded")), RateLimitError
        )
        assert failure.retryable

    async def test_gemini_signals_it_with_a_status_rather_than_a_type(self) -> None:
        failure = await refused(
            gemini(429, _error(code=429, status="RESOURCE_EXHAUSTED")), RateLimitError
        )
        assert failure.retryable

    async def test_a_429_that_names_no_code_at_all_is_still_a_rate_limit(self) -> None:
        assert (await refused(openai(429, {}), RateLimitError)).retryable

    async def test_the_wait_the_vendor_asked_for_is_carried_rather_than_computed(self) -> None:
        failure = await refused(
            anthropic(429, _error(type="rate_limit_error"), **{"retry-after": "12"}),
            RateLimitError,
        )
        assert failure.retry_after == 12.0


class TestAQuotaIsNotARateLimit:
    """A limit clears on its own; a quota clears when somebody pays. Retrying is the same
    call every time, so it is classified as the configuration failure it is."""

    async def test_an_exhausted_quota_is_not_retried(self) -> None:
        failure = await refused(
            openai(429, _error(type="insufficient_quota", code="insufficient_quota")),
            RateLimitError,
        )
        assert not failure.retryable
        assert failure.quota

    async def test_an_ordinary_rate_limit_does_not_claim_to_be_one(self) -> None:
        failure = await refused(openai(429, _error(code="rate_limit_exceeded")), RateLimitError)
        assert not failure.quota


class TestABrokenKeyIsNotAmplified:
    async def test_a_401_is_an_authentication_failure_and_is_never_retried(self) -> None:
        failure = await refused(
            openai(401, _error(code="invalid_api_key", message="bad key")), AuthenticationError
        )
        assert not failure.retryable

    async def test_a_403_is_one_too(self) -> None:
        failure = await refused(
            anthropic(403, _error(type="permission_error")), AuthenticationError
        )
        assert not failure.retryable

    async def test_gemini_says_it_in_capitals(self) -> None:
        await refused(gemini(401, _error(status="UNAUTHENTICATED")), AuthenticationError)


class TestAPromptTooLongForTheWindow:
    async def test_openai_reporting_it_is_not_retried(self) -> None:
        failure = await refused(
            openai(400, _error(code="context_length_exceeded")), ContextWindowExceededError
        )
        assert not failure.retryable

    async def test_anthropic_reporting_it_by_size(self) -> None:
        await refused(anthropic(413, _error(type="request_too_large")), ContextWindowExceededError)

    async def test_the_declared_window_is_carried_where_the_vendor_gave_no_number(self) -> None:
        async with openai(200, {}) as declared:
            window = declared.capabilities.context_window_tokens
        failure = await refused(
            openai(400, _error(code="context_length_exceeded")), ContextWindowExceededError
        )
        assert failure.limit == window


class TestContentTheVendorWouldNotProcess:
    async def test_a_content_filter_is_its_own_type(self) -> None:
        failure = await refused(
            openai(400, _error(code="content_filter", message="refused")), ContentFilteredError
        )
        assert not failure.retryable

    async def test_a_policy_violation_is_the_same_event(self) -> None:
        await refused(openai(403, _error(code="content_policy_violation")), ContentFilteredError)

    async def test_gemini_blocking_a_prompt_is_too(self) -> None:
        await refused(gemini(400, _error(status="BLOCKED")), ContentFilteredError)


class TestARequestTheVendorRejected:
    async def test_a_400_is_invalid_rather_than_transient(self) -> None:
        failure = await refused(
            anthropic(400, _error(type="invalid_request_error")), InvalidRequestError
        )
        assert not failure.retryable

    async def test_a_404_names_a_model_that_is_not_there(self) -> None:
        await refused(openai(404, _error(code="model_not_found")), InvalidRequestError)

    async def test_gemini_rejects_arguments_in_capitals(self) -> None:
        await refused(gemini(400, _error(status="INVALID_ARGUMENT")), InvalidRequestError)


class TestSomethingUpstreamIsNotThere:
    async def test_a_503_is_waited_for(self) -> None:
        failure = await refused(openai(503, _error(code="unavailable")), ProviderUnavailableError)
        assert failure.retryable

    async def test_anthropic_overloaded_is_the_same_event_under_its_own_status(self) -> None:
        failure = await refused(
            anthropic(529, _error(type="overloaded_error")), ProviderUnavailableError
        )
        assert failure.retryable

    async def test_a_408_is_a_timeout_the_vendor_reported_rather_than_one_we_hit(self) -> None:
        assert (await refused(openai(408, {}), ProviderTimeoutError)).retryable


class TestAFailureNobodyMapped:
    """The default is explicit: an unrecognised failure is a `ProviderError` whose
    retryability follows its status, never a swallowed one and never a guess."""

    async def test_an_unknown_code_under_a_500_stays_retryable(self) -> None:
        failure = await refused(openai(500, _error(code="teapot_overheated")), ProviderError)
        assert type(failure) is ProviderError
        assert failure.retryable

    async def test_an_unknown_code_under_a_418_does_not(self) -> None:
        failure = await refused(openai(418, _error(code="teapot")), ProviderError)
        assert type(failure) is ProviderError
        assert not failure.retryable

    async def test_a_gateways_html_page_is_classified_by_its_status_alone(self) -> None:
        """The 502 in front of the vendor is a proxy's error page, not a vendor body."""
        proxied = OpenAIProvider(
            OPENAI_MODEL,
            secrets=FakeSecrets({"OPENAI_API_KEY": "test-key"}),
            transport=httpx.MockTransport(
                lambda _: httpx.Response(502, text="<html>bad gateway</html>")
            ),
        )
        failure = await refused(proxied, ProviderUnavailableError)
        assert failure.details["code"] == ""
        assert "bad gateway" not in str(failure.details)

    async def test_a_body_that_is_not_an_object_does_not_stop_the_classification(self) -> None:
        assert not (await refused(openai(400, ["nope"]), InvalidRequestError)).retryable

    async def test_an_error_field_that_is_a_bare_string_is_read_as_the_message(self) -> None:
        await refused(openai(400, {"error": "nope"}), InvalidRequestError)


class TestWhatTheFailureCarries:
    async def test_it_names_the_provider_the_model_and_the_call(self) -> None:
        failure = await refused(
            openai(429, _error(code="rate_limit_exceeded"), **{"x-request-id": "req-7"}),
            RateLimitError,
        )
        assert (failure.provider, failure.model, failure.request_id) == (
            "openai",
            OPENAI_MODEL,
            "req-7",
        )

    async def test_the_vendors_own_code_is_kept_because_a_ticket_is_answered_against_it(
        self,
    ) -> None:
        failure = await refused(anthropic(429, _error(type="rate_limit_error")), RateLimitError)
        assert failure.details["code"] == "rate_limit_error"


class TestNothingTheModelWasSentComesBackInTheError:
    """A 400 body quotes the request that caused it. Copied into an exception, that is
    prompt content in every log line the exception reaches."""

    async def test_the_vendors_message_is_not_in_the_error(self) -> None:
        quoted = "the defendant's address is 14 Pine Road"
        failure = await refused(
            openai(400, _error(code="invalid_request_error", message=quoted)), InvalidRequestError
        )
        assert quoted not in str(failure)
        assert quoted not in str(failure.details)

    async def test_nor_is_the_raw_body(self) -> None:
        quoted = "14 Pine Road"
        failure = await refused(openai(400, {"error": {"echo": quoted}}), InvalidRequestError)
        assert quoted not in str(failure)
        assert quoted not in str(failure.details)

    async def test_an_operator_who_needs_it_can_ask_for_it(self) -> None:
        quoted = "the model was sent 14 Pine Road"
        provider = OpenAIProvider(
            OPENAI_MODEL,
            secrets=FakeSecrets({"OPENAI_API_KEY": "test-key"}),
            transport=_answering(400, _error(code="invalid_request_error", message=quoted), {}),
            redact_vendor_messages=False,
        )
        failure = await refused(provider, InvalidRequestError)
        assert failure.details["message"] == quoted

    async def test_the_status_and_the_code_are_enough_to_read_it_without_the_body(self) -> None:
        failure = await refused(openai(429, _error(code="rate_limit_exceeded")), RateLimitError)
        assert "openai" in str(failure)
        assert "429" in str(failure)
        assert "rate_limit_exceeded" in str(failure)


class TestTheRetryPolicyReadsTheTaxonomyRatherThanTheBody:
    """The point of one taxonomy is that the layer above never sees a vendor string. The
    retry policy asks the error, and the error already knows."""

    async def test_a_rate_limit_is_tried_again(self) -> None:
        failure = await refused(anthropic(429, _error(type="rate_limit_error")), RateLimitError)
        assert _plan().retryable(failure)

    async def test_a_spent_quota_and_a_broken_key_are_not(self) -> None:
        quota = await refused(openai(429, _error(code="insufficient_quota")), RateLimitError)
        key = await refused(openai(401, _error(code="invalid_api_key")), AuthenticationError)
        assert not _plan().retryable(quota)
        assert not _plan().retryable(key)

    async def test_the_wait_the_vendor_asked_for_is_the_one_taken(self) -> None:
        failure = await refused(
            anthropic(429, _error(type="rate_limit_error"), **{"retry-after": "7"}), RateLimitError
        )
        assert _plan().delay_for(1, retry_after=failure.retry_after) == pytest.approx(7.0)


def _plan() -> RetryPlan:
    return RetryPlan(RetryConfig(max_attempts=3))
