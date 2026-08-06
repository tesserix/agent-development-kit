"""What a vendor failure arrives as, and the two things done before the call is sent.

Five scenarios: three vendors signalling one rate limit three ways; a spent quota that is
not worth retrying; what a failure carries and what it deliberately does not; which wait
ran out; and one allowance shared by two providers on the same key.

Run it with `python examples/resilience.py`. A stub stands in for each vendor, so nothing
here reaches the network and no key is needed.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import httpx

from tesserix_adk.core import (
    Message,
    ModelRequest,
    ProviderError,
    RateLimitError,
    RetryConfig,
    TextPart,
)
from tesserix_adk.models.providers import (
    PHASE_DEFAULTS,
    AnthropicProvider,
    GeminiProvider,
    OpenAIProvider,
)
from tesserix_adk.runtime import RateLimiter, RetryPlan
from tesserix_adk.testing import FakeClock, FakeSecrets

if TYPE_CHECKING:
    from collections.abc import Callable

Body = dict[str, Any]
Provider = AnthropicProvider | OpenAIProvider | GeminiProvider

ANTHROPIC = "claude-sonnet-4-5"
OPENAI = "gpt-4o"
GEMINI = "gemini-2.0-flash"

PLAN = RetryPlan(RetryConfig(max_attempts=3))


def anthropic(status: int, body: Body, **headers: str) -> AnthropicProvider:
    """An Anthropic provider whose endpoint is a stub answering `status`."""
    return AnthropicProvider(
        ANTHROPIC, secrets=_key("ANTHROPIC"), transport=_answering(status, body, headers)
    )


def openai(status: int, body: Body, **headers: str) -> OpenAIProvider:
    """An OpenAI provider whose endpoint is a stub answering `status`."""
    return OpenAIProvider(
        OPENAI, secrets=_key("OPENAI"), transport=_answering(status, body, headers)
    )


def gemini(status: int, body: Body, **headers: str) -> GeminiProvider:
    """A Gemini provider whose endpoint is a stub answering `status`."""
    return GeminiProvider(
        GEMINI, secrets=_key("GEMINI"), transport=_answering(status, body, headers)
    )


async def one_event_three_vocabularies() -> None:
    """A rate limit is one operational event whoever sent it, and one type here."""
    for provider in (
        anthropic(429, _error(type="rate_limit_error")),
        openai(429, _error(code="rate_limit_exceeded")),
        gemini(429, _error(code=429, status="RESOURCE_EXHAUSTED")),
    ):
        failure = await _failed(provider)
        print(  # noqa: T201
            f"{provider.name:<10}",
            f"{type(failure).__name__} code={failure.details['code']!r}",
            f"retried={PLAN.retryable(failure)}",
        )


async def a_quota_is_not_a_rate_limit() -> None:
    """A rate clears by waiting. An allowance clears when somebody pays."""
    spent = await _failed(openai(429, _error(code="insufficient_quota")))
    quota = spent.quota if isinstance(spent, RateLimitError) else None
    print(f"{'openai':<10} quota={quota} retried={PLAN.retryable(spent)}")  # noqa: T201

    broken = await _failed(openai(401, _error(code="invalid_api_key")))
    print(f"{'openai':<10} {type(broken).__name__} retried={PLAN.retryable(broken)}")  # noqa: T201


async def what_the_failure_carries() -> None:
    """A 400 body quotes the request that caused it, and the request is the prompt."""
    address = "the defendant lives at 14 Pine Road"
    failure = await _failed(
        openai(400, _error(code="invalid_request_error", message=address), **{"x-request-id": "r7"})
    )
    print(  # noqa: T201
        f"{'openai':<10}",
        f"provider={failure.provider} model={failure.model} request={failure.request_id}",
        f"prompt_leaked={address in str(failure) + str(failure.details)}",
    )


async def which_wait_ran_out() -> None:
    """A dead host and a slow model are one exception in httpx and two different events."""
    print(  # noqa: T201
        f"{'defaults':<10} connect={PHASE_DEFAULTS.connect}s read={PHASE_DEFAULTS.read}s"
    )
    for ran_out in (httpx.ConnectTimeout("no answer"), httpx.ReadTimeout("still thinking")):
        provider = OpenAIProvider(
            OPENAI, secrets=_key("OPENAI"), transport=httpx.MockTransport(_raising(ran_out))
        )
        failure = await _failed(provider)
        print(f"{'openai':<10} {type(failure).__name__} phase={failure.details['phase']}")  # noqa: T201


async def one_key_one_allowance() -> None:
    """Two providers on one key have one allowance between them, not one each."""
    clock = FakeClock()
    shared = RateLimiter(requests_per_minute=2, clock=clock)
    for model in (OPENAI, "gpt-4o-mini", OPENAI):
        async with OpenAIProvider(
            model, secrets=_key("OPENAI"), limiter=shared, transport=_answering(200, _reply(), {})
        ) as provider:
            await provider.complete(_asked(model))
    print(f"{'openai':<10} calls=3 waited={[round(s, 1) for s in clock.slept]}s")  # noqa: T201


async def _failed(provider: Provider) -> ProviderError:
    async with provider:
        try:
            await provider.complete(_asked(provider.model))
        except ProviderError as refused:
            return refused
    raise RuntimeError("the stub was meant to fail")


def _answering(status: int, body: Body, headers: dict[str, str]) -> httpx.MockTransport:
    return httpx.MockTransport(lambda _: httpx.Response(status, json=body, headers=headers))


def _raising(failure: Exception) -> Callable[[httpx.Request], httpx.Response]:
    def handler(_: httpx.Request) -> httpx.Response:
        raise failure

    return handler


def _key(vendor: str) -> FakeSecrets:
    return FakeSecrets({f"{vendor}_API_KEY": "not-a-real-key"})


def _error(**fields: object) -> Body:
    return {"error": fields}


def _reply() -> dict[str, Any]:
    return {
        "id": "cmpl-1",
        "object": "chat.completion",
        "choices": [
            {"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "ok"}}
        ],
        "usage": {"prompt_tokens": 3, "completion_tokens": 1},
    }


def _asked(model: str) -> ModelRequest:
    return ModelRequest(
        model=model, messages=(Message(role="user", content=[TextPart(text="did it rain")]),)
    )


async def main() -> None:
    """Run every scenario in order."""
    await one_event_three_vocabularies()
    await a_quota_is_not_a_rate_limit()
    await what_the_failure_carries()
    await which_wait_ran_out()
    await one_key_one_allowance()


if __name__ == "__main__":
    asyncio.run(main())
