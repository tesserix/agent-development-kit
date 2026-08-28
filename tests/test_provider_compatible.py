"""The provider for endpoints the operator runs itself, against a stub server.

A self-hosted endpoint speaks OpenAI's wire format and then deviates from it: no key, no
usage, no tool-call ids, a stop reason it forgot to send, a context window smaller than
the one it advertises. Each of those is a wrong answer rather than an error if the kit
papers over it, so each has a test here saying what the kit does instead.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import httpx
import pytest
from pydantic import BaseModel

from tesserix_adk.core import (
    Agent,
    CapabilityError,
    ConfigurationError,
    Message,
    ModelCapabilities,
    ModelRequest,
    ModelResponseError,
    ProviderError,
    ProviderTimeoutError,
    ProviderUnavailableError,
    RunState,
    StopReason,
    StreamEnd,
    TextPart,
)
from tesserix_adk.models.providers import (
    GROK,
    GROQ,
    OLLAMA,
    OPENROUTER,
    VLLM,
    XAI,
    CompatibilityPreset,
    OpenAICompatibleProvider,
)
from tesserix_adk.runtime import AgentRunner
from tesserix_adk.testing import FakeClock, FakeSecrets, FakeToolRegistry

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from tesserix_adk.core import StreamEvent

MODEL = "qwen2.5-7b-instruct"
IN_CLUSTER = "http://vllm.models.svc.cluster.local:8000"
SERVES = ModelCapabilities(
    tool_calling=True,
    structured_output=True,
    streaming=True,
    context_window_tokens=32768,
    max_output_tokens=2048,
)


class Weather(BaseModel):
    """The shape an agent asks this endpoint for.

    Args:
        city: Where the reading is from.
        summary: The reading, in a few words.
    """

    city: str
    summary: str


def served(**overrides: Any) -> dict[str, Any]:
    return {
        "id": "cmpl-1",
        "object": "chat.completion",
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": "it rained"},
            }
        ],
        "usage": {"prompt_tokens": 31, "completion_tokens": 4},
    } | overrides


def answering(
    body: dict[str, Any] | None = None, status: int = 200, headers: dict[str, str] | None = None
) -> tuple[httpx.MockTransport, list[httpx.Request]]:
    """A stub endpoint answering every call the same way, keeping what it was sent."""
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(status, json=body if body is not None else served(), headers=headers)

    return httpx.MockTransport(handler), seen


def raising(failure: Exception) -> httpx.MockTransport:
    def handler(_: httpx.Request) -> httpx.Response:
        raise failure

    return httpx.MockTransport(handler)


def provider(
    transport: httpx.MockTransport | None = None,
    *,
    capabilities: ModelCapabilities | None = SERVES,
    **kwargs: Any,
) -> OpenAICompatibleProvider:
    return OpenAICompatibleProvider(
        MODEL,
        base_url=kwargs.pop("base_url", IN_CLUSTER),
        capabilities=capabilities,
        transport=transport if transport is not None else answering()[0],
        **kwargs,
    )


def asked(**kwargs: Any) -> ModelRequest:
    return ModelRequest(
        model=MODEL,
        messages=(Message(role="user", content=[TextPart(text="did it rain")]),),
        **kwargs,
    )


async def collected(events: AsyncIterator[StreamEvent]) -> list[StreamEvent]:
    return [event async for event in events]


def streaming(*frames: str) -> tuple[httpx.MockTransport, list[httpx.Request]]:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        body = "".join(f"{frame}\n\n" for frame in frames)
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"}, content=body.encode()
        )

    return httpx.MockTransport(handler), seen


class TestAnEndpointHasToBeDescribedBecauseItCannotBeAsked:
    async def test_the_address_is_required(self) -> None:
        """There is no default host for a service only the operator knows the name of."""
        with pytest.raises(ConfigurationError, match="base_url"):
            OpenAICompatibleProvider(MODEL, base_url="", capabilities=SERVES)

    async def test_the_capabilities_are_required(self) -> None:
        """A deployment's flags decide these, and no probe of the endpoint reports them."""
        with pytest.raises(ConfigurationError, match="capabilities"):
            provider(capabilities=None)

    async def test_the_provider_is_named_for_the_server_and_not_for_the_vendor(self) -> None:
        """Traffic to a self-hosted box recorded as OpenAI is traffic on the wrong bill."""
        assert provider(preset=VLLM).name == "vllm"
        assert provider(preset=OLLAMA).name == "ollama"

    async def test_the_name_can_be_the_deployment_rather_than_the_server(self) -> None:
        """Two vLLM deployments of different models are two things to route between."""
        assert provider(preset=VLLM, name="vllm-cheap").name == "vllm-cheap"

    async def test_the_capabilities_are_the_ones_given_and_not_a_catalogue_guess(self) -> None:
        declared = ModelCapabilities(tool_calling=True)
        assert provider(capabilities=declared).capabilities == declared


class TestAnEndpointThatWantsNoKey:
    async def test_no_authorization_header_is_sent_when_no_variable_is_named(self) -> None:
        """An in-cluster endpoint behind a mesh has no key, and inventing one is a 401."""
        transport, seen = answering()
        await provider(transport).complete(asked())
        assert "authorization" not in seen[0].headers

    async def test_a_named_variable_is_read_and_sent(self) -> None:
        transport, seen = answering()
        model = provider(
            transport,
            api_key_variable="VLLM_API_KEY",
            secrets=FakeSecrets({"VLLM_API_KEY": "in-cluster-key"}),
        )
        await model.complete(asked())
        assert seen[0].headers["authorization"] == "Bearer in-cluster-key"

    async def test_a_named_variable_that_is_unset_is_refused_before_the_call(self) -> None:
        transport, seen = answering()
        model = provider(transport, api_key_variable="VLLM_API_KEY", secrets=FakeSecrets({}))
        with pytest.raises(ConfigurationError, match="VLLM_API_KEY"):
            await model.complete(asked())
        assert seen == []

    async def test_service_dns_is_reached_without_a_proxy(self) -> None:
        transport, seen = answering()
        await provider(transport).complete(asked())
        assert str(seen[0].url) == f"{IN_CLUSTER}/v1/chat/completions"


class TestHostedCompatibleProviders:
    def test_grok_is_the_discoverable_name_for_xais_preset(self) -> None:
        assert GROK is XAI

    @pytest.mark.parametrize(
        ("preset", "base_url", "expected_url"),
        [
            (GROQ, "https://api.groq.com", "https://api.groq.com/openai/v1/chat/completions"),
            (XAI, "https://api.x.ai", "https://api.x.ai/v1/chat/completions"),
            (
                OPENROUTER,
                "https://openrouter.ai",
                "https://openrouter.ai/api/v1/chat/completions",
            ),
        ],
        ids=["groq", "xai-grok", "openrouter"],
    )
    async def test_each_preset_uses_its_documented_chat_completions_path(
        self,
        preset: CompatibilityPreset,
        base_url: str,
        expected_url: str,
    ) -> None:
        transport, seen = answering()
        await provider(
            transport,
            preset=preset,
            base_url=base_url,
            api_key_variable="",
        ).complete(asked())
        assert str(seen[0].url) == expected_url

    @pytest.mark.parametrize(
        ("preset", "key_variable", "expected_url"),
        [
            (GROQ, "GROQ_API_KEY", "https://api.groq.com/openai/v1/chat/completions"),
            (XAI, "XAI_API_KEY", "https://api.x.ai/v1/chat/completions"),
            (
                OPENROUTER,
                "OPENROUTER_API_KEY",
                "https://openrouter.ai/api/v1/chat/completions",
            ),
        ],
        ids=["groq", "grok", "openrouter"],
    )
    async def test_hosted_presets_supply_the_usual_url_and_key_variable(
        self,
        preset: CompatibilityPreset,
        key_variable: str,
        expected_url: str,
    ) -> None:
        transport, seen = answering()
        model = OpenAICompatibleProvider(
            MODEL,
            preset=preset,
            capabilities=SERVES,
            secrets=FakeSecrets({key_variable: "hosted-key"}),
            transport=transport,
        )
        await model.complete(asked())
        assert str(seen[0].url) == expected_url
        assert seen[0].headers["authorization"] == "Bearer hosted-key"

    async def test_gateway_metadata_headers_are_sent_without_replacing_authentication(
        self,
    ) -> None:
        transport, seen = answering()
        model = OpenAICompatibleProvider(
            MODEL,
            preset=OPENROUTER,
            capabilities=SERVES,
            secrets=FakeSecrets({"OPENROUTER_API_KEY": "hosted-key"}),
            headers={
                "HTTP-Referer": "https://agents.example.com",
                "X-OpenRouter-Title": "Acme agents",
                "Authorization": "Bearer ignored",
            },
            transport=transport,
        )
        await model.complete(asked())
        assert seen[0].headers["http-referer"] == "https://agents.example.com"
        assert seen[0].headers["x-openrouter-title"] == "Acme agents"
        assert seen[0].headers["authorization"] == "Bearer hosted-key"


class TestWhatTheEndpointCannotDoIsRefusedRatherThanEmulated:
    async def test_an_agent_wanting_a_shape_the_endpoint_cannot_enforce_is_refused(self) -> None:
        """Prompting for JSON and parsing the prose back is a schema nobody enforced."""
        model = provider(capabilities=ModelCapabilities(tool_calling=True))
        runner = AgentRunner(provider=model, clock=FakeClock())
        with pytest.raises(CapabilityError, match="structured_output"):
            await runner.run(
                Agent(name="forecaster", instructions="Answer.", model=MODEL, output_type=Weather),
                "did it rain",
                tenant="acme",
            )

    async def test_a_tool_calling_agent_against_the_same_endpoint_runs(self) -> None:
        """The refusal is of the one capability, not of the endpoint."""
        transport, _ = answering()
        model = provider(transport, capabilities=ModelCapabilities(tool_calling=True))
        finished = await AgentRunner(
            provider=model, tools=FakeToolRegistry({"lookup": lambda: "clear"}), clock=FakeClock()
        ).run(
            Agent(
                name="forecaster",
                instructions="Answer.",
                model=MODEL,
                free_text=True,
                tools=("lookup",),
            ),
            "did it rain",
            tenant="acme",
        )
        assert finished.state is RunState.COMPLETED

    async def test_the_kit_may_be_told_to_emulate_anyway(self) -> None:
        """An operator who has read what it costs can have it; the default is not silent."""
        transport, _ = answering(served(choices=[_choice('{"city": "Delhi", "summary": "clear"}')]))
        model = provider(transport, capabilities=ModelCapabilities(), emulates=True)
        finished = await AgentRunner(provider=model, clock=FakeClock()).run(
            Agent(name="forecaster", instructions="Answer.", model=MODEL, output_type=Weather),
            "did it rain",
            tenant="acme",
        )
        assert finished.output == Weather(city="Delhi", summary="clear")


class TestWhenTheEndpointIsNotThere:
    async def test_an_unreachable_endpoint_is_unavailable_and_retryable(self) -> None:
        """A pod that has not finished starting is worth another attempt; a 400 is not."""
        model = provider(raising(httpx.ConnectError("no route to host")))
        with pytest.raises(ProviderUnavailableError) as refused:
            await model.complete(asked())
        assert refused.value.retryable is True

    async def test_a_gateway_error_while_the_weights_load_is_unavailable(self) -> None:
        transport, _ = answering({"error": "model is loading"}, status=503)
        with pytest.raises(ProviderUnavailableError) as refused:
            await provider(transport).complete(asked())
        assert refused.value.status == 503

    async def test_the_endpoints_own_wait_is_believed_so_a_cold_start_is_not_stampeded(
        self,
    ) -> None:
        """Retrying a loading model every second is how it never finishes loading."""
        transport, _ = answering({"error": "loading"}, status=503, headers={"retry-after": "45"})
        with pytest.raises(ProviderUnavailableError) as refused:
            await provider(transport).complete(asked())
        assert refused.value.retry_after == 45.0

    async def test_a_slow_first_token_hits_the_deadline_rather_than_hanging(self) -> None:
        model = provider(raising(httpx.ReadTimeout("still loading weights")))
        with pytest.raises(ProviderTimeoutError):
            await model.complete(asked())

    async def test_the_deadline_is_longer_than_a_hosted_vendors_by_default(self) -> None:
        """First-token latency on a cold self-hosted model is minutes, not seconds."""
        assert VLLM.timeout > 60.0
        assert OLLAMA.timeout > 60.0

    async def test_a_refusal_the_endpoint_meant_is_not_retried(self) -> None:
        """A window smaller than the declared one comes back as a 400, and stays a 400."""
        transport, _ = answering(
            {"error": {"message": "This model's maximum context length is 4096 tokens"}}, status=400
        )
        with pytest.raises(ProviderError) as refused:
            await provider(transport).complete(asked())
        assert refused.value.retryable is False
        assert refused.value.details["status"] == "400"
        assert "maximum context length" not in str(refused.value.details)


class TestAnErrorDressedAsAnAnswer:
    async def test_an_error_object_under_a_200_is_still_an_error(self) -> None:
        """Some compatible servers answer 200 with the failure in the body, and mean it."""
        transport, _ = answering({"error": {"message": "no such model", "type": "invalid"}})
        with pytest.raises(ProviderError, match="no such model"):
            await provider(transport).complete(asked())

    async def test_no_answer_is_fabricated_from_it(self) -> None:
        transport, _ = answering({"error": {"message": "no such model"}})
        with pytest.raises(ProviderError) as refused:
            await provider(transport).complete(asked())
        assert refused.value.status is None

    async def test_a_body_with_neither_choices_nor_an_error_is_unreadable(self) -> None:
        transport, _ = answering({"object": "chat.completion"})
        with pytest.raises(ModelResponseError):
            await provider(transport).complete(asked())


class TestUsageNobodyReported:
    async def test_a_missing_usage_object_is_estimated_rather_than_zero(self) -> None:
        """Zero tokens reads as a free call, and a call on a GPU somebody paid for is not."""
        transport, _ = answering(served(usage=None))
        response = await provider(transport).complete(asked())
        assert response.usage.input_tokens > 0
        assert response.usage.output_tokens > 0

    async def test_estimated_usage_says_so(self) -> None:
        """A ledger that cannot tell a count from a guess reports a guess as a count."""
        transport, _ = answering(served(usage=None))
        response = await provider(transport).complete(asked())
        assert response.usage.estimated is True

    async def test_zero_counts_are_estimated_too(self) -> None:
        transport, _ = answering(served(usage={"prompt_tokens": 0, "completion_tokens": 0}))
        response = await provider(transport).complete(asked())
        assert response.usage.estimated is True

    async def test_counts_the_endpoint_reported_are_kept_as_reported(self) -> None:
        response = await provider().complete(asked())
        assert (response.usage.input_tokens, response.usage.output_tokens) == (31, 4)
        assert response.usage.estimated is False

    async def test_a_self_hosted_call_has_no_price_rather_than_a_price_of_zero(self) -> None:
        """The GPU costs money the kit cannot see, and zero would be a false statement."""
        assert (await provider().complete(asked())).usage.cost is None

    async def test_an_estimate_stays_an_estimate_when_a_run_totals_it(self) -> None:
        transport, _ = answering(served(usage=None))
        finished = await AgentRunner(provider=provider(transport), clock=FakeClock()).run(
            Agent(name="asker", instructions="Answer.", model=MODEL, free_text=True),
            "did it rain",
            tenant="acme",
        )
        assert finished.usage.estimated is True


class TestWhereTheCompatibleServersDeviate:
    async def test_strict_is_not_claimed_for_a_server_that_does_not_implement_it(self) -> None:
        """vLLM guides decoding by the schema and rejects the vendor's `strict` flag."""
        transport, seen = answering()
        await provider(transport, preset=VLLM).complete(
            asked(output_schema={"type": "object", "properties": {}})
        )
        assert _sent(seen[0])["response_format"]["json_schema"]["strict"] is False

    async def test_a_server_that_ignores_stream_options_is_not_sent_them(self) -> None:
        """Ollama refuses a body field it does not know, and the stream never opens."""
        transport, seen = streaming(
            'data: {"choices":[{"delta":{"content":"it rained"},"finish_reason":"stop"}]}'
        )
        model = provider(transport, preset=OLLAMA)
        await collected(await model.stream(asked()))
        assert "stream_options" not in _sent(seen[0])

    async def test_a_server_that_reports_usage_on_request_is_asked_to(self) -> None:
        transport, seen = streaming(
            'data: {"choices":[{"delta":{"content":"it rained"},"finish_reason":"stop"}]}'
        )
        model = provider(transport, preset=VLLM)
        await collected(await model.stream(asked()))
        assert _sent(seen[0])["stream_options"] == {"include_usage": True}

    async def test_a_tool_call_with_no_id_is_given_one_rather_than_refused(self) -> None:
        """Ollama sends none, and a result with nothing to match it back to is unusable."""
        transport, _ = answering(
            served(
                choices=[
                    {
                        "index": 0,
                        "finish_reason": "tool_calls",
                        "message": {
                            "role": "assistant",
                            "tool_calls": [
                                {
                                    "type": "function",
                                    "function": {"name": "lookup", "arguments": "{}"},
                                }
                            ],
                        },
                    }
                ]
            )
        )
        response = await provider(transport, preset=OLLAMA).complete(asked())
        assert [call.name for call in response.tool_calls] == ["lookup"]
        assert response.tool_calls[0].id

    async def test_a_body_with_no_choices_is_unreadable_rather_than_patched(self) -> None:
        """Minting ids into a body that has no answer in it invents the answer too."""
        transport, _ = answering({"object": "chat.completion", "choices": None})
        with pytest.raises(ModelResponseError):
            await provider(transport, preset=OLLAMA).complete(asked())

    async def test_a_choice_with_no_message_is_left_alone_rather_than_rebuilt(self) -> None:
        """There is nothing to mint an id onto, and inventing a message invents an answer."""
        transport, _ = answering({"choices": [{"index": 0}], "usage": {"prompt_tokens": 3}})
        response = await provider(transport, preset=OLLAMA).complete(asked())
        assert response.content == ""
        assert response.tool_calls == ()

    async def test_a_stop_reason_the_server_forgot_is_read_off_the_answer(self) -> None:
        """`unknown` would end a run that had asked for a tool and never run it."""
        transport, _ = answering(
            served(
                choices=[
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {"name": "lookup", "arguments": "{}"},
                                }
                            ],
                        },
                    }
                ]
            )
        )
        response = await provider(transport, preset=OLLAMA).complete(asked())
        assert response.stop_reason is StopReason.TOOL_CALLS

    async def test_an_answer_with_no_stop_reason_and_no_call_is_a_finished_turn(self) -> None:
        transport, _ = answering(served(choices=[_choice("it rained")]))
        response = await provider(transport, preset=OLLAMA).complete(asked())
        assert response.stop_reason is StopReason.END_TURN

    async def test_a_server_that_reports_its_stop_reason_is_believed(self) -> None:
        transport, _ = answering(served(choices=[_choice("it rained", finish_reason="length")]))
        response = await provider(transport, preset=OLLAMA).complete(asked())
        assert response.stop_reason is StopReason.MAX_TOKENS


class TestTheSameEndpointStreams:
    async def test_the_events_are_the_kits_own(self) -> None:
        model = provider(
            streaming(
                'data: {"choices":[{"delta":{"content":"it "}}]}',
                'data: {"choices":[{"delta":{"content":"rained"},"finish_reason":"stop"}]}',
            )[0],
            preset=VLLM,
        )
        end = (await collected(await model.stream(asked())))[-1]
        assert isinstance(end, StreamEnd)
        assert end.response.content == "it rained"

    async def test_a_stream_that_reported_no_usage_is_estimated_too(self) -> None:
        transport, _ = streaming(
            'data: {"choices":[{"delta":{"content":"it rained"},"finish_reason":"stop"}]}'
        )
        end = (await collected(await provider(transport).stream(asked())))[-1]
        assert isinstance(end, StreamEnd)
        assert end.response.usage.estimated is True

    async def test_streaming_is_refused_where_it_was_not_declared(self) -> None:
        model = provider(capabilities=ModelCapabilities(tool_calling=True))
        with pytest.raises(CapabilityError, match="streaming"):
            await model.stream(asked())


def _choice(content: str, finish_reason: str | None = None) -> dict[str, Any]:
    choice: dict[str, Any] = {
        "index": 0,
        "message": {"role": "assistant", "content": content},
    }
    if finish_reason is not None:
        choice["finish_reason"] = finish_reason
    return choice


def _sent(request: httpx.Request) -> dict[str, Any]:
    body = json.loads(request.content)
    assert isinstance(body, dict)
    return body
