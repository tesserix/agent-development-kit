"""The llama.cpp server, which is how an operator without a GPU runs anything at all.

It speaks OpenAI's wire format through `llama-server`, with its own accents and its own
tuning. What is tested here is the part the kit owns: the flags it renders, the prompt
cache it asks for, the fit it refuses before loading, and the fact that an agent written
against vLLM does not change to run here.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import httpx
import pytest

from tesserix_adk.core import (
    Agent,
    Message,
    ModelCapabilities,
    ModelRequest,
    NoOutput,
    RunState,
    TextPart,
)
from tesserix_adk.models.gguf import GgufModel, ModelTooLargeError, Quantization
from tesserix_adk.models.providers import VLLM, LlamaCppProvider, LlamaCppTuning
from tesserix_adk.models.providers.compatible import OpenAICompatibleProvider
from tesserix_adk.runtime import AgentRunner
from tesserix_adk.testing import FakeClock

if TYPE_CHECKING:
    from tesserix_adk.core.protocols import ModelProvider

GIB = 1024**3
MODEL = "llama-3.1-8b-instruct"
LOCAL = "http://127.0.0.1:8080"
SERVES = ModelCapabilities(
    tool_calling=True,
    structured_output=True,
    streaming=True,
    context_window_tokens=8_192,
    max_output_tokens=1_024,
)


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


def answering() -> tuple[httpx.MockTransport, list[httpx.Request]]:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=served())

    return httpx.MockTransport(handler), seen


def provider(transport: httpx.MockTransport | None = None, **kwargs: Any) -> LlamaCppProvider:
    return LlamaCppProvider(
        MODEL,
        base_url=kwargs.pop("base_url", LOCAL),
        capabilities=kwargs.pop("capabilities", SERVES),
        transport=transport if transport is not None else answering()[0],
        **kwargs,
    )


def asked(text: str = "did it rain") -> ModelRequest:
    return ModelRequest(
        model=MODEL, messages=(Message(role="user", content=[TextPart(text=text)]),)
    )


def body_of(request: httpx.Request) -> dict[str, Any]:
    parsed: dict[str, Any] = json.loads(request.content)
    return parsed


class TestTheServerIsDescribedNotProbed:
    async def test_a_full_turn_reports_what_it_used(self) -> None:
        """A CPU run that reports nothing cannot be budgeted, which is most of the point."""
        answered = await provider().complete(asked())
        assert answered.content == "it rained"
        assert answered.usage.input_tokens == 31
        assert answered.usage.output_tokens == 4

    async def test_the_provider_names_itself_llama_cpp_not_openai(self) -> None:
        """A box under a desk billed to the vendor is spend attributed to the wrong budget."""
        assert provider().provider_name == "llama.cpp"

    async def test_no_authorization_header_is_sent_to_a_local_server(self) -> None:
        transport, seen = answering()
        await provider(transport).complete(asked())
        assert "authorization" not in seen[0].headers

    async def test_the_timeout_is_long_because_the_first_token_waits_for_the_weights(
        self,
    ) -> None:
        assert provider().timeouts.read >= 600.0


class TestThePromptCache:
    async def test_the_cache_is_asked_for_by_default(self) -> None:
        """Without it every turn re-evaluates the whole prefix, which on CPU is the run."""
        transport, seen = answering()
        await provider(transport).complete(asked())
        assert body_of(seen[0])["cache_prompt"] is True

    async def test_a_shared_prefix_arrives_byte_identical_on_the_second_turn(self) -> None:
        """A prefix that shifts by one byte is a cache that never hits."""
        transport, seen = answering()
        served_by = provider(transport)
        system = Message(role="system", content=[TextPart(text="answer in one word")])
        for question in ("did it rain", "did it snow"):
            await served_by.complete(
                ModelRequest(
                    model=MODEL,
                    messages=(system, Message(role="user", content=[TextPart(text=question)])),
                )
            )
        first, second = (body_of(request)["messages"][0] for request in seen)
        assert json.dumps(first) == json.dumps(second)

    async def test_the_cache_can_be_turned_off_for_a_server_that_shares_a_slot(self) -> None:
        transport, seen = answering()
        await provider(transport, tuning=LlamaCppTuning(prompt_cache=False)).complete(asked())
        assert "cache_prompt" not in body_of(seen[0])


class TestTuningIsConfigurationNotFolklore:
    def test_the_flags_are_rendered_from_the_same_description_the_kit_uses(self) -> None:
        """An operator tuning by hand and a kit checking the fit must read one description."""
        tuning = LlamaCppTuning(threads=8, batch_size=512, context_tokens=8_192)
        assert tuning.server_arguments() == (
            "--threads",
            "8",
            "--batch-size",
            "512",
            "--ctx-size",
            "8192",
        )

    def test_a_thread_count_over_the_physical_cores_is_refused(self) -> None:
        """Oversubscribing a CPU backend makes it slower, reliably and counter-intuitively."""
        with pytest.raises(ValueError, match="threads"):
            LlamaCppTuning(threads=0)

    def test_the_context_the_server_was_started_with_is_what_the_fit_is_checked_against(
        self,
    ) -> None:
        tuning = LlamaCppTuning(context_tokens=32_768)
        with pytest.raises(ModelTooLargeError, match="32768-token context"):
            provider(
                tuning=tuning,
                weights=GgufModel(name=MODEL, parameters_b=8.03),
                available_bytes=8 * GIB,
            )

    def test_an_undescribed_flag_is_left_to_llama_cpps_own_default(self) -> None:
        """Rendering a guess at a thread count is worse than rendering nothing."""
        assert LlamaCppTuning().server_arguments() == ("--ctx-size", "4096")

    def test_a_micro_batch_is_rendered_where_a_deployment_tunes_it_apart(self) -> None:
        tuning = LlamaCppTuning(batch_size=2_048, micro_batch_size=512, context_tokens=4_096)
        assert tuning.server_arguments() == (
            "--batch-size",
            "2048",
            "--ubatch-size",
            "512",
            "--ctx-size",
            "4096",
        )

    def test_the_provider_carries_the_tuning_it_was_given(self) -> None:
        """What the server was started with is answerable from the provider, not folklore."""
        tuning = LlamaCppTuning(threads=4)
        assert provider(tuning=tuning).tuning is tuning

    def test_a_batch_size_below_the_micro_batch_is_refused(self) -> None:
        with pytest.raises(ValueError, match="micro-batch"):
            LlamaCppTuning(batch_size=64, micro_batch_size=512)


class TestAModelBiggerThanTheMachine:
    def test_it_is_refused_at_construction_rather_than_oom_killed_mid_run(self) -> None:
        with pytest.raises(ModelTooLargeError) as refused:
            provider(weights=GgufModel(name=MODEL, parameters_b=70.0), available_bytes=8 * GIB)
        assert "70.0" in str(refused.value) or MODEL in str(refused.value)

    def test_a_model_that_fits_is_constructed_without_complaint(self) -> None:
        assert (
            provider(
                weights=GgufModel(name=MODEL, parameters_b=8.03), available_bytes=16 * GIB
            ).provider_name
            == "llama.cpp"
        )

    def test_nothing_is_checked_where_the_operator_described_nothing(self) -> None:
        """The kit does not guess a machine's memory, and does not refuse for want of one."""
        assert provider(weights=GgufModel(name=MODEL, parameters_b=70.0)).provider_name

    def test_the_heavier_quantization_is_the_one_that_gets_refused(self) -> None:
        light = GgufModel(name=MODEL, parameters_b=8.03, quantization=Quantization.Q4_K_M)
        with pytest.raises(ModelTooLargeError):
            provider(weights=light.at(Quantization.F16), available_bytes=10 * GIB)
        assert provider(weights=light, available_bytes=10 * GIB).provider_name


class TestTheSameAgentRunsSomewhereElse:
    async def test_an_agent_written_for_vllm_runs_here_unchanged(self) -> None:
        """The whole promise of the CPU path: the agent is not a different agent."""
        agent: Agent[NoOutput] = Agent(
            name="reporter", instructions="answer in one word", model=MODEL, free_text=True
        )

        async def ran(against: ModelProvider) -> str:
            run = await AgentRunner(provider=against, clock=FakeClock()).run(
                agent, "did it rain", tenant="acme"
            )
            assert run.state is RunState.COMPLETED
            return "".join(
                part.text for part in run.messages[-1].content if isinstance(part, TextPart)
            )

        on_cpu = await ran(provider())
        on_gpu = await ran(
            OpenAICompatibleProvider(
                MODEL,
                base_url="http://vllm.models.svc.cluster.local:8000",
                capabilities=SERVES,
                preset=VLLM,
                transport=answering()[0],
            )
        )
        assert on_cpu == on_gpu == "it rained"
