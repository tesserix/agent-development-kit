"""Every vendor adapter against the shared `ModelProvider` suite.

The per-vendor suites assert each translation. This asserts the thing the runtime assumes
of anything it is handed: a stable capability record, a name, a token count that does not
go backwards, and a response object at the end.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from tesserix_adk.models.providers import AnthropicProvider, GeminiProvider, OpenAIProvider
from tesserix_adk.testing import (
    FakeSecrets,
    HttpCassette,
    HttpExchange,
    HttpReplay,
    ModelProviderConformance,
)

if TYPE_CHECKING:
    from tesserix_adk.core.protocols import ModelProvider

# The suite asks one provider for several completions, and a spare costs nothing.
_ANSWERS = 8


def _replay(provider: str, path: str, body: dict[str, Any]) -> HttpReplay:
    return HttpReplay(
        HttpCassette(
            provider=provider,
            exchanges=tuple(HttpExchange(path=path, body=body) for _ in range(_ANSWERS)),
        )
    )


class TestAnthropicConformance(ModelProviderConformance):
    def make_provider(self) -> ModelProvider:
        replay = _replay(
            "anthropic",
            "/v1/messages",
            {
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": "hello"}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 3, "output_tokens": 1},
            },
        )
        return AnthropicProvider(
            "claude-sonnet-4-5",
            secrets=FakeSecrets({"ANTHROPIC_API_KEY": "test-key"}),
            transport=replay.transport,
        )


class TestOpenAIConformance(ModelProviderConformance):
    def make_provider(self) -> ModelProvider:
        replay = _replay(
            "openai",
            "/v1/chat/completions",
            {
                "id": "chatcmpl_1",
                "object": "chat.completion",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "hello"},
                    }
                ],
                "usage": {"prompt_tokens": 3, "completion_tokens": 1},
            },
        )
        return OpenAIProvider(
            "gpt-4o",
            secrets=FakeSecrets({"OPENAI_API_KEY": "test-key"}),
            transport=replay.transport,
        )


class TestGeminiConformance(ModelProviderConformance):
    def make_provider(self) -> ModelProvider:
        replay = _replay(
            "gemini",
            # The suite asks under its own model name, and Gemini puts the model in the path.
            "/v1beta/models/conformance:generateContent",
            {
                "candidates": [
                    {
                        "content": {"role": "model", "parts": [{"text": "hello"}]},
                        "finishReason": "STOP",
                    }
                ],
                "usageMetadata": {"promptTokenCount": 3, "candidatesTokenCount": 1},
            },
        )
        return GeminiProvider(
            "gemini-2.5-flash",
            secrets=FakeSecrets({"GEMINI_API_KEY": "test-key"}),
            transport=replay.transport,
        )
