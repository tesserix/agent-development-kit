"""Operator-triggered, cost-bounded live smoke for the complete reference agents.

Without ``--live`` this file performs no network call and needs no credential. The
``manual-reference-agents`` workflow is the intended live entry point.
"""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass

from tesserix_adk.core import Agent, BudgetLimits, DeadlineConfig, ModelCapabilities
from tesserix_adk.models.providers import (
    GROK,
    GROQ,
    OPENROUTER,
    AnthropicProvider,
    GeminiProvider,
    OpenAICompatibleProvider,
    OpenAIProvider,
)
from tesserix_adk.runtime import AgentRunner

HostedProvider = AnthropicProvider | GeminiProvider | OpenAICompatibleProvider | OpenAIProvider


@dataclass(frozen=True)
class Options:
    """Validated command-line options for one bounded provider call."""

    live: bool
    provider: str
    model: str
    api_key_variable: str
    max_input_tokens: int
    max_output_tokens: int


def positive(raw: str) -> int:
    """Parse a strictly positive integer for a hard token ceiling."""
    value = int(raw)
    if value < 1:
        raise argparse.ArgumentTypeError("token ceilings must be positive")
    return value


def options(argv: list[str] | None = None) -> Options:
    """Parse the intentionally small live-smoke interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true")
    parser.add_argument(
        "--provider",
        choices=("openai", "anthropic", "gemini", "groq", "grok", "openrouter"),
        default="openrouter",
    )
    parser.add_argument("--model", default="")
    parser.add_argument("--api-key-variable", default="ADK_PROVIDER_API_KEY")
    parser.add_argument("--max-input-tokens", type=positive, default=512)
    parser.add_argument("--max-output-tokens", type=positive, default=64)
    parsed = parser.parse_args(argv)
    return Options(
        live=bool(parsed.live),
        provider=str(parsed.provider),
        model=str(parsed.model),
        api_key_variable=str(parsed.api_key_variable),
        max_input_tokens=int(parsed.max_input_tokens),
        max_output_tokens=int(parsed.max_output_tokens),
    )


def provider_for(config: Options) -> HostedProvider:
    """Build the selected provider behind the common model-provider contract."""
    capabilities = ModelCapabilities(
        context_window_tokens=config.max_input_tokens + config.max_output_tokens + 512,
        max_output_tokens=config.max_output_tokens,
    )
    common = {
        "capabilities": capabilities,
        "api_key_variable": config.api_key_variable,
        "timeout": 20.0,
    }
    if config.provider == "openai":
        return OpenAIProvider(config.model, **common)  # type: ignore[arg-type]
    if config.provider == "anthropic":
        return AnthropicProvider(config.model, **common)  # type: ignore[arg-type]
    if config.provider == "gemini":
        return GeminiProvider(config.model, **common)  # type: ignore[arg-type]
    presets = {"groq": GROQ, "grok": GROK, "openrouter": OPENROUTER}
    return OpenAICompatibleProvider(
        config.model,
        preset=presets[config.provider],
        **common,  # type: ignore[arg-type]
    )


async def run(config: Options) -> int:
    """Make at most one bounded model call and print only non-payload evidence."""
    if not config.live:
        print("offline default: no provider called; use the protected manual workflow")  # noqa: T201
        return 0
    if not config.model:
        raise ValueError("--model is required with --live")

    provider = provider_for(config)
    agent = Agent(
        name="reference-live-smoke",
        instructions="Reply with the single word READY.",
        model=config.model,
        free_text=True,
        budget=BudgetLimits(
            max_model_calls=1,
            max_input_tokens=config.max_input_tokens,
            max_output_tokens=config.max_output_tokens,
        ),
        deadlines=DeadlineConfig(run_seconds=25, model_call_seconds=20),
    )
    try:
        result = await AgentRunner(provider=provider).run(
            agent,
            "Return the readiness marker.",
            tenant="reference-validation",
            user="github-actions",
        )
    finally:
        await provider.aclose()
    print(  # noqa: T201
        f"provider={provider.name} state={result.state.value} "
        f"input_tokens={result.usage.input_tokens} output_tokens={result.usage.output_tokens}"
    )
    return 0 if result.state.value == "completed" else 1


def main(argv: list[str] | None = None) -> int:
    """Run the async smoke from a command line."""
    return asyncio.run(run(options(argv)))


if __name__ == "__main__":
    raise SystemExit(main())
