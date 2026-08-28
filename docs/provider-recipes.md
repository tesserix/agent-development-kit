# Provider recipes

Every provider below implements the same `ModelProvider` protocol. The agent runner sees
the provider name, capabilities, completions, streams, token count, and normalized errors;
it does not receive a vendor client.

## Declare the deployed model

Capability values belong to the exact model and endpoint, not to the company serving it:

```python
from tesserix_adk.core import ModelCapabilities

CAPABILITIES = ModelCapabilities(
    tool_calling=True,
    structured_output=True,
    parallel_tool_calls=False,
    vision=False,
    streaming=True,
    context_window_tokens=32_000,
    max_output_tokens=4_096,
)
```

Change every value to match the deployment. Defaults are off or unknown. A false
`structured_output=True` is worse than a refused request because it tells the runtime
that a schema is enforced when it is not.

The native adapters can use the built-in model catalogue when a model is known, but an
explicit record is recommended for pinned production deployments. Compatible and
self-hosted endpoints require one.

## Native APIs

These adapters translate their vendors' actual wire formats:

```python
from tesserix_adk.models.providers import (
    AnthropicProvider,
    GeminiProvider,
    OpenAIProvider,
)

openai = OpenAIProvider("your-openai-model", capabilities=CAPABILITIES)
anthropic = AnthropicProvider("your-anthropic-model", capabilities=CAPABILITIES)
gemini = GeminiProvider("your-gemini-model", capabilities=CAPABILITIES)
```

| Adapter | Base URL | Credential variable | Wire API |
|---|---|---|---|
| `OpenAIProvider` | `https://api.openai.com` | `OPENAI_API_KEY` | Chat Completions |
| `AnthropicProvider` | `https://api.anthropic.com` | `ANTHROPIC_API_KEY` | Messages |
| `GeminiProvider` | `https://generativelanguage.googleapis.com` | `GEMINI_API_KEY` | generateContent |

All accept `base_url`, `api_key_variable`, an injected `SecretProvider`, timeouts,
an HTTP transport, limiter, and shared client pool. A custom base URL is suitable only
when the target still speaks that adapter's wire contract.

## Groq

Groq exposes an OpenAI-compatible endpoint under `/openai/v1`:

```python
from tesserix_adk.models.providers import GROQ, OpenAICompatibleProvider

provider = OpenAICompatibleProvider(
    "your-groq-model",
    preset=GROQ,
    capabilities=CAPABILITIES,
)
```

The preset uses `https://api.groq.com/openai/v1/chat/completions` and
`GROQ_API_KEY`.

## xAI / Grok

Grok is the model family; xAI is the provider recorded on runs:

```python
from tesserix_adk.models.providers import GROK, OpenAICompatibleProvider

provider = OpenAICompatibleProvider(
    "your-grok-model",
    preset=GROK,
    capabilities=CAPABILITIES,
)
```

`GROK` is an alias for the `XAI` preset. It uses
`https://api.x.ai/v1/chat/completions` and `XAI_API_KEY`.

## OpenRouter

OpenRouter routes to many model families. The capability declaration must describe the
selected model, not the router as a whole:

```python
from tesserix_adk.models.providers import OPENROUTER, OpenAICompatibleProvider

provider = OpenAICompatibleProvider(
    "provider/model-name",
    preset=OPENROUTER,
    capabilities=CAPABILITIES,
    headers={
        "HTTP-Referer": "https://your-application.example",
        "X-Title": "Your Application",
    },
)
```

The preset uses `https://openrouter.ai/api/v1/chat/completions` and
`OPENROUTER_API_KEY`. Attribution headers are optional. `Authorization` and
`Content-Type` cannot be overridden through `headers`; the adapter owns them.

## vLLM

```python
from tesserix_adk.models.providers import VLLM, OpenAICompatibleProvider

provider = OpenAICompatibleProvider(
    "your-served-model",
    base_url="http://vllm.models.svc.cluster.local:8000",
    preset=VLLM,
    capabilities=CAPABILITIES,
    api_key_variable="",  # explicitly unauthenticated inside the trusted network
)
```

Declare the context length and features from the vLLM launch flags. If a gateway protects
the service, name the gateway's key variable instead of disabling authentication.

## Ollama

```python
from tesserix_adk.models.providers import OLLAMA, OpenAICompatibleProvider

provider = OpenAICompatibleProvider(
    "your-ollama-model",
    base_url="http://127.0.0.1:11434",
    preset=OLLAMA,
    capabilities=CAPABILITIES,
    api_key_variable="",
)
```

The preset omits unsupported stream-usage options, mints deterministic tool-call IDs when
the server omits them, and allows a longer cold-model timeout.

## Text Generation Inference

```python
from tesserix_adk.models.providers import TGI, OpenAICompatibleProvider

provider = OpenAICompatibleProvider(
    "your-tgi-model",
    base_url="http://tgi.models.svc.cluster.local:8080",
    preset=TGI,
    capabilities=CAPABILITIES,
    api_key_variable="",
)
```

Use this path only for a TGI deployment that enables its OpenAI-compatible Chat
Completions surface.

## llama.cpp

```python
from tesserix_adk.models.providers import LlamaCppProvider, LlamaCppTuning

provider = LlamaCppProvider(
    "your-gguf-model",
    base_url="http://127.0.0.1:8080",
    capabilities=CAPABILITIES,
    tuning=LlamaCppTuning(
        threads=8,
        batch_size=512,
        context_tokens=16_384,
        prompt_cache=True,
    ),
)
```

The dedicated adapter enables prompt caching and can refuse a GGUF model before launch
when its weights and available memory are supplied. See [CPU inference](cpu-inference.md).

## A generic OpenAI-compatible gateway

A gateway often adds its own path, provider name, metadata headers, and key variable:

```python
from tesserix_adk.models.providers import (
    CompatibilityPreset,
    OpenAICompatibleProvider,
)

COMPANY_GATEWAY = CompatibilityPreset(
    name="company-gateway",
    completions_path="/models/v1/chat/completions",
    stream_usage_option=True,
    timeout=60.0,
)

provider = OpenAICompatibleProvider(
    "deployed-model",
    base_url="https://models.example.com",
    preset=COMPANY_GATEWAY,
    capabilities=CAPABILITIES,
    api_key_variable="COMPANY_GATEWAY_API_KEY",
    headers={"x-application": "support-agent"},
)
```

Use `completions_path` for a gateway prefix because an absolute request path replaces a
path embedded in `base_url`. Static headers are for routing and attribution, never for
secret values. If authentication is not Bearer, implement an adapter or inject a transport
that owns the authenticated request.

## One application, selectable providers

Keep provider selection at application wiring and return the protocol:

```python
from tesserix_adk.models import ModelProvider
from tesserix_adk.models.providers import (
    GROQ,
    OPENROUTER,
    XAI,
    AnthropicProvider,
    GeminiProvider,
    OpenAICompatibleProvider,
    OpenAIProvider,
)

HOSTED_COMPATIBLE = {
    "groq": GROQ,
    "xai": XAI,
    "grok": XAI,
    "openrouter": OPENROUTER,
}


def provider_for(name: str, model: str) -> ModelProvider:
    if name == "openai":
        return OpenAIProvider(model, capabilities=CAPABILITIES)
    if name == "anthropic":
        return AnthropicProvider(model, capabilities=CAPABILITIES)
    if name == "gemini":
        return GeminiProvider(model, capabilities=CAPABILITIES)
    if name in HOSTED_COMPATIBLE:
        return OpenAICompatibleProvider(
            model,
            preset=HOSTED_COMPATIBLE[name],
            capabilities=CAPABILITIES,
        )
    raise ValueError(f"unsupported provider: {name}")
```

In a real application, load each model's own capability record rather than sharing the
illustrative `CAPABILITIES` object. The `AgentRunner` construction and tool registry do
not change.

For policy-based selection and fallback across several providers, use task classes and
the routing table described in [Routing](routing.md). A fallback is allowed only when the
trust boundary and side-effect safety permit it.

## Provider lifecycle and failure behavior

- Close owned HTTP clients with `await provider.aclose()` or `async with provider`.
- Share `ClientPool` and `RateLimiter` objects for deployments using the same endpoint
  and credential.
- A credential is read at request time, so rotation can land without a process restart.
- Timeouts, rate limits, authentication failures, invalid requests, content filtering,
  unavailable services, malformed model responses, and interrupted streams arrive as
  typed kit errors.
- Compatible endpoints returning an error inside HTTP 200 are refused.
- Missing usage is estimated and marked as estimated; self-hosted cost remains unknown,
  not zero.
- Missing tool-call IDs and stop reasons are normalized only where the preset declares
  that deviation.

See [Resilience](resilience.md), [Connection pooling](connection-pooling.md), and
[Fallback](fallback.md).

## APIs that need dedicated adapters

URL replacement is not enough for APIs with another request, authentication, or streaming
contract. First-party adapters are still needed for:

- Azure OpenAI deployment paths, API versions, and `api-key` or Entra authentication;
- Amazon Bedrock's signed requests and provider-specific payloads;
- Vertex AI's Google Cloud authentication and regional publisher paths;
- provider-native Cohere, Mistral, or other non-compatible APIs;
- gateways that expose only Responses, gRPC, WebSocket, or another proprietary contract.

Any of these can integrate today by implementing `ModelProvider` and inheriting
`ModelProviderConformance` in its test suite. That extension seam is stable; claiming
the adapter already exists would not be.
