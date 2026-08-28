# Providers and capabilities

One protocol, `ModelProvider`, sits between the kit and every vendor. A provider
translates its own wire format into `ModelRequest` and `ModelResponse` and declares, as
data, what its model can do. Nothing above the provider layer names a vendor type, and
nothing in the kit finds out a model's limits by exceeding them.

Import the surface from `tesserix_adk.models`:

```python
from tesserix_adk.models import (
    Capability,
    ModelCapabilities,
    ModelProvider,
    ModelRef,
    ModelRequest,
    ModelResponse,
    ModelSpec,
)
```

The types themselves are defined in `core` — a protocol typed over types that live above
it could not be implemented from below — and re-exported here, which is where a provider
author looks for them.

## What a provider implements

| Member | Contract |
|---|---|
| `name` | Stable identifier, recorded on the run and used in routing. |
| `capabilities` | A `ModelCapabilities` record. Read at wiring time and before each request. |
| `complete(request)` | One completion. Vendor errors are translated into the kit's hierarchy. |
| `stream(request)` | Incremental events. Refuses with `CapabilityError` unless `streaming` is declared. |
| `count_tokens(messages)` | How many tokens the messages occupy, by this provider's own count. |

`count_tokens` belongs to the provider because every vendor counts differently, and it is
what the declared context window is checked against. A provider that ships no tokeniser
can use `tesserix_adk.testing.estimate_tokens`, which counts characters.

## The capability record

```python
ModelCapabilities(
    structured_output=True,
    tool_calling=True,
    parallel_tool_calls=False,
    vision=False,
    streaming=True,
    context_window_tokens=200_000,
    max_output_tokens=8_192,
)
```

Every field defaults to off or unknown. Silence is not a claim: a capability nobody
declared is one the kit will not assume, because assuming it moves the failure to the
first request that needed it — in production, on someone's budget, mid-run.

`declared` is the set of capabilities that are on. `supports(requirement)` answers a
question; `require(requirement, provider=..., model=...)` raises a `CapabilityError`
naming all three, because "unsupported" on its own leaves the caller to work out which of
the three to change.

New capabilities arrive as fields with defaults, never as required arguments. That is a
1.0 promise: a provider written against an earlier version keeps compiling.

## What is checked, and when

| Check | When | Failure |
|---|---|---|
| Provider satisfies the protocol | `AgentRunner(...)` | `ProtocolConformanceError` |
| A tool registry against `tool_calling` | `AgentRunner(...)` | `CapabilityError` |
| An agent naming tools against `tool_calling` | Before the first request | `CapabilityError` |
| An image part against `vision` | Before each request | `CapabilityError` |
| Prompt length against `context_window_tokens` | Before each request | `ContextWindowExceededError` |
| Schema enforcement against `structured_output` | Before each request | Falls back to the prompt |
| A response that is not a `ModelResponse` | After each call | `ModelResponseError` |

Wiring failures are raised where the wiring is, which is the thing the caller can still
change. A window nobody declared is not a limit to check — the run proceeds.

The context-window check exists because a vendor handed an over-long prompt truncates it
and answers anyway, so the first sign of the problem is an answer that ignores the
beginning of the case. `ContextWindowExceededError` carries `counted` and `limit`.

`structured_output` is the one capability whose absence degrades rather than refuses: the
schema goes into the prompt instead and the answer is validated the same way. See
[`docs/structured-output.md`](structured-output.md).

`ModelResponseError` is distinct from `SchemaViolationError`. A well-formed answer in the
wrong shape is repairable, and goes to the repair flow. A payload that is not an answer
at all is a provider implementation fault: it carries the raw payload and the provider's
request id, and nothing is invented from it.

## Addressing a model

```python
ModelRef.parse("anthropic:claude-sonnet-5")   # provider:model
ModelSpec(provider="vllm", model="llama-3.3-70b").with_capabilities(vision=False)
```

The provider is part of a model's identity rather than a lookup, because a vendor API and
an OpenAI-compatible proxy serve the same model ids and are not the same model.
Defaulting the provider is how a proxy's traffic ends up recorded against the vendor, so
`ModelRef.parse` refuses a reference without one.

A self-hosted endpoint serves the weights it was given rather than the ones on the model
card. `ModelCapabilities.declaring(**overrides)` and `ModelSpec.with_capabilities(...)`
narrow a record from configuration, so a deployment never needs a subclass to say what it
actually runs.

## Proving a provider

A provider is substitutable only if it behaves the way the runtime assumes, and
structural typing cannot express "a declaration does not change between reads". Inherit
the suite:

```python
from tesserix_adk.testing import ModelProviderConformance


class TestVLLMProvider(ModelProviderConformance):
    def make_provider(self):
        return VLLMProvider(endpoint="http://localhost:8000")
```

Adding a member to the protocol means adding its case to the suite in the same change, so
every implementation learns about it by failing rather than by drifting.

For tests that need no provider at all, `ScriptedProvider` declares its capabilities
explicitly — `capabilities=CAPABLE.declaring(structured_output=True)` — so capability
gating can be proven without a network.

Runnable: [`examples/providers.py`](https://github.com/tesserix/agent-development-kit/blob/main/examples/providers.py).

## The vendor adapters

Three ship with the kit, under `tesserix_adk.models.providers`:

```python
from tesserix_adk.models.providers import AnthropicProvider, GeminiProvider, OpenAIProvider

provider = AnthropicProvider("claude-sonnet-4-5")   # key from ANTHROPIC_API_KEY
```

Each takes a model id as the vendor spells it, reads its capabilities and prices from the
[model catalogue](models.md), and resolves its key on every call rather than at
construction, so a rotated key is picked up without a restart. Nothing else is required:

| Provider | Endpoint | Key |
|---|---|---|
| `AnthropicProvider` | `/v1/messages` | `ANTHROPIC_API_KEY` |
| `OpenAIProvider` | `/v1/chat/completions` | `OPENAI_API_KEY` |
| `GeminiProvider` | `/v1beta/models/{model}:generateContent` | `GEMINI_API_KEY` |

Options are shared: `capabilities` overrides what the catalogue says, `secrets` injects a
`SecretProvider`, `api_key_variable` renames the variable, `base_url` reaches a proxy or a
self-hosted endpoint, `timeout` bounds the call, and `transport` replaces the HTTP
transport, which is what the recorded tests use.

They speak HTTP directly rather than through vendor SDKs. `httpx` is already a dependency,
so each adapter is one request shape and one response shape instead of a second dependency
graph and a second translation — and the traffic can be recorded at the HTTP layer, which
is where the interesting half of an adapter's behaviour lives.

### Where the three differ

The differences the adapters absorb, so nothing above them has to:

| | Anthropic | OpenAI | Gemini |
|---|---|---|---|
| System prompt | Top-level `system` | A `system` turn | `systemInstruction` |
| Structured output | Forced tool, unwrapped on the way back | `response_format`, `strict` only when the schema qualifies | `responseSchema`, pruned of keywords the vendor rejects |
| Tool results | Merged into one user turn of `tool_result` blocks | One `tool` turn each | `functionResponse`, matched back by tool name |
| Tool call ids | Sent | Sent | **Not sent** — minted by the adapter |
| Stop reason | Reported | Reported | `STOP` either way, so it is read off the parts |

An adapter that believed Gemini's `STOP` would return a finished turn for a model that had
asked for a tool, and the caller would never run it.

### Recording the traffic

`HttpCassette` records exchanges at the HTTP layer and `HttpReplay` serves them back
through an `httpx` transport, so the whole matrix runs in CI with no network and no keys:

```python
from tesserix_adk.testing import FakeSecrets, HttpCassette, HttpExchange, HttpReplay

replay = HttpReplay(HttpCassette(provider="openai", exchanges=(HttpExchange(...),)))
provider = OpenAIProvider(
    "gpt-4o", secrets=FakeSecrets({"OPENAI_API_KEY": "test"}), transport=replay.transport
)
```

`replay.sent` is the list of requests the adapter actually made. Asserting on it is how a
dropped system prompt or a mis-shaped tool result is caught — a provider-level recording
cannot see any of it, because by then the translation has already happened.

Runnable: [`examples/vendor_providers.py`](https://github.com/tesserix/agent-development-kit/blob/main/examples/vendor_providers.py).

## Hosted OpenAI-compatible APIs

Groq, xAI/Grok, and OpenRouter use the compatible adapter with reviewed endpoint and
credential presets:

```python
from tesserix_adk.core import ModelCapabilities
from tesserix_adk.models.providers import GROQ, OpenAICompatibleProvider

provider = OpenAICompatibleProvider(
    "your-groq-model",
    preset=GROQ,
    capabilities=ModelCapabilities(
        tool_calling=True,
        streaming=True,
        context_window_tokens=32_000,
    ),
)
```

| Preset | Completion URL | Credential |
|---|---|---|
| `GROQ` | `https://api.groq.com/openai/v1/chat/completions` | `GROQ_API_KEY` |
| `XAI` / `GROK` | `https://api.x.ai/v1/chat/completions` | `XAI_API_KEY` |
| `OPENROUTER` | `https://openrouter.ai/api/v1/chat/completions` | `OPENROUTER_API_KEY` |

Capabilities remain per deployed model. A router serving hundreds of models does not have
one capability record. Static routing/attribution headers are supported, while
`Authorization` and `Content-Type` remain owned by the adapter.

Copyable recipes, including custom gateway paths, are in
[Provider recipes](provider-recipes.md).

## Endpoints you run yourself

vLLM, Ollama and TGI serve the same wire format, so they go through the same adapter:

```python
from tesserix_adk.core import ModelCapabilities
from tesserix_adk.models.providers import VLLM, OpenAICompatibleProvider

provider = OpenAICompatibleProvider(
    "qwen2.5-7b-instruct",
    base_url="http://vllm.models.svc.cluster.local:8000",
    capabilities=ModelCapabilities(
        tool_calling=True, structured_output=True, streaming=True, context_window_tokens=32768
    ),
    preset=VLLM,
)
```

Two arguments have no default. `base_url`, because there is no host to guess for a service
only you have named — in-cluster service DNS is the expected form, and no egress is needed
for it. And `capabilities`, because the deployment's own flags decide them: whether tool
calling was enabled, what `--max-model-len` was set to. Nothing the endpoint reports says.

The provider is named for the server rather than for OpenAI, so a run against a box in the
cluster is not recorded against the vendor's bill. `name=` distinguishes two deployments.

`api_key_variable` is optional. Leave it out and no `Authorization` header is sent at all,
which is the in-cluster case; name a variable and it is read per call like any other key.

### What the presets absorb

| | vLLM | Ollama | TGI |
|---|---|---|---|
| `strict` on `response_format` | Not claimed | Not claimed | Not claimed |
| `stream_options.include_usage` | Sent | Not sent | Not sent |
| Tool-call ids | Sent by the server | **Minted** by the adapter | Sent by the server |
| Default timeout | 120s | 300s | 120s |

Three deviations are handled for every preset, because every one of them is a wrong answer
rather than an error if it is passed on:

- **An error under a 200.** Several compatible servers answer `{"error": ...}` with a 200
  and mean it. It raises `ProviderError`; no response is assembled from a failure.
- **A missing stop reason.** `unknown` on a turn that asked for a tool ends the run with
  the call never made, so the reason is read off what actually came back.
- **Missing usage.** Zero tokens reads as a free call, and a call on a GPU somebody is
  paying for is not free. The counts are estimated and `Usage.estimated` is `True`, so a
  ledger can tell a count from a guess. The cost stays `None` — the kit does not know what
  your GPU hour is worth, and zero would be a false statement.

### Nothing is emulated unless you ask

Where a model cannot enforce a schema itself, the kit can ask for JSON in the prompt and
validate the reply. Against a small self-hosted model that is a schema enforced by nobody,
so `OpenAICompatibleProvider` refuses instead: an agent with an `output_type` against an
endpoint that has not declared `structured_output` raises `CapabilityError` naming it,
before the run starts. Pass `emulates=True` to have the old behaviour anyway.

### When it is not there

A connection that never landed, and 502/503/504, raise `ProviderUnavailableError` — a
`ProviderError`, so existing handling still catches it, and always `retryable`. Any
`Retry-After` the endpoint sent is on the error and is believed in preference to a
computed backoff: retrying a model that is still loading its weights as fast as the policy
allows is how it never finishes loading.

What every other vendor failure arrives as, what it carries and what it deliberately does
not, and the timeouts and rate limiting in front of the call, are in
[resilience.md](resilience.md).

Runnable: [`examples/self_hosted_provider.py`](https://github.com/tesserix/agent-development-kit/blob/main/examples/self_hosted_provider.py).

Azure OpenAI, Amazon Bedrock, Vertex AI, and other APIs with different authentication,
paths, payloads, or streaming contracts need dedicated `ModelProvider` adapters.
Changing `base_url` alone does not make another wire protocol compatible.
