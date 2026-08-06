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

Runnable: [`examples/providers.py`](../examples/providers.py).
