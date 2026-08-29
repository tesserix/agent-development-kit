# Tesserix Agent Development Kit

[![CI](https://github.com/tesserix/agent-development-kit/actions/workflows/ci.yml/badge.svg)](https://github.com/tesserix/agent-development-kit/actions/workflows/ci.yml)
[![Security](https://github.com/tesserix/agent-development-kit/actions/workflows/security.yml/badge.svg)](https://github.com/tesserix/agent-development-kit/actions/workflows/security.yml)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/)
[![Apache--2.0](https://img.shields.io/badge/license-Apache--2.0-green.svg)](LICENSE)

`tesserix-adk` is a typed Python toolkit for building, running, testing, and connecting
production AI agents. It keeps the agent declaration separate from model providers,
tools, storage, gateways, and transports, so those parts can change without rewriting the
agent.

The kit is deliberately a library, not a hosted control plane and not a process-owning
framework. An application imports the pieces it needs and keeps ownership of deployment,
networking, identity, and data.

## Why the Tesserix Agent Development Kit is different

- **Provider truth instead of provider guesses.** Every deployment declares tool calling,
  structured output, vision, streaming, and context limits. Unsupported work fails before
  a paid request is sent.
- **Policy in code, not prompt prose.** Tenant boundaries, tool allowlists, approvals,
  idempotency, budgets, deadlines, retries, guardrails, and output validation are runtime
  controls.
- **Portable by protocol.** OpenAI, Anthropic, Gemini, Groq, xAI/Grok, OpenRouter, vLLM,
  Ollama, TGI, llama.cpp, custom gateways, and any `ModelProvider` implementation enter the
  same runner.
- **Testable without a network.** Scripted providers, transport recordings, conformance
  suites, deterministic clocks, and evaluation gates are first-class package surfaces.
- **Interoperable without conflation.** MCP tools, Tesserix's typed peer protocol, and the
  official Agent2Agent (A2A) protocol are separate integrations with explicit trust
  boundaries.
- **Lean installation.** Vendor SDKs and infrastructure clients do not enter the base
  dependency graph. Integrations are optional extras.

## Five-minute start

Python 3.12 or newer is required. Development and release verification use CPython 3.14,
while CI keeps every declared minor from 3.12 through 3.14 compatible.
PyPI trusted publishing is not enabled yet, so install the exact wheel from the public
v0.53.1 release. Standard `pip` and `uv` install the same distribution.

With `pip`, create an isolated environment and import the underscore-named Python package:

```bash
python -m venv .venv
.venv/bin/python -m pip install "tesserix-adk @ https://github.com/tesserix/agent-development-kit/releases/download/v0.53.1/tesserix_adk-0.53.1-py3-none-any.whl"
.venv/bin/python -c "import tesserix_adk; print(tesserix_adk.__version__)"
```

With `uv`, add the same immutable wheel to an application project and commit the generated
lockfile:

```bash
uv add "tesserix-adk @ https://github.com/tesserix/agent-development-kit/releases/download/v0.53.1/tesserix_adk-0.53.1-py3-none-any.whl"
uv run python -c "import tesserix_adk; print(tesserix_adk.__version__)"
```

The distribution name is `tesserix-adk`; Python code imports `tesserix_adk`. To contribute
to the kit itself, use the source checkout:

```bash
git clone https://github.com/tesserix/agent-development-kit.git
cd agent-development-kit
uv sync --frozen
```

Create an agent, a typed tool, and one provider-backed runner:

```python
import asyncio

from tesserix_adk import Agent, AgentRunner, ToolRegistry, tool
from tesserix_adk.core import ModelCapabilities
from tesserix_adk.models.providers import OpenAIProvider


@tool(idempotency="read_only")
def current_weather(city: str) -> str:
    """Return the current weather for a city."""
    return f"{city} is 21°C and clear"


async def main() -> None:
    model = "gpt-4.1-mini"
    agent = Agent(
        name="weather-agent",
        instructions="Use current_weather, then answer in one sentence.",
        model=model,
        free_text=True,
        tools=("current_weather",),
        idempotent_tools=("current_weather",),
    )
    capabilities = ModelCapabilities(
        tool_calling=True,
        streaming=True,
        context_window_tokens=128_000,
    )

    async with OpenAIProvider(model, capabilities=capabilities) as provider:
        runner = AgentRunner(
            provider=provider,
            tools=ToolRegistry((current_weather,)),
        )
        run = await runner.run(agent, "What should I pack for Melbourne?", tenant="demo")
        print(run.text)


asyncio.run(main())
```

Set `OPENAI_API_KEY` before running it. Capability values are deployment facts: use the
limits and features of the exact model and endpoint you deploy, not the illustrative
values above.

To prove the complete tool loop with no key and no network from that checkout:

```bash
uv run python examples/getting_started.py
```

Continue with [Getting started](docs/getting-started.md) and [Build a custom
agent](docs/custom-agent.md). If code already exists in another framework, use the
[framework interoperability guide](docs/framework-interop.md) to choose import, wrap,
MCP, or official A2A without losing identity or task lifecycle.

`Agent[OutputT]` is the stable text-input contract. Applications that already have a
Pydantic request model use `TypedAgent[InputT, OutputT]` with `runner.run_typed(...)` or
`runner.stream_typed(...)`; both surfaces enter the same budgets, guardrails, tools,
identity, tracing and provider-neutral execution loop. See [Typing](docs/typing.md).

## Providers

The runtime depends on `ModelProvider`, not on a vendor SDK.

| Provider or deployment | Adapter | Authentication default |
|---|---|---|
| OpenAI | `OpenAIProvider` | `OPENAI_API_KEY` |
| Anthropic | `AnthropicProvider` | `ANTHROPIC_API_KEY` |
| Google Gemini API | `GeminiProvider` | `GEMINI_API_KEY` |
| Groq | `OpenAICompatibleProvider(..., preset=GROQ)` | `GROQ_API_KEY` |
| xAI / Grok | `OpenAICompatibleProvider(..., preset=XAI)` | `XAI_API_KEY` |
| OpenRouter | `OpenAICompatibleProvider(..., preset=OPENROUTER)` | `OPENROUTER_API_KEY` |
| vLLM, Ollama, TGI | `OpenAICompatibleProvider` with its preset | Operator-defined |
| llama.cpp | `LlamaCppProvider` | Operator-defined |
| OpenAI-compatible gateway | Custom `CompatibilityPreset` | Operator-defined |
| Any other API | Implement `ModelProvider` | Adapter-defined |

See [Provider recipes](docs/provider-recipes.md) for copyable configurations and the
important limitations. Azure OpenAI, Amazon Bedrock, Vertex AI, and other APIs that do
not expose a compatible wire contract need a dedicated adapter; changing a URL alone is
not treated as compatibility.

## Integrations and interoperability

Install only what the application uses. From a source checkout, select extras explicitly:

```bash
uv sync --frozen --extra a2a
uv sync --frozen --extra google-adk
uv sync --frozen --extra mcp
uv sync --frozen --extra redis --extra postgres --extra temporal
```

For an application dependency, select a tagged artifact and add the same extra names as
described in [Keep agents current safely](docs/keeping-current.md). Do not depend on the
moving `main` branch.

| Boundary | What the kit provides |
|---|---|
| Model gateway | Base URL, endpoint-path presets, protected auth headers, custom metadata headers, and injectable HTTP transport |
| MCP | Client/server surfaces, stdio and HTTP transports, scoped credentials, resilience, and AgentGateway routing |
| Official A2A 1.x | Official Agent Cards, clients, registries, custom gateway bindings, and an `AgentRunner` server bridge through `tesserix-adk[a2a]` |
| Google Agent Development Kit | Imports `FunctionTool`, wraps `BaseAgent`, and connects either runtime through official A2A via `tesserix-adk[google-adk]` |
| Tesserix peer protocol | Typed discovery, delegation, invocation, trust containment, and peer tools under `tesserix_adk.a2a` |
| State and durability | Redis, PostgreSQL, pgvector, Temporal-facing workflow primitives, NATS JetStream patterns, checkpoints, leases, outbox, and replay controls |

Official A2A support includes a bounded task executor, verified principal binding, final
artifacts, and cancellation. The application still mounts the official request handler
and routes, injects a tenant-scoped `TaskStore`, and owns authentication, persistence,
subscriptions, crash recovery, and push delivery. Agent Card security metadata describes
a contract; it does not enforce one. See [Official A2A interoperability](docs/a2a.md) and
the [Google Agent Development Kit bridge](docs/google-adk.md).
The [framework interoperability guide](docs/framework-interop.md) provides one decision
path for importing tools, wrapping agents, and exporting Tesserix agents to any runtime.

## Reliability model

The defaults are intentionally conservative:

- tools are unavailable until explicitly registered and allowlisted;
- tenant is required for every run;
- retries are opt-in and side effects require idempotency policy;
- missing model capabilities fail closed;
- provider errors are translated into one typed hierarchy;
- secrets are resolved at use time and protected headers cannot be overridden;
- untrusted tool, retrieval, MCP, and peer content remains data across the boundary;
- package API, event schemas, dependency decisions, replay safety, typing, coverage, and
  release compatibility are gated in CI.

Read [Architecture](docs/architecture.md), [Security](SECURITY.md), and the
[public-readiness review](docs/public-readiness-review.md) before a production rollout.
The [end-to-end agent lifecycle](docs/agent-lifecycle.md) shows how authoring, evaluations,
registry approval, canary execution, runtime controls, recovery and feedback fit together.

## Documentation

- [Published documentation](https://tesserix.github.io/agent-development-kit/)
- [Documentation home](docs/index.md)
- [Getting started](docs/getting-started.md)
- [Capability map](docs/capabilities.md)
- [Command-line guide](docs/cli.md)
- [Runnable cookbook](docs/cookbook/index.md)
- [Build a custom agent](docs/custom-agent.md)
- [End-to-end agent lifecycle](docs/agent-lifecycle.md)
- [Provider recipes](docs/provider-recipes.md)
- [Integrations and gateways](docs/integrations.md)
- [Framework interoperability](docs/framework-interop.md)
- [Official A2A interoperability](docs/a2a.md)
- [Google Agent Development Kit bridge](docs/google-adk.md)
- [Agent architecture escalation ladder](docs/escalation-ladder.md)
- [Testing](docs/testing.md)
- [Keep agents current safely](docs/keeping-current.md)
- [Repository governance](docs/repository-governance.md)
- [Code of Conduct](CODE_OF_CONDUCT.md)
- [Contributing](CONTRIBUTING.md)

The package is currently pre-1.0. Stability is declared per subpackage in
[Stability](docs/stability.md). The package installs the project-qualified
`tesserix-adk` command for self-contained operations and never claims the ambiguous
`adk` executable. `python -m tesserix_adk.cli ...` remains equivalent.

## Name and license

Google also publishes an Agent Development Kit. The distinct distribution name is
`tesserix-adk` and the import namespace is `tesserix_adk`. In prose, “Tesserix Agent
Development Kit” means this project and “Google Agent Development Kit” means Google's
independent project. They can interoperate through the
[official A2A bridge](docs/google-adk.md).

Licensed under [Apache License 2.0](LICENSE).
