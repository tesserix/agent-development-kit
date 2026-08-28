# Getting started

This path proves a complete agent/tool run locally, then replaces the fake model with a
real provider. It needs Python 3.12 or newer and [uv](https://docs.astral.sh/uv/).

## 1. Install

PyPI trusted publishing is not enabled yet. Use the public source checkout to learn and
test the current API; use an exact tagged artifact for an application dependency as
described in [Keep agents current safely](keeping-current.md).

```bash
git clone https://github.com/tesserix/agent-development-kit.git
cd agent-development-kit
uv sync --frozen
```

Optional integrations are extras and do not enter the base environment:

```bash
uv sync --frozen --extra a2a
uv sync --frozen --extra mcp --extra postgres --extra redis --extra temporal
```

## 2. Prove the offline path

The getting-started example scripts the model's tool call and final answer. It performs
the same declaration, registration, validation, dispatch, and run loop as a live model,
but it opens no connection and reads no credential.

```bash
uv run python examples/getting_started.py
```

Expected shape:

```text
tool: Melbourne is 21°C and clear
tesserix-adk <version>
agent: weather-agent
answer: Pack a light jacket.
```

Open [the example](https://github.com/tesserix/agent-development-kit/blob/main/examples/getting_started.py) and notice the four public imports:
`Agent`, `AgentRunner`, `ToolRegistry`, and `tool`. A `FakeModelProvider` is the
only test-specific piece.

## 3. Connect a live model

Choose the adapter that matches the wire protocol:

```python
from tesserix_adk.core import ModelCapabilities
from tesserix_adk.models.providers import OpenAIProvider

model = "gpt-4.1-mini"
provider = OpenAIProvider(
    model,
    capabilities=ModelCapabilities(
        tool_calling=True,
        streaming=True,
        context_window_tokens=128_000,
    ),
)
```

Set `OPENAI_API_KEY` in the process environment or inject a `SecretProvider`. Do not
put keys in `adk.toml`, source code, Agent metadata, gateway headers, or Agent Cards.

Replace only the fake:

```python
runner = AgentRunner(
    provider=provider,
    tools=ToolRegistry((current_weather,)),
)
run = await runner.run(
    agent,
    "What should I pack for Melbourne?",
    tenant="demo",
)
print(run.text)
await provider.aclose()
```

Prefer `async with provider:` when the provider lifecycle belongs to one block. Share a
provider or a configured client pool when one application serves many runs.

For Groq, xAI/Grok, OpenRouter, self-hosted models, and gateways, use the same runner with
the configurations in [Provider recipes](provider-recipes.md).

## 4. Understand the two required declarations

An agent must choose exactly one model strategy:

- `model="model-id"` calls the runner's direct provider; or
- `task_class="planning"` lets a configured router select a provider/model.

It must also choose exactly one answer strategy:

- `free_text=True` for prose; or
- `output_type=YourPydanticModel` for a validated typed result.

These conditions fail at construction, not after the first provider call.

## 5. Keep the first production deployment small

Start with one agent and the fewest tools that solve the task. Before production:

- use a real tenant identifier and acting principal;
- set model and run deadlines;
- enable retries only for failures worth paying for again;
- declare every tool's idempotency and approval policy;
- add a shared idempotency store before effectful tool use;
- declare exact model capabilities and context/output limits;
- test provider failures and malformed tool calls offline;
- export redacted traces and measure tokens, cost, and latency;
- run the isolation and evaluation suites for the agent.

The detailed walkthrough is [Build a custom agent](custom-agent.md). The repository's
readiness findings and remaining protocol gaps are in the
[public-readiness review](public-readiness-review.md).

## CLI note

The package currently exposes focused Python modules for inspection and evaluation; it
does not install a global `adk` executable. Use the Python API for application startup
and only explicitly runnable entry points such as
`python -m tesserix_adk.cli evals`. Other CLI helpers accept application-supplied
storage/build callbacks and are meant to be wrapped by the embedding product.
