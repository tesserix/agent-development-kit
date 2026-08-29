# Getting started

This path proves a complete agent/tool run locally, then replaces the fake model with a
real provider. It needs Python 3.12 or newer and either standard `pip` or
[uv](https://docs.astral.sh/uv/). The repository uses `uv` and CPython 3.14 for its own
development while testing the full declared 3.12–3.14 range.

## 1. Install

PyPI trusted publishing is not enabled yet. Install the exact tagged v0.53.0 wheel from
the public GitHub Release rather than asking an index for a project that is not there.
The distribution name uses a hyphen; the Python import uses an underscore.

With `pip`:

```bash
python -m venv .venv
.venv/bin/python -m pip install "tesserix-adk @ https://github.com/tesserix/agent-development-kit/releases/download/v0.53.0/tesserix_adk-0.53.0-py3-none-any.whl"
.venv/bin/python -c "import tesserix_adk; print(tesserix_adk.__version__)"
```

With `uv`, from an application project:

```bash
uv add "tesserix-adk @ https://github.com/tesserix/agent-development-kit/releases/download/v0.53.0/tesserix_adk-0.53.0-py3-none-any.whl"
uv run python -c "import tesserix_adk; print(tesserix_adk.__version__)"
```

Optional integrations use the same extras with either installer. For example,
`tesserix-adk[a2a,google-adk,mcp]` installs the official A2A, Google Agent Development
Kit bridge, and MCP dependencies. The release workflow clean-installs every individual
extra and `all` with `pip` before publishing succeeds.

Use the public source checkout only when contributing to this repository:

```bash
git clone https://github.com/tesserix/agent-development-kit.git
cd agent-development-kit
uv sync --frozen
```

## 2. Prove the offline path

The getting-started example scripts the model's tool call and structured final answer. It
performs the same declaration, registration, validation, budget enforcement, dispatch,
and streamed run loop as a live model, but it opens no connection and reads no credential.

```python
--8<-- "examples/getting_started.py"
```

The example contains 24 executable application statements; CI measures that ceiling so
the first path cannot quietly grow into a framework of its own.

```bash
uv run python examples/getting_started.py
```

Expected shape:

```text
trace: 0 run_started
...
trace: <n> tool_call_started
trace: <n> tool_call_finished
...
trace: <n> run_completed
... suggestion='Pack a light jacket.' ...
```

The `PackingTip` Pydantic model is the application contract. Invalid provider prose or a
wrong field fails as a typed schema error; the runtime does not parse, coerce, or invent a
replacement. `FakeModelProvider` is the only test-specific piece.

### What the short example got for free

- The run records the agent, provider, tenant, acting user, usage, and every typed event.
- `BudgetLimits` caps model and tool calls before work is dispatched.
- Tool arguments are validated and terminal rendering travels through the same redacted
  event surface used by telemetry.
- Provider, budget, guardrail, and schema failures are typed and fail closed; none becomes
  a plausible assistant answer.

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
If the setting is absent, the typed `ConfigurationError` names `OPENAI_API_KEY` and points
back to the credential-free `FakeModelProvider` path before any HTTP request is attempted.

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

The detailed walkthrough is [Build a custom agent](custom-agent.md). Continue with the
[cookbook](cookbook/index.md) for the agent, tool, budget, structured-output, and testing
primitives introduced here. The repository's
readiness findings and remaining protocol gaps are in the
[public-readiness review](public-readiness-review.md).

## Optional: generate the first typed files

Inside an existing Python project with `pyproject.toml`, list the templates and generate
one agent without changing the project's layout or dependency manager:

```bash
tesserix-adk new --list
tesserix-adk new agent support-agent --template tool-using
uv run pytest -q test_support_agent.py
uv run mypy --strict support_agent.py support_tools.py test_support_agent.py
```

Every agent template uses `TypedAgent[Input, Output]`, a bounded budget, resolved typed
configuration, run-rooted instrumentation, and an offline fake-provider test:

| Template | Added composition |
|---|---|
| `single` | One structured agent plus a separate schema-derived, read-only local tool |
| `tool-using` | The same safe baseline plus an explicit reusable `ToolRegistry` factory |
| `multi-agent` | A typed specialist plus an explicit `Roster` boundary for a supervisor |
| `mcp-client` | A local fallback tool and transport-neutral `McpClient` factory |

`tesserix-adk new tool NAME` generates a typed standalone tool and contract test. The
command validates every target before writing anything; an existing path aborts the whole
operation unless `--force` is explicit, and a failed replacement restores the prior bytes.
The generated dependency hint pins the installed kit version so a template never mixes API
eras with the runtime that created it.

## CLI note

The package installs the project-qualified `tesserix-adk` command for self-contained
inspection and evaluation operations and never claims the ambiguous `adk` executable.
Use the Python API for application startup; `python -m tesserix_adk.cli ...` is equivalent.
Other helpers accept application-supplied storage/build callbacks and remain embedding
product surfaces rather than dispatcher commands.
