# Testing agents

Agent behavior should be testable without provider availability, credentials, cost, or
timing variance. Tesserix ADK ships strict fakes, recorded HTTP transports, behavioral
conformance suites, tool spies, isolation suites, and evaluation gates.

## Run this repository's gates

```bash
make test       # pytest
make lint       # Ruff, formatting, import boundaries
make typecheck  # mypy --strict
make check      # complete CI-equivalent quality gate
make audit      # locked dependencies against advisory policy
make secrets    # credential-shape scan
make licences   # dependency licence policy
make docs-check # strict documentation build
uv build        # source and wheel artefacts
```

`make check` is intentionally substantial. Use targeted tests while iterating, then the
complete gate before review.

## Script the model

```python
from tesserix_adk import Agent, AgentRunner, ToolRegistry, tool
from tesserix_adk.testing import FakeModelProvider, ScriptedTurn


@tool(idempotency="read_only")
def lookup(city: str) -> str:
    """Look up one city."""
    return f"{city}: clear"


async def test_the_agent_uses_the_lookup() -> None:
    provider = FakeModelProvider(
        ScriptedTurn.calling("lookup", {"city": "Melbourne"}),
        ScriptedTurn.saying("It is clear."),
    )
    runner = AgentRunner(
        provider=provider,
        tools=ToolRegistry((lookup,)),
    )
    agent = Agent(
        name="weather",
        instructions="Use lookup.",
        model="test-model",
        free_text=True,
        tools=("lookup",),
        idempotent_tools=("lookup",),
    )

    run = await runner.run(agent, "Weather?", tenant="test")

    assert run.text == "It is clear."
    assert provider.calls == 2
    assert provider.remaining == 0
```

The fake is strict by default: an unscripted call fails rather than silently returning a
plausible answer. Scripted turns can answer, request a tool, or inject typed provider
faults and malformed responses.

## Enable the pytest plugin

In a consuming application's `conftest.py`:

```python
pytest_plugins = ["tesserix_adk.testing.pytest_plugin"]
```

The plugin:

- blocks outbound TCP and DNS by default;
- supplies `fake_model` and `fake_model_factory`;
- supplies replay-first cassette fixtures;
- requires owner, reason, and expiry on quarantined tests.

A real integration test must opt into network access explicitly. Keep it in a separate CI
lane so the unit suite never becomes dependent on an external provider.

## Record provider traffic at the HTTP boundary

Native and compatible providers accept an injected `httpx` transport. `HttpCassette`
and `HttpReplay` exercise the real request/response translation without opening a
socket:

```python
from pathlib import Path

from tesserix_adk.models.providers import OpenAIProvider
from tesserix_adk.testing import FakeSecrets, HttpCassette, HttpReplay

replay = HttpReplay(HttpCassette.load(Path("tests/cassettes/openai-weather.json")))
provider = OpenAIProvider(
    "gpt-test",
    secrets=FakeSecrets({"OPENAI_API_KEY": "test-value"}),
    transport=replay.transport,
)
```

Assert on `replay.sent` to catch path, header, system-prompt, tool-result, schema, and
streaming regressions. Recorded credentials are refused, not redacted after writing.
Replay is the default mode, so a test cannot start spending because an environment flag
was omitted.

## Prove a custom provider

A provider is substitutable only if its behavior matches runtime assumptions. Inherit
`ModelProviderConformance` in the adapter's test suite:

```python
from tesserix_adk.testing import ModelProviderConformance


class TestCompanyProvider(ModelProviderConformance):
    def make_provider(self):
        return company_provider_with_recorded_transport()
```

Conformance suites also exist for clocks, budgets, tracers, guardrails, memory, state,
checkpoints, leases, work queues, idempotency stores, spend ledgers, search indexes, and
tenant propagation.

## Assert tools and context

`ToolSpy` records the validated arguments, tenant, user, idempotency key, result or
error, and elapsed time for each invocation. Assertions include:

- `assert_tool_sequence`;
- `assert_tool_called_once_with`;
- `assert_no_tool_called`;
- `assert_idempotency_key_stable`;
- `assert_context_propagated`.

Use `scoped_run` to bind a real tenant/caller context rather than constructing a partial
mock. Approval stubs, failing tools, slow tools, and concurrency probes cover the common
failure paths.

See [Tool assertions](tool-assertions.md).

## Test tenancy and untrusted boundaries

At minimum, use confusable markers for two tenants and prove that:

- memory, state, cache, work, checkpoints, cards, tasks, and artifacts cannot cross;
- a payload tenant cannot override the authenticated tenant;
- tool, retrieval, MCP, and peer content cannot create instructions or authority;
- traces and errors do not export payload content or credentials;
- cancellation, retry, fallback, and replay preserve tenant and caller identity.

Use the [Isolation suite](isolation-suite.md), [Guard testing](guard-testing.md), and
[Prompt injection](prompt-injection.md) fixtures.

## Evaluation before rollout

Unit tests prove mechanics. Evaluation datasets prove task behavior:

1. Store versioned `EvalCase` rows in an `EvalSuite`.
2. Run through `SuiteRunner` with deterministic case/run identities.
3. Measure correctness, schema validity, tool sequence, grounding, refusals, tokens, cost,
   and latency.
4. Record a baseline with dataset, agent, prompt, model, and cassette provenance.
5. Gate changes against named thresholds and per-case regressions.
6. Calibrate any LLM judge against human labels before it can block a change.

Read [Evaluation datasets](eval-datasets.md), [Metrics](eval-metrics.md), and
[Baseline gate](eval-baseline.md).

## Release smoke tests

A public release should be verified from built artefacts, not from the source tree:

- install the wheel alone and import the root convenience surface;
- run `examples/getting_started.py` from the installed wheel;
- install each extra alone and import its integration;
- serialize an official A2A card with the official SDK;
- construct Groq, xAI/Grok, and OpenRouter providers with fake secrets/transports;
- verify the sdist contains licence, README, typing marker, and documentation metadata;
- build the documentation with `--strict`.

See [Verifying a release](verifying.md).
