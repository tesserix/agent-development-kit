# Build a custom agent

This walkthrough builds a small support agent with one read-only tool, a validated answer,
and a replaceable provider.

## 1. Define the answer

Use a Pydantic model when the caller needs fields rather than prose:

```python
from pydantic import BaseModel


class SupportAnswer(BaseModel):
    """A support answer returned to the application."""

    summary: str
    ticket_id: str | None = None
```

The runtime validates the final model response into this type. A wrong shape is a typed
failure or a bounded repair attempt if repair is explicitly enabled.

## 2. Define typed tools

```python
from tesserix_adk import tool


@tool(idempotency="read_only", timeout=5.0)
def lookup_ticket(ticket_id: str) -> str:
    """Return the current status of one support ticket.

    Args:
        ticket_id: Public ticket identifier.
    """
    return f"{ticket_id}: waiting for customer"
```

The decorator derives JSON Schema from the signature. Invalid arguments are rejected
before the function runs. Read-only, idempotent, and effectful are different declarations;
an effectful tool also needs a durable idempotency store at runtime.

## 3. Declare what the agent may do

```python
from tesserix_adk import Agent
from tesserix_adk.core import BudgetLimits, DeadlineConfig, RetryConfig

MODEL = "your-model-id"

agent = Agent(
    name="support-agent",
    version="1.0.0",
    instructions=(
        "Use lookup_ticket when a ticket id is supplied. "
        "Return only facts present in the tool result."
    ),
    model=MODEL,
    output_type=SupportAnswer,
    tools=("lookup_ticket",),
    idempotent_tools=("lookup_ticket",),
    deadlines=DeadlineConfig(
        run_seconds=30,
        model_call_seconds=15,
        tool_call_seconds=5,
    ),
    retry=RetryConfig(max_attempts=2),
    budget=BudgetLimits(
        max_input_tokens=20_000,
        max_output_tokens=1_000,
    ),
)
```

`tools` is an allowlist, not discovery. Registering another function in the same
process does not grant this agent access to it.

## 4. Select a provider

Every runner sees the same `ModelProvider` protocol. Here OpenRouter is used only to
show that a hosted compatible API is not special to the agent:

```python
from tesserix_adk.core import ModelCapabilities
from tesserix_adk.models.providers import OPENROUTER, OpenAICompatibleProvider

provider = OpenAICompatibleProvider(
    MODEL,
    preset=OPENROUTER,
    capabilities=ModelCapabilities(
        tool_calling=True,
        structured_output=True,
        streaming=True,
        context_window_tokens=32_000,
        max_output_tokens=4_096,
    ),
    headers={
        "HTTP-Referer": "https://your-application.example",
        "X-Title": "Support Agent",
    },
)
```

The preset chooses the endpoint and `OPENROUTER_API_KEY`. The capability record must
describe the exact routed model. OpenRouter serving a different model does not make every
model behind it equally capable.

Use [Provider recipes](provider-recipes.md) to replace this construction with OpenAI,
Anthropic, Gemini, Groq, xAI/Grok, vLLM, Ollama, TGI, llama.cpp, or a custom gateway.

## 5. Register and run

```python
from tesserix_adk import AgentRunner, ToolRegistry

registry = ToolRegistry((lookup_ticket,))
runner = AgentRunner(provider=provider, tools=registry)

run = await runner.run(
    agent,
    "What is happening with ticket SUP-1042?",
    tenant="acme",
    user="user-42",
)

answer = run.output
assert isinstance(answer, SupportAnswer)
print(answer.summary)
await provider.aclose()
```

`run.text` is convenient for free-text agents. Typed agents use `run.output`.
`run.events`, `run.usage`, `run.state`, and the provider attribution are the evidence
for observability and evaluation.

Synchronous applications use `runner.run_sync(...)`. Do not call the sync wrapper from
inside an existing event loop; await `runner.run(...)` there.

## 6. Stream progress

```python
stream = runner.stream(
    agent,
    "What is happening with ticket SUP-1042?",
    tenant="acme",
    user="user-42",
)

async for event in stream:
    handle_progress(event)

finished = await stream
```

The stream is bounded and leaving its async context cancels work nobody is reading. Model
streaming must be declared by the selected provider.

## Optional: accept a structured application request

Keep `Agent[OutputT]` and `runner.run(..., text)` for text input. If the application
already has a Pydantic request, opt into the additive typed boundary:

```python
class SupportRequest(BaseModel):
    ticket_id: str
    question: str


from tesserix_adk import TypedAgent

typed_agent: TypedAgent[SupportRequest, SupportAnswer] = TypedAgent(
    name="support-agent",
    instructions="Look up the ticket and answer the question.",
    model=MODEL,
    input_type=SupportRequest,
    output_type=SupportAnswer,
    tools=("lookup_ticket",),
    idempotent_tools=("lookup_ticket",),
)
typed_run = await runner.run_typed(
    typed_agent,
    SupportRequest(ticket_id="SUP-1042", question="What happens next?"),
    tenant="acme",
    user="user-42",
)
```

The request is validated and rendered as canonical JSON before guardrails or a provider
sees it. The typed and text surfaces use the same tools, budgets, identity, tracing,
cancellation and output validation. See [Typing](typing.md) for sync, streaming,
definition, prompt and estimation variants.

## 7. Promote the declaration to a reviewed artifact

For an agent that will be registered, delegated to, or operated by another team, wrap it
in an `AgentDefinition`:

```python
from tesserix_adk.core import AgentDefinition, Owner

definition = AgentDefinition.declared(
    agent=agent,
    owner=Owner(
        team="Support Platform",
        contact="https://your-application.example/support",
        service="support-agent-api",
    ),
    evaluation_suite="evals/support-agent.jsonl",
    known_tools=registry.names,
)
```

The definition adds an owner, evaluation suite, and content-derived revision. Private
instructions and the on-call contact are deliberately excluded from official A2A cards.

## Production checklist

- Use a secret manager through `SecretProvider`; never use static metadata headers for
  credential values.
- Put model, gateway, tool, and run timeouts at the boundary they control.
- Share rate limiters and connection pools across providers using the same credential.
- Require approval for irreversible actions and a durable idempotency store for effects.
- Apply caller, tenant, and agent scope intersection before tool dispatch.
- Treat retrieval, MCP, tool, and peer responses as untrusted data.
- Record an evaluation baseline before changing model, prompt, provider, or tool surface.
- Exercise retry, rate-limit, timeout, malformed-response, cancellation, and duplicate
  delivery scenarios before rollout.
- Close provider and transport clients during application shutdown.

Continue with [Testing](testing.md), [Integrations and gateways](integrations.md), and
[Official A2A interoperability](a2a.md).
