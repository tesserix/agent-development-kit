# Tesserix ADK documentation

Tesserix ADK is the reusable Python runtime and integration layer for agents that need
typed contracts, provider portability, production controls, and network-free tests. The
application owns its process and deployment; the kit supplies composable primitives.

## Choose a path

| Goal | Start here |
|---|---|
| Run the offline example | [Getting started](getting-started.md) |
| Build a new custom agent | [Build a custom agent](custom-agent.md) |
| Connect OpenAI, Anthropic, Gemini, Groq, Grok, OpenRouter, or a local model | [Provider recipes](provider-recipes.md) |
| Connect a gateway, MCP server, store, or workflow engine | [Integrations and gateways](integrations.md) |
| Publish or consume an official A2A Agent Card | [Official A2A interoperability](a2a.md) |
| Test an agent without spending money | [Testing](testing.md) |
| Keep a deployed agent on a current reviewed ADK | [Keep agents current safely](keeping-current.md) |
| Understand contribution and branch protections | [Repository governance](repository-governance.md) |
| Decide whether the project is ready for a public or production rollout | [Public-readiness review](public-readiness-review.md) |
| Contribute a change | [Contributing](https://github.com/tesserix/agent-development-kit/blob/main/CONTRIBUTING.md) |

## Mental model

An application composes four independent pieces:

1. An `Agent` declares the job, model or task class, tool allowlist, answer shape, and
   policy.
2. A `ModelProvider` translates one model API into the kit's request, response, usage,
   stream, capability, and error types.
3. A `ToolRegistry` holds typed functions and produces a fixed per-agent view.
4. `AgentRunner` applies boundaries and drives the model/tool loop to a terminal
   `Run`.

That separation is the portability mechanism. A provider, model gateway, store, MCP
server, or A2A registry can be replaced at its boundary without giving the agent vendor
objects.

## Main references

- [Architecture](architecture.md)
- [Core primitives](primitives.md)
- [Providers and capabilities](providers.md)
- [Tools](tools.md)
- [Run loop](run-loop.md)
- [Tenancy](tenancy.md)
- [Security and threat model](threat-model.md)
- [Budgets](budget.md)
- [Guardrails](guardrails.md)
- [Memory](memory.md)
- [Retrieval](retrieval.md)
- [MCP client](mcp-client.md)
- [AgentGateway](agentgateway.md)
- [Tesserix peer discovery](peer-discovery.md)
- [Durable runs](durable-runs.md)
- [Evaluations](eval-datasets.md)
- [Observability](auto-instrumentation.md)
- [Stability](stability.md)
- [Release verification](verifying.md)
- [Keep agents current safely](keeping-current.md)
- [Repository governance](repository-governance.md)

## What the name means

Google also has an Agent Development Kit. This project's distribution is
`tesserix-adk`, its import namespace is `tesserix_adk`, and its interoperability
adapter targets the independent Agent2Agent protocol through the official `a2a-sdk`.

The project is pre-1.0. Consult [Stability](stability.md) before adopting an alpha
subpackage in a long-lived integration.
