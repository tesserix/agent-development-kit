# Runnable cookbook

Every recipe below is the source file CI executes. The documentation includes the file
rather than copying it, and `make recipe-coverage-check` maps every exported public name
to one of these reviewed starting points. A new export therefore fails until its recipe
mapping is regenerated and reviewed with the API-surface change.

Optional integrations state their extra in the heading. All recipes run without network
access or credentials; integration transports are replaced with deterministic fakes.

## Agent construction and structured output

```python
--8<-- "examples/getting_started.py"
```

## Typed core primitives

```python
--8<-- "examples/typed_primitives.py"
```

## Tool definition and validation

```python
--8<-- "examples/tools.py"
```

## Registry allowlists

```python
--8<-- "examples/tool_allowlist.py"
```

## Run, working, profile, episodic, and semantic memory

```python
--8<-- "examples/memory.py"
```

## Ordered guardrails with an injection attempt

```python
--8<-- "examples/guardrails.py"
```

## Budget ceilings

```python
--8<-- "examples/budget.py"
```

## MCP client (`mcp` extra for a live transport)

```python
--8<-- "examples/mcp_client.py"
```

## MCP server (`mcp` extra for a live transport)

```python
--8<-- "examples/mcp_server.py"
```

## MCP authentication context (`mcp` extra for a live transport)

```python
--8<-- "examples/mcp_auth_context.py"
```

## Peer invocation and delegated scope

```python
--8<-- "examples/peer_invocation.py"
```

## Retrieval, citations, and untrusted content

```python
--8<-- "examples/retrieval.py"
```

## Durable workflow (`temporal` extra for a live worker)

```python
--8<-- "examples/durable_run.py"
```

## Evaluation suite

```python
--8<-- "examples/eval_suite.py"
```

## Integration transports

```python
--8<-- "examples/transports.py"
```

## Local redacted trace

```python
--8<-- "examples/local_trace_view.py"
```

## Code intelligence with provenance

```python
--8<-- "examples/code_intelligence.py"
```

## Provider substitution

```python
--8<-- "examples/providers.py"
```

## Automatic telemetry

```python
--8<-- "examples/auto_instrumentation.py"
```

## Runtime loop

```python
--8<-- "examples/run_loop.py"
```

## Deterministic fake provider

```python
--8<-- "examples/fake_model_provider.py"
```
