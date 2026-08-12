# Claim-check tool results

A tool that returns a contract, a log file or a scraped page puts that content in the
conversation, and the conversation is re-sent on every iteration after it. The model read
it once; the prefill pays for it every turn. On CPU that is the difference between a run
that finishes and one that does not.

So an oversized result is *checked in*: the content goes to a store, and what enters the
conversation is an extractive head and a handle.

```python
store = MemoryClaimCheckStore(clock=clock)
runner = AgentRunner(
    provider=provider,
    tools=registry.view(allow=("read_contract", "fetch_result"), agent="counsel"),
    claim_check=ClaimCheck(store=store),
)
```

The `fetch_result` tool is what redeems a handle, and it has to be in the agent's tools
or the handles are dead ends:

```python
registry = ToolRegistry((read_contract, claim_check_tool(store)))
```

## What the model sees

```
Clause 1. This agreement is governed by...

[92104 characters, 91592 not shown. Call fetch_result with handle='claim:9f2c…' to read the rest.]
```

The head is cut at a boundary the content itself provides — a blank line, a newline, the
end of a sentence — rather than mid-word, because a head that reads as damage gets fetched
whether or not it was needed, which is the cost this exists to avoid. Most of the time the
head is the whole answer and the handle is never redeemed.

## Thresholds

| | Default | What it means |
|---|---|---|
| `threshold_chars` | 4096 | Below this nothing happens. A handle costs a tool call; a small result is cheaper to just read. |
| `head_chars` | 512 | How much stays in the conversation. |
| `ttl_seconds` | 3600 | How long the content can still be fetched. |

A head no smaller than the threshold is refused at construction: the substitution would be
as large as what it replaced.

Per-tool thresholds are for the tools that need them. A tool returning a whole PDF and one
returning a row count are not the same decision:

```python
ClaimCheck(store=store, per_tool={"read_contract": ClaimCheckPolicy(threshold_chars=1_024)})
```

## Scope

A handle is scoped to the tenant and the run that made it, and the scope is hashed into
the handle as well as checked on the lookup — a handle from another run is not merely
refused, it cannot be derived. Identical content within one run derives one handle, so a
tool called twice is stored once.

A fetch outside that scope, past the retention window, or for a handle nobody stored, all
answer the same way: `ClaimUnavailableError`, which the retrieval tool turns into a
`claim_unavailable` refusal. Distinguishing "gone" from "not yours" would tell a caller
which handles other runs hold, and the model can do nothing different with either.

## Reading it back

`fetch_result(handle, offset=0)` returns a window — `DEFAULT_FETCH_CHARS`, 4096 — not the
document. A fetch that returned the whole thing would put back into the conversation
exactly what checking it in took out, one tool call later. Long content is read by
advancing `offset`; an offset past the end reads empty, and a negative one is refused.

## What is never done

The content is not summarised, and the head is not paraphrased. What comes back is what
the tool returned. A model handed a plausible substitute for a document it asked to read
has no way to know it is reasoning about nothing, so the tool refuses rather than
approximates.

Checking in happens *after* the [result boundary](tool-results.md), never instead of it:
what is stored has already been validated against the tool's declared type and had
structural forgery neutralised. Storing what the boundary refused would be storing an
attack for later.

## Stores

`MemoryClaimCheckStore` holds handles in a dict and loses them on restart, which for
content scoped to a run in flight is the right shape and for a run that must survive a
failover is not. A deployment that needs the second binds its own `ClaimCheckStore` — the
protocol is `put`, `fetch`, `forget`, and `forget(tenant=…)` is where right-to-erasure
reaches this content.

The run loop records a `tool_result_stored` event naming the size and the handle, never
the content. Without a `claim_check` bound, an oversized result is cut at
`max_tool_result_chars` and the rest is gone, which is the behaviour that was there before.
