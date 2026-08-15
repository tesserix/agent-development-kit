# Reversible compression and the retrieval handle

Every lossy compression scheme is a bet that the model will not need the part that was
removed. The bet is usually right and occasionally catastrophically wrong: the one elided
log line was the one naming the failing host, and the agent now answers confidently from a
hole in its context without knowing the hole is there.

The fix is not better compression. It is making compression reversible. `ReversibleRouter`
wraps a [`ContentRouter`](content-compression.md), retains the original, and appends a
handle to the compressed form that the model can redeem through a tool. An irreversible
quality risk becomes a recoverable extra turn, which is what makes an aggressive ratio
defensible at all.

## Admitting

```python
from tesserix_adk.memory import ContentRouter, ReversibleRouter
from tesserix_adk.runtime import MemoryClaimCheckStore

router = ReversibleRouter(ContentRouter(), MemoryClaimCheckStore(), audit=sink)

admitted = await router.admit(
    tool_output, budget_tokens=4_000, tenant="acme", run_id=run.id, untrusted=True
)
```

`admitted.content` is the compressed form followed by a note naming the size of the
original and the tool that reads it:

```
[compressed from 8848 characters. Call expand_content with handle='claim:…' for the original.]
```

Content the router passed through is left entirely alone — there is nothing to reverse, so
no handle is issued and nothing is stored.

## Redeeming

```python
from tesserix_adk.tools import expand_content_tool

registry.register(expand_content_tool(router, budget_tokens=2_000))
```

The tool resolves the handle within the caller's `ToolContext`, so the scope is the run's
rather than the model's to claim. Directly:

```python
expanded = await router.expand(handle, tenant="acme", run_id=run.id, budget_tokens=2_000)
expanded.content, expanded.chars, expanded.truncated
```

Retrieval is itself subject to admission: an original larger than the budget left comes
back as the leading window that fits, marked `truncated`, rather than putting back into the
prompt exactly what compression took out.

## What a handle cannot do

The handle is the claim check the kit already issues for oversized tool results, so the
scope rules come with it. The tenant and the run are hashed into the handle *and* checked
at the lookup, so a handle cannot be transplanted, derived for another run, or used to read
across an isolation boundary.

`ClaimUnavailableError` is raised for every failure — unknown handle, another tenant,
another run, past its retention window, or a string the model invented. They are
deliberately indistinguishable to the caller: telling a model which condition failed tells
it what handles other runs hold, and it would do nothing different either way.

Every expansion is recorded through the `AuditSink`: the handle, who asked, and how many
characters came back. Refused expansions are recorded too. The content itself never reaches
the record.

## Known limitations

* Retention is per run. Nothing is shared across runs or processes; a handle from yesterday
  is gone, and that is the intended lifetime for content nobody validated.
* Eviction is by age. The store expires handles rather than leaving them dangling, but a
  deployment holding very large originals should bound its own store.
* Originals inherit the erasure obligations of the content they came from: erasing a run's
  content means calling `forget` on the store for that run.
* The windowing in `expand` cuts on characters, so a truncated expansion may end
  mid-token. It is a window, not a summary.
