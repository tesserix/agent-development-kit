# Tool errors

A tool that failed and a tool that declined are different answers, and the run loop has to
be able to tell them apart.

Without the distinction every tool problem arrives as a generic exception, and the runtime
cannot separate "the supplier API was briefly unavailable, trying again is sensible" from
"this booking is not cancellable, stop asking". What follows is a run retrying a refusal
until the iteration cap fires — spending the budget to be told the same thing, and in the
worst case re-attempting an action the downstream had already declined.

## The taxonomy

| | |
|---|---|
| `ToolFailure(tool, code, transient=…)` | The tool tried and could not finish. `transient=True` is the author saying nothing landed and the same call could succeed. |
| `ToolRefusal(tool, code, message)` | The tool worked and declined. An answer, not a fault. |
| `ToolTimedOutError` | The call outran its declared ceiling. |
| `ToolNotPermittedError` / `ToolNotFoundError` | A permission decision, and a deployment mistake. |
| `ToolArgumentValidationError` | The model's arguments did not match the schema; nothing ran. |
| `ToolResultError` | What came back may not enter the run. |

Every one carries a stable `code`, a `retryable` flag and an optional `retry_after`. A code
is required at construction: a failure nobody can name is a failure nobody can write a
policy about.

`transient` defaults to `False`. An author who has not thought about whether repeating the
call repeats a side effect has not established that it does not.

## A refusal is data, not an error

`ToolRefusal` reaches the model **once**, through the untrusted-result envelope, with its
reason code:

```
<untrusted-data source="tool_refusal">
booking_not_cancellable: this fare is non-refundable
</untrusted-data>
```

It is never retried, and it does not fail the run. A reason string authored to read like an
instruction is carried as data like any other tool output — see
[`docs/tool-results.md`](tool-results.md).

## Translating what a tool author did not write

Tools raise their libraries' exceptions. `ToolErrorMap` translates them declaratively,
because an author writing `except` blocks by hand eventually writes a bare one, and a bare
one classifies a bug as a retryable failure.

```python
from tesserix_adk.tools.errors import ToolErrorMap, permanent, refusal, transient

UPSTREAM = ToolErrorMap(
    {
        httpx.ConnectError: transient("upstream_unavailable", retry_after=2.0),
        httpx.HTTPStatusError: permanent("upstream_rejected"),
    },
    statuses={409: refusal("booking_not_cancellable", "This fare is non-refundable.")},
)

@tool
async def cancel(booking: str) -> str:
    """Cancel a booking, where the supplier allows it."""
    try:
        return await supplier.cancel(booking)
    except Exception as failure:
        raise UPSTREAM.classify(failure, tool="cancel") from None
```

The most specific rule on the raised type's MRO wins, so a base class can catch a family and
a subclass can differ. Where no type rule matches, an exception carrying `status_code` or
`status` is read against `statuses`.

Three things the map will not do:

* **Guess.** An unmapped exception becomes a permanent `unmapped_failure`. The kit does not
  know whether repeating the call repeats a side effect, and guessing that it does not is
  how a run pays twice for one booking.
* **Swallow cancellation.** Anything outside `Exception` — `CancelledError` included — is
  re-raised. A run being torn down is not a fault to retry.
* **Overrule the author.** An exception already in the taxonomy is returned untouched.

Messages are scrubbed as they are translated: a token or an identifier in an upstream
exception string never survives into the typed error, and so never reaches a span, a run
event or memory.

## What the run loop does with each

| | |
|---|---|
| `retryable=True` | Retried, with jittered backoff, honouring `retry_after`, bounded by the run's deadline and budget. |
| `retryable=False` | Not retried. The failure reaches the model as an error result, or fails the run under `ToolFailurePolicy.FAIL_RUN`. |
| `ToolRefusal` | Delivered once as data. Never retried. |
| Anything untyped | Retried only where the agent declared the tool idempotent — unchanged from before. |

A `retry_after` longer than the run has left is not slept through: the call fails closed
immediately rather than waking past a deadline it cannot meet.

`max_tool_attempts` on `AgentRunner` (12 by default) caps how many retries **one tool** may
consume across a whole run, so a dependency failing transiently on every call cannot own the
iteration budget.

## What is recorded

The `tool_error` run event names the code, what the model was told, and how many attempts it
took. `ToolCallSpan` carries `code` alongside `outcome`, where `declined` is the tool saying
no and `refused` is the permission decision — two different things that would otherwise share
a word. Neither carries the raw exception message.

## Stability

* Codes are public API. A consumer branching on `booking_not_cancellable` is doing what the
  code is for.
* New codes are added in minor versions.
* Removing a code, or repurposing an existing one to mean something else, is a major
  version. A code that quietly changes meaning breaks a policy nobody edited.

Run [`examples/tool_errors.py`](https://github.com/tesserix/agent-development-kit/blob/main/examples/tool_errors.py) for the taxonomy, the map and the
retry decisions end to end.
