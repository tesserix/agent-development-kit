# Asserting tool calls

Reading a transcript catches the call that is missing and misses the call that ran under the
wrong tenant. The second is the one that matters: an agent acting with broader access than
its caller is a disclosure, not a wrong answer, and no log line makes it visible.

```python
spy = ToolSpy(registry)
with scoped_run(tenant="acme", user="ada", scopes=("search_flights", "hold_seat")):
    await agent.run("hold me a seat to Singapore", tools=spy)

assert_tool_sequence(spy, "search_flights", "hold_seat")
assert_context_propagated(spy, tenant="acme", user="ada")
```

## The spy wraps, it does not replace

`ToolSpy` records and dispatches onward, so the registry under test keeps its own argument
validation, allowlist and failure behaviour. A spy that answered calls itself would prove
only that the spy works.

Every call is recorded as a `RecordedCall`: the name as it was asked for, the validated
arguments, the tenant and user bound at dispatch, the run id, the idempotency key, what came
back or what was raised, and how long the await took. The record is written in a `finally`,
so a call that raised — or one a timeout cancelled — is still an attempt and still counted.

## Each failure names the first divergence

```
call 2: expected 'hold_seat', 'refund' was called
'search' was called with limit=10, expected 5
'refund' was called with {'id': '1'} and should not have been
```

Not two sequences and a diff to work out by eye. `assert_tool_called_once_with` is once
rather than at-least-once on purpose: a second call to a tool that acts is a duplicated side
effect, which is the defect worth catching.

## The run context is real

`scoped_run` binds the tenant through `tenant_scope` and the ambient through `carrying` —
what the runtime itself does. A fixture that merely recorded the tenant it was handed would
pass for an implementation that never reads it.

```python
with scoped_run(tenant="acme", declares=("refund",), scopes=("search",)) as run:
    run.allowlist.check("refund")  # ToolNotPermittedError, before dispatch
```

`declares` is what the agent was built to call and `scopes` is what the caller's token
covers; the allowlist is their intersection, resolved once by `ToolAllowlist`. A scope the
run does not hold refuses *before* dispatch, because an allowlist enforced afterwards is a
side effect already made. `declares` defaults to `scopes`, so the common case takes one
argument.

## Retries are proved not to duplicate

```python
with scoped_run(tenant="acme", idempotency_key="charge-1"):
    ...  # two attempts, one key
assert_idempotency_key_stable(spy, "charge")
```

Keys are compared within an argument digest: identical arguments are what makes two calls a
retry rather than two pieces of work, so calls with differing arguments may differ. A
repeated call carrying no key at all fails — absence is not stability, it is a duplicate
waiting for a retry.

## Failure injection

| Helper | Injects |
|---|---|
| `failing_tool(error)` | The caller's own domain error, so the assertion is about what the runtime surfaces |
| `slow_tool(seconds=...)` | A call long enough for the test's own timeout to fire |
| `peak_concurrency()` | A probe reporting how many calls really overlapped |

Counting calls proves nothing about how many ran at once, so `ConcurrencyProbe` measures the
overlap itself and a fan-out cap that is configured but not applied fails the test.

## Approvals, in both states

`approving()` and `denying(reason=...)` answer immediately and record what they were asked,
so the granted and refused halves of a human-gated path are both testable without a queue.
`ApprovalStub(granted=None)` is the third state: nobody answered. It denies, with the same
wording the real gate uses when it runs out of patience, because a gate that hangs reports
nothing and a gate that defaults to yes is not a gate.

## Known limitations

`assert_context_propagated` asserts over recorded tool calls. Propagation into memory and
telemetry surfaces is asserted by the isolation suite instead — see
[`isolation-suite.md`](isolation-suite.md), which seeds two tenants and counts any surface
nobody read as a failure.

A runnable version of all of the above is `examples/tool_assertions.py`.
