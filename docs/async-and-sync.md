# Async and sync

The kit is async all the way down. A run is mostly waiting — on a provider, on a tool, on
a ledger — and a process that waits by blocking serves one run where it could serve
hundreds. But consumers are not all async, and the usual improvisations fail in the same
three ways every time: a sync helper called inside a running loop, a blocking body called
from a coroutine, and identity dropped the moment work hops to a thread.

Each has a name here rather than a convention nobody reads.

## Going in: calling the runtime from sync code

| From | Use |
|---|---|
| An async service | `await runner.run(...)`, `runner.stream(...)` |
| A script with no loop | `runner.run_sync(...)`, `runner.stream_sync(...)` |
| Sync code inside a running loop | Nothing — see below |

`run_sync` is the same run `run` drives, on a loop of its own. There is one implementation
of the loop and not a second one that drifts from it: everything the async path does —
budgets, hooks, retries, cancellation, progress — the sync path does, because it *is* the
async path.

It drives its own loop rather than calling `asyncio.run`, which clears the thread's event
loop on exit and so would break whatever else on that thread had set one.

`stream_sync` returns every progress event the run produced, as a tuple, once it has
finished. A sync caller cannot be handed events as they happen without a thread and a
queue it did not ask for; where progress has to be acted on while the run is still going,
that is `stream`, awaited.

## The refusal

Calling `run_sync` or `stream_sync` from inside a running event loop raises
`RunningLoopError`. It does not nest a second loop, and it does not block the loop it is
standing on:

```python
try:
    run = runner.run_sync(agent, "…", tenant="acme")
except RunningLoopError as refusal:
    refusal.async_name  # 'AgentRunner.run'
```

Nesting runs two schedulers over one set of tasks. Blocking deadlocks the loop against the
work it is waiting for — and a deadlock says nothing about which line caused it, which is
why the refusal names both the helper called and the call to use instead. It is also a
`RuntimeError`, so code already guarding against "this event loop is already running"
keeps working.

### Notebooks

A notebook cell runs inside a live loop, so `run_sync` refuses there. Two supported
patterns, in order of preference:

```python
run = await runner.run(agent, "…", tenant="acme")     # Jupyter awaits at top level
run = await asyncio.to_thread(                         # from a plain sync cell
    lambda: runner.run_sync(agent, "…", tenant="acme")
)
```

Both are the same run. `nest_asyncio` and its relatives are not supported: patching the
loop to re-enter itself makes ordering, cancellation and timeouts unpredictable in a way
that surfaces later as a run that cannot be explained.

## Going out: a body that blocks

A blocking call inside a coroutine does not slow its own run. It stops the loop, so it
slows *every* run sharing the process, and the latency lands on requests that did nothing
wrong. Two halves answer this.

### Declare it, and it gets a thread

```python
pool = WorkerPool(Workers(size=8, queue_seconds=30.0))

async def lookup(city: str) -> str:
    return await pool.call("lookup", lambda: legacy_client.fetch(city))
```

The pool is bounded on purpose. An unbounded pool turns a slow dependency into a thread
per in-flight request and fails on memory rather than on the queue. A body that waits
longer than `queue_seconds` for a worker is refused with `WorkersBusyError`, which names
the pressure; growing the pool instead would hide it in latency and fail later, harder,
and on someone else's request.

Cancelling the await stops the waiting, not the thread: a running body has no interrupt,
so it is abandoned rather than claimed undone — the same rule as a tool caught in flight
in [`run-progress.md`](run-progress.md#stopping-a-stream). A body that may run long should
cooperate:

```python
def slow_body() -> str:
    for page in pages:
        current_ambient().raise_if_cancelled()
        ...
```

A body may run its own event loop — `asyncio.run(...)` inside it is fine, because it is on
a thread of its own and not on the runtime's loop.

### Don't declare it, and it gets caught

Every tool call is watched by a `LoopMonitor`, which measures the loop's own lag while the
tool runs. Work that stopped the loop for longer than `stall_seconds` fails with
`EventLoopStalledError` naming the tool, and the run records it against that tool rather
than leaving unattributed tail latency on whichever request was next:

```
tool lookup stalled the event loop for 0.412s; run a blocking body on a worker pool
rather than on the loop
```

Lag, not duration: a tool that legitimately awaits for a minute never trips it, because
the loop kept turning throughout. A body that failed on its own terms is reported on its
own terms — the monitor does not relabel someone else's exception.

`AgentRunner(monitor=None)` turns the instrumentation off, which is a deployment deciding
to take the tail latency rather than attribute it.

## Identity across the hop

A tool body receives the arguments the model chose. Anything the model did not choose —
the tenant, the run, the caller's switch — has to arrive another way or be dropped at the
first hop onto a thread. It arrives as an `Ambient`, bound for the duration of the call:

```python
def body() -> str:
    who = current_ambient()      # run_id, tenant, user, cancellation
    ...
```

`current_ambient()` returns `None` outside a run rather than an invented default: a tenant
nobody set is the bug this exists to make visible, not a value to guess.

Each `pool.call` copies the calling context, so the ambient crosses onto the worker thread
and nothing the body sets crosses back or reaches the next body on that thread. Two runs
sharing one worker cannot read each other's tenant.

## Cancellation from a signal

`run_sync` runs the loop on the calling thread, so a signal handler that flips the
caller's token is seen by the run it belongs to:

```python
token = CancellationToken()
signal.signal(signal.SIGINT, lambda *_: token.cancel("interrupted at the terminal"))
run = runner.run_sync(agent, "…", tenant="acme", cancellation=token)
run.state   # RunState.CANCELLED
```

The run resolves cancelled with its spend recorded, rather than the process dying with an
unattributed bill.

## Known limitations

- A `WorkerPool` belongs to the loop it was first used on. Sharing one across `run_sync`
  calls that each build their own loop is not supported; build the pool where the loop is.
- `stream_sync` collects the whole run before returning, so a run whose progress must be
  acted on mid-flight needs `stream`.
- Stall detection measures the loop, not the tool: a process pathologically starved by
  something outside the kit will be attributed to whichever tool was running.
