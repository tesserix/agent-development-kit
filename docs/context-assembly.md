# Context assembly

Application code concatenates history until the provider truncates it, and the provider
truncates by position. What falls off is the trip constraint stated in turn three, not
the small talk in turn ninety, and nothing records that it happened.

`ContextAssembler` builds the prompt from a declared plan, under a budget taken from the
provider's own window, and reports what it did to make it fit.

## The plan

```python
plan = ContextPlan(
    sections=(
        SectionPlan(name="system", share=0.1),
        SectionPlan(name="constraints", share=0.2, pinned=True),
        SectionPlan(name="profile", share=0.1),
        SectionPlan(name="recent", share=0.6, compaction="summarise-span"),
    ),
    reserve_output_tokens=1024,
)
assembled = await ContextAssembler(plan, provider=provider).assemble(
    {"system": [...], "constraints": [...], "profile": [...], "recent": history}
)
```

Sections appear in the prompt in plan order. A share is a fraction of the whole budget,
not of what earlier sections left over, so adding a section cannot silently shrink the
ones after it. Shares that add up to more than the budget are refused when the plan is
built, not discovered at the first long conversation.

Passing messages under a name the plan does not declare is a `ValueError`. A section the
plan declares and the caller omits is reported as empty rather than skipped.

## The budget is the provider's answer

Token counting goes through `ModelProvider.count_tokens`, which is the tokeniser of the
model that will read the prompt. Four-characters-to-the-token is wrong for every model,
and wrong in the direction that overflows.

The budget is `plan.budget_tokens` where one is set, otherwise the provider's declared
`context_window_tokens`, minus `reserve_output_tokens` either way. Both are read on every
assembly, so a model swapped mid-session for one with a smaller window is budgeted for
correctly rather than overflowed once and fixed afterwards. A provider that declares no
window and a plan that sets no budget is a `CapabilityError`: there is no safe guess.

## Pinning

```python
history = [pinned(Message(role="user", content=[TextPart(text="allergic to peanuts")])), ...]
```

A pin travels on the message, because a message and a separate list of pinned ids go
through different hands and only one of them survives a `model_copy`. A whole section can
be pinned too, which is the usual home for standing constraints.

Pinned content is allocated its room before any share is worked out, and no strategy may
evict it. Where pinned content alone exceeds the budget, assembly raises
`ContextBudgetError` rather than deciding which of the caller's constraints was optional.

## Compaction, not truncation

| Strategy | What it does | Costs |
|---|---|---|
| `drop-oldest` | Drops unpinned entries from the front until the rest fits | Nothing |
| `pin-and-fold` | Keeps pins verbatim, folds the rest into one short note | Nothing |
| `summarise-span` | Replaces the oldest unpinned span with a model-written summary | One model call |

`drop-oldest` and `pin-and-fold` are built in. `summarise-span` needs a provider of its
own — often a smaller, cheaper model than the run's — so it is supplied rather than
assumed:

```python
ContextAssembler(
    plan,
    provider=provider,
    strategies={"summarise-span": SummariseSpan(provider=small, model="haiku")},
    memory=store,
    scope=scope,
)
```

A section naming a strategy the assembler does not have is refused at construction, not
at the first assembly that needs it.

Given a store and a scope, every summary is written back to episodic memory with
`source="compaction:<section>"` and the ids it stands for, so the span it replaced can be
traced afterwards. A summary nobody kept is a summary paid for twice.

Writing your own is one method:

```python
class KeepQuestions:
    async def compact(self, entries, *, budget_tokens, count) -> CompactionOutcome:
        kept = tuple(e for e in entries if e.pinned or "?" in text_of(e))
        return CompactionOutcome(entries=kept, evicted=tuple(...))
```

## Failing closed

A prompt over the budget is never emitted, and a summary is never invented:

- The summarisation call fails or times out → `ContextBudgetError`, with the transport
  failure as its `__cause__`.
- The summariser returns nothing usable → `ContextBudgetError`. Half a summary presented
  as the record of a conversation is worse than a conversation that would not fit.
- A strategy hands back more than it was allowed → `ContextBudgetError` at the end of
  assembly, so a third-party strategy cannot quietly overflow the window.
- Cancellation while a summarisation call is in flight stays cancellation. It is not
  reported as a budget failure, because nothing about the budget went wrong.

Compaction only ever sees the messages it is given. A span carrying redaction
placeholders is summarised from the placeholders; nothing re-reads an unredacted
original, because the assembler never had one.

## What the result says

```python
assembled.messages          # the prompt, in plan order
assembled.tokens            # by the provider's own count
assembled.budget_tokens     # what there was room for
assembled.sections          # one SectionOutcome each: kept, evicted, summarised
assembled.span_attributes() # counts for a trace, and nothing that was said
```

`span_attributes()` carries token counts and how many entries were evicted or summarised.
It carries no message content: a trace of what a prompt cost is useful on every run, and
a trace of what was in it is a copy of the conversation in whatever system holds traces.

Assembly is deterministic. The same plan, the same messages and the same strategies give
the same prompt, which is what lets a recorded run replay identically in an eval.

## Not here

Retrieving external documents into a section belongs to the RAG epic; screening untrusted
content placed in one belongs to guardrails. This decides what fits, not what is true.
