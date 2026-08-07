# The agent definition

An `Agent` says what the job is. An `AgentDefinition` says what was *agreed*: the same
declaration, plus the owner who answers for it, the suite that checks it, the prompt entry
it was written for, and the shape it answers in.

```python
definition = AgentDefinition.declared(
    agent=Agent(name="clerk", instructions=text, model="llama-3.1-8b", free_text=True),
    owner=Owner(team="search", contact="search@example.gov", service="aequitas-search"),
    evaluation_suite="suites/clerk.yaml",
    known_tools=registry.names(),
)
run = await runner.run(definition, "what does page 12 say?", tenant="acme")
```

Worked through with no network: `examples/agent_definition.py`.

## Why the agent alone is not enough

An agent constructed inline is an agent whose model policy, tool allowlist, limits and
owner live in the call sites that build it. Nothing about it can be reviewed in one place,
diffed between two releases, or named by a run that has already finished — and when nobody
can review it, routing and retry behaviour drift into the prompt string, which is where
they stop being visible at all.

The definition is a frozen Pydantic model, so review, diff and storage all read the same
object the runtime does.

## What it insists on

| Field | Refused when | Because |
|---|---|---|
| `owner` | absent | an agent nobody answers for is an agent nobody fixes at three in the morning |
| `owner.contact` | not an address or URL | a name cannot be paged |
| `evaluation_suite` | absent | regressions in an unevaluated agent are found by its users |
| `agent.tools` | names a tool `known_tools` does not hold | discovering it at first execution in production is discovering it late |
| execution limits | a ceiling of zero | that disables the agent rather than constraining it, and is nearly always a typo |

`known_tools` is optional because a definition is often authored before the registry that
will serve it exists; passing `None` checks nothing and says so, rather than pretending.

An empty `tools` means *no* tools. "All of them" is never inferred from silence.

## The revision

`revision` is a 12-character digest of everything the definition says. It is derived, never
declared, so an edit cannot pass as the revision a past run recorded — changing the
instructions, the allowlist, the owner or the answer schema all produce a new one.

Two versions of one name coexist without collision: `key` is `name@version`, and the
revisions differ regardless.

The digest covers the serialised form, which is why `output_schema` exists. `output_type`
is a Python class excluded from serialisation, so without a stored schema two definitions
answering in materially different shapes would digest identically. The schema is derived
from `output_type` at construction and kept as data; a schema supplied explicitly is kept
as given, which is what lets a definition read back out of a store say what shape it was
agreed to answer in.

## Pinning it to the run

`AgentRunner.run` and `run_sync` take a definition wherever they take an agent. The
revision lands on `Run.definition_revision` and on every span as `adk.definition`, so
chargeback and traces both name the exact artifact that spent the money — `agent_version`
alone cannot, since a version can be edited in place between two runs that both claim it.

A run started from a bare agent records `None` and is attributed as `unknown`. That is a
visible hole rather than a hidden one: `Attribution.unknowns` names it.

## Known limitations

- Reconstructing an `Agent` with a typed answer out of JSON needs the Python class; the
  stored definition names the schema but cannot rebuild the type.
- The prompt registry itself is #56. `instructions_ref` names an entry; the text still
  lives on `agent.instructions`.
- There is no YAML or UI authoring surface. The typed object comes first.
