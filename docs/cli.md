# Command-line guide

The distribution installs the project-qualified `tesserix-adk` command. It provides local,
self-contained project operations; it does not start an application server or take over an
application's process. `python -m tesserix_adk.cli ...` is equivalent.

```bash
tesserix-adk --help
```

The root help command exits successfully. The current subcommand parsers treat
`tesserix-adk <command> --help` as command-line misuse and return exit code `2`; use the
command syntax below when invoking an individual command.

## Installed commands

| Command | Purpose |
|---|---|
| `config` | Show resolved configuration provenance or validate every typed value. |
| `doctor` | Run startup checks against resolved configuration and the local environment. |
| `eval` | Discover and execute a versioned evaluation suite, offline by default. |
| `evals` | Compare, bootstrap, or promote already measured evaluation baselines. |
| `inspect` | Inspect, diff, replay, or export a committed local run artifact. |
| `new` | Generate a typed agent or tool plus an offline contract test. |
| `run` | Run one application-supplied local agent with a redacted streamed trace. |
| `trace` | Render a recorded trace as a tree or JSON. |

The dispatcher returns non-zero when input is invalid, a gate refuses, or a command cannot
produce trustworthy evidence. Scripts should check the exit status rather than parsing a
success-looking line.

## Configure and diagnose

Validate configuration before starting a live provider:

```bash
tesserix-adk config validate
tesserix-adk config show
tesserix-adk doctor
```

`config show` reports the winning source for each value and redacts secret material.
`doctor` first requires a valid configuration, then runs the registered environment checks;
it does not guess a missing provider endpoint or credential.

## Scaffold typed files

List the built-in templates or generate one agent inside an existing Python project:

```bash
tesserix-adk new --list
tesserix-adk new agent support-agent --template tool-using
tesserix-adk new tool lookup-ticket
```

Generation validates every target before writing. Existing paths stop the whole operation
unless `--force` is explicit, and a failed replacement restores the prior bytes. See
[Getting started](getting-started.md#optional-generate-the-first-typed-files) for the
template matrix and verification commands.

## Run and record a local agent

`run` resolves an application-owned `module:attribute`; the application still constructs
the agent, provider, tools, and policies:

```bash
tesserix-adk run support_agent:target \
  --input "Summarise ticket SUP-1042" \
  --tenant acme \
  --user ada \
  --record runs/sup-1042.jsonl
```

Interactive approval is available only when the target wires it. Use `--no-interactive`
in automation so an approval-required call fails closed instead of waiting for a terminal.
Use `--json` for newline-delimited machine-readable progress.

## Inspect and render evidence

Inspect the committed artifact produced by `run`:

```bash
tesserix-adk inspect runs/sup-1042.jsonl --errors-only
tesserix-adk inspect runs/sup-1042.jsonl --export-eval-case
```

`inspect --replay` uses the artifact's embedded cassette and remains offline. `--diff`
compares two artifacts, while `--step`, `--tool`, and `--since` narrow the view.

The separate `trace` command renders the redacted trace format described in
[Local trace view](local-trace-view.md):

```bash
tesserix-adk trace trace.json --depth 3
tesserix-adk trace trace.json --json
```

## Run evaluations

`eval` executes one versioned `EvalSuite` JSONL file. Deterministic mode is the default and
must not open a network connection:

```bash
tesserix-adk eval evals/support.jsonl \
  --target support_eval:target \
  --report junit \
  --report-path reports/support.xml \
  --output artifacts/support
```

Live execution requires `--live`, an explicit per-invocation `--live-ceiling`, and
confirmation with `--yes`. The command refuses before provider work when the estimate is
over the ceiling.

`evals` operates on baseline artifacts measured by a consumer's harness:

```bash
tesserix-adk evals compare \
  --baseline evals/baseline.json \
  --candidate artifacts/candidate.json \
  --policy evals/policy.json
```

Use `evals bootstrap` to record the first baseline and `evals promote` after a reviewed
candidate is accepted. See [Evaluation datasets](eval-datasets.md),
[evaluation baselines](eval-baseline.md), and [the prompt gate](prompt-gate.md).

## Application-wired command helpers

The package also exports callable helpers for deployment-owned stores and registries, such
as approvals, checkpoint resume, dead letters, prompt aliases, Agent Cards, and MCP surface
inspection. They are Python embedding surfaces, not additional installed commands, because
the kit cannot discover an application's database, authorization policy, or agent builder.
Their feature pages show how an application can expose them through its own operator CLI.
