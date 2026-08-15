# The prompt registry

Prompts living as inline f-strings across product repositories cost twice. Nobody can say
which text produced a given run, so a behaviour regression cannot be traced to the edit that
caused it; and the same system prompt gets copied into a second product, where it diverges
without anyone deciding that it should.

A prompt here is an artifact: a name, an immutable version, a content digest, the variables it
declares, and the metadata a reviewer needs.

## Keeping prompts in a project

One TOML file per prompt name, versioned in the consuming project's own repository:

```toml
# prompts/greeting.toml
owner = "platform"
task_class = "conversation"
requires = ["tool_calling"]

[aliases]
current = "3"
candidate = "4"

[versions.3]
body = "You greet the customer by name."
variables = ["customer"]

[versions.4]
body = "You greet the customer by name, once."
variables = ["customer"]
```

```python
from tesserix_adk.core import FilePromptRegistry

registry = FilePromptRegistry("prompts")
prompt = await registry.get("greeting", alias="current")
agent = prompt.instruct(agent)
```

`PromptRegistry` is the public surface, and a remote or database-backed registry is an additive
implementation of the same protocol.

## Aliases resolve, versions record

An alias like `current` is resolved when the prompt is read and never recorded. Telemetry gets
the concrete version, because a run that recorded `current` says nothing a month later — and a
run already resolved keeps the version it started on when the alias is repointed under it.

`prompt.instruct(agent)` sets the agent's instructions and records the reference on it. The
runtime carries that onto the run, and `spend_of` puts it on every model-call span and cost
row as `adk.prompt` and `adk.prompt_digest`. No project wires anything.

The body is never exported. Prompt text goes to the provider, not to a telemetry backend where
people who were never cleared to read prompts can search it.

## What is refused

`PromptNotFoundError` for a name, version or alias that does not exist, listing the versions
that do. The kit never falls back to an empty prompt, a nearest match or a default system
prompt: an agent that silently ran on something other than what it named is exactly the
untraceable behaviour change the registry exists to prevent.

`PromptRejectedError` for a prompt that exists but may not be served — an unreadable file, an
empty body, text shaped like a credential, or a published version edited in place. The last one
is the digest doing its job: a version is immutable, an edit is a new version, and an in-place
change is surfaced rather than silently used. Development that edits prompts in a loop can pass
`sealed=False` deliberately.

`prompt.fits(window_tokens)` flags a body that would dominate the target model's declared
window, taking half of it as the ceiling — a prompt that fills the window leaves nothing to
retrieve into or answer with.

## Tenant overrides

A tenant's own copy lives at `tenants/<tenant>/<name>.toml` and is preferred where it exists,
falling back to the shared prompt. The tenant and the name are validated as plain path
components rather than joined onto the directory, so neither can be used to read another
tenant's prompts.

## Known limitations

* Rendering variables is not here: `variables` is the declaration a renderer checks against.
* Digest sealing is per registry instance and per process. It catches an edit under a running
  service, not one made while nothing was reading.
* Aliases are resolved per read. A deployment wanting one resolution for a whole release should
  resolve once at startup and pass the concrete version.
