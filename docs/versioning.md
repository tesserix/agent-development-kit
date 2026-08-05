# Versioning and deprecation policy

Four products pin this kit, so a breaking change is paid for in other teams' sprints.
This page says what counts as breaking, how long notice you get, and which gate stops a
release that ignores either. Live deprecations are listed in
[deprecations.md](deprecations.md), generated from the code.

## What counts as breaking

Breaking is about what a working consumer has to change, not about how the diff looks.

- Removing or renaming any name in `docs/api-surface.txt`
- Changing a public signature: parameters, order, keyword-only status, return type
- Adding a member to a protocol — every consumer implementation stops conforming
- Narrowing an error type, or raising a different one for the same failure
- Changing a default in a way that changes behaviour: a stricter guardrail default, a
  lower budget ceiling, a shorter timeout
- Tightening validation so input that used to be accepted is rejected
- Changing a configuration key's name, meaning, or default

Not breaking: adding a name, adding an optional keyword argument with a
behaviour-preserving default, widening what is accepted, adding a configuration key, and
anything under `tesserix_adk.experimental`, which carries no promise at all.

A behavioural change with an unchanged signature is still breaking. It is worse than a
signature change, because the consumer's type checker will not find it.

## Version numbers

| Release | Contains |
|---|---|
| major | Removals and other breaking changes |
| minor | New surface, new configuration keys, new deprecations |
| patch | Fixes that change no surface and no documented behaviour |

Before 1.0 the minor takes the major's role: `0.x` is the alpha channel, breaking changes
land in a minor, and the guarantees below hold within it. Nothing else about the policy is
relaxed — the notice period and the deprecation record are the same, because pinning
`0.4.*` and getting a surprise is no cheaper for a consumer than pinning `1.4.*` and
getting one.

## The deprecation window

A removal is announced at least **two minor releases** before it happens, and happens only
in a release allowed to break. Announce in `0.1.0` and the earliest removal is `0.3.0`;
announce in `1.4.0` and it is `2.0.0`.

Mark the symbol, do not delete it:

```python
from tesserix_adk.core import deprecate


@deprecate(
    since="0.1.0",
    removal="0.3.0",
    alternative="tesserix_adk.runtime.Runner",
    reason="the sync path cannot cancel a tool call",
)
def run_blocking() -> None:
    """Run an agent to completion."""
```

The call keeps working. It emits `AdkDeprecationWarning` — a `DeprecationWarning`
subclass, so existing filters catch it — once per call site, attributed to the caller's
frame rather than the kit's, because a warning pointing into the kit's internals tells
nobody what to change. The window itself is checked when the decorator runs, so a promise
the policy forbids fails at import rather than in review.

## Preparing an upgrade before it lands

Set this in your CI and a deprecation fails the build months before the removal ships:

```bash
TESSERIX_ADK_DEPRECATIONS_AS_ERRORS=1 pytest
```

Every live deprecation, with its removal version, is in
[deprecations.md](deprecations.md).

## The gates

| Gate | Refuses |
|---|---|
| `make api-check` | A surface change that is not in the committed snapshot |
| `make deprecations` (CI) | A page that disagrees with the decorators, or a removal version that has already shipped |
| `make release-check` | A removed or reshaped symbol with no deprecation record, a removal earlier than promised, or a version bump too small for what changed |

`release-check` compares the surface and the deprecations page published at the last
release tag with the ones about to ship, so "we forgot to deprecate it" is caught before
the release rather than by a consumer's build.

## Support window

- **Python**: every version supported upstream, dropped no earlier than its upstream
  end-of-life. Dropping one is a minor release before 1.0 and a major after it.
- **Previous minor**: fixes are backported to the previous minor for the shorter of three
  months or the next minor.

## The security exception

A security fix may need to break behaviour faster than the window allows. It ships as
soon as it is ready, with an advisory naming the vulnerability, a migration note, and a
changelog entry saying the window was not met and why. It is the only exception, it is
never used for design changes, and the deprecation record is still written so the change
is in `deprecations.md` alongside everything else.
