"""The reviewed inventory of every escape from the typing guarantee.

`mypy --strict` proves what the checker can see. It does not prove that an exported symbol
is annotated at all, that an `Any` in a public signature was a decision, or that a
`type: ignore` was reviewed by anyone. Those three are what erode a library's typing claim
between releases, one plausible exception at a time.

So each one is declared in `typing-policy.toml` with a reason, an owner and a review date,
and the gate fails both ways: an escape the policy does not know about, and a policy entry
the code no longer contains. `make typing-gate` runs it; CI runs the same command.
"""

from __future__ import annotations

import inspect
import re
import sys
import tomllib
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, Any

from tools.api_surface import collect_surface, public_modules

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence
    from types import ModuleType

__all__ = [
    "ALLOWED_ANY_KINDS",
    "POLICY",
    "Escape",
    "TypingPolicyError",
    "annotation_gaps",
    "ignores_in_source",
    "load_policy",
    "public_anys",
    "violations",
]

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
POLICY = ROOT / "typing-policy.toml"

# Why an `Any` in an exported signature is allowed to stand. Anything outside this set is
# a reason someone invented at review time to get a build green.
ALLOWED_ANY_KINDS = frozenset(
    {
        # A JSON document is heterogeneous by definition; narrowing it would be a lie.
        "json",
        # A placeholder until a named story lands the real type. Carries `removed_by`.
        "provisional",
        # `**kwargs` on a sink that forwards whatever it is handed without reading it.
        "variadic",
    }
)

_IGNORE = re.compile(r"#\s*type:\s*ignore\[([^\]]+)\]")

REQUIRED_IGNORE_FIELDS = ("path", "code", "reason", "owner", "review_by")
REQUIRED_ANY_FIELDS = ("symbol", "kind", "reason", "owner", "review_by")


class TypingPolicyError(Exception):
    """Raised when the policy cannot be read. Reading none silently would disable the gate."""


@dataclass(frozen=True)
class Escape:
    """One `# type: ignore[code]` in the source, as the gate sees it."""

    path: str
    code: str
    line: int


def load_policy(path: Path = POLICY) -> dict[str, Any]:
    """Read the policy, or say why it could not be read."""
    try:
        loaded: dict[str, Any] = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as failure:
        raise TypingPolicyError(f"{path}: {failure}") from failure
    return loaded


def ignores_in_source(root: Path = SRC) -> tuple[Escape, ...]:
    """Every `# type: ignore[code]` under `root`, one entry per code on the line."""
    found: list[Escape] = []
    for file in sorted(root.rglob("*.py")):
        relative = file.relative_to(ROOT).as_posix()
        for number, text in enumerate(file.read_text(encoding="utf-8").splitlines(), start=1):
            match = _IGNORE.search(text)
            if match is None:
                continue
            found.extend(
                Escape(path=relative, code=code.strip(), line=number)
                for code in match.group(1).split(",")
            )
    return tuple(found)


def public_anys(surface: dict[str, str]) -> tuple[str, ...]:
    """Every *definition* whose exported signature mentions `Any`.

    Keyed by where the symbol is defined rather than where it is exported: `Guardrail`
    reachable as both `core.Guardrail` and `core.protocols.Guardrail` is one decision to
    review, and an inventory that lists it twice is one that drifts apart.
    """
    modules = {module.__name__: module for module in public_modules()}
    defined = set()
    for exported, described in surface.items():
        if "Any" not in described:
            continue
        module_name, _, name = exported.rpartition(".")
        obj = getattr(modules.get(module_name), name, None)
        defined.add(_defined_at(obj) or exported)
    return tuple(sorted(defined))


def _defined_at(obj: object) -> str | None:
    module = getattr(obj, "__module__", None)
    qualname = getattr(obj, "__qualname__", None)
    return f"{module}.{qualname}" if module and qualname else None


def annotation_gaps(
    check: Callable[[object], bool] | None = None,
    modules: Sequence[ModuleType] | None = None,
) -> list[str]:
    """Return one message per exported callable with a parameter or return left unannotated.

    Args:
        check: Treats an annotation as missing when it returns true. Injected so the
            reporting can be exercised without an unannotated symbol in the kit.
        modules: The modules to walk. Injected for the same reason.
    """
    missing = check or (lambda annotation: annotation is inspect.Parameter.empty)
    gaps: list[str] = []
    for module in modules if modules is not None else public_modules():
        for name in sorted(getattr(module, "__all__", [])):
            obj = getattr(module, name, None)
            gaps.extend(_gaps_of(f"{module.__name__}.{name}", obj, missing))
    return gaps


def _gaps_of(name: str, obj: object, missing: Callable[[object], bool]) -> list[str]:
    # A typing alias is callable, and inspecting one reports the `(*args, **kwargs)` of the
    # machinery rather than anything the kit declared.
    if getattr(obj, "__module__", "") == "typing":
        return []
    if inspect.isclass(obj):
        members = [(n, m) for n, m in sorted(vars(obj).items()) if not n.startswith("_")]
        return [g for n, m in members for g in _gaps_of(f"{name}.{n}", m, missing)]
    if not callable(obj):
        return []
    try:
        signature = inspect.signature(obj)
    except (TypeError, ValueError):  # A builtin or a C callable has no readable signature.
        return []
    gaps = [
        f"{name}: parameter {parameter} is not annotated"
        for parameter, spec in signature.parameters.items()
        if parameter != "self" and missing(spec.annotation)
    ]
    if missing(signature.return_annotation):
        gaps.append(f"{name}: the return is not annotated")
    return gaps


def violations(
    *,
    policy: dict[str, Any] | None = None,
    found: Sequence[Escape] | None = None,
    anys: Sequence[str] | None = None,
    ignores: Sequence[dict[str, Any]] | None = None,
) -> list[str]:
    """Return one message per way the code and the policy disagree.

    Arguments other than `policy` substitute what the gate would read from the repository,
    so each rule can be exercised without planting a real escape hatch in the kit.
    """
    loaded = policy if policy is not None else load_policy()
    declared_ignores = list(ignores if ignores is not None else loaded.get("ignore", []))
    declared_anys = list(loaded.get("any", []))
    owners = set(loaded.get("owners", []))

    in_source = tuple(found if found is not None else ignores_in_source())
    in_surface = tuple(anys if anys is not None else public_anys(collect_surface()))

    messages: list[str] = []
    messages += _field_problems(declared_ignores, REQUIRED_IGNORE_FIELDS, "ignore")
    messages += _field_problems(declared_anys, REQUIRED_ANY_FIELDS, "any")
    messages += _owner_problems(declared_ignores + declared_anys, owners)
    messages += _review_problems(declared_ignores + declared_anys)
    messages += _ignore_problems(in_source, declared_ignores)
    messages += _any_problems(in_surface, declared_anys)
    return messages


def _field_problems(
    entries: Iterable[dict[str, Any]], required: Sequence[str], kind: str
) -> list[str]:
    problems = []
    for entry in entries:
        absent = [field for field in required if not entry.get(field)]
        if absent:
            named = entry.get("symbol") or entry.get("path") or "<unnamed>"
            problems.append(f"{kind} entry {named} is missing: {', '.join(absent)}")
    return problems


def _owner_problems(entries: Iterable[dict[str, Any]], owners: set[str]) -> list[str]:
    return [
        f"entry {entry.get('symbol') or entry.get('path')} has an unknown owner "
        f"{entry.get('owner')}: reassign it rather than inheriting it silently"
        for entry in entries
        if entry.get("owner") and entry["owner"] not in owners
    ]


def _review_problems(entries: Iterable[dict[str, Any]], today: date | None = None) -> list[str]:
    now = today or date.today()
    return [
        f"entry {entry.get('symbol') or entry.get('path')} is overdue for review "
        f"({entry['review_by']}): an exception nobody revisits is a permanent one"
        for entry in entries
        if isinstance(entry.get("review_by"), date) and entry["review_by"] < now
    ]


def _ignore_problems(found: Sequence[Escape], declared: Sequence[dict[str, Any]]) -> list[str]:
    pairs = {(escape.path, escape.code) for escape in found}
    listed = {(entry.get("path"), entry.get("code")) for entry in declared}
    problems = [
        f"{escape.path}:{escape.line} ignores [{escape.code}] and is not declared in "
        f"typing-policy.toml"
        for escape in found
        if (escape.path, escape.code) not in listed
    ]
    problems += [
        f"typing-policy.toml declares an ignore [{code}] in {path} that no longer exists"
        for path, code in sorted(listed - pairs)
    ]
    return problems


def _any_problems(found: Sequence[str], declared: Sequence[dict[str, Any]]) -> list[str]:
    listed = {str(entry["symbol"]) for entry in declared if entry.get("symbol")}
    problems = [
        f"{symbol} has an undeclared Any in its public signature"
        for symbol in found
        if symbol not in listed
    ]
    problems += [
        f"typing-policy.toml declares an Any in {symbol} that is no longer there"
        for symbol in sorted(listed - set(found))
    ]
    problems += [
        f"{entry['symbol']} declares kind {entry.get('kind')!r}, which is not one of "
        f"{', '.join(sorted(ALLOWED_ANY_KINDS))}"
        for entry in declared
        if entry.get("kind") not in ALLOWED_ANY_KINDS
    ]
    problems += [
        f"{entry['symbol']} is provisional but names no story that removes it"
        for entry in declared
        if entry.get("kind") == "provisional"
        and not str(entry.get("removed_by", "")).startswith("#")
    ]
    return problems


def main() -> int:
    """Report every disagreement between the code and the policy."""
    problems = violations() + annotation_gaps()
    for problem in problems:
        sys.stderr.write(f"{problem}\n")
    if problems:
        sys.stderr.write(
            f"\n{len(problems)} typing-policy violations. Fix the type, or add a reviewed "
            f"entry to typing-policy.toml with a reason and an owner.\n"
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
