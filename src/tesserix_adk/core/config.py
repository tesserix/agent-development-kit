"""One typed configuration, resolved the same way everywhere, with per-key provenance.

Precedence, highest first: explicit code arguments, `TESSERIX_ADK_*` environment
variables, `adk.toml` or the `[tool.tesserix-adk]` table in `pyproject.toml`, then
defaults. Resolution happens once at startup and the result is frozen — a changed
setting needs a restart, not a reload.

Secrets are readable from the environment only. A config file gets committed; a
`SecretStr` field supplied by one is rejected rather than warned about.
"""

from __future__ import annotations

import os
import re
import tomllib
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Annotated, Any, Literal, get_args

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    SecretStr,
    ValidationError,
    model_validator,
)

from tesserix_adk.core.errors import ConfigurationError

__all__ = [
    "ENV_PREFIX",
    "AdkConfig",
    "BudgetConfig",
    "ConfigError",
    "ConfigProblem",
    "ConfigResolution",
    "DeadlineConfig",
    "Layer",
    "LoopConfig",
    "Provenance",
    "ProviderConfig",
    "RedactionConfig",
    "RetryConfig",
    "StoreConfig",
    "TelemetryConfig",
    "leaf_keys",
    "load_config",
    "resolve_config",
    "secret_keys",
]

ENV_PREFIX = "TESSERIX_ADK_"
NESTED_DELIMITER = "__"
TOOL_TABLE = "tesserix-adk"
FILE_NAME = "adk.toml"
MASK = "**********"

Layer = Literal["code", "env", "file", "default"]

# Highest precedence first. The order is the contract; everything else derives from it.
PRECEDENCE: tuple[Layer, ...] = ("code", "env", "file")

# The default search root, rendered portably in the public API snapshot.
_CWD = "."
_NUMBER = re.compile(r"-?\d+(\.\d+)?")


def _seconds_when_bare_number(value: Any) -> Any:  # noqa: ANN401
    """Every environment value is a string, so `45` must mean 45 seconds, not a parse error."""
    return float(value) if isinstance(value, str) and _NUMBER.fullmatch(value.strip()) else value


Duration = Annotated[timedelta, BeforeValidator(_seconds_when_bare_number)]


class ProviderConfig(BaseModel):
    """Where the model provider lives and how long to wait for it.

    `endpoint` has no default: an invented endpoint is how an agent starts half
    configured and fails on its first call instead of at startup.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    endpoint: str
    api_key: SecretStr | None = None
    request_timeout: Duration = timedelta(seconds=30)


class BudgetConfig(BaseModel):
    """Spend ceilings enforced per run. Reaching one ends the run; it is never a warning."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    max_tokens_per_run: int = 100_000
    max_cost_usd_per_run: float = 1.0


class LoopConfig(BaseModel):
    """Caps on the shape of a run: how deep, how wide, and how often the same call.

    Unlike a deadline, these are bounded by default. A ceiling on wall-clock time the kit
    invented would kill good runs on slow hardware; a ceiling on recursion and repetition
    only ever stops a run that has stopped making progress — and a run with neither is one
    nobody can interrupt short of a provider quota.

    Args:
        max_depth: How deep a chain of agents calling agents may go. A child run cannot
            raise this: an agent's own config narrows what it was given and never widens
            it, or a runaway agent could vote itself more rope.
        max_tool_calls_per_turn: How many tool calls one model response may ask for. The
            whole turn is refused rather than trimmed, because half a fan-out is a set of
            side effects nobody chose.
        max_tool_calls_per_run: How many tool calls the whole run may make.
        max_repeated_calls: How many times one tool may be called with the same arguments
            before the run is treated as cycling. Tools declared in
            `Agent.idempotent_tools` are exempt: polling one endpoint with the same
            arguments is the design, not a cycle.

    Example:
        >>> LoopConfig(max_depth=9).narrowed_to(LoopConfig(max_depth=2)).max_depth
        2
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    max_depth: int = 4
    max_tool_calls_per_turn: int = 8
    max_tool_calls_per_run: int = 32
    max_repeated_calls: int = 3

    @model_validator(mode="after")
    def _every_cap_permits_something(self) -> LoopConfig:
        offending = sorted(name for name in _LOOP_CAPS if getattr(self, name) < 1)
        if offending:
            raise ValueError(
                f"cap must be at least 1: {', '.join(offending)}. Zero reads as 'never do "
                f"this at all', which is not a bound on a run but a run that cannot work"
            )
        return self

    def narrowed_to(self, other: LoopConfig | None) -> LoopConfig:
        """Return the tighter of each cap, so an inherited bound is never loosened."""
        if other is None:
            return self
        return LoopConfig(
            **{name: min(getattr(self, name), getattr(other, name)) for name in _LOOP_CAPS}
        )


_LOOP_CAPS = (
    "max_depth",
    "max_tool_calls_per_turn",
    "max_tool_calls_per_run",
    "max_repeated_calls",
)


class DeadlineConfig(BaseModel):
    """Wall-clock ceilings for a run, in seconds. `None` means no ceiling at that layer.

    Nothing is bounded by default. A model call on CPU inference can legitimately take
    minutes where the same call on a GPU takes a second, so a ceiling the kit invented
    would kill good runs on the hardware this kit is aimed at. Declare the ones you want.

    Args:
        run_seconds: Wall-clock ceiling for the whole run, across every model call, tool
            call and check it makes.
        model_call_seconds: Ceiling for one model call.
        tool_call_seconds: Ceiling for one tool call.
        hook_seconds: Ceiling for one policy hook. A hook that outruns it stops the run,
            because a check still running is not a check that passed.
        grace_seconds: How long a cancelled step is given to unwind before the run stops
            waiting for it and reports the work orphaned.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    run_seconds: float | None = None
    model_call_seconds: float | None = None
    tool_call_seconds: float | None = None
    hook_seconds: float | None = None
    grace_seconds: float = 5.0

    @model_validator(mode="after")
    def _every_ceiling_is_positive(self) -> DeadlineConfig:
        offending = sorted(
            name
            for name in _DEADLINE_CEILINGS
            if (value := getattr(self, name)) is not None and value <= 0
        )
        if offending:
            raise ValueError(
                f"ceiling must be positive: {', '.join(offending)}. Zero reads as "
                f"'no time at all', which cancels every run before it starts; None is "
                f"how a layer is left unbounded"
            )
        return self


_DEADLINE_CEILINGS = (
    "run_seconds",
    "model_call_seconds",
    "tool_call_seconds",
    "hook_seconds",
    "grace_seconds",
)


class RetryConfig(BaseModel):
    """When a failed attempt is worth making again, and how long to wait first.

    Nothing is retried by default. A retry is a second charge on someone's account and a
    second chance to duplicate a side effect, so it is declared rather than assumed.

    Args:
        max_attempts: Total attempts, not retries. 1 is one attempt and no retry.
        base_delay_seconds: The first backoff window. The delay is drawn uniformly from
            `[0, window]` — full jitter, so a fleet recovering from one provider blip
            does not retry in unison and cause the next one.
        multiplier: How much the window widens per attempt.
        max_delay_seconds: Where the window stops widening.
        max_retry_after_seconds: The longest `Retry-After` that is honoured. Beyond it
            the run stops rather than waiting: a provider asking for an hour is reporting
            a quota, not a blip, and neither waiting it out nor retrying sooner is right.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    max_attempts: int = 1
    base_delay_seconds: float = 0.5
    multiplier: float = 2.0
    max_delay_seconds: float = 30.0
    max_retry_after_seconds: float = 60.0

    @model_validator(mode="after")
    def _the_policy_can_be_followed(self) -> RetryConfig:
        if self.max_attempts < 1:
            raise ValueError(
                f"max_attempts counts attempts rather than retries, so it must permit at "
                f"least one attempt; got {self.max_attempts}"
            )
        if self.multiplier < 1:
            raise ValueError(
                f"multiplier must be at least 1; got {self.multiplier}. A window that "
                f"shrinks retries harder the worse things get, which is the storm"
            )
        offending = sorted(
            name
            for name in ("base_delay_seconds", "max_delay_seconds", "max_retry_after_seconds")
            if getattr(self, name) < 0
        )
        if offending:
            raise ValueError(f"delay must not be negative: {', '.join(offending)}")
        return self


class TelemetryConfig(BaseModel):
    """OpenTelemetry export. Case content is never an attribute; see docs/contributing.md."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    enabled: bool = True
    endpoint: str | None = None
    sample_ratio: float = 1.0


class RedactionConfig(BaseModel):
    """Redaction applied before anything is logged or exported."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    enabled: bool = True
    extra_patterns: tuple[str, ...] = ()


class StoreConfig(BaseModel):
    """Endpoints for the optional stores. Absent means the integration is not in use."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    redis_url: str | None = None
    postgres_dsn: SecretStr | None = None


class AdkConfig(BaseModel):
    """The kit's resolved configuration. Frozen, fully typed, validated at startup."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    provider: ProviderConfig
    budget: BudgetConfig = BudgetConfig()
    deadlines: DeadlineConfig = DeadlineConfig()
    loop: LoopConfig = LoopConfig()
    retry: RetryConfig = RetryConfig()
    telemetry: TelemetryConfig = TelemetryConfig()
    redaction: RedactionConfig = RedactionConfig()
    stores: StoreConfig = StoreConfig()


@dataclass(frozen=True)
class ConfigProblem:
    """One reason the configuration cannot be used.

    Args:
        key: Dotted key, e.g. `budget.max_tokens_per_run`.
        layer: Layer that supplied the offending value, or None when nothing supplied it.
        reason: What is wrong, in the words a reader can act on.
        literal: The offending value as written, masked for secret keys.
    """

    key: str
    layer: Layer | None
    reason: str
    literal: str | None = None

    def __str__(self) -> str:
        """One line naming the key, the layer that supplied it and the offending value."""
        source = self.layer or "unset"
        written = f" (got {self.literal!r})" if self.literal is not None else ""
        return f"{self.key} [{source}]: {self.reason}{written}"


class ConfigError(ConfigurationError):
    """Raised when configuration cannot be resolved, listing every problem at once.

    Fixing one problem, restarting, and discovering the next is how a ten-minute
    deployment becomes an afternoon.

    Args:
        problems: Every problem found, in the order they were detected.
    """

    def __init__(self, problems: tuple[ConfigProblem, ...]) -> None:
        self.problems = problems
        listed = "\n  ".join(str(p) for p in problems)
        super().__init__(f"configuration could not be resolved:\n  {listed}")


@dataclass(frozen=True)
class Provenance:
    """Where one resolved key came from, and what it beat.

    Args:
        layer: The layer that supplied the winning value.
        value: The resolved value as displayed, masked for secret keys.
        overridden: Lower-precedence layers that also supplied the key, in order.
    """

    layer: Layer
    value: str
    overridden: tuple[tuple[Layer, str], ...] = ()


@dataclass(frozen=True)
class ConfigResolution:
    """A resolved configuration together with the provenance of every key."""

    config: AdkConfig
    provenance: dict[str, Provenance] = field(default_factory=dict)

    def explain(self) -> str:
        """Render one line per key: the layer that supplied it and what it overrode."""
        width = max((len(k) for k in self.provenance), default=0)
        lines = []
        for key in sorted(self.provenance):
            entry = self.provenance[key]
            beaten = "".join(f"  (overrides {layer}={v})" for layer, v in entry.overridden)
            lines.append(f"{key:<{width}}  {entry.layer:<7}  {entry.value}{beaten}")
        return "\n".join(lines)


def leaf_keys(model: type[BaseModel], prefix: str = "") -> dict[str, Any]:
    """Map every dotted leaf key of a config model to its field.

    Nested models become dotted prefixes; everything else is a leaf.
    """
    keys: dict[str, Any] = {}
    for name, info in model.model_fields.items():
        dotted = f"{prefix}{name}"
        annotation = info.annotation
        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
            keys.update(leaf_keys(annotation, f"{dotted}."))
        else:
            keys[dotted] = info
    return keys


def _is_secret(annotation: object) -> bool:
    return annotation is SecretStr or SecretStr in get_args(annotation)


def secret_keys(model: type[BaseModel]) -> set[str]:
    """Dotted keys whose values are secret: environment-only, never rendered."""
    return {key for key, info in leaf_keys(model).items() if _is_secret(info.annotation)}


def _nested_prefixes(model: type[BaseModel], prefix: str = "") -> list[str]:
    """Dotted paths of the nested models, so validation reports a missing leaf as a leaf."""
    prefixes = []
    for name, info in model.model_fields.items():
        annotation = info.annotation
        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
            prefixes.append(f"{prefix}{name}")
            prefixes.extend(_nested_prefixes(annotation, f"{prefix}{name}."))
    return prefixes


def _flatten(data: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for name, value in data.items():
        dotted = f"{prefix}{name}"
        if isinstance(value, dict):
            flat.update(_flatten(value, f"{dotted}."))
        else:
            flat[dotted] = value
    return flat


def _from_env(env: dict[str, str]) -> dict[str, Any]:
    return {
        name.removeprefix(ENV_PREFIX).lower().replace(NESTED_DELIMITER, "."): value
        for name, value in env.items()
        if name.startswith(ENV_PREFIX)
    }


def _discover(start: Path) -> Path | None:
    """Walk upward for `adk.toml`, then for a pyproject carrying the tool table."""
    for directory in [start, *start.parents]:
        candidate = directory / FILE_NAME
        if candidate.is_file():
            return candidate
        pyproject = directory / "pyproject.toml"
        if pyproject.is_file() and _tool_table(pyproject) is not None:
            return pyproject
    return None


def _tool_table(path: Path) -> dict[str, Any] | None:
    with path.open("rb") as handle:
        parsed = tomllib.load(handle)
    table = parsed.get("tool", {}).get(TOOL_TABLE)
    return table if isinstance(table, dict) else None


def _read_file(path: Path) -> tuple[dict[str, Any], tuple[ConfigProblem, ...]]:
    if not path.is_file():
        return {}, (ConfigProblem(str(path), "file", "config file not found"),)
    if path.name == "pyproject.toml":
        return _tool_table(path) or {}, ()
    with path.open("rb") as handle:
        loaded: dict[str, Any] = tomllib.load(handle)
    return loaded, ()


def _display(key: str, value: object, secrets: set[str]) -> str:
    """Render a value for provenance and error messages, never disclosing a secret."""
    return MASK if key in secrets else str(value)


def resolve_config(
    overrides: dict[str, Any] | None = None,
    *,
    env: dict[str, str] | None = None,
    path: Path | str | None = None,
    start: Path | str | None = _CWD,
) -> ConfigResolution:
    """Resolve configuration from code, environment and file, and record where each key came from.

    Args:
        overrides: Explicit values, nested as the model is. Highest precedence.
        env: Environment mapping. Defaults to the process environment.
        path: An explicit config file. Skips discovery; a missing file is an error.
        start: Directory to search upward from. `None` disables file discovery entirely.

    Returns:
        The resolved config and the provenance of every key.

    Raises:
        ConfigError: One or more keys are unknown, missing, invalid, or a secret was
            supplied by a file. Every problem is reported together, before any I/O the
            configuration would have driven.
    """
    env = dict(os.environ) if env is None else env

    problems: list[ConfigProblem] = []
    layers: dict[Layer, dict[str, Any]] = {
        "code": _flatten(overrides or {}),
        "env": _from_env(env),
        "file": {},
    }

    chosen = _config_file(path, start)
    if chosen is not None:
        data, file_problems = _read_file(chosen)
        problems.extend(file_problems)
        layers["file"] = _flatten(data)

    known = leaf_keys(AdkConfig)
    secrets = secret_keys(AdkConfig)

    merged: dict[str, tuple[Layer, Any]] = {}
    for layer in PRECEDENCE:
        for key, value in layers[layer].items():
            if key not in known:
                problems.append(ConfigProblem(key, layer, "unknown key"))
            elif layer == "file" and key in secrets:
                problems.append(
                    ConfigProblem(key, layer, "secrets must be supplied by the environment")
                )
            elif key not in merged:
                merged[key] = (layer, value)

    data_in: dict[str, Any] = {prefix: {} for prefix in _nested_prefixes(AdkConfig)}
    for key, (_, value) in merged.items():
        head, _, tail = key.rpartition(".")
        target = data_in[head] if head else data_in
        target[tail] = value

    try:
        config = AdkConfig(**data_in)
    except ValidationError as err:
        for detail in err.errors():
            key = ".".join(str(part) for part in detail["loc"])
            layer_of = merged.get(key, (None, None))[0]
            literal = _display(key, merged[key][1], secrets) if key in merged else None
            problems.append(ConfigProblem(key, layer_of, detail["msg"], literal))
        raise ConfigError(tuple(problems)) from err

    if problems:
        raise ConfigError(tuple(problems))

    return ConfigResolution(config, _provenance(config, layers, merged, secrets))


def _provenance(
    config: AdkConfig,
    layers: dict[Layer, dict[str, Any]],
    merged: dict[str, tuple[Layer, Any]],
    secrets: set[str],
) -> dict[str, Provenance]:
    """One entry per key, including the layers that lost, so an override is visible."""
    provenance = {}
    for key in leaf_keys(AdkConfig):
        winner = merged.get(key)
        layer: Layer = winner[0] if winner is not None else "default"
        beaten = PRECEDENCE[PRECEDENCE.index(layer) + 1 :] if winner is not None else ()
        overridden = tuple(
            (other, _display(key, layers[other][key], secrets))
            for other in beaten
            if key in layers[other]
        )
        value = _display(key, _value_at(config, key), secrets)
        provenance[key] = Provenance(layer, value, overridden)
    return provenance


def _config_file(path: Path | str | None, start: Path | str | None) -> Path | None:
    if path is not None:
        return Path(path)
    return _discover(Path(start)) if start is not None else None


def _value_at(config: AdkConfig, key: str) -> object:
    value: object = config
    for part in key.split("."):
        value = getattr(value, part)
    return value


def load_config(
    overrides: dict[str, Any] | None = None,
    *,
    env: dict[str, str] | None = None,
    path: Path | str | None = None,
    start: Path | str | None = _CWD,
) -> AdkConfig:
    """Resolve configuration and return it, discarding the provenance.

    Use `resolve_config` when the provenance matters, such as when supporting a
    deployment nobody can explain.

    Raises:
        ConfigError: As `resolve_config`.
    """
    return resolve_config(overrides, env=env, path=path, start=start).config
