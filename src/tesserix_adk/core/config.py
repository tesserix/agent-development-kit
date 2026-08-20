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
from typing import TYPE_CHECKING, Annotated, Any, Literal, get_args

from pydantic import (
    BaseModel,
    BeforeValidator,
    Field,
    SecretStr,
    TypeAdapter,
    ValidationError,
    field_validator,
    model_validator,
)

from tesserix_adk.core.budget import BudgetLimits
from tesserix_adk.core.errors import ConfigurationError, SchemaViolationError
from tesserix_adk.core.models import AdkModel, parsed_from_strings

if TYPE_CHECKING:
    from pydantic.fields import FieldInfo

__all__ = [
    "ENV_PREFIX",
    "AdkConfig",
    "ConfigError",
    "ConfigProblem",
    "ConfigResolution",
    "DeadlineConfig",
    "Layer",
    "LoopConfig",
    "McpConfig",
    "McpServerConfig",
    "Provenance",
    "ProviderConfig",
    "RedactionConfig",
    "RepairConfig",
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

# The same portable name the MCP gateway routes use: no path, no quoting, no surprises.
_SERVER_NAME = re.compile(r"^[A-Za-z0-9_-]+$")

# ISO 8601 durations, parsed by pydantic itself rather than by a regex of our own.
_ISO_DURATION = TypeAdapter(timedelta).validate_strings


def _duration(value: Any) -> Any:  # noqa: ANN401
    """A duration arrives as a string wherever it comes from a file, an env var or JSON.

    `45` means 45 seconds; `PT45S` means the same thing spelt in ISO 8601. Both are read
    here, in front of the strict validation, so that nothing downstream has to guess.
    """
    if not isinstance(value, str):
        return value
    text = value.strip()
    return timedelta(seconds=float(text)) if _NUMBER.fullmatch(text) else _ISO_DURATION(text)


Duration = Annotated[timedelta, BeforeValidator(_duration)]


class ProviderConfig(AdkModel):
    """Where the model provider lives and how long to wait for it.

    `endpoint` has no default: an invented endpoint is how an agent starts half
    configured and fails on its first call instead of at startup.
    """

    endpoint: str
    api_key: SecretStr | None = None
    request_timeout: Duration = timedelta(seconds=30)


class LoopConfig(AdkModel):
    """How often a run may make the same call before it is treated as going nowhere.

    How deep and how wide a run may go are ceilings on spend and live in `BudgetLimits`
    with the money — one policy, so a cap cannot be raised in the place nobody read.
    Repetition is not spend: it is the shape of a run that has stopped making progress,
    and it is bounded by default because a run with neither bound is one nobody can
    interrupt short of a provider quota.

    Args:
        max_repeated_calls: How many times one tool may be called with the same arguments
            before the run is treated as cycling. Tools declared in
            `Agent.idempotent_tools` are exempt: polling one endpoint with the same
            arguments is the design, not a cycle.

    Example:
        >>> LoopConfig().narrowed_to(LoopConfig(max_repeated_calls=2)).max_repeated_calls
        2
    """

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


_LOOP_CAPS = ("max_repeated_calls",)


class DeadlineConfig(AdkModel):
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


class ConcurrencyConfig(AdkModel):
    """How wide one turn's tool calls may run, and how long any one of them may take.

    A turn that asked for four independent lookups should not cost four round trips, and
    firing all four at once is how one agent turn breaks a partner's rate limit. The lanes
    below are stood in together — a call holds its tenant's, its run's and its tool's at
    once — so the tightest declared bound is the one that decides.

    Args:
        max_concurrent_tools: How many of one turn's tool calls may be in flight together.
        per_tool: Tighter caps for named tools, across every run of this runner. A partner
            that permits one call at a time is declared here rather than in each agent.
        per_tenant: How many tool calls one tenant may have in flight across all its runs.
            `None` leaves a tenant bounded only by its runs.
        per_tool_seconds: Ceiling for one named tool's call, in seconds. Independent of
            its siblings, so a slow tool spends its own ceiling and not the batch's, and
            the call that overran is reported rather than the run stopped.

    Example:
        >>> narrowed = ConcurrencyConfig().narrowed_to(ConcurrencyConfig(max_concurrent_tools=2))
        >>> narrowed.max_concurrent_tools
        2
    """

    max_concurrent_tools: int = 8
    per_tool: dict[str, int] = Field(default_factory=dict)
    per_tenant: int | None = None
    per_tool_seconds: dict[str, float] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _every_lane_admits_someone(self) -> ConcurrencyConfig:
        empty = sorted(
            name
            for name, width in (
                ("max_concurrent_tools", self.max_concurrent_tools),
                ("per_tenant", self.per_tenant),
                *((f"per_tool[{tool!r}]", width) for tool, width in self.per_tool.items()),
            )
            if width is not None and width < 1
        )
        if empty:
            raise ValueError(
                f"a lane must admit at least one call: {', '.join(empty)}. Zero reads as "
                f"'never call this at all', which stops every run that needs it rather "
                f"than bounding it"
            )
        return self

    @model_validator(mode="after")
    def _every_ceiling_is_positive(self) -> ConcurrencyConfig:
        offending = sorted(tool for tool, seconds in self.per_tool_seconds.items() if seconds <= 0)
        if offending:
            raise ValueError(
                f"per-tool ceiling must be positive: {', '.join(offending)}. Zero reads as "
                f"'no time at all', which fails the call before it is made"
            )
        return self

    def narrowed_to(self, other: ConcurrencyConfig | None) -> ConcurrencyConfig:
        """Return the tighter of each lane and ceiling, so an agent cannot widen a runner's."""
        if other is None:
            return self
        tools = {
            name: min(width, other.per_tool.get(name, width))
            for name, width in self.per_tool.items()
        } | {name: width for name, width in other.per_tool.items() if name not in self.per_tool}
        seconds = {
            name: min(limit, other.per_tool_seconds.get(name, limit))
            for name, limit in self.per_tool_seconds.items()
        } | {
            name: limit
            for name, limit in other.per_tool_seconds.items()
            if name not in self.per_tool_seconds
        }
        tenants = [width for width in (self.per_tenant, other.per_tenant) if width is not None]
        return ConcurrencyConfig(
            max_concurrent_tools=min(self.max_concurrent_tools, other.max_concurrent_tools),
            per_tool=tools,
            per_tenant=min(tenants) if tenants else None,
            per_tool_seconds=seconds,
        )


class RetryConfig(AdkModel):
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


class RepairConfig(AdkModel):
    """How many times an answer that did not validate may be sent back with the reason.

    A repair is not a retry: the model is told the exact field paths that failed and what
    was wrong with each, so a second attempt is informed rather than hopeful. It is also
    not coercion — running out of attempts fails the run, and no default is ever filled in
    on the model's behalf.

    Args:
        enabled: Whether to repair at all. Turning it off here rather than deleting the
            block is how a high-stakes agent records the decision instead of inheriting it.
        max_attempts: How many repair attempts follow the first violation. Each one is a
            further model call, charged to the run's usage and its budget like any other.

    Example:
        >>> RepairConfig().max_attempts
        2
    """

    enabled: bool = True
    max_attempts: int = 2

    @model_validator(mode="after")
    def _a_budget_of_zero_says_off_in_the_wrong_field(self) -> RepairConfig:
        if self.max_attempts < 1:
            raise ValueError(
                f"max_attempts must permit at least one repair; got {self.max_attempts}. "
                f"Repair is turned off with enabled=False, which reads as a decision "
                f"rather than as a budget somebody forgot to raise"
            )
        return self


class TelemetryConfig(AdkModel):
    """OpenTelemetry export. Case content is never an attribute; see docs/contributing.md."""

    enabled: bool = True
    endpoint: str | None = None
    sample_ratio: float = 1.0


class RedactionConfig(AdkModel):
    """Redaction applied before anything is logged or exported."""

    enabled: bool = True
    extra_patterns: tuple[str, ...] = ()


class StoreConfig(AdkModel):
    """Endpoints for the optional stores. Absent means the integration is not in use."""

    redis_url: str | None = None
    postgres_dsn: SecretStr | None = None


class McpServerConfig(AdkModel):
    """One declared MCP server, so adopting its tools is configuration rather than code.

    Args:
        name: What this server is called here. It names the tools' origin in the untrusted
            envelope and in every rejection, so it is the operator's word, not the server's.
        endpoint: Where it lives, in whatever form the transport reads. Empty is an
            in-process session the consumer supplies itself.
        allow: The tools this process may adopt, by the server's own names for them. Empty
            adopts none; `*` adopts whatever is advertised, which is the setting to leave
            behind once the server is known.
        deny: The tools this process may never adopt, whatever the allowlist says.
        prefix: What this server's tools are namespaced under, so two servers that both
            advertise `search` can be told apart. Empty keeps the server's own names.
        timeout_seconds: The ceiling on one call to this server.
        max_tools: How many of its tools may reach a model at all. A server that grows a
            hundred tools should not silently spend a context window.
        max_schema_bytes: The ceiling on this server's schemas taken together, which is
            what a tool list actually costs a context window.
        max_result_bytes: The ceiling on one result, applied before it enters a prompt.
        transport: How the server is reached. Moving one from a developer's subprocess to a
            shared cluster endpoint is this field and nothing else.
        command: The child's argv, for a stdio server. Explicit, never a shell string.
        env_allow: The environment variables the child inherits. Everything else is left
            behind, so a subprocess cannot pick up a credential nobody meant to give it.
        max_message_bytes: The ceiling on one transport message, so a server writing
            without end costs a typed failure rather than this process's memory.
        read_timeout_seconds: How long a read may stall before the call is failed. A stream
            that goes quiet without closing is the failure this catches.
        max_in_flight: How many requests may be outstanding on one connection at once.
        required: Whether an agent can be assembled without this server. A required server
            that cannot be reached fails assembly; an optional one is absent, and its
            absence is stated to the model as data rather than guessed around.
        connect_timeout_seconds: The ceiling on the handshake, so an unreachable server
            costs a startup pause rather than a startup hang.
        discovery_timeout_seconds: The ceiling on reading its tool list.
        retry: How often a transport fault against this server is worth another attempt,
            and how long to wait first. One attempt is no retry at all.
        breaker_failures: Consecutive faults after which this server stops being called,
            so model turns are not spent on doomed calls.
        breaker_reset_seconds: How long the breaker stays shut before one probe is let
            through. Recovery is measured, never assumed.
    """

    name: str
    endpoint: str = ""
    allow: tuple[str, ...] = ()
    deny: tuple[str, ...] = ()
    prefix: str = ""
    timeout_seconds: float = Field(default=15.0, gt=0, le=300)
    max_tools: int = Field(default=40, ge=1, le=128)
    max_schema_bytes: int = Field(default=256 * 1024, ge=1, le=4 * 1024 * 1024)
    max_result_bytes: int = Field(default=64 * 1024, ge=1, le=4 * 1024 * 1024)
    transport: Literal["stdio", "http"] = "http"
    command: tuple[str, ...] = ()
    env_allow: tuple[str, ...] = ()
    max_message_bytes: int = Field(default=1024 * 1024, ge=1, le=16 * 1024 * 1024)
    read_timeout_seconds: float = Field(default=30.0, gt=0, le=300)
    max_in_flight: int = Field(default=8, ge=1, le=64)
    required: bool = True
    connect_timeout_seconds: float = Field(default=10.0, gt=0, le=120)
    discovery_timeout_seconds: float = Field(default=10.0, gt=0, le=120)
    retry: RetryConfig = Field(default_factory=lambda: RetryConfig(max_attempts=2))
    breaker_failures: int = Field(default=5, ge=1, le=100)
    breaker_reset_seconds: float = Field(default=30.0, gt=0, le=3600)

    @field_validator("name")
    @classmethod
    def _portable_name(cls, value: str) -> str:
        if not _SERVER_NAME.fullmatch(value):
            raise ValueError("an MCP server name may contain only letters, digits, '_' and '-'")
        return value

    @field_validator("prefix")
    @classmethod
    def _portable_prefix(cls, value: str) -> str:
        if value and not _SERVER_NAME.fullmatch(value):
            raise ValueError("an MCP tool prefix may contain only letters, digits, '_' and '-'")
        return value

    @model_validator(mode="after")
    def _one_answer_per_tool(self) -> McpServerConfig:
        """Refuse a declaration that both allows and denies a tool, or denies everything."""
        both = sorted(set(self.allow) & set(self.deny))
        if both:
            raise ValueError(f"{', '.join(both)} is both allowed and denied")
        if "*" in self.deny:
            raise ValueError("denying '*' is an empty allowlist; leave allow empty instead")
        return self

    @model_validator(mode="after")
    def _reachable_as_declared(self) -> McpServerConfig:
        """Refuse a declaration that names a transport it does not say how to reach."""
        if self.transport == "stdio" and not self.command:
            raise ValueError("a stdio MCP server needs the argv of the process to run")
        if self.transport == "http" and self.command:
            raise ValueError("an http MCP server is reached by endpoint, not by argv")
        return self


class McpConfig(AdkModel):
    """The MCP servers this process may reach. Empty is a process that reaches none."""

    servers: tuple[McpServerConfig, ...] = ()

    @model_validator(mode="after")
    def _one_entry_per_server(self) -> McpConfig:
        names = [server.name for server in self.servers]
        if len(names) != len(set(names)):
            raise ValueError("an MCP server may be declared only once")
        return self

    def server(self, name: str) -> McpServerConfig:
        """Return the declared server called `name`.

        Raises:
            ConfigurationError: No server is declared under that name, which is a typo in
                configuration rather than something to fall back from.
        """
        found = next((server for server in self.servers if server.name == name), None)
        if found is None:
            raise ConfigurationError(f"no MCP server named {name} is declared")
        return found


class AdkConfig(AdkModel):
    """The kit's resolved configuration. Frozen, fully typed, validated at startup."""

    provider: ProviderConfig
    budget: BudgetLimits = BudgetLimits.conservative()
    deadlines: DeadlineConfig = DeadlineConfig()
    loop: LoopConfig = LoopConfig()
    retry: RetryConfig = RetryConfig()
    telemetry: TelemetryConfig = TelemetryConfig()
    redaction: RedactionConfig = RedactionConfig()
    stores: StoreConfig = StoreConfig()
    mcp: McpConfig = McpConfig()

    @field_validator("budget")
    @classmethod
    def _a_stated_ceiling_does_not_remove_the_others(cls, budget: BudgetLimits) -> BudgetLimits:
        """A file that names one dimension is not a file that unbounded the rest."""
        return budget.filled()


@dataclass(frozen=True)
class ConfigProblem:
    """One reason the configuration cannot be used.

    Args:
        key: Dotted key, e.g. `budget.max_input_tokens`.
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


def _parsed(info: FieldInfo, value: object) -> Any:  # noqa: ANN401
    """Parse one environment string against its field's own annotation.

    Every value in the environment is a string, so this is where the kit's models stop
    being strict — once, explicitly, at the layer that has no types to offer.
    """
    annotation = Annotated[(info.annotation, *info.metadata)] if info.metadata else info.annotation
    return parsed_from_strings(annotation, value)


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
    for key, (layer, value) in merged.items():
        head, _, tail = key.rpartition(".")
        target = data_in[head] if head else data_in
        try:
            target[tail] = _parsed(known[key], value) if layer == "env" else value
        except SchemaViolationError as failure:
            reason = next(iter(failure.problems.values()), "invalid value")
            problems.append(ConfigProblem(key, layer, reason, _display(key, value, secrets)))

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
