"""One typed function is the whole tool: the schema the model reads is derived from it.

A tool advertised by a hand-written dictionary and implemented by a Python function is two
declarations of one shape, and the second one is the one nobody updates. The drift is
silent — the model keeps sending the argument that was renamed six months ago — and it
surfaces as a call the code refuses in production. Here the signature is the only
declaration: `@tool` reads it and the docstring, and what the model is told cannot say
anything the function does not accept.

Refusals happen at decoration, which is import time. A parameter no schema can describe is
a tool the model calls wrongly for as long as the process lives, and the call that suffers
is nowhere near the definition that caused it.

Every name exported here is semver-governed: it appears in `docs/api-surface.txt`, so a
change to it shows up in a pull request's diff and follows `docs/versioning.md`.
"""

from __future__ import annotations

import asyncio
import functools
import inspect
import weakref
from collections.abc import Awaitable, Coroutine
from dataclasses import dataclass, field
from types import UnionType
from typing import TYPE_CHECKING, Any, Protocol, Union, cast, get_args, get_origin, overload

from tesserix_adk.core.errors import (
    CapabilityError,
    SchemaGenerationError,
    ToolDefinitionError,
    ToolExecutionError,
)
from tesserix_adk.core.hooks import ApprovalPolicy, ApprovalPredicate
from tesserix_adk.core.idempotency import Idempotency, IdempotencyPolicy
from tesserix_adk.core.schema import (
    INLINE_REFS,
    SchemaDialect,
    annotations_of,
    schema_for,
)
from tesserix_adk.tools.context import ToolContext
from tesserix_adk.tools.validation import (
    STRICT,
    ArgumentPolicy,
    ArgumentValidator,
    ToolArgumentValidator,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from tesserix_adk.runtime.blocking import WorkerPool

__all__ = ["Tool", "tool"]

# One name, one live tool. The claim is given back when the tool is collected, so the rule
# is what shadowing actually means — two tools answering to one name at the same time —
# rather than a module that can only ever be imported once.
_CLAIMED: dict[str, str] = {}
_TOKENS: dict[str, object] = {}
_RELEASES: dict[str, weakref.finalize[[str, object], Tool[Any, Any]]] = {}


@dataclass(frozen=True, slots=True, weakref_slot=True)
class Tool[**P, R]:
    """A callable the model can be told about, with the schema derived from its signature.

    Built by `@tool`; constructing one directly is possible but skips every check the
    decorator makes. Awaiting the tool calls the underlying function with its own typed
    signature; `invoke` is the model-facing path, taking the arguments a provider chose as
    a mapping and injecting the context the model never sees.

    Args:
        name: What the model calls it.
        description: What it is for, one line, from the docstring summary or the override.
        parameters_schema: The model-facing arguments, without the injected context.
        returns_schema: The result, where the function annotates one.
        is_async: Whether the underlying function is a coroutine function. A synchronous
            one is run off the event loop rather than on it.
        function: The undecorated callable.
        validator: What `invoke` holds the model's arguments to before the body runs.
        context_parameter: The parameter `invoke` fills with a `ToolContext`, where the
            function asked for one.
        context_required: Whether that parameter has to be filled. A context parameter
            with a default is a tool that also works outside a run; one without is a tool
            that must be refused rather than called with a tenant nobody set.
        timeout: The ceiling one call may take, in seconds, where the author declared one.
            Enforced by whoever dispatches the tool; awaiting the tool directly is the
            typed path and is not bounded here.
        parallel_safe: Whether two of this tool's calls may be in flight together. A tool
            whose effect depends on the order it is called in declares itself here; no
            signature says it.
        idempotency: What repeating this call does, and which arguments identify it.
            `None` is undeclared: no record is held and retries behave as they always have.
        approval: When a human must decide before the body runs. Declared here rather than
            per agent, because the tool knows it moves money and every agent that adopts it
            would otherwise have to remember.
        returns_type: What the function annotated its result as, awaited, or `None` where
            it annotated nothing. The schema is what a model reads; this is what a result
            can actually be held to before it enters a conversation.
        origin: Stable import provenance for a translated tool. Native tools leave this
            empty and the registry derives their module and qualified name.
        max_concurrency: A tighter per-tool lane declared by an import boundary. The
            registry applies it automatically; `None` uses its configured policy.
    """

    name: str
    description: str
    parameters_schema: dict[str, Any]
    returns_schema: dict[str, Any] | None
    is_async: bool
    function: Callable[P, Awaitable[R]] | Callable[P, R]
    validator: ArgumentValidator
    context_parameter: str | None = None
    context_required: bool = False
    timeout: float | None = None
    parallel_safe: bool = True
    approval: ApprovalPolicy = field(default_factory=ApprovalPolicy)
    idempotency: IdempotencyPolicy | None = None
    returns_type: Any = None
    origin: str = ""
    max_concurrency: int | None = None

    async def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        """Call the function with its own signature, awaiting it either way.

        Uniform on purpose: a consumer moving a tool from synchronous to asynchronous
        should not have to change every call site, and a synchronous body left on the
        event loop stalls every other run sharing it.
        """
        return await self._dispatch(args, kwargs, None)

    async def invoke(
        self,
        arguments: Mapping[str, Any] | str | bytes,
        context: ToolContext | None = None,
        *,
        workers: WorkerPool | None = None,
    ) -> R:
        """Call the tool with the arguments a model chose, and return its result.

        This is the untrusted path, so the arguments are held to the signature the model
        was shown before the body is entered: unknown fields are refused, nothing absent
        is filled in, and each argument arrives as its declared type. `__call__` is the
        typed path, where the type checker has already done it.

        Args:
            arguments: What the model asked for, as a mapping or as the JSON text some
                providers send instead.
            context: The run this call belongs to. Defaults to the ambient the runtime
                bound, so a tool invoked inside a run gets the right tenant.
            workers: Where a synchronous body runs. Without one it still leaves the event
                loop, but nothing bounds how many bodies run at once.

        Raises:
            ToolArgumentValidationError: If the arguments do not match the tool's schema.
                The body is not entered and nothing is partially applied.
            ToolExecutionError: If the function requires a context and none was supplied
                or bound. Guessing a tenant is worse than refusing the call.
            Exception: Whatever the function raised, unchanged.
        """
        checked = self.validator.arguments(arguments)
        return await self._dispatch((), {**checked, **self._injected(context)}, workers)

    def requires_approval(self, arguments: Mapping[str, Any] | str | bytes) -> bool:
        """Whether this call may not run without a human.

        The policy's predicate is evaluated over validated arguments, so a threshold reads
        the number the body would receive rather than the string a provider sent.
        Arguments that will not validate are read as requiring approval: a call the gate
        cannot understand is not a call to wave through.
        """
        if self.approval.required:
            return True
        if self.approval.when is None:
            return False
        try:
            checked = self.validator.arguments(arguments)
        except Exception:
            return True
        return self.approval.applies_to(checked)

    def release(self) -> None:
        """Give up this tool's claim on its name, so another definition may take it.

        Rarely needed: a tool that goes out of scope releases its own name. This is for a
        tool being replaced while something still holds a reference to it. Releasing twice,
        or releasing a name another tool has since taken, does nothing.
        """
        claim = _RELEASES.get(self.name)
        held = claim.peek() if claim is not None else None
        if claim is not None and held is not None and held[0] is self:
            claim()

    def _injected(self, context: ToolContext | None) -> dict[str, Any]:
        """The context argument, where the function asked for one and only then."""
        if self.context_parameter is None:
            return {}
        supplied = context if context is not None else ToolContext.current()
        if supplied is None and not self.context_required:
            return {}
        if supplied is None:
            raise ToolExecutionError(
                f"{self.name} takes a ToolContext and none was given: there is no ambient "
                f"run to take one from, and a guessed tenant is worse than a refused call.",
                details={"tool": self.name, "parameter": self.context_parameter},
            )
        return {self.context_parameter: supplied}

    async def _dispatch(
        self, args: tuple[Any, ...], kwargs: dict[str, Any], workers: WorkerPool | None
    ) -> R:
        """Run the body, off the loop where it is synchronous, and await what it produced."""
        if self.is_async:
            call = cast("Callable[..., Awaitable[R]]", self.function)
            return await call(*args, **kwargs)
        body = functools.partial(cast("Callable[..., R]", self.function), *args, **kwargs)
        produced = await (workers.call(self.name, body) if workers else asyncio.to_thread(body))
        if inspect.isawaitable(produced):
            return cast("R", await produced)
        return produced


class _Decorator(Protocol):
    """What `@tool(...)` returns: the same decorator, holding the overrides."""

    @overload
    def __call__[**P, R](self, function: Callable[P, Awaitable[R]], /) -> Tool[P, R]: ...

    @overload
    def __call__[**P, R](self, function: Callable[P, R], /) -> Tool[P, R]: ...

    def __call__(self, function: Callable[..., Any], /) -> Tool[Any, Any]:
        """Describe `function` and return the tool built from it."""
        ...


@overload
def tool[**P, R](function: Callable[P, Awaitable[R]], /) -> Tool[P, R]: ...


@overload
def tool[**P, R](function: Callable[P, R], /) -> Tool[P, R]: ...


@overload
def tool(
    *,
    name: str | None = ...,
    description: str | None = ...,
    dialect: SchemaDialect = ...,
    arguments: ArgumentPolicy = ...,
    timeout: float | None = ...,
    parallel_safe: bool = ...,
    requires_approval: bool | ApprovalPredicate | ApprovalPolicy = ...,
    idempotency: Idempotency | str | IdempotencyPolicy | None = ...,
) -> _Decorator: ...


def tool(
    function: Callable[..., Any] | None = None,
    /,
    *,
    name: str | None = None,
    description: str | None = None,
    dialect: SchemaDialect = INLINE_REFS,
    arguments: ArgumentPolicy = STRICT,
    timeout: float | None = None,
    parallel_safe: bool = True,
    requires_approval: bool | ApprovalPredicate | ApprovalPolicy = False,
    idempotency: Idempotency | str | IdempotencyPolicy | None = None,
) -> Tool[Any, Any] | _Decorator:
    """Turn a typed function into a tool, deriving its schema from the signature.

    Usable bare or with overrides: `@tool` and `@tool(name="lookup_fare")` both work.
    Argument descriptions come from the Google-style `Args:` block of the docstring, and
    the summary line becomes the tool's description; a missing docstring costs the
    descriptions, never the tool.

    Args:
        function: The function to describe, when used bare.
        name: What the model calls it. Defaults to the function's name.
        description: One line saying what it is for. Defaults to the docstring summary,
            and to the tool's name where there is no docstring.
        dialect: The schema dialect to emit. Defaults to inlining every nested model,
            which is what providers that refuse `$ref` need.
        arguments: How strictly `invoke` reads what the model sent. Strict by default, so
            a tool's contract does not depend on which provider answered.
        timeout: How long one call may take, in seconds. Enforced by the registry that
            dispatches it, which may tighten it further. `None` leaves the tool bounded
            only by whatever ceiling the run itself has.
        parallel_safe: Whether two of its calls may be in flight together. Declare `False`
            for a tool whose effect depends on the order it is called in.
        idempotency: What repeating a call does — `"read_only"`, `"idempotent"` or
            `"effectful"`, or an `IdempotencyPolicy` naming the arguments that identify one
            call. Undeclared by default, which changes nothing about how the tool is run.
        requires_approval: Whether a human must decide before the body runs. `True` gates
            every call; a predicate over the validated arguments gates conditionally, for a
            threshold. Evaluated by this code and never by the model.

    Returns:
        The tool, or the decorator that builds it when overrides were given.

    Raises:
        ToolDefinitionError: If the signature cannot be described — an unannotated or
            variadic parameter, a type with no JSON Schema representation, a generator, a
            model that refers to itself, or a name another live tool already answers to.
            Raised where the tool is declared, not on the call that first sends it.

    Example:
        >>> import asyncio
        >>> @tool
        ... def fare(leg: str) -> str:
        ...     '''Price a leg.
        ...
        ...     Args:
        ...         leg: The hop to price.
        ...     '''
        ...     return f"{leg}: 40 EUR"
        >>> fare.parameters_schema["required"]
        ['leg']
        >>> asyncio.run(fare.invoke({"leg": "Osaka to Kyoto"}))
        'Osaka to Kyoto: 40 EUR'
    """
    overrides = _Overrides(
        name,
        description,
        dialect,
        arguments,
        timeout,
        parallel_safe,
        _policy(requires_approval),
        _repeatable(idempotency),
    )
    if function is None:
        return _overriding(overrides)
    return _built(function, overrides)


@dataclass(frozen=True, slots=True)
class _Overrides:
    """What `@tool(...)` was given, carried to the function it ends up decorating."""

    name: str | None
    description: str | None
    dialect: SchemaDialect
    arguments: ArgumentPolicy
    timeout: float | None
    parallel_safe: bool
    approval: ApprovalPolicy
    idempotency: IdempotencyPolicy | None


def _overriding(overrides: _Overrides) -> _Decorator:
    """The decorator `@tool(...)` becomes, carrying the overrides to the function."""

    def decorate(function: Callable[..., Any]) -> Tool[Any, Any]:
        return _built(function, overrides)

    return cast("_Decorator", decorate)


def _built(function: Callable[..., Any], overrides: _Overrides) -> Tool[Any, Any]:
    """Describe one function, refusing anything the model could not be told about."""
    _refuse_unsuitable(function)
    called = overrides.name or getattr(function, "__name__", "")
    if not called:
        raise ToolDefinitionError(
            "a tool needs a name and this callable has none. Pass name= explicitly.",
            tool="",
        )
    hints = _hints(function, called)
    injected, required = _context_parameter(function, called, hints)
    parameters = _parameters(function, called, overrides.dialect, injected)
    _refuse_a_key_over_arguments_that_do_not_exist(called, overrides.idempotency, parameters)
    _refuse_an_impossible_ceiling(called, overrides.timeout)
    built = Tool[Any, Any](
        name=called,
        description=overrides.description or _summary(function) or called,
        parameters_schema=parameters,
        returns_schema=_returns(hints, called, overrides.dialect),
        is_async=inspect.iscoroutinefunction(function),
        function=function,
        validator=ToolArgumentValidator(
            function,
            tool=called,
            exclude=(injected,) if injected else (),
            policy=overrides.arguments,
        ),
        context_parameter=injected,
        context_required=required,
        timeout=overrides.timeout,
        parallel_safe=overrides.parallel_safe,
        approval=overrides.approval,
        idempotency=overrides.idempotency,
        returns_type=_awaited(hints["return"]) if "return" in hints else None,
    )
    _claim(called, function, built)
    return built


def _policy(declared: bool | ApprovalPredicate | ApprovalPolicy) -> ApprovalPolicy:
    """Read what `requires_approval=` was given as the policy it describes."""
    if isinstance(declared, ApprovalPolicy):
        return declared
    if isinstance(declared, bool):
        return ApprovalPolicy(required=declared)
    return ApprovalPolicy(when=declared)


def _repeatable(declared: Idempotency | str | IdempotencyPolicy | None) -> IdempotencyPolicy | None:
    """Read what `idempotency=` was given as the policy it describes.

    `None` stays `None`: a tool that has not been classified is not the same as one
    classified as safe, and the dispatcher treats it as it always has.
    """
    if declared is None or isinstance(declared, IdempotencyPolicy):
        return declared
    return IdempotencyPolicy(Idempotency(declared))


def _refuse_a_key_over_arguments_that_do_not_exist(
    called: str, policy: IdempotencyPolicy | None, parameters: Mapping[str, Any]
) -> None:
    """A key over an argument the tool does not take is a key that never derives."""
    if policy is None or not policy.key_arguments:
        return
    stray = sorted(set(policy.key_arguments) - set(parameters.get("properties", {})))
    if stray:
        raise ToolDefinitionError(
            f"{called} keys its idempotency on {', '.join(stray)}, which it does not take, "
            f"so no call could ever derive a key and every one would fail closed.",
            tool=called,
        )


def _refuse_an_impossible_ceiling(called: str, timeout: float | None) -> None:
    """A ceiling of zero or less reads as no time at all, which fails every call."""
    if timeout is not None and timeout <= 0:
        raise ToolDefinitionError(
            f"{called} declares a timeout of {timeout:g}s, which is no time at all: every "
            f"call would fail before it was made. Leave it unset to declare no ceiling.",
            tool=called,
        )


def _refuse_unsuitable(function: object) -> None:
    """Refuse what cannot be a tool at all, before asking what its arguments look like."""
    if not callable(function):
        raise ToolDefinitionError(
            f"a {type(function).__name__} is not callable, so there is no tool to build.",
            tool="",
        )
    if inspect.isgeneratorfunction(function) or inspect.isasyncgenfunction(function):
        raise ToolDefinitionError(
            f"{getattr(function, '__name__', 'this function')} is a generator, and a tool "
            f"has to return one result the caller can validate and hand back to the model. "
            f"Collect the items and return them.",
            tool=getattr(function, "__name__", ""),
        )


def _hints(function: Callable[..., Any], called: str) -> dict[str, Any]:
    """The annotations, resolved. A string annotation nobody resolved describes nothing."""
    try:
        return annotations_of(function)
    except (NameError, TypeError) as unresolved:
        raise ToolDefinitionError(
            f"{called} has an annotation that cannot be resolved, so the model would be "
            f"told a type nobody can check. ({unresolved})",
            tool=called,
        ) from unresolved


def _context_parameter(
    function: Callable[..., Any], called: str, hints: Mapping[str, Any]
) -> tuple[str | None, bool]:
    """The one parameter the runtime fills and whether it has to be filled.

    Two would be one too many: the runtime has one context to give, and filling both from
    it makes the second parameter a synonym rather than an argument.
    """
    parameters = inspect.signature(function).parameters
    found = [name for name in parameters if _is_context(hints.get(name))]
    if len(found) > 1:
        raise ToolDefinitionError(
            f"{called} takes {len(found)} ToolContext parameters and the runtime has one "
            f"context to give. Keep one: {', '.join(found)}.",
            tool=called,
            parameter=found[1],
        )
    if not found:
        return None, False
    return found[0], parameters[found[0]].default is inspect.Parameter.empty


def _is_context(annotation: object) -> bool:
    """`ToolContext`, or an optional one. Anything else is the model's to choose."""
    if annotation is ToolContext:
        return True
    if get_origin(annotation) in (Union, UnionType):
        return any(argument is ToolContext for argument in get_args(annotation))
    return False


def _parameters(
    function: Callable[..., Any], called: str, dialect: SchemaDialect, injected: str | None
) -> dict[str, Any]:
    """The model-facing arguments, without the context and without the summary.

    The summary is the tool's description and a provider sends it separately, so repeating
    it inside the parameters is tokens spent saying the same thing on every call.
    """
    schema = _described(function, called, dialect, exclude=(injected,) if injected else ())
    schema.pop("description", None)
    return schema


def _returns(
    hints: Mapping[str, Any], called: str, dialect: SchemaDialect
) -> dict[str, Any] | None:
    """The result's schema, or `None` where the function does not annotate one."""
    if "return" not in hints:
        return None
    return _described(_awaited(hints["return"]), called, dialect)


def _awaited(annotation: Any) -> Any:  # noqa: ANN401 — whatever the function annotated
    """What the result is once awaited: a caller of `invoke` never sees the awaitable."""
    origin = get_origin(annotation)
    if origin is None or not inspect.isclass(origin) or not issubclass(origin, Awaitable):
        return annotation
    arguments = get_args(annotation)
    return arguments[2] if issubclass(origin, Coroutine) else arguments[0]


def _described(
    target: Any,  # noqa: ANN401 — a type or a callable, whatever the caller annotated
    called: str,
    dialect: SchemaDialect,
    exclude: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Generate a schema, turning every way that can fail into one typed refusal."""
    try:
        return schema_for(target, dialect=dialect, exclude=exclude)
    except SchemaGenerationError as undescribable:
        raise ToolDefinitionError(
            f"{called} cannot be described to a model: {undescribable}",
            tool=called,
            parameter=undescribable.field,
        ) from undescribable
    except CapabilityError as unrepresentable:
        raise ToolDefinitionError(
            f"{called} cannot be described to a model: {unrepresentable}",
            tool=called,
        ) from unrepresentable


def _summary(function: Callable[..., Any]) -> str:
    """The docstring's first line. A docstring in some other shape simply has no summary."""
    doc = inspect.getdoc(function)
    return doc.splitlines()[0].strip() if doc else ""


def _claim(called: str, function: Callable[..., Any], built: Tool[Any, Any]) -> None:
    """Take the name, refusing it to a second live tool. Re-decorating one function is fine."""
    origin = f"{getattr(function, '__module__', '?')}.{getattr(function, '__qualname__', '?')}"
    held = _CLAIMED.get(called)
    if held is not None and held != origin:
        raise ToolDefinitionError(
            f"{called} is already the name of a live tool ({held}), and two tools "
            f"answering to one name means the model reaches whichever the registry "
            f"happened to keep. Rename one, or release the first.",
            tool=called,
        )
    token = object()
    _CLAIMED[called] = origin
    _TOKENS[called] = token
    _RELEASES[called] = weakref.finalize(built, _released, called, token)


def _released(called: str, token: object) -> None:
    """Give the name back, unless a later tool has since taken it."""
    if _TOKENS.get(called) is token:
        del _CLAIMED[called], _TOKENS[called], _RELEASES[called]
