"""Whose work this is, carried by the execution context rather than passed by hand.

A tenant threaded through call sites is a tenant that arrives on most of them. The one it
misses is a background task, a thread offload or a helper written in a hurry, and the
symptom is a query with no tenant filter that nothing in the type system notices and
nothing at runtime refuses.

So the tenant is established once at a process boundary — an HTTP handler, a queue
consumer, a workflow activity, a run — and read from the context everywhere below.
Absence raises: no default tenant, no "all tenants", because a default is one typo away
from being every tenant. Crossing into another tenant has to be declared, so the
legitimate administrative case is auditable and the accidental one is a refusal.

The context does not replace the scoping arguments the stores take. A `MemoryScope` still
carries its tenant as a value somebody can read in a diff; what this removes is the call
site that had to remember which one to put there.
"""

from __future__ import annotations

import contextvars
import functools
from contextlib import contextmanager
from typing import TYPE_CHECKING, Self

from pydantic import field_validator

from tesserix_adk.core.errors import MissingTenantContextError, TenantCrossingError
from tesserix_adk.core.models import AdkModel

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

__all__ = ["TenantContext", "bound", "current_tenant", "tenant_here", "tenant_scope"]

_CURRENT: contextvars.ContextVar[TenantContext | None] = contextvars.ContextVar(
    "tesserix_adk_tenant", default=None
)


class TenantContext(AdkModel):
    """The isolation boundary the work in hand belongs to.

    Frozen, like every boundary model here: a context a helper can edit is a context one
    helper can rewrite for everything running below it.

    Args:
        tenant: The isolation boundary. Required, and never blank.
        user: The acting principal within the tenant, where there is one.
        locale: The language and region the answer is for, where it matters.
        region: Where the data may be processed, for a residency-scoped deployment.
        correlation_id: The caller's own identifier for this piece of work, so a refusal
            can be found in their logs as well as in ours.
        crossing: Why this context is for a different tenant than the one it was entered
            from. Set by `tenant_scope`, never by hand, and empty for ordinary work.
        partial: Whether optional fields were dropped to fit a transport's header ceiling
            on the way here. The tenant is always intact; a consumer that reads `locale`
            or `correlation_id` can tell absence from loss.
    """

    tenant: str
    user: str | None = None
    locale: str | None = None
    region: str | None = None
    correlation_id: str | None = None
    crossing: str | None = None
    partial: bool = False

    @field_validator("tenant")
    @classmethod
    def _named(cls, value: str) -> str:
        """Blank is not a tenant; it is a filter that matches whatever the store joins."""
        if not value.strip():
            raise ValueError("tenant must name a tenant")
        return value

    def acting_as(self, user: str) -> Self:
        """The same context for a named principal.

        Args:
            user: Who is acting.

        Returns:
            A new context. Nothing is mutated.

        Example:
            >>> TenantContext(tenant="acme").acting_as("ada").user
            'ada'
        """
        return self.model_copy(update={"user": user})


def current_tenant(*, where: str = "current_tenant") -> TenantContext:
    """The context bound here, or a refusal.

    Args:
        where: What was about to happen, so the refusal names the egress point rather
            than only the accessor.

    Returns:
        The context established by the nearest enclosing `tenant_scope`.

    Raises:
        MissingTenantContextError: Where nothing is bound. Never a default tenant and
            never an unscoped read — an operation that cannot say whose data it is
            about is refused, which is the whole point of this module.

    Example:
        >>> with tenant_scope("acme"):
        ...     current_tenant().tenant
        'acme'
    """
    here = _CURRENT.get()
    if here is None:
        raise MissingTenantContextError(
            f"{where} needs a tenant context and none is bound; enter tenant_scope(...) at "
            f"the boundary that knows whose work this is",
            where=where,
        )
    return here


def tenant_here() -> TenantContext | None:
    """The context bound here, or None outside one.

    For code that legitimately runs outside a run — a health check, a corpus job — and has
    to branch on that rather than fail. Anything that touches tenant data uses
    `current_tenant` instead.

    Example:
        >>> tenant_here() is None
        True
    """
    return _CURRENT.get()


@contextmanager
def tenant_scope(
    what: TenantContext | str,
    *,
    user: str | None = None,
    crossing: str | None = None,
) -> Iterator[TenantContext]:
    """Bind a tenant for the duration of the block, restoring what was there before.

    Args:
        what: The context to bind, or a tenant name to build one from.
        user: Who is acting, where `what` is a name.
        crossing: Why this block is for a different tenant than the enclosing one. An
            administrative operation is legitimate; one nobody declared is an incident,
            so the declaration is required rather than assumed.

    Yields:
        The bound context, which is what `current_tenant` returns inside the block.

    Raises:
        TenantCrossingError: Where the block names a different tenant than the enclosing
            scope and no `crossing` was given.

    Example:
        >>> with tenant_scope("acme") as here:
        ...     here.tenant
        'acme'
    """
    outer = _CURRENT.get()
    if isinstance(what, str) and outer is not None and outer.tenant == what and outer.user == user:
        yield outer  # already bound; a second identical context is an allocation nobody reads
        return
    context = what if isinstance(what, TenantContext) else TenantContext(tenant=what)
    if user is not None:
        context = context.acting_as(user)
    if outer is not None and outer.tenant != context.tenant:
        if crossing is None:
            raise TenantCrossingError(
                f"{context.tenant!r} is not the tenant this work is under ({outer.tenant!r}); "
                f"an administrative crossing has to say why",
                tenant=outer.tenant,
                into=context.tenant,
            )
        context = context.model_copy(update={"crossing": crossing})
    token = _CURRENT.set(context)
    try:
        yield context
    finally:
        _CURRENT.reset(token)


def bound[**ArgsT, ResultT](call: Callable[ArgsT, ResultT]) -> Callable[ArgsT, ResultT]:
    """Wrap `call` so it runs under the context bound now, wherever it runs later.

    `asyncio.to_thread`, `create_task` and `TaskGroup` copy the context themselves, so
    nothing here is needed for them. `loop.run_in_executor` and a bare
    `ThreadPoolExecutor.submit` do not, and a body that lands on a pool thread with
    whatever the previous job left bound is the failure this module exists to prevent.

    Args:
        call: What will run elsewhere.

    Returns:
        The same callable, carrying a snapshot of the context taken now.
    """
    snapshot = contextvars.copy_context()

    @functools.wraps(call)
    def carrying(*args: ArgsT.args, **kwargs: ArgsT.kwargs) -> ResultT:
        return snapshot.run(call, *args, **kwargs)

    return carrying
