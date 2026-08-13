"""Whose memory it is. Required at every read and every write, with no overload without it.

A store that can be called without a scope will be, on the one path nobody tested, and the
symptom is another tenant's data in a prompt. The type makes that unspellable rather than
discouraged.
"""

from __future__ import annotations

from typing import Self

from pydantic import field_validator

from tesserix_adk.core.models import AdkModel
from tesserix_adk.core.tenancy import current_tenant

__all__ = ["MemoryScope"]


class MemoryScope(AdkModel):
    """The addressing of a memory: tenant, then who and what within it.

    Args:
        tenant_id: Required. There is no default and no "shared" sentinel — a default
            tenant is one typo away from being every tenant.
        user_id: The person, where the memory is theirs rather than the tenant's.
        session_id: The conversation, for working memory that dies with it.
        agent: Which agent wrote it, so two agents sharing a tenant do not share a
            scratch space by accident.
    """

    tenant_id: str
    user_id: str | None = None
    session_id: str | None = None
    agent: str | None = None

    @field_validator("tenant_id")
    @classmethod
    def _named(cls, value: str) -> str:
        """Blank is not a tenant; it is a scope that matches whatever the adapter joins."""
        if not value.strip():
            raise ValueError("tenant_id must name a tenant")
        return value

    @classmethod
    def here(
        cls,
        *,
        user_id: str | None = None,
        session_id: str | None = None,
        agent: str | None = None,
    ) -> Self:
        """The scope for the tenant bound to the execution context.

        The tenant stays a value on the record — what this removes is the call site that
        had to remember which one to put there, which is the call site that forgets.

        Args:
            user_id: The person, where the memory is theirs. Defaults to the acting
                principal on the context, where there is one.
            session_id: The conversation.
            agent: Which agent wrote it.

        Returns:
            A scope carrying the context's tenant.

        Raises:
            MissingTenantContextError: Where no context is bound. Refused rather than
                widened: an unscoped recall reads every tenant's memory and looks like
                an answer.
        """
        context = current_tenant(where="MemoryScope.here")
        return cls(
            tenant_id=context.tenant,
            user_id=user_id if user_id is not None else context.user,
            session_id=session_id,
            agent=agent,
        )

    @property
    def path(self) -> tuple[str, str, str, str]:
        """The scope as an adapter's key prefix. Absent parts are empty, never skipped."""
        return (self.tenant_id, self.user_id or "", self.session_id or "", self.agent or "")
