"""Deriving an activity's numbers from what the tool itself declared.

A global heartbeat constant kills exactly the calls it was meant to protect, so the
windows follow the timeout the tool declared rather than a value picked once for the
worker. Tuning any of these is a value change: the surface is the policy model, and the
numbers behind it are not part of the contract.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from tesserix_adk.core.activities import (
    DEFAULT_ACTIVITY_POLICIES,
    MAXIMUM_HEARTBEAT_SECONDS,
    MINIMUM_HEARTBEAT_SECONDS,
    ActivityClass,
    ActivityPolicy,
)

if TYPE_CHECKING:
    from tesserix_adk.tools.decorator import Tool

__all__ = ["activity_policy_for", "attempts_for_tool", "policy_for_tool"]


def activity_policy_for(
    activity_class: ActivityClass,
    *,
    timeout: float | None = None,
    base: ActivityPolicy | None = None,
) -> ActivityPolicy:
    """The policy for one activity, narrowed by a declared timeout.

    Args:
        activity_class: What kind of work it is.
        timeout: What the tool said one call may take. Absent, the class default holds.
        base: A policy a consumer already tuned. Its retry settings are kept; only the
            windows are derived, so overriding a policy is not undone by this call.

    Returns:
        The policy to run the activity under.

    Example:
        >>> activity_policy_for(ActivityClass.TOOL, timeout=5.0).start_to_close_seconds
        5.0
    """
    policy = base or DEFAULT_ACTIVITY_POLICIES[activity_class]
    if timeout is None:
        return policy
    return policy.model_copy(
        update={
            "activity_class": activity_class,
            "start_to_close_seconds": timeout,
            "heartbeat_timeout_seconds": _heartbeat_inside(timeout, policy),
        }
    )


def policy_for_tool(tool: Tool[Any, Any], *, base: ActivityPolicy | None = None) -> ActivityPolicy:
    """The policy for a tool activity, taking the tool's own declared timeout.

    Example:
        >>> from tesserix_adk.tools import tool as as_tool
        >>> @as_tool(timeout=5.0)
        ... def fare(leg: str) -> str:
        ...     '''Price a leg.
        ...
        ...     Args:
        ...         leg: The hop to price.
        ...     '''
        ...     return "40 EUR"
        >>> policy_for_tool(fare).start_to_close_seconds
        5.0
    """
    return activity_policy_for(ActivityClass.TOOL, timeout=tool.timeout, base=base)


def attempts_for_tool(
    tool: Tool[Any, Any], *, keyed: bool = False, base: ActivityPolicy | None = None
) -> int:
    """How many attempts this tool gets, from what it declared about repeating it.

    A tool declared effectful gets one attempt unless the caller supplies an idempotency
    key, and a tool that declared nothing is treated as effectful.
    """
    return policy_for_tool(tool, base=base).attempts_for(tool.idempotency, keyed=keyed)


def _heartbeat_inside(timeout: float, policy: ActivityPolicy) -> float:
    """A heartbeat window that fits in `timeout` without becoming a hair trigger."""
    if policy.heartbeat_timeout_seconds == 0:
        return 0.0
    window = min(max(timeout / 10, MINIMUM_HEARTBEAT_SECONDS), MAXIMUM_HEARTBEAT_SECONDS)
    return min(window, timeout / 2)
