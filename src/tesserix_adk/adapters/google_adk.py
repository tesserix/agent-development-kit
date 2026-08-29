"""Explicit interoperability with Google's Agent Development Kit over official A2A."""

from __future__ import annotations

from typing import TYPE_CHECKING

from tesserix_adk.core.extras import require_extra

if TYPE_CHECKING:
    from a2a.types import AgentCard
    from google.adk.agents.remote_a2a_agent import RemoteA2aAgent

__all__ = ["google_adk_remote_agent"]


def google_adk_remote_agent(
    *,
    name: str,
    agent_card: AgentCard | str,
    description: str = "",
    timeout_seconds: float = 60.0,
) -> RemoteA2aAgent:
    """Create a Google ADK remote agent for a Tesserix official A2A endpoint.

    The helper selects Google ADK's current A2A 1.x implementation instead of its legacy
    compatibility path. Authentication still belongs in Google ADK's credential or A2A
    client configuration; this function never accepts or stores a token.

    Raises:
        ValueError: If ``timeout_seconds`` is not positive.
        MissingExtraError: If ``tesserix-adk[google-adk]`` is not installed.
    """
    require_extra("google-adk", "google.adk")
    if timeout_seconds <= 0:
        raise ValueError("a Google ADK remote-agent timeout must be positive")

    from google.adk.agents.remote_a2a_agent import RemoteA2aAgent

    return RemoteA2aAgent(
        name=name,
        description=description,
        agent_card=agent_card,
        timeout=timeout_seconds,
        use_legacy=False,
    )
