"""Where a provider's key comes from, and when it is read.

Two rules, and the second is the one that bites. A key comes from the environment or from
an injected secret provider, never from a config file or a table: a repository is where
secrets are found by whoever clones it, and a database read at start-up is one an operator
cannot rotate. And a key is read at the moment it is used, never captured at construction,
because a process holding a copy of a revoked key keeps presenting it until it restarts.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from tesserix_adk.core.errors import ConfigurationError

if TYPE_CHECKING:
    from tesserix_adk.core.protocols import SecretProvider

__all__ = ["Credential", "EnvironmentSecrets"]


class EnvironmentSecrets:
    """The default secret provider: the process environment, read on every request.

    A blank value is treated as absent. A deployment that exported an empty variable
    believes it is configured, and the request it sends says otherwise on the vendor's
    terms rather than the kit's.
    """

    def secret(self, name: str) -> str | None:
        """Return the value of `name`, or `None` where it is unset or blank."""
        value = os.environ.get(name, "")
        return value if value.strip() else None


class Credential:
    """One provider key, resolved on each use.

    Args:
        variable: The environment variable, which is also the name asked of an injected
            provider. It is not the secret, so it appears in errors and in `repr`.
        secrets: Where to look. Defaults to the environment.
    """

    def __init__(self, variable: str, *, secrets: SecretProvider | None = None) -> None:
        self.variable = variable
        self._secrets: SecretProvider = secrets or EnvironmentSecrets()

    def __repr__(self) -> str:
        """Names the variable and never its value: a key in a traceback is a leaked key."""
        return f"Credential({self.variable!r})"

    def value(self) -> str:
        """Return the key as it stands now.

        Raises:
            ConfigurationError: If nothing answers for `variable`. Raised before the
                request, because an unauthenticated call is a 401 that reads like a
                vendor problem.
        """
        found = self._secrets.secret(self.variable)
        if found is None or not found.strip():
            raise ConfigurationError(
                f"{self.variable} is not set. Provider keys come from the environment or "
                f"an injected secret provider, never from a config file in the repository."
            )
        return found
