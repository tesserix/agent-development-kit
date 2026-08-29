"""Prompts as versioned artifacts, so a run can be tied to the exact text that produced it.

Prompts live as inline f-strings across product repositories, which has two consequences and
both are expensive. Nobody can say which text produced a given run, so a behaviour regression
cannot be traced to the edit that caused it. And the same system prompt gets copied into two
products, which then diverge without anyone deciding that they should.

A prompt here is an artifact: a name, an immutable version, a content digest, the variables it
declares, and the metadata a reviewer needs. Aliases like `current` resolve to a concrete
version at load time, so telemetry records the version rather than a moving label — a run that
recorded `current` says nothing a month later.

The registry is the public surface. This one reads files a project keeps in its own repository;
a database-backed or remote one is an additive implementation of the same protocol.
"""

from __future__ import annotations

import hashlib
import tomllib
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from pydantic import Field

from tesserix_adk.core.errors import PromptNotFoundError, PromptRejectedError
from tesserix_adk.core.models import AdkModel, InputT, OutputT
from tesserix_adk.core.redaction import looks_sensitive

if TYPE_CHECKING:
    from tesserix_adk.core.agent import Agent, TypedAgent

__all__ = [
    "FilePromptRegistry",
    "PromptDefinition",
    "PromptRef",
    "PromptRegistry",
    "prompt_digest",
]

_LABEL = "adk.prompt"
_DIGEST = "adk.prompt_digest"


def _plain(component: str) -> bool:
    """Whether a name may be joined onto a directory at all, rather than walk out of it."""
    return bool(component) and "/" not in component and ".." not in component


def prompt_digest(body: str) -> str:
    """The content address of prompt text.

    Args:
        body: The prompt text, exactly as it will be sent.

    Returns:
        A short digest of the content and nothing else, so two prompts with the same text
        share it. Attribution stays distinct because a reference carries the name as well.

    Example:
        >>> prompt_digest("Greet them.") == prompt_digest("Greet them.")
        True
    """
    return hashlib.blake2b(body.encode("utf-8"), digest_size=16).hexdigest()


class PromptRef(AdkModel):
    """What a run and its spans record about the prompt that produced them.

    The body is deliberately absent. Prompt text goes to the provider, not to a telemetry
    backend where people who were never cleared to read prompts can search it.

    Args:
        name: Which prompt.
        version: Which concrete version of it — never an alias, which moves.
        digest: The content address of the text, so an in-place edit is detectable.
    """

    name: str = Field(min_length=1)
    version: str = Field(min_length=1)
    digest: str = ""

    @property
    def label(self) -> str:
        """The prompt as one string, for a log line or a changelog entry."""
        return f"{self.name}@{self.version}"

    def attributes(self) -> dict[str, str]:
        """The reference as span attributes: the label, the digest, and nothing else."""
        return {_LABEL: self.label, _DIGEST: self.digest}


class PromptDefinition(AdkModel):
    """One published version of one prompt.

    Args:
        name: What the prompt is called. Two prompts with identical text keep separate
            names, because attribution is about which product asked, not which bytes.
        version: The published label. Immutable: an edit is a new version.
        body: The text itself, frozen like everything else here.
        variables: The variables the body declares, for a renderer to check against.
        owner: Who answers for it when its behaviour is questioned.
        task_class: The kind of work it is for, as a hint to routing.
        requires: What it needs of whatever model answers — tool calling, a long window.
    """

    name: str = Field(min_length=1)
    version: str = Field(min_length=1)
    body: str = Field(min_length=1)
    variables: tuple[str, ...] = ()
    owner: str = ""
    task_class: str | None = None
    requires: tuple[str, ...] = ()

    @property
    def digest(self) -> str:
        """The content address of the body."""
        return prompt_digest(self.body)

    @property
    def ref(self) -> PromptRef:
        """What to record about this prompt on a run, a span or a cost row."""
        return PromptRef(name=self.name, version=self.version, digest=self.digest)

    @property
    def estimated_tokens(self) -> int:
        """Roughly what the body costs, for comparing against a model's declared window."""
        return -(-len(self.body) // 4)

    def fits(self, window_tokens: int, *, share: float = 0.5) -> bool:
        """Whether this prompt leaves room to work in.

        Args:
            window_tokens: The target model's declared context window.
            share: How much of the window the prompt may take. Half by default: a prompt
                filling the window leaves nothing to retrieve into or answer with.

        Returns:
            Whether it fits. The estimate is approximate, so this is a flag to act on
            rather than an assertion to build on.
        """
        return self.estimated_tokens <= window_tokens * share

    def instruct(self, agent: Agent[OutputT]) -> Agent[OutputT]:
        """Return `agent` running on this prompt, with the reference recorded on it.

        Args:
            agent: The agent to instruct.

        Returns:
            A copy whose instructions are this body and whose `prompt` names this version,
            so the runtime carries it onto the run and every model-call span without the
            project wiring anything.
        """
        return agent.model_copy(update={"instructions": self.body, "prompt": self.ref})

    def instruct_typed(self, agent: TypedAgent[InputT, OutputT]) -> TypedAgent[InputT, OutputT]:
        """Apply this prompt while preserving a typed agent's input contract."""
        return agent.model_copy(update={"instructions": self.body, "prompt": self.ref})


@runtime_checkable
class PromptRegistry(Protocol):
    """Where prompt text is read from. The protocol is the stable surface."""

    async def get(
        self, name: str, *, version: str = "", alias: str = "", tenant: str = ""
    ) -> PromptDefinition:
        """One concrete version of one prompt.

        Args:
            name: Which prompt.
            version: Which version, where the caller pinned one.
            alias: Which alias to resolve instead, resolved here and never recorded.
            tenant: Whose override to prefer, where a deployment has per-tenant prompts.

        Returns:
            The prompt, always at a concrete version.

        Raises:
            PromptNotFoundError: Where the name, version or alias does not exist. Never an
                empty prompt, a nearest match, or a default.
            PromptRejectedError: Where the stored prompt may not be served.
        """
        ...

    async def list_versions(self, name: str, *, tenant: str = "") -> tuple[str, ...]:
        """Every published version of `name`, in order.

        Args:
            name: Which prompt.
            tenant: Whose override to read, where there is one.

        Returns:
            The versions.

        Raises:
            PromptNotFoundError: Where the prompt does not exist.
        """
        ...


class FilePromptRegistry:
    """Prompts as one TOML file per name, versions and aliases inside it.

    ```toml
    owner = "platform"

    [aliases]
    current = "3"

    [versions.3]
    body = "You greet the customer by name."
    variables = ["customer"]
    ```

    A tenant override lives at `tenants/<tenant>/<name>.toml` and is preferred where it
    exists. The tenant and the name are checked against the directory rather than joined
    onto it: on somebody's deployment both arrive from a request.

    Args:
        directory: Where the files live.
        sealed: Whether a published version is immutable. Under the default, a body that
            changed after it was first read is refused, because a version edited in place
            makes every run that recorded it unattributable. Development that edits prompts
            in a loop can turn it off deliberately.
    """

    def __init__(self, directory: Path | str, *, sealed: bool = True) -> None:
        self._directory = Path(directory).resolve()
        self._sealed = sealed
        self._digests: dict[tuple[str, str, str], str] = {}
        self.files_read = 0

    async def get(
        self, name: str, *, version: str = "", alias: str = "", tenant: str = ""
    ) -> PromptDefinition:
        """Read one prompt, resolving an alias to the version it currently points at."""
        document = self._document(name, tenant=tenant)
        versions = self._versions(document)
        if alias:
            aliases = document.get("aliases", {})
            resolved = str(aliases.get(alias, "")) if isinstance(aliases, dict) else ""
            if not resolved:
                raise PromptNotFoundError(
                    f"the prompt {name!r} has no alias {alias!r}",
                    name=name,
                    available=tuple(versions),
                )
            version = resolved
        if version not in versions:
            raise PromptNotFoundError(
                f"the prompt {name!r} has no version {version!r}",
                name=name,
                available=tuple(versions),
            )
        return self._defined(name, version, document, versions[version], tenant=tenant)

    async def list_versions(self, name: str, *, tenant: str = "") -> tuple[str, ...]:
        """Every published version of one prompt, in the order the file declares."""
        return tuple(self._versions(self._document(name, tenant=tenant)))

    def _document(self, name: str, *, tenant: str) -> dict[str, Any]:
        """The file for one prompt, preferring the tenant's own copy."""
        for path, expected in self._candidates(name, tenant=tenant):
            if path.parent == expected and path.is_file():
                self.files_read += 1
                try:
                    return tomllib.loads(path.read_text(encoding="utf-8"))
                except tomllib.TOMLDecodeError as unreadable:
                    raise PromptRejectedError(
                        f"the prompt {name!r} could not be read: {unreadable}", name=name
                    ) from unreadable
        raise PromptNotFoundError(f"there is no prompt named {name!r}", name=name)

    def _candidates(self, name: str, *, tenant: str) -> tuple[tuple[Path, Path], ...]:
        """Where to look and where each must be, tenant override first.

        A name or a tenant that is not a plain component has nowhere to look at all, and
        each candidate carries the directory it must sit directly in.
        """
        if not _plain(name):
            return ()
        shared = ((self._directory / f"{name}.toml").resolve(), self._directory)
        if not tenant:
            return (shared,)
        if not _plain(tenant):
            return (shared,)
        own = self._directory / "tenants" / tenant
        return ((own / f"{name}.toml").resolve(), own), shared

    def _versions(self, document: dict[str, Any]) -> dict[str, dict[str, Any]]:
        """The versions block, or nothing where the file declares none."""
        versions = document.get("versions", {})
        return versions if isinstance(versions, dict) else {}

    def _defined(
        self,
        name: str,
        version: str,
        document: dict[str, Any],
        published: dict[str, Any],
        *,
        tenant: str,
    ) -> PromptDefinition:
        """Validate one published version, and refuse to serve what should not be served."""
        body = str(published.get("body", ""))
        if not body:
            raise PromptRejectedError(f"the prompt {name}@{version} has no body", name=name)
        if looks_sensitive(body):
            raise PromptRejectedError(
                f"the prompt {name}@{version} contains something shaped like a secret",
                name=name,
            )
        self._unchanged(name, version, body, tenant=tenant)
        return PromptDefinition(
            name=name,
            version=version,
            body=body,
            variables=tuple(published.get("variables", ())),
            owner=str(document.get("owner", "")),
            task_class=document.get("task_class"),
            requires=tuple(document.get("requires", ())),
        )

    def _unchanged(self, name: str, version: str, body: str, *, tenant: str) -> None:
        """Refuse a published version whose text moved under it."""
        digest = prompt_digest(body)
        seen = self._digests.setdefault((tenant, name, version), digest)
        if self._sealed and seen != digest:
            raise PromptRejectedError(
                f"the published prompt {name}@{version} was edited in place: "
                f"{seen} became {digest}. Publish a new version instead.",
                name=name,
            )
        self._digests[(tenant, name, version)] = digest
