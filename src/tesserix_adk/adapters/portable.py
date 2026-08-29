"""Portable Agent Definition v1 exporters for Registry and DevAI.

The output is a normal Agentic Registry artifact. Runtime location and every registry
dependency are immutable, credential values are references rather than secrets, and the
same contract is emitted for Tesserix ADK and third-party frameworks. Framework objects
are inspected through their public attributes, so exporting one never makes that SDK a
runtime dependency of this package.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any, Literal, Required, Self, TypedDict, Unpack
from urllib.parse import urlsplit

from pydantic import Field, field_validator, model_validator

from tesserix_adk.core.errors import ConfigurationError
from tesserix_adk.core.models import AdkModel

if TYPE_CHECKING:
    from tesserix_adk.core.definition import AgentDefinition
    from tesserix_adk.core.models import OutputT

__all__ = [
    "PortableAgentManifest",
    "PortableExportError",
    "PortableExportOptions",
    "PortableReference",
    "PortableRuntime",
    "container_runtime",
    "export_a2a_agent",
    "export_google_adk_agent",
    "export_langgraph_agent",
    "export_oci_agent",
    "export_openai_agent",
    "export_tesserix_agent",
    "remote_runtime",
]

API_VERSION = "registry.agentic.dev/v1alpha1"
_DNS_LABEL = re.compile(r"^[a-z0-9](?:[-a-z0-9]{0,61}[a-z0-9])?$")
_OCI_DIGEST = re.compile(r"^[^\s@]+@sha256:[a-fA-F0-9]{64}$")
_REFERENCE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]{0,199}$")
_CREDENTIAL_REFERENCE = re.compile(r"^[a-z][a-z0-9+.-]*://[^\s]+$")
_SECRET_KEYS = frozenset({"apikey", "clientsecret", "credentials", "password", "secret", "token"})


class PortableExportError(ConfigurationError):
    """A framework object cannot be exported without inventing mutable metadata."""


class PortableExportOptions(TypedDict, total=False):
    """Shared keyword contract for third-party framework exporters."""

    namespace: Required[str]
    version: Required[str]
    runtime: Required[PortableRuntime]
    name: str
    description: str
    model_provider: str
    include_system_prompt: bool
    tools: Iterable[PortableReference]
    skills: Iterable[PortableReference]
    mcp_servers: Iterable[PortableReference]
    prompts: Iterable[PortableReference]
    visibility: Literal["private", "internal", "public"]


class PortableReference(AdkModel):
    """An immutable Registry dependency reference."""

    ref: str = Field(min_length=1, max_length=200)
    version: str = Field(min_length=1, max_length=200)

    @field_validator("ref")
    @classmethod
    def _valid_reference(cls, value: str) -> str:
        if not _REFERENCE.fullmatch(value):
            raise ValueError("a Registry dependency ref must be a stable artifact name")
        return value

    @field_validator("version")
    @classmethod
    def _immutable_version(cls, value: str) -> str:
        if value.strip().lower() == "latest" or not value.strip():
            raise ValueError("a Registry dependency must pin an immutable version")
        return value

    def rendered(self) -> dict[str, str]:
        """Return the Registry wire shape."""
        return {"ref": self.ref, "version": self.version}


class PortableRuntime(AdkModel):
    """An immutable container or authenticated remote Agent runtime."""

    runtime_type: Literal["container", "remote"]
    protocol: Literal["a2a", "http"] = "a2a"
    image: str = ""
    url: str = ""
    port: int | None = Field(default=None, ge=1, le=65535)
    path: str = ""
    health_path: str = ""
    credential_ref: str = ""

    @field_validator("path", "health_path")
    @classmethod
    def _safe_path(cls, value: str) -> str:
        if value and (
            not value.startswith("/") or value.startswith("//") or ".." in value.split("/")
        ):
            raise ValueError("a runtime path must be absolute and contain no traversal")
        return value

    @model_validator(mode="after")
    def _complete_runtime(self) -> Self:
        if self.runtime_type == "container":
            if not _OCI_DIGEST.fullmatch(self.image):
                raise ValueError("a container image must be pinned by a sha256 digest")
            if self.url or self.credential_ref:
                raise ValueError("a container runtime cannot declare a remote URL or credential")
            return self
        parsed = urlsplit(self.url)
        if parsed.scheme != "https" or not parsed.netloc or parsed.username is not None:
            raise ValueError("a remote runtime URL must be absolute HTTPS without user info")
        if self.image:
            raise ValueError("a remote runtime cannot declare a container image")
        if not _CREDENTIAL_REFERENCE.fullmatch(self.credential_ref):
            raise ValueError("a remote runtime requires a server-managed credential reference")
        return self

    def rendered(self) -> dict[str, object]:
        """Return only the fields belonging to this runtime variant."""
        rendered: dict[str, object] = {
            "type": self.runtime_type,
            "protocol": self.protocol,
        }
        if self.runtime_type == "container":
            rendered["image"] = self.image
        else:
            rendered["url"] = self.url
            rendered["auth"] = {
                "type": "bearer",
                "credentialRef": self.credential_ref,
            }
        if self.port is not None:
            rendered["port"] = self.port
        if self.path:
            rendered["path"] = self.path
        if self.health_path:
            rendered["healthPath"] = self.health_path
        return rendered


def container_runtime(
    *,
    image: str,
    protocol: Literal["a2a", "http"] = "a2a",
    port: int = 8080,
    path: str = "/a2a/v1",
    health_path: str = "/healthz",
) -> PortableRuntime:
    """Declare a digest-pinned OCI runtime."""
    return PortableRuntime(
        runtime_type="container",
        protocol=protocol,
        image=image,
        port=port,
        path=path,
        health_path=health_path,
    )


def remote_runtime(
    *,
    url: str,
    credential_ref: str,
    protocol: Literal["a2a", "http"] = "a2a",
    port: int | None = None,
    path: str = "",
    health_path: str = "",
) -> PortableRuntime:
    """Declare an HTTPS runtime whose bearer secret is resolved by DevAI."""
    return PortableRuntime(
        runtime_type="remote",
        protocol=protocol,
        url=url,
        credential_ref=credential_ref,
        port=port,
        path=path,
        health_path=health_path,
    )


class PortableAgentManifest(AdkModel):
    """One Registry Agent artifact ready for ``agentic validate/apply``."""

    name: str = Field(min_length=1, max_length=200)
    version: str = Field(min_length=1, max_length=200)
    namespace: str = Field(min_length=1, max_length=63)
    framework: str = Field(min_length=1, max_length=100)
    runtime: PortableRuntime
    visibility: Literal["private", "internal", "public"] = "private"
    labels: dict[str, str] = Field(default_factory=dict)
    annotations: dict[str, str] = Field(default_factory=dict)
    spec_fields: dict[str, Any] = Field(default_factory=dict)

    @field_validator("name")
    @classmethod
    def _name_is_stable(cls, value: str) -> str:
        if not _DNS_LABEL.fullmatch(value):
            raise ValueError("a portable agent name must be a lowercase DNS label")
        return value

    @field_validator("namespace")
    @classmethod
    def _namespace_is_stable(cls, value: str) -> str:
        if not _DNS_LABEL.fullmatch(value):
            raise ValueError("a Registry namespace must be a lowercase DNS label")
        return value

    @field_validator("version")
    @classmethod
    def _version_is_immutable(cls, value: str) -> str:
        if value.strip().lower() == "latest":
            raise ValueError("a portable agent must use an immutable version, not latest")
        return value

    def to_dict(self) -> dict[str, object]:
        """Render the exact Registry envelope and Portable Definition v1 spec."""
        metadata: dict[str, object] = {
            "name": self.name,
            "namespace": self.namespace,
            "tag": self.version,
            "visibility": self.visibility,
            "labels": {"framework": self.framework, **self.labels},
        }
        if self.annotations:
            metadata["annotations"] = dict(self.annotations)
        spec = {
            "definitionVersion": "v1",
            "framework": self.framework,
            "runtime": self.runtime.rendered(),
            **self.spec_fields,
        }
        _reject_secret_material(spec)
        return {
            "apiVersion": API_VERSION,
            "kind": "Agent",
            "metadata": metadata,
            "spec": spec,
        }


def export_tesserix_agent(
    definition: AgentDefinition[OutputT],
    *,
    namespace: str,
    runtime: PortableRuntime,
    model_provider: str = "",
    tool_versions: Mapping[str, str] | None = None,
    skills: Iterable[PortableReference] = (),
    mcp_servers: Iterable[PortableReference] = (),
    prompts: Iterable[PortableReference] = (),
    description: str = "",
    include_system_prompt: bool = False,
    visibility: Literal["private", "internal", "public"] = "private",
    labels: Mapping[str, str] | None = None,
) -> PortableAgentManifest:
    """Export a reviewed Tesserix definition without silently moving dependencies."""
    agent = definition.agent
    tool_versions = tool_versions or {}
    declared_tools = set(agent.tools)
    supplied_tools = set(tool_versions)
    if declared_tools != supplied_tools:
        missing = sorted(declared_tools - supplied_tools)
        extra = sorted(supplied_tools - declared_tools)
        detail = []
        if missing:
            detail.append("missing pinned versions for " + ", ".join(missing))
        if extra:
            detail.append("versions supplied for undeclared tools " + ", ".join(extra))
        raise PortableExportError("; ".join(detail))
    if agent.model and not model_provider.strip():
        raise PortableExportError("model_provider is required for a definition with a fixed model")

    fields: dict[str, Any] = {
        "definitionRevision": definition.revision,
        "owner": {"team": definition.owner.team, "service": definition.owner.service},
        "evaluationSuite": definition.evaluation_suite,
    }
    if description.strip():
        fields["description"] = description.strip()
    if agent.model:
        fields["model"] = {"provider": model_provider.strip(), "name": agent.model}
    elif agent.task_class:
        fields["taskClass"] = agent.task_class
    if include_system_prompt:
        fields["systemPrompt"] = agent.instructions
    if definition.output_schema is not None:
        fields["outputSchema"] = definition.output_schema
    if definition.memory_policy is not None:
        fields["memoryPolicy"] = definition.memory_policy
    _add_references(fields, "tools", _references(tool_versions))
    _add_references(fields, "skills", tuple(skills))
    _add_references(fields, "mcpServers", tuple(mcp_servers))
    _add_references(fields, "prompts", tuple(prompts))
    return PortableAgentManifest(
        name=definition.name,
        version=definition.version,
        namespace=namespace,
        framework="tesserix-adk",
        runtime=runtime,
        visibility=visibility,
        labels={"owner": definition.owner.team, **dict(labels or {})},
        annotations={
            "registry.tesserix.dev/definition-revision": definition.revision,
            "registry.tesserix.dev/evaluation-suite": definition.evaluation_suite,
            "registry.tesserix.dev/owner-contact": definition.owner.contact,
            "registry.tesserix.dev/owner-service": definition.owner.service,
        },
        spec_fields=fields,
    )


def export_a2a_agent(
    card: Mapping[str, object] | object,
    *,
    namespace: str,
    runtime: PortableRuntime,
    framework: str = "a2a",
    visibility: Literal["private", "internal", "public"] = "private",
) -> PortableAgentManifest:
    """Export a generic or official A2A card as a runnable Registry Agent."""
    public = _as_mapping(card)
    name = _text(public.get("name") or public.get("agent"), "A2A card name")
    version = _text(public.get("version"), "A2A card version")
    skills = public.get("skills", [])
    if not isinstance(skills, list | tuple):
        raise PortableExportError("A2A card skills must be a list")
    fields: dict[str, Any] = {"a2a": public}
    if skills:
        fields["skills"] = list(skills)
    if description := public.get("description"):
        fields["description"] = str(description)
    return PortableAgentManifest(
        name=name,
        version=version,
        namespace=namespace,
        framework=framework,
        runtime=runtime,
        visibility=visibility,
        spec_fields=fields,
    )


def export_oci_agent(
    *,
    name: str,
    version: str,
    namespace: str,
    image: str,
    protocol: Literal["a2a", "http"] = "a2a",
    port: int = 8080,
    path: str = "/a2a/v1",
    health_path: str = "/healthz",
    visibility: Literal["private", "internal", "public"] = "private",
) -> PortableAgentManifest:
    """Export an unknown framework through its digest-pinned OCI boundary."""
    return PortableAgentManifest(
        name=name,
        version=version,
        namespace=namespace,
        framework="oci",
        runtime=container_runtime(
            image=image,
            protocol=protocol,
            port=port,
            path=path,
            health_path=health_path,
        ),
        visibility=visibility,
    )


def export_langgraph_agent(
    graph: object,
    **kwargs: Unpack[PortableExportOptions],
) -> PortableAgentManifest:
    """Export a compiled LangGraph graph through public attributes only."""
    return _export_foreign(
        graph,
        framework="langgraph",
        instruction_attribute="instructions",
        **kwargs,
    )


def export_google_adk_agent(
    agent: object,
    **kwargs: Unpack[PortableExportOptions],
) -> PortableAgentManifest:
    """Export a Google ADK agent without importing Google ADK."""
    return _export_foreign(
        agent,
        framework="google-adk",
        instruction_attribute="instruction",
        **kwargs,
    )


def export_openai_agent(
    agent: object,
    **kwargs: Unpack[PortableExportOptions],
) -> PortableAgentManifest:
    """Export an OpenAI Agents SDK agent without importing that SDK."""
    return _export_foreign(
        agent,
        framework="openai-agents",
        instruction_attribute="instructions",
        **kwargs,
    )


def _export_foreign(
    value: object,
    *,
    framework: str,
    instruction_attribute: str,
    namespace: str,
    version: str,
    runtime: PortableRuntime,
    name: str = "",
    description: str = "",
    model_provider: str = "",
    include_system_prompt: bool = False,
    tools: Iterable[PortableReference] = (),
    skills: Iterable[PortableReference] = (),
    mcp_servers: Iterable[PortableReference] = (),
    prompts: Iterable[PortableReference] = (),
    visibility: Literal["private", "internal", "public"] = "private",
) -> PortableAgentManifest:
    resolved_name = name or str(getattr(value, "name", ""))
    if not resolved_name:
        raise PortableExportError(f"{framework} agent has no public name; pass name explicitly")
    resolved_description = description or str(getattr(value, "description", ""))
    fields: dict[str, Any] = {}
    if resolved_description.strip():
        fields["description"] = resolved_description.strip()
    model = getattr(value, "model", "")
    model_name = model if isinstance(model, str) else getattr(model, "model", "")
    if model_name:
        fields["model"] = {"provider": model_provider or framework, "name": str(model_name)}
    if include_system_prompt:
        instructions = getattr(value, instruction_attribute, "")
        if not isinstance(instructions, str) or not instructions.strip():
            raise PortableExportError(
                f"{framework} agent has no string {instruction_attribute} to publish"
            )
        fields["systemPrompt"] = instructions
    _add_references(fields, "tools", tuple(tools))
    _add_references(fields, "skills", tuple(skills))
    _add_references(fields, "mcpServers", tuple(mcp_servers))
    _add_references(fields, "prompts", tuple(prompts))
    return PortableAgentManifest(
        name=resolved_name,
        version=version,
        namespace=namespace,
        framework=framework,
        runtime=runtime,
        visibility=visibility,
        spec_fields=fields,
    )


def _references(versions: Mapping[str, str]) -> tuple[PortableReference, ...]:
    return tuple(PortableReference(ref=name, version=version) for name, version in versions.items())


def _add_references(
    fields: dict[str, Any], name: str, references: Iterable[PortableReference]
) -> None:
    rendered = [reference.rendered() for reference in references]
    if rendered:
        fields[name] = rendered


def _as_mapping(value: Mapping[str, Any] | object) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        rendered = dump(mode="json", by_alias=True, exclude_none=True)
        if isinstance(rendered, Mapping):
            return dict(rendered)
    raise PortableExportError("an A2A card must be a mapping or expose model_dump")


def _text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PortableExportError(f"{field} is required")
    return value.strip()


def _reject_secret_material(value: object, path: str = "spec") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = str(key).lower().replace("_", "").replace("-", "")
            if normalized in _SECRET_KEYS:
                raise PortableExportError(
                    f"{path}.{key} looks like secret material; "
                    "publish a credential reference instead"
                )
            _reject_secret_material(child, f"{path}.{key}")
    elif isinstance(value, list | tuple):
        for index, child in enumerate(value):
            _reject_secret_material(child, f"{path}[{index}]")
