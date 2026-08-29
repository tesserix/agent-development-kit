from __future__ import annotations

from dataclasses import dataclass

import pytest

from tesserix_adk.adapters.portable import (
    PortableExportError,
    PortableReference,
    container_runtime,
    export_a2a_agent,
    export_google_adk_agent,
    export_langgraph_agent,
    export_oci_agent,
    export_openai_agent,
    export_tesserix_agent,
    remote_runtime,
)
from tesserix_adk.core import Agent, AgentDefinition, Owner

DIGEST = "a" * 64


def definition() -> AgentDefinition:
    return AgentDefinition(
        agent=Agent(
            name="support-agent",
            version="1.2.0",
            instructions="Resolve the support request.",
            model="gpt-5-mini",
            tools=("ticket-search",),
            free_text=True,
        ),
        owner=Owner(team="support", contact="support@example.com", service="support-agent"),
        evaluation_suite="evals/support.yaml",
    )


def test_tesserix_export_is_a_registry_portable_agent() -> None:
    manifest = export_tesserix_agent(
        definition(),
        namespace="acme-ai",
        runtime=container_runtime(
            image=f"ghcr.io/acme/support@sha256:{DIGEST}",
            port=8080,
            path="/a2a/v1",
            health_path="/readyz",
        ),
        model_provider="openai",
        tool_versions={"ticket-search": "2.1.0"},
        skills=(PortableReference(ref="triage", version="1.0.0"),),
    ).to_dict()

    assert manifest["apiVersion"] == "registry.agentic.dev/v1alpha1"
    assert manifest["kind"] == "Agent"
    assert manifest["metadata"] == {
        "name": "support-agent",
        "namespace": "acme-ai",
        "tag": "1.2.0",
        "visibility": "private",
        "labels": {"framework": "tesserix-adk", "owner": "support"},
        "annotations": {
            "registry.tesserix.dev/definition-revision": definition().revision,
            "registry.tesserix.dev/evaluation-suite": "evals/support.yaml",
            "registry.tesserix.dev/owner-contact": "support@example.com",
            "registry.tesserix.dev/owner-service": "support-agent",
        },
    }
    spec = object_dict(manifest["spec"])
    assert spec["definitionVersion"] == "v1"
    assert spec["framework"] == "tesserix-adk"
    assert str(object_dict(spec["runtime"])["image"]).endswith(DIGEST)
    assert spec["tools"] == [{"ref": "ticket-search", "version": "2.1.0"}]
    assert spec["skills"] == [{"ref": "triage", "version": "1.0.0"}]
    assert "systemPrompt" not in spec


def test_tesserix_export_refuses_moving_or_missing_dependencies() -> None:
    with pytest.raises(PortableExportError, match="ticket-search"):
        export_tesserix_agent(
            definition(),
            namespace="acme-ai",
            runtime=container_runtime(image=f"ghcr.io/acme/support@sha256:{DIGEST}"),
            model_provider="openai",
        )
    with pytest.raises(ValueError, match="immutable"):
        PortableReference(ref="ticket-search", version="latest")


def test_runtime_contract_rejects_mutable_images_and_inline_remote_tokens() -> None:
    with pytest.raises(ValueError, match="sha256"):
        container_runtime(image="ghcr.io/acme/support:latest")
    with pytest.raises(ValueError, match="HTTPS"):
        remote_runtime(url="http://agents.example.com/a2a", credential_ref="openbao://agents/a")
    with pytest.raises(TypeError):
        remote_runtime(  # type: ignore[call-arg]
            url="https://agents.example.com/a2a",
            credential_ref="openbao://agents/a",
            token=object(),
        )


def test_generic_a2a_export_keeps_public_card_and_authenticated_runtime() -> None:
    runtime_url = "https://agents.example.com/support/a2a"
    card = {
        "name": "remote-support",
        "version": "3.0.0",
        "description": "Resolves incidents",
        "url": runtime_url,
        "skills": [{"id": "triage", "name": "Triage", "description": "Triage an alert"}],
    }
    manifest = export_a2a_agent(
        card,
        namespace="acme-ai",
        runtime=remote_runtime(
            url=runtime_url, credential_ref="openbao://agent-runtime/acme-support"
        ),
    ).to_dict()

    metadata = object_dict(manifest["metadata"])
    spec = object_dict(manifest["spec"])
    runtime = object_dict(spec["runtime"])
    skills = spec["skills"]
    assert isinstance(skills, list)
    assert metadata["name"] == "remote-support"
    assert spec["framework"] == "a2a"
    assert object_dict(spec["a2a"])["url"] == runtime_url
    assert runtime["auth"] == {
        "type": "bearer",
        "credentialRef": "openbao://agent-runtime/acme-support",
    }
    assert object_dict(skills[0])["id"] == "triage"


@dataclass
class ForeignAgent:
    name: str
    description: str
    instruction: str = ""
    instructions: str = ""
    model: str = ""


def object_dict(value: object) -> dict[str, object]:
    assert isinstance(value, dict)
    assert all(isinstance(key, str) for key in value)
    return value


@pytest.mark.parametrize(
    ("exporter", "framework", "foreign"),
    [
        (export_langgraph_agent, "langgraph", ForeignAgent("graph-agent", "Graph agent")),
        (export_google_adk_agent, "google-adk", ForeignAgent("google-agent", "Google agent")),
        (export_openai_agent, "openai-agents", ForeignAgent("openai-agent", "OpenAI agent")),
    ],
)
def test_foreign_framework_adapters_need_no_framework_import(
    exporter: object, framework: str, foreign: ForeignAgent
) -> None:
    manifest = exporter(  # type: ignore[operator]
        foreign,
        namespace="acme-ai",
        version="1.0.0",
        runtime=container_runtime(image=f"ghcr.io/acme/{foreign.name}@sha256:{DIGEST}"),
    ).to_dict()
    assert manifest["spec"]["framework"] == framework
    assert manifest["metadata"]["name"] == foreign.name


def test_oci_fallback_still_emits_a_complete_portable_contract() -> None:
    manifest = export_oci_agent(
        name="custom-agent",
        version="1.0.0",
        namespace="acme-ai",
        image=f"ghcr.io/acme/custom@sha256:{DIGEST}",
        protocol="http",
        path="/invoke",
    ).to_dict()
    assert manifest["spec"] == {
        "definitionVersion": "v1",
        "framework": "oci",
        "runtime": {
            "type": "container",
            "protocol": "http",
            "image": f"ghcr.io/acme/custom@sha256:{DIGEST}",
            "port": 8080,
            "path": "/invoke",
            "healthPath": "/healthz",
        },
    }
