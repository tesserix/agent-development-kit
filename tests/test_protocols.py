import re
from pathlib import Path

import pytest

from tesserix_adk.core import (
    AdkError,
    BudgetPolicy,
    Clock,
    Guardrail,
    MemoryStore,
    ModelProvider,
    ProtocolConformanceError,
    Tracer,
    verify_conformance,
)

PROTOCOLS = [ModelProvider, Guardrail, MemoryStore, Tracer, BudgetPolicy, Clock]


class NotAStore:
    """Missing every member."""


class HalfAStore:
    async def get(self, key: str) -> str | None:
        return None


def test_protocol_conformance_error_is_an_adk_error() -> None:
    assert issubclass(ProtocolConformanceError, AdkError)


def test_verify_conformance_accepts_a_complete_implementation() -> None:
    from tesserix_adk.testing import FakeMemoryStore

    verify_conformance(FakeMemoryStore(), MemoryStore)  # must not raise


def test_verify_conformance_names_every_missing_member() -> None:
    with pytest.raises(ProtocolConformanceError) as exc:
        verify_conformance(HalfAStore(), MemoryStore)

    assert exc.value.missing == ("delete", "put")
    assert exc.value.protocol == "MemoryStore"
    assert "delete" in str(exc.value)
    assert "put" in str(exc.value)


def test_verify_conformance_reports_all_members_when_none_are_present() -> None:
    with pytest.raises(ProtocolConformanceError) as exc:
        verify_conformance(NotAStore(), MemoryStore)

    assert set(exc.value.missing) == {"get", "put", "delete"}


def test_a_non_callable_attribute_does_not_satisfy_a_protocol_method() -> None:
    class Sabotage:
        get = "not callable"
        put = "not callable"
        delete = "not callable"

    with pytest.raises(ProtocolConformanceError) as exc:
        verify_conformance(Sabotage(), MemoryStore)

    assert set(exc.value.missing) == {"get", "put", "delete"}


@pytest.mark.parametrize("protocol", PROTOCOLS, ids=lambda p: p.__name__)
def test_every_protocol_is_runtime_checkable(protocol: type) -> None:
    assert getattr(protocol, "_is_runtime_protocol", False)


@pytest.mark.parametrize("protocol", PROTOCOLS, ids=lambda p: p.__name__)
def test_every_protocol_member_is_documented(protocol: type) -> None:
    from tesserix_adk.core.protocols import members_of

    assert protocol.__doc__, f"{protocol.__name__} has no docstring"
    for member in members_of(protocol):
        assert getattr(protocol, member).__doc__, f"{protocol.__name__}.{member} undocumented"


def test_core_protocols_module_performs_no_io() -> None:
    """A protocol module that imports a transport has already lost substitutability."""
    source = (
        Path(__file__).resolve().parents[1] / "src" / "tesserix_adk" / "core" / "protocols.py"
    ).read_text(encoding="utf-8")
    forbidden = ["import httpx", "import requests", "import socket", "import openai", "open("]
    assert [f for f in forbidden if f in source] == []


def test_docs_table_covers_every_protocol() -> None:
    doc = (Path(__file__).resolve().parents[1] / "docs" / "protocols.md").read_text(
        encoding="utf-8"
    )
    documented = set(re.findall(r"^\| `([A-Za-z]+)` \|", doc, re.MULTILINE))
    assert {p.__name__ for p in PROTOCOLS} <= documented
