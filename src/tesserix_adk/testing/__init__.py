"""Fakes and fixtures published for consumers to test against."""

from importlib import import_module
from typing import TYPE_CHECKING

from tesserix_adk.testing.cassette import (
    Cassette,
    CassetteMissError,
    CassetteVersionError,
    Interaction,
    RecordedError,
    RecordingProvider,
    ReplayingProvider,
    assert_same_run,
    redacted,
)
from tesserix_adk.testing.embedding import FakeEmbedder
from tesserix_adk.testing.fakes import (
    CAPABLE,
    BudgetExceededError,
    FakeBudgetPolicy,
    FakeClock,
    FakeGuardrail,
    FakeKeyValueStore,
    FakeMeter,
    FakeSecrets,
    FakeTenantLedger,
    FakeToolRegistry,
    FakeTracer,
    MetricPoint,
    RecordedEvent,
    ScriptedProvider,
    SequentialIds,
    StallingProvider,
    ToolExecutionError,
    estimate_tokens,
)
from tesserix_adk.testing.http_cassette import (
    HTTP_CASSETTE_FORMAT,
    HttpCassette,
    HttpExchange,
    HttpReplay,
    SentRequest,
)
from tesserix_adk.testing.injection import INJECTION_FIXTURES, InjectionFixture
from tesserix_adk.testing.memory import InMemoryMemoryStore
from tesserix_adk.testing.retrieval import FakeIndex, Indexed

if TYPE_CHECKING:
    from tesserix_adk.testing.conformance import (
        BudgetPolicyConformance,
        CheckpointStoreConformance,
        ClockConformance,
        IdempotencyStoreConformance,
        KeyValueStoreConformance,
        MemoryStoreConformance,
        ModelProviderConformance,
        SearchIndexConformance,
        SpendLedgerConformance,
        StateStoreConformance,
        TenantPropagationConformance,
        TracerConformance,
        WorkQueueConformance,
    )
    from tesserix_adk.testing.pytest_plugin import NetworkAccessInTestError, QuarantineError

__all__ = [
    "CAPABLE",
    "HTTP_CASSETTE_FORMAT",
    "INJECTION_FIXTURES",
    "BudgetExceededError",
    "BudgetPolicyConformance",
    "Cassette",
    "CassetteMissError",
    "CassetteVersionError",
    "CheckpointStoreConformance",
    "ClockConformance",
    "FakeBudgetPolicy",
    "FakeClock",
    "FakeEmbedder",
    "FakeGuardrail",
    "FakeIndex",
    "FakeKeyValueStore",
    "FakeMeter",
    "FakeSecrets",
    "FakeTenantLedger",
    "FakeToolRegistry",
    "FakeTracer",
    "HttpCassette",
    "HttpExchange",
    "HttpReplay",
    "IdempotencyStoreConformance",
    "InMemoryMemoryStore",
    "Indexed",
    "InjectionFixture",
    "Interaction",
    "KeyValueStoreConformance",
    "MemoryStoreConformance",
    "MetricPoint",
    "ModelProviderConformance",
    "NetworkAccessInTestError",
    "QuarantineError",
    "RecordedError",
    "RecordedEvent",
    "RecordingProvider",
    "ReplayingProvider",
    "ScriptedProvider",
    "SearchIndexConformance",
    "SentRequest",
    "SequentialIds",
    "SpendLedgerConformance",
    "StallingProvider",
    "StateStoreConformance",
    "TenantPropagationConformance",
    "ToolExecutionError",
    "TracerConformance",
    "WorkQueueConformance",
    "assert_same_run",
    "estimate_tokens",
    "redacted",
]

# These modules import pytest at module scope, and pytest is a test-time dependency nobody
# installing the wheel is given — importing it here put it on the path of every consumer
# who only wanted a fake.
_DEFERRED = {
    **{name: "conformance" for name in __all__ if name.endswith("Conformance")},
    "NetworkAccessInTestError": "pytest_plugin",
    "QuarantineError": "pytest_plugin",
}


def __getattr__(name: str) -> object:
    """Load a pytest-dependent name on first use, so the import costs pytest only then."""
    module = _DEFERRED.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(import_module(f"{__name__}.{module}"), name)
