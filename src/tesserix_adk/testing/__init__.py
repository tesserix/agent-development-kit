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
from tesserix_adk.testing.guardrails import (
    CORPUS_VERSION,
    GUARD_CORPUS,
    GuardCase,
    GuardFamily,
    GuardMetrics,
    GuardThresholds,
    RecordedGuard,
    assert_allows,
    assert_blocks,
    assert_fails_closed,
    assert_pipeline_order,
    assert_redacts,
    assert_synthetic,
    measure,
    sampled,
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
from tesserix_adk.testing.retrieval import POISONED_CORPUS, FakeIndex, Indexed

if TYPE_CHECKING:
    from tesserix_adk.testing.conformance import (
        BudgetPolicyConformance,
        CheckpointStoreConformance,
        ClockConformance,
        GuardrailConformance,
        IdempotencyStoreConformance,
        KeyValueStoreConformance,
        LeaseStoreConformance,
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

from tesserix_adk.testing.isolation import (
    CONFUSABLE_FIXTURES,
    DEFAULT_TENANTS,
    SENTINEL_KINDS,
    IsolationScenario,
    Leak,
    LeakReport,
    Observed,
    SeededDocument,
    Step,
    Surface,
    TenantFixture,
    assert_no_leak,
    interleaved,
    sentinel_for,
)

__all__ = [
    "CAPABLE",
    "CONFUSABLE_FIXTURES",
    "CORPUS_VERSION",
    "DEFAULT_TENANTS",
    "GUARD_CORPUS",
    "HTTP_CASSETTE_FORMAT",
    "INJECTION_FIXTURES",
    "POISONED_CORPUS",
    "SENTINEL_KINDS",
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
    "GuardCase",
    "GuardFamily",
    "GuardMetrics",
    "GuardThresholds",
    "GuardrailConformance",
    "HttpCassette",
    "HttpExchange",
    "HttpReplay",
    "IdempotencyStoreConformance",
    "InMemoryMemoryStore",
    "Indexed",
    "InjectionFixture",
    "Interaction",
    "IsolationScenario",
    "KeyValueStoreConformance",
    "Leak",
    "LeakReport",
    "LeaseStoreConformance",
    "MemoryStoreConformance",
    "MetricPoint",
    "ModelProviderConformance",
    "NetworkAccessInTestError",
    "Observed",
    "QuarantineError",
    "RecordedError",
    "RecordedEvent",
    "RecordedGuard",
    "RecordingProvider",
    "ReplayingProvider",
    "ScriptedProvider",
    "SearchIndexConformance",
    "SeededDocument",
    "SentRequest",
    "SequentialIds",
    "SpendLedgerConformance",
    "StallingProvider",
    "StateStoreConformance",
    "Step",
    "Surface",
    "TenantFixture",
    "TenantPropagationConformance",
    "ToolExecutionError",
    "TracerConformance",
    "WorkQueueConformance",
    "assert_allows",
    "assert_blocks",
    "assert_fails_closed",
    "assert_no_leak",
    "assert_pipeline_order",
    "assert_redacts",
    "assert_same_run",
    "assert_synthetic",
    "estimate_tokens",
    "interleaved",
    "measure",
    "redacted",
    "sampled",
    "sentinel_for",
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
