"""Evaluation harness, gold sets and quality gates."""

from tesserix_adk.evals.dataset import DATASET_FORMAT, EvalCase, EvalSuite
from tesserix_adk.evals.gate import (
    DEFAULT_POLICY,
    Bypass,
    GatePolicy,
    GateReport,
    Measured,
    MetricMove,
    Tolerance,
    Verdict,
    gate,
)
from tesserix_adk.evals.suite import (
    CaseExecutor,
    CaseResult,
    CaseStatus,
    SuiteResult,
    SuiteRunner,
)

__all__ = [
    "DATASET_FORMAT",
    "DEFAULT_POLICY",
    "Bypass",
    "CaseExecutor",
    "CaseResult",
    "CaseStatus",
    "EvalCase",
    "EvalSuite",
    "GatePolicy",
    "GateReport",
    "Measured",
    "MetricMove",
    "SuiteResult",
    "SuiteRunner",
    "Tolerance",
    "Verdict",
    "gate",
]
