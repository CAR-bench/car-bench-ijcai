"""Public API for the domain-agnostic PACT planning and execution kernel.

The primary surface is the typed plan IR, live-contract verifier, immutable
evidence ledger, semantic compiler, and deterministic obligation runtime.
Legacy rule-controller symbols remain lazily importable in a source checkout
for older development tests, but they are intentionally not part of the PACT
API or the final Track 2 image.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

from .evidence_ledger import (
    ActionAuthorization,
    ConfirmationEvent,
    EventDisposition,
    EvidenceLedger,
    EvidenceRecord,
    EvidenceSource,
    EvidenceStatus,
    ExternalResultEvent,
    ExternalResultStatus,
    InputEvent,
)
from .obligation_runtime import (
    AskAction,
    ConfirmAction,
    EmissionRejectedError,
    ExternalCallAction,
    NodeExecutionStatus,
    ObligationRuntime,
    PlanRunStatus,
    RespondAction,
    RuntimeAction,
)
from .plan_ir import (
    ActNode,
    AskNode,
    ConfirmNode,
    ObserveNode,
    PlanIR,
    PlanNode,
    RespondNode,
    ResponseOutcome,
)
from .plan_verifier import (
    CapabilityDescriptor,
    CapabilityEffect,
    CapabilitySnapshot,
    PlanVerificationIssue,
    PlanVerificationReport,
    PlanVerifier,
    VerificationPolicy,
)
from .semantic_compiler import (
    CompilationAttempt,
    CompilationRequest,
    CompilationResult,
    CompilerBackendError,
    CompilerInputError,
    CompilerTokenUsage,
    PlanRepairExhaustedError,
    SemanticCompilationError,
    SemanticCompiler,
    SemanticCompilerLimits,
)
from .tool_index import ToolIndex, ValidationIssue


__all__ = [
    "ActionAuthorization",
    "ActNode",
    "AskAction",
    "AskNode",
    "CapabilityDescriptor",
    "CapabilityEffect",
    "CapabilitySnapshot",
    "CompilationAttempt",
    "CompilationRequest",
    "CompilationResult",
    "CompilerBackendError",
    "CompilerInputError",
    "CompilerTokenUsage",
    "ConfirmAction",
    "ConfirmationEvent",
    "ConfirmNode",
    "EmissionRejectedError",
    "EventDisposition",
    "EvidenceLedger",
    "EvidenceRecord",
    "EvidenceSource",
    "EvidenceStatus",
    "ExternalCallAction",
    "ExternalResultEvent",
    "ExternalResultStatus",
    "InputEvent",
    "NodeExecutionStatus",
    "ObligationRuntime",
    "ObserveNode",
    "PlanIR",
    "PlanNode",
    "PlanRepairExhaustedError",
    "PlanRunStatus",
    "PlanVerificationIssue",
    "PlanVerificationReport",
    "PlanVerifier",
    "RespondAction",
    "RespondNode",
    "ResponseOutcome",
    "RuntimeAction",
    "SemanticCompilationError",
    "SemanticCompiler",
    "SemanticCompilerLimits",
    "ToolIndex",
    "ValidationIssue",
    "VerificationPolicy",
]


def __getattr__(name: str) -> Any:
    """Load pre-PACT controller symbols only when older code requests them."""

    if name == "NextAction":
        try:
            symbol = import_module(".actions", __name__).NextAction
        except ModuleNotFoundError as exc:  # absent from the minimal image
            raise AttributeError(
                "NextAction belongs to the legacy controller, which is not "
                "included in the PACT runtime image"
            ) from exc
        globals()[name] = symbol
        return symbol

    if name == "PolicyAwareController":
        try:
            symbol = import_module(".controller", __name__).PolicyAwareController
        except ModuleNotFoundError as exc:  # absent from the minimal image
            raise AttributeError(
                "PolicyAwareController is not included in the PACT runtime image"
            ) from exc
        globals()[name] = symbol
        return symbol

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Expose only the supported PACT API during introspection."""

    return sorted(set(globals()) | set(__all__))
