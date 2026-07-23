"""Static verification against a revocable, live capability contract.

The language model is allowed to propose a :class:`PlanIR`, but it is not
trusted to decide whether an external operation exists or whether its
arguments satisfy the evaluator-provided JSON Schema.  This module takes that
decision out of the model path: it snapshots the current capability surface
and returns a structured, value-redacted verification report.

No operation is classified by its name.  Callers may explicitly mark a
capability as requiring confirmation when they construct the snapshot, or may
advertise the generic ``x-pact-requires-confirmation`` function extension.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from collections.abc import Collection
from typing import Any, Iterable, Literal, Mapping

from jsonschema import Draft202012Validator  # type: ignore[import-untyped]
from jsonschema.exceptions import SchemaError  # type: ignore[import-untyped]
from pydantic import ValidationError as PydanticValidationError

from .evidence_ledger import EvidenceRecord, EvidenceSource, EvidenceStatus
from .plan_ir import (
    ActNode,
    AskNode,
    ConfirmNode,
    ObserveNode,
    PlanIR,
    ResponseOutcome,
)
from .tool_index import ToolIndex


IssueStage = Literal["capability", "ir", "bounds", "schema", "safety", "evidence"]
IssueSeverity = Literal["error", "warning"]
CapabilityEffect = Literal["unknown", "observe", "act"]
REQUIRES_CONFIRMATION_MARKER = "REQUIRES_CONFIRMATION"


def description_requires_confirmation(description: str) -> bool:
    """Recognize the evaluator's anchored confirmation contract marker.

    Only a marker at the beginning of a trusted function description counts.
    User text and model output never flow through this parser.  The boundary
    check avoids treating a longer identifier that merely shares the prefix as
    a confirmation requirement.
    """

    normalized = description.lstrip()
    if not normalized.startswith(REQUIRES_CONFIRMATION_MARKER):
        return False
    suffix = normalized[len(REQUIRES_CONFIRMATION_MARKER) :]
    return not suffix or suffix[0] in {",", ":", " ", "\t", "\r", "\n"}


@dataclass(frozen=True)
class PlanVerificationIssue:
    """Machine-readable reason that a plan cannot be executed safely.

    Argument values are intentionally omitted.  ``instance_path`` and
    ``schema_path`` identify only structural locations, making reports safe to
    retain for diagnostics and bounded compiler repair.
    """

    code: str
    stage: IssueStage
    message: str
    severity: IssueSeverity = "error"
    node_id: str | None = None
    operation: str | None = None
    instance_path: tuple[str | int, ...] = ()
    schema_path: tuple[str | int, ...] = ()
    constraint: str | None = None


@dataclass(frozen=True)
class CapabilityDescriptor:
    """Immutable description of one callable operation contract."""

    operation: str
    description: str
    schema_digest: str
    schema_json: str
    requires_confirmation: bool = False
    effect: CapabilityEffect = "unknown"

    @property
    def parameter_schema(self) -> dict[str, Any]:
        """Return a fresh copy of the parameter schema."""

        schema = json.loads(self.schema_json)
        if not isinstance(schema, dict):  # pragma: no cover - guarded at build time
            raise TypeError("capability parameter schema must be an object")
        return schema


@dataclass(frozen=True)
class CapabilitySnapshot:
    """Content-addressed view of the capabilities available for one turn.

    A snapshot is deliberately rebuilt whenever the incoming tool list
    changes.  Therefore a capability that disappears is revoked immediately;
    a verifier never falls back to a previously observed schema.
    """

    digest: str
    capabilities: tuple[CapabilityDescriptor, ...]
    issues: tuple[PlanVerificationIssue, ...] = ()

    @classmethod
    def from_tools(
        cls,
        tools: Iterable[Mapping[str, Any]],
        *,
        critical_operations: Iterable[str] = (),
        operation_effects: Mapping[str, CapabilityEffect] | None = None,
    ) -> "CapabilitySnapshot":
        """Build a deterministic snapshot from live function-tool schemas.

        Invalid and duplicate capability declarations are not silently
        accepted.  They remain visible as snapshot issues and make every plan
        verification fail closed.
        """

        critical = frozenset(critical_operations)
        trusted_effects = operation_effects or {}
        candidates: dict[
            str,
            list[tuple[dict[str, Any], bool, CapabilityEffect]],
        ] = {}
        issues: list[PlanVerificationIssue] = []

        for position, raw_tool in enumerate(tools):
            tool = _json_object_copy(raw_tool)
            if tool is None:
                issues.append(
                    PlanVerificationIssue(
                        code="malformed_capability",
                        stage="capability",
                        message=f"Capability declaration at position {position} is invalid.",
                    )
                )
                continue

            if tool.get("type") != "function":
                issues.append(
                    PlanVerificationIssue(
                        code="malformed_capability",
                        stage="capability",
                        message=(
                            f"Capability declaration at position {position} is not "
                            "a function tool."
                        ),
                    )
                )
                continue

            function = tool.get("function")
            if not isinstance(function, dict):
                issues.append(
                    PlanVerificationIssue(
                        code="malformed_capability",
                        stage="capability",
                        message=f"Capability declaration at position {position} has no function contract.",
                    )
                )
                continue

            operation = function.get("name")
            if not isinstance(operation, str) or not operation:
                issues.append(
                    PlanVerificationIssue(
                        code="malformed_capability",
                        stage="capability",
                        message=f"Capability declaration at position {position} has no operation name.",
                    )
                )
                continue

            if "parameters" not in function:
                issues.append(
                    PlanVerificationIssue(
                        code="invalid_capability_schema",
                        stage="capability",
                        message="Capability declaration has no parameter schema.",
                        operation=operation,
                        constraint="schema",
                    )
                )
                continue

            parameters = function["parameters"]
            if not isinstance(parameters, dict):
                issues.append(
                    PlanVerificationIssue(
                        code="invalid_capability_schema",
                        stage="capability",
                        message="Capability parameters must be a JSON Schema object.",
                        operation=operation,
                        constraint="schema",
                    )
                )
                continue

            description = function.get("description", "")
            if not isinstance(description, str):
                issues.append(
                    PlanVerificationIssue(
                        code="malformed_capability",
                        stage="capability",
                        message="Capability description must be a string.",
                        operation=operation,
                    )
                )
                continue

            try:
                Draft202012Validator.check_schema(parameters)
                schema_json = _canonical_json(parameters)
            except (SchemaError, TypeError, ValueError):
                issues.append(
                    PlanVerificationIssue(
                        code="invalid_capability_schema",
                        stage="capability",
                        message="Capability parameters are not a valid Draft 2020-12 schema.",
                        operation=operation,
                        constraint="schema",
                    )
                )
                continue

            extension = function.get("x-pact-requires-confirmation", False)
            requires_confirmation = (
                operation in critical
                or extension is True
                or description_requires_confirmation(description)
            )
            raw_effect = trusted_effects.get(
                operation,
                function.get("x-pact-effect", "unknown"),
            )
            if raw_effect not in {"unknown", "observe", "act"}:
                issues.append(
                    PlanVerificationIssue(
                        code="invalid_capability_effect",
                        stage="capability",
                        message="Capability effect classification is invalid.",
                        operation=operation,
                    )
                )
                continue
            effect: CapabilityEffect = raw_effect
            if requires_confirmation and effect == "observe":
                issues.append(
                    PlanVerificationIssue(
                        code="conflicting_capability_effect",
                        stage="capability",
                        message=(
                            "A confirmation-required capability cannot be "
                            "classified as observation-only."
                        ),
                        operation=operation,
                    )
                )
                continue
            if requires_confirmation:
                effect = "act"
            normalized_tool = {
                "type": "function",
                "function": {
                    "name": operation,
                    "description": description,
                    "parameters": json.loads(schema_json),
                },
            }
            candidates.setdefault(operation, []).append(
                (normalized_tool, requires_confirmation, effect)
            )

        descriptors: list[CapabilityDescriptor] = []
        normalized_contracts: list[dict[str, Any]] = []
        for operation in sorted(candidates):
            declarations = candidates[operation]
            if len(declarations) != 1:
                issues.append(
                    PlanVerificationIssue(
                        code="duplicate_capability",
                        stage="capability",
                        message="A callable operation must have exactly one live contract.",
                        operation=operation,
                    )
                )
                continue

            tool, requires_confirmation, effect = declarations[0]
            parameters = tool["function"]["parameters"]
            schema_json = _canonical_json(parameters)
            schema_digest = _sha256(schema_json)
            descriptors.append(
                CapabilityDescriptor(
                    operation=operation,
                    description=tool["function"]["description"],
                    schema_digest=schema_digest,
                    schema_json=schema_json,
                    requires_confirmation=requires_confirmation,
                    effect=effect,
                )
            )
            normalized_contracts.append(
                {
                    "operation": operation,
                    "description": tool["function"]["description"],
                    "parameters": parameters,
                    "requires_confirmation": requires_confirmation,
                    "effect": effect,
                }
            )

        digest = _sha256(_canonical_json(normalized_contracts))
        return cls(
            digest=digest,
            capabilities=tuple(descriptors),
            issues=tuple(issues),
        )

    @property
    def names(self) -> frozenset[str]:
        return frozenset(capability.operation for capability in self.capabilities)

    @property
    def is_valid(self) -> bool:
        return not self.issues

    def get(self, operation: str) -> CapabilityDescriptor | None:
        return next(
            (
                capability
                for capability in self.capabilities
                if capability.operation == operation
            ),
            None,
        )

    def tool_index(self) -> ToolIndex:
        """Construct the trusted Draft 2020-12 guard for this snapshot."""

        tools = [
            {
                "type": "function",
                "function": {
                    "name": capability.operation,
                    "description": capability.description,
                    "parameters": capability.parameter_schema,
                },
            }
            for capability in self.capabilities
        ]
        return ToolIndex(tools)


@dataclass(frozen=True)
class VerificationPolicy:
    """Domain-independent resource and safety bounds for compiled plans."""

    max_nodes: int = 32
    max_dependency_depth: int = 12
    # A graph edge is only a sequencing relation, not by itself proof of a
    # semantic precondition.  Keep this optional until PlanIR can cite trusted
    # goal/evidence predicates explicitly.
    require_action_precondition: bool = False
    require_all_action_evidence: bool = True

    def __post_init__(self) -> None:
        if self.max_nodes < 1:
            raise ValueError("max_nodes must be positive")
        if self.max_dependency_depth < 1:
            raise ValueError("max_dependency_depth must be positive")


@dataclass(frozen=True)
class PlanVerificationReport:
    """Fail-closed result of validating one candidate plan."""

    capability_digest: str
    plan_id: str | None
    plan: PlanIR | None
    issues: tuple[PlanVerificationIssue, ...]

    @property
    def accepted(self) -> bool:
        return self.plan is not None and not any(
            issue.severity == "error" for issue in self.issues
        )


@dataclass(frozen=True)
class EvidenceAvailability:
    """Trusted evidence view supplied by the deterministic runtime.

    ``successful_external_keys`` contains ledger records created only after a
    successful external result.  Raw keys are accepted separately by
    :meth:`PlanVerifier.verify` so adapter code cannot accidentally treat
    model-authored strings as evidence records.
    """

    record_keys: frozenset[str]
    successful_external_keys: frozenset[str]
    successful_action_keys: frozenset[str]
    false_confirmation_keys: frozenset[str]
    true_confirmation_records: tuple[EvidenceRecord, ...]
    failure_keys: frozenset[str]
    trusted_external_keys: frozenset[str]
    issues: tuple[PlanVerificationIssue, ...] = ()

    @property
    def keys(self) -> frozenset[str]:
        """All record-backed and explicitly trusted evidence identifiers."""

        return self.record_keys | self.trusted_external_keys

    @classmethod
    def from_records(
        cls,
        records: Iterable[EvidenceRecord],
        *,
        trusted_external_keys: Iterable[str] = (),
    ) -> "EvidenceAvailability":
        by_key: dict[str, EvidenceRecord] = {}
        record_keys: set[str] = set()
        external: set[str] = set()
        actions: set[str] = set()
        false_confirmations: set[str] = set()
        failures: set[str] = set()
        conflicts: set[str] = set()
        issues: list[PlanVerificationIssue] = []

        for record in records:
            existing = by_key.get(record.key)
            if existing is not None and existing != record:
                conflicts.add(record.key)
                external.discard(record.key)
                actions.discard(record.key)
                false_confirmations.discard(record.key)
                failures.discard(record.key)
                by_key.pop(record.key, None)
                issues.append(
                    PlanVerificationIssue(
                        code="conflicting_evidence_provenance",
                        stage="evidence",
                        message="An evidence key has conflicting provenance records.",
                    )
                )
                continue
            if record.key in conflicts:
                continue
            by_key[record.key] = record
            record_keys.add(record.key)
            if (
                record.source == EvidenceSource.EXTERNAL
                and record.status == EvidenceStatus.SUCCESS
            ):
                external.add(record.key)
                if record.producer_kind == "act":
                    actions.add(record.key)
            elif (
                record.source == EvidenceSource.INPUT
                and record.status == EvidenceStatus.FAILURE
                and record.producer_kind == "confirm"
                and record.value is False
            ):
                false_confirmations.add(record.key)
            if record.status == EvidenceStatus.FAILURE:
                failures.add(record.key)

        trusted: set[str] = set()
        for key in trusted_external_keys:
            if not isinstance(key, str) or not key.strip():
                issues.append(
                    PlanVerificationIssue(
                        code="invalid_trusted_evidence_key",
                        stage="evidence",
                        message="Trusted evidence keys must be non-empty strings.",
                    )
                )
                continue
            trusted.add(key)

        return cls(
            record_keys=frozenset(record_keys),
            successful_external_keys=frozenset(external),
            successful_action_keys=frozenset(actions),
            false_confirmation_keys=frozenset(false_confirmations),
            true_confirmation_records=tuple(
                record
                for record in by_key.values()
                if record.source == EvidenceSource.INPUT
                and record.status == EvidenceStatus.OBSERVED
                and record.producer_kind == "confirm"
                and record.value is True
            ),
            failure_keys=frozenset(failures),
            trusted_external_keys=frozenset(trusted),
            issues=tuple(issues),
        )


class PlanVerifier:
    """Verify typed plans before any external operation is emitted."""

    def __init__(
        self,
        snapshot: CapabilitySnapshot,
        *,
        policy: VerificationPolicy | None = None,
    ) -> None:
        self._snapshot = snapshot
        self._policy = policy or VerificationPolicy()

    def verify(
        self,
        candidate: PlanIR | Mapping[str, Any],
        *,
        available_evidence: Iterable[EvidenceRecord] = (),
        trusted_evidence_keys: Iterable[str] = (),
        required_completion_evidence: Collection[str] = (),
    ) -> PlanVerificationReport:
        """Return all independent static issues without executing the plan."""

        plan, parse_issues, candidate_plan_id = _parse_plan(candidate)
        evidence = EvidenceAvailability.from_records(
            available_evidence,
            trusted_external_keys=trusted_evidence_keys,
        )
        issues = list(self._snapshot.issues)
        issues.extend(evidence.issues)
        issues.extend(parse_issues)
        if plan is None:
            return PlanVerificationReport(
                capability_digest=self._snapshot.digest,
                plan_id=candidate_plan_id,
                plan=None,
                issues=tuple(issues),
            )

        if len(plan.nodes) > self._policy.max_nodes:
            issues.append(
                PlanVerificationIssue(
                    code="node_limit_exceeded",
                    stage="bounds",
                    message="Plan exceeds the configured node limit.",
                )
            )

        depths = _dependency_depths(plan)
        for node in plan.nodes:
            if depths[node.id] > self._policy.max_dependency_depth:
                issues.append(
                    PlanVerificationIssue(
                        code="dependency_depth_exceeded",
                        stage="bounds",
                        message="Plan dependency depth exceeds the configured limit.",
                        node_id=node.id,
                    )
                )

        tool_index = self._snapshot.tool_index()
        terminal = plan.terminal
        terminal_evidence = frozenset(terminal.requires_evidence)
        outstanding_completion = frozenset(required_completion_evidence)
        for key in sorted(outstanding_completion - evidence.successful_action_keys):
            issues.append(
                PlanVerificationIssue(
                    code="invalid_completion_obligation",
                    stage="evidence",
                    message=(
                        "A carried completion obligation must identify a successful "
                        "external action record."
                    ),
                    instance_path=(key,),
                )
            )
        if outstanding_completion and terminal.outcome != ResponseOutcome.COMPLETED:
            issues.append(
                PlanVerificationIssue(
                    code="completion_obligation_wrong_outcome",
                    stage="evidence",
                    message=(
                        "A carried action-success obligation requires a completed "
                        "terminal outcome."
                    ),
                    node_id=terminal.id,
                )
            )
        if terminal.outcome == ResponseOutcome.COMPLETED:
            for key in sorted(outstanding_completion - terminal_evidence):
                issues.append(
                    PlanVerificationIssue(
                        code="required_completion_evidence_missing",
                        stage="evidence",
                        message=(
                            "Completion omits a carried-forward evidence obligation."
                        ),
                        node_id=terminal.id,
                        instance_path=(key,),
                    )
                )

        declared_evidence_inputs = frozenset(plan.evidence_inputs)
        for key in sorted(declared_evidence_inputs - evidence.record_keys):
            issues.append(
                PlanVerificationIssue(
                    code="evidence_input_unavailable",
                    stage="evidence",
                    message="Plan declares evidence absent from the trusted ledger view.",
                    instance_path=(key,),
                )
            )

        external_producer_by_key = {
            node.success_evidence_key: node
            for node in plan.nodes
            if isinstance(node, (ObserveNode, ActNode))
        }
        input_producer_keys = {
            node.evidence_key
            for node in plan.nodes
            if isinstance(node, (AskNode, ConfirmNode))
        }
        produced_evidence_keys = frozenset(external_producer_by_key) | frozenset(
            input_producer_keys
        )
        for key in sorted(produced_evidence_keys & evidence.record_keys):
            issues.append(
                PlanVerificationIssue(
                    code="evidence_key_reuse",
                    stage="evidence",
                    message=(
                        "A new evidence producer cannot reuse a key already present "
                        "in the immutable ledger."
                    ),
                    instance_path=(key,),
                )
            )
        derived_failure_keys = {f"{key}.failure" for key in external_producer_by_key}
        for key in sorted(
            derived_failure_keys & (produced_evidence_keys | evidence.record_keys)
        ):
            issues.append(
                PlanVerificationIssue(
                    code="failure_evidence_key_collision",
                    stage="evidence",
                    message=(
                        "A runtime failure-evidence key collides with an existing "
                        "or planned producer key."
                    ),
                    instance_path=(key,),
                )
            )
        if terminal.outcome == ResponseOutcome.COMPLETED:
            for key in sorted(terminal_evidence):
                if key in external_producer_by_key:
                    continue
                if key in input_producer_keys:
                    issues.append(
                        PlanVerificationIssue(
                            code="completion_evidence_not_external",
                            stage="evidence",
                            message=(
                                "Completion evidence must come from a successful "
                                "external result."
                            ),
                            node_id=terminal.id,
                            instance_path=(key,),
                        )
                    )
                elif key not in evidence.record_keys:
                    issues.append(
                        PlanVerificationIssue(
                            code="completion_evidence_unavailable",
                            stage="evidence",
                            message=(
                                "Completion cites evidence absent from the trusted "
                                "ledger view."
                            ),
                            node_id=terminal.id,
                            instance_path=(key,),
                        )
                    )
                elif key not in evidence.successful_action_keys:
                    issues.append(
                        PlanVerificationIssue(
                            code="completion_evidence_not_action",
                            stage="evidence",
                            message=(
                                "Carried completion evidence must come from a "
                                "successful external action."
                            ),
                            node_id=terminal.id,
                            instance_path=(key,),
                        )
                    )

        if terminal.outcome == ResponseOutcome.DECLINED and not (
            terminal_evidence & evidence.false_confirmation_keys
        ):
            issues.append(
                PlanVerificationIssue(
                    code="declined_without_false_input",
                    stage="evidence",
                    message=(
                        "A declined outcome requires an available false user-input "
                        "evidence record."
                    ),
                    node_id=terminal.id,
                )
            )

        if terminal.outcome == ResponseOutcome.FAIL_SAFE and not (
            terminal_evidence & declared_evidence_inputs & evidence.failure_keys
        ):
            issues.append(
                PlanVerificationIssue(
                    code="fail_safe_failure_provenance_unavailable",
                    stage="evidence",
                    message=(
                        "The current ledger schema cannot yet attest a typed failure "
                        "record; fail-safe output remains allowed."
                    ),
                    severity="warning",
                    node_id=terminal.id,
                )
            )

        for node in plan.nodes:
            if not isinstance(node, (ObserveNode, ActNode)):
                continue

            capability = self._snapshot.get(node.operation)
            if capability is None:
                issues.append(
                    PlanVerificationIssue(
                        code="operation_unavailable",
                        stage="capability",
                        message="Plan references an operation absent from the live snapshot.",
                        node_id=node.id,
                        operation=node.operation,
                    )
                )
                continue

            schema_issue = tool_index.validation_issue(node.operation, node.arguments)
            if schema_issue is not None:
                issues.append(
                    PlanVerificationIssue(
                        code=f"arguments_{schema_issue.code}",
                        stage="schema",
                        message="Operation arguments do not satisfy the live capability schema.",
                        node_id=node.id,
                        operation=node.operation,
                        instance_path=schema_issue.instance_path,
                        schema_path=schema_issue.schema_path,
                        constraint=schema_issue.constraint,
                    )
                )

            expected_kind = capability.effect
            if expected_kind != "unknown" and node.kind != expected_kind:
                issues.append(
                    PlanVerificationIssue(
                        code="operation_kind_mismatch",
                        stage="safety",
                        message=(
                            "Plan node kind conflicts with the trusted capability "
                            "effect classification."
                        ),
                        node_id=node.id,
                        operation=node.operation,
                    )
                )

            if not isinstance(node, ActNode):
                continue

            ancestors = _ancestor_ids(node.id, plan)
            if self._policy.require_action_precondition and not ancestors:
                issues.append(
                    PlanVerificationIssue(
                        code="action_without_precondition",
                        stage="safety",
                        message="A state-changing operation must have an explicit prerequisite.",
                        node_id=node.id,
                        operation=node.operation,
                    )
                )

            has_current_confirmation = False
            for ancestor in ancestors:
                ancestor_node = plan.node_by_id[ancestor]
                if (
                    isinstance(ancestor_node, ConfirmNode)
                    and node.id in ancestor_node.authorizes
                ):
                    has_current_confirmation = True
                    break
            argument_digest = _sha256(_canonical_json(node.arguments))
            carried_confirmation = any(
                record.key in declared_evidence_inputs
                and record.schema_digest == self._snapshot.digest
                and any(
                    authorization.operation == node.operation
                    and authorization.argument_digest == argument_digest
                    for authorization in record.authorizations
                )
                for record in evidence.true_confirmation_records
            )
            if (
                capability.requires_confirmation
                and not has_current_confirmation
                and not carried_confirmation
            ):
                issues.append(
                    PlanVerificationIssue(
                        code="critical_action_without_confirmation",
                        stage="safety",
                        message="A critical operation requires a confirmation prerequisite.",
                        node_id=node.id,
                        operation=node.operation,
                    )
                )
            if (
                self._policy.require_all_action_evidence
                and node.success_evidence_key not in terminal_evidence
            ):
                issues.append(
                    PlanVerificationIssue(
                        code="action_evidence_not_terminal",
                        stage="evidence",
                        message="Completion must cite success evidence for every action.",
                        node_id=node.id,
                        operation=node.operation,
                    )
                )

        return PlanVerificationReport(
            capability_digest=self._snapshot.digest,
            plan_id=plan.plan_id,
            plan=plan,
            issues=tuple(issues),
        )


def _parse_plan(
    candidate: PlanIR | Mapping[str, Any],
) -> tuple[PlanIR | None, list[PlanVerificationIssue], str | None]:
    if isinstance(candidate, PlanIR):
        return candidate, [], candidate.plan_id

    candidate_plan_id = candidate.get("plan_id")
    plan_id = candidate_plan_id if isinstance(candidate_plan_id, str) else None
    try:
        plan = PlanIR.model_validate(candidate)
    except PydanticValidationError as exc:
        issues = [
            PlanVerificationIssue(
                code="invalid_plan_ir",
                stage="ir",
                message=error.get("msg", "Candidate does not satisfy PlanIR."),
                instance_path=tuple(error.get("loc", ())),
                constraint=str(error.get("type", "validation")),
            )
            for error in exc.errors(include_input=False, include_url=False)
        ]
        return None, issues, plan_id
    return plan, [], plan.plan_id


def _dependency_depths(plan: PlanIR) -> dict[str, int]:
    depths: dict[str, int] = {}
    for node_id in plan.topological_order():
        node = plan.node_by_id[node_id]
        depths[node_id] = (
            1 + max(depths[dependency] for dependency in node.depends_on)
            if node.depends_on
            else 1
        )
    return depths


def _ancestor_ids(node_id: str, plan: PlanIR) -> frozenset[str]:
    ancestors: set[str] = set()
    frontier = list(plan.node_by_id[node_id].depends_on)
    while frontier:
        ancestor = frontier.pop()
        if ancestor in ancestors:
            continue
        ancestors.add(ancestor)
        frontier.extend(plan.node_by_id[ancestor].depends_on)
    return frozenset(ancestors)


def _json_object_copy(value: Mapping[str, Any]) -> dict[str, Any] | None:
    try:
        serialized = _canonical_json(value)
        parsed = json.loads(serialized)
    except (TypeError, ValueError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


__all__ = [
    "CapabilityEffect",
    "CapabilityDescriptor",
    "CapabilitySnapshot",
    "EvidenceAvailability",
    "IssueSeverity",
    "PlanVerificationIssue",
    "PlanVerificationReport",
    "PlanVerifier",
    "VerificationPolicy",
    "description_requires_confirmation",
]
