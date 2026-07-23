from __future__ import annotations

import hashlib
import json
from typing import Any, Literal

import pytest

from carbench_agent_core.plan_ir import (
    ActNode,
    AskNode,
    ConfirmNode,
    ObserveNode,
    PlanIR,
    RespondNode,
    ResponseOutcome,
)
from carbench_agent_core.evidence_ledger import (
    ActionAuthorization,
    EvidenceRecord,
    EvidenceSource,
    EvidenceStatus,
)
from carbench_agent_core.plan_verifier import (
    CapabilitySnapshot,
    EvidenceAvailability,
    PlanVerifier,
    VerificationPolicy,
)


def tool(
    name: str,
    *,
    properties: dict[str, Any] | None = None,
    required: list[str] | None = None,
    confirmation: bool = False,
    description: str = "Synthetic capability.",
) -> dict[str, Any]:
    function: dict[str, Any] = {
        "name": name,
        "description": description,
        "parameters": {
            "type": "object",
            "properties": properties or {},
            "required": required or [],
            "additionalProperties": False,
        },
    }
    if confirmation:
        function["x-pact-requires-confirmation"] = True
    return {"type": "function", "function": function}


def safe_plan() -> PlanIR:
    return PlanIR(
        plan_id="synthetic-plan",
        nodes=(
            ObserveNode(
                id="inspect",
                operation="read_resource",
                arguments={"resource_id": "alpha"},
                success_evidence_key="resource.observed",
            ),
            ConfirmNode(
                id="authorize",
                depends_on=("inspect",),
                prompt="Apply the requested update?",
                evidence_key="input.confirmed",
                authorizes=("apply",),
            ),
            ActNode(
                id="apply",
                depends_on=("authorize",),
                operation="update_resource",
                arguments={"resource_id": "alpha", "level": 2},
                success_evidence_key="resource.updated",
            ),
            RespondNode(
                id="finish",
                depends_on=("apply",),
                text="The update completed.",
                outcome=ResponseOutcome.COMPLETED,
                requires_evidence=("resource.updated",),
            ),
        ),
    )


@pytest.fixture
def live_tools() -> list[dict[str, Any]]:
    return [
        tool(
            "read_resource",
            properties={"resource_id": {"type": "string", "minLength": 1}},
            required=["resource_id"],
        ),
        tool(
            "update_resource",
            properties={
                "resource_id": {"type": "string", "minLength": 1},
                "level": {"type": "integer", "minimum": 0, "maximum": 4},
            },
            required=["resource_id", "level"],
            confirmation=True,
        ),
    ]


def issue_codes(report: object) -> set[str]:
    return {issue.code for issue in report.issues}  # type: ignore[attr-defined]


def test_snapshot_is_order_independent_and_content_addressed(
    live_tools: list[dict[str, Any]],
) -> None:
    forward = CapabilitySnapshot.from_tools(live_tools)
    reverse = CapabilitySnapshot.from_tools(reversed(live_tools))

    assert forward.is_valid
    assert forward.digest == reverse.digest
    assert forward.names == frozenset({"read_resource", "update_resource"})
    update_capability = forward.get("update_resource")
    assert update_capability is not None
    assert update_capability.requires_confirmation is True
    assert update_capability.effect == "act"

    changed = CapabilitySnapshot.from_tools(
        [
            live_tools[0],
            tool(
                "update_resource",
                properties={"level": {"type": "integer", "maximum": 9}},
                required=["level"],
                confirmation=True,
            ),
        ]
    )
    revoked = CapabilitySnapshot.from_tools(live_tools[:1])

    assert changed.digest != forward.digest
    assert revoked.digest != forward.digest
    assert "update_resource" not in revoked.names

    description_changed = [
        {
            **item,
            "function": {
                **item["function"],
                "description": item["function"]["description"] + " Changed.",
            },
        }
        for item in live_tools
    ]
    assert CapabilitySnapshot.from_tools(description_changed).digest != forward.digest


def test_snapshot_recognizes_only_anchored_confirmation_contract_marker() -> None:
    marked = CapabilitySnapshot.from_tools(
        [
            tool(
                "mutate_resource",
                description=(
                    "  REQUIRES_CONFIRMATION, applies the requested mutation."
                ),
            )
        ]
    )
    embedded = CapabilitySnapshot.from_tools(
        [
            tool(
                "read_resource",
                description="Mentions REQUIRES_CONFIRMATION only as documentation.",
            )
        ]
    )
    longer_identifier = CapabilitySnapshot.from_tools(
        [
            tool(
                "read_other",
                description="REQUIRES_CONFIRMATIONLESS synthetic marker.",
            )
        ]
    )

    assert marked.get("mutate_resource").requires_confirmation is True  # type: ignore[union-attr]
    assert marked.get("mutate_resource").effect == "act"  # type: ignore[union-attr]
    assert embedded.get("read_resource").requires_confirmation is False  # type: ignore[union-attr]
    assert longer_identifier.get("read_other").requires_confirmation is False  # type: ignore[union-attr]
    unmarked = CapabilitySnapshot.from_tools([tool("mutate_resource")])
    assert marked.digest != unmarked.digest


def test_snapshot_rejects_invalid_and_duplicate_contracts() -> None:
    invalid = tool("broken")
    invalid["function"]["parameters"] = {"type": "unknown-json-type"}
    snapshot = CapabilitySnapshot.from_tools([invalid, tool("same"), tool("same")])

    assert not snapshot.is_valid
    assert {issue.code for issue in snapshot.issues} == {
        "invalid_capability_schema",
        "duplicate_capability",
    }
    assert "broken" not in snapshot.names
    assert "same" not in snapshot.names


def test_snapshot_rejects_missing_parameter_schema() -> None:
    snapshot = CapabilitySnapshot.from_tools(
        [{"type": "function", "function": {"name": "underspecified"}}]
    )

    assert not snapshot.is_valid
    assert {issue.code for issue in snapshot.issues} == {"invalid_capability_schema"}


def test_snapshot_rejects_non_function_tool_declaration() -> None:
    declaration = tool("not_callable")
    declaration["type"] = "resource"

    snapshot = CapabilitySnapshot.from_tools([declaration])

    assert not snapshot.is_valid
    assert snapshot.names == frozenset()
    assert {issue.code for issue in snapshot.issues} == {"malformed_capability"}


def test_verifier_accepts_plan_against_live_draft_2020_12_contract(
    live_tools: list[dict[str, Any]],
) -> None:
    snapshot = CapabilitySnapshot.from_tools(live_tools)
    report = PlanVerifier(snapshot).verify(safe_plan())

    assert report.accepted
    assert report.plan == safe_plan()
    assert report.capability_digest == snapshot.digest
    assert report.issues == ()


def test_verifier_fails_closed_when_operation_is_revoked(
    live_tools: list[dict[str, Any]],
) -> None:
    snapshot = CapabilitySnapshot.from_tools(live_tools[:1])
    report = PlanVerifier(snapshot).verify(safe_plan())

    assert not report.accepted
    assert "operation_unavailable" in issue_codes(report)
    unavailable = next(
        issue for issue in report.issues if issue.code == "operation_unavailable"
    )
    assert unavailable.node_id == "apply"
    assert unavailable.operation == "update_resource"


def test_verifier_reports_schema_paths_without_argument_values(
    live_tools: list[dict[str, Any]],
) -> None:
    plan = safe_plan().model_copy(
        update={
            "nodes": tuple(
                node.model_copy(
                    update={"arguments": {"resource_id": "alpha", "level": 7}}
                )
                if isinstance(node, ActNode)
                else node
                for node in safe_plan().nodes
            )
        }
    )
    report = PlanVerifier(CapabilitySnapshot.from_tools(live_tools)).verify(plan)

    issue = next(issue for issue in report.issues if issue.stage == "schema")
    assert issue.code == "arguments_out_of_range"
    assert issue.instance_path == ("level",)
    assert issue.constraint == "maximum"
    assert "7" not in issue.message


def test_verifier_checks_observation_arguments_too(
    live_tools: list[dict[str, Any]],
) -> None:
    plan = safe_plan().model_copy(
        update={
            "nodes": tuple(
                node.model_copy(update={"arguments": {}})
                if isinstance(node, ObserveNode)
                else node
                for node in safe_plan().nodes
            )
        }
    )
    report = PlanVerifier(CapabilitySnapshot.from_tools(live_tools)).verify(plan)

    issue = next(issue for issue in report.issues if issue.node_id == "inspect")
    assert issue.code == "arguments_missing_required_argument"
    assert issue.instance_path == ("resource_id",)


def test_verifier_enforces_node_and_dependency_depth_bounds(
    live_tools: list[dict[str, Any]],
) -> None:
    report = PlanVerifier(
        CapabilitySnapshot.from_tools(live_tools),
        policy=VerificationPolicy(max_nodes=3, max_dependency_depth=2),
    ).verify(safe_plan())

    assert "node_limit_exceeded" in issue_codes(report)
    depth_issues = [
        issue for issue in report.issues if issue.code == "dependency_depth_exceeded"
    ]
    assert {issue.node_id for issue in depth_issues} == {"apply", "finish"}


def test_verifier_can_enforce_explicit_action_precondition_policy(
    live_tools: list[dict[str, Any]],
) -> None:
    plan = PlanIR(
        plan_id="direct-action",
        nodes=(
            ActNode(
                id="apply",
                operation="update_resource",
                arguments={"resource_id": "alpha", "level": 1},
                success_evidence_key="resource.updated",
            ),
            RespondNode(
                id="finish",
                depends_on=("apply",),
                text="Update completed.",
                outcome=ResponseOutcome.COMPLETED,
                requires_evidence=("resource.updated",),
            ),
        ),
    )
    snapshot = CapabilitySnapshot.from_tools(live_tools, critical_operations=())
    report = PlanVerifier(
        snapshot,
        policy=VerificationPolicy(require_action_precondition=True),
    ).verify(plan)

    apply_issues = {issue.code for issue in report.issues if issue.node_id == "apply"}
    assert "action_without_precondition" in apply_issues


def test_default_policy_does_not_treat_graph_roots_as_missing_preconditions() -> None:
    plan = PlanIR(
        plan_id="direct-action",
        nodes=(
            ActNode(
                id="apply",
                operation="apply_update",
                arguments={"enabled": True},
                success_evidence_key="update.applied",
            ),
            RespondNode(
                id="finish",
                depends_on=("apply",),
                text="The update completed.",
                outcome=ResponseOutcome.COMPLETED,
                requires_evidence=("update.applied",),
            ),
        ),
    )
    snapshot = CapabilitySnapshot.from_tools(
        [
            tool(
                "apply_update",
                properties={"enabled": {"type": "boolean"}},
                required=["enabled"],
            )
        ]
    )

    report = PlanVerifier(snapshot).verify(plan)

    assert report.accepted


def test_critical_action_requires_confirmation_ancestor(
    live_tools: list[dict[str, Any]],
) -> None:
    plan = PlanIR(
        plan_id="unconfirmed-plan",
        nodes=(
            ObserveNode(
                id="inspect",
                operation="read_resource",
                arguments={"resource_id": "alpha"},
                success_evidence_key="resource.observed",
            ),
            ActNode(
                id="apply",
                depends_on=("inspect",),
                operation="update_resource",
                arguments={"resource_id": "alpha", "level": 2},
                success_evidence_key="resource.updated",
            ),
            RespondNode(
                id="finish",
                depends_on=("apply",),
                text="The update completed.",
                outcome=ResponseOutcome.COMPLETED,
                requires_evidence=("resource.updated",),
            ),
        ),
    )
    report = PlanVerifier(CapabilitySnapshot.from_tools(live_tools)).verify(plan)

    assert "critical_action_without_confirmation" in issue_codes(report)


def test_recompiled_critical_action_accepts_carried_confirmation_provenance(
    live_tools: list[dict[str, Any]],
) -> None:
    plan = PlanIR(
        plan_id="recompiled-action",
        evidence_inputs=("input.confirmed",),
        nodes=(
            ActNode(
                id="apply",
                operation="update_resource",
                arguments={"resource_id": "alpha", "level": 2},
                success_evidence_key="resource.updated",
            ),
            RespondNode(
                id="finish",
                depends_on=("apply",),
                text="The update completed.",
                outcome=ResponseOutcome.COMPLETED,
                requires_evidence=("resource.updated",),
            ),
        ),
    )
    snapshot = CapabilitySnapshot.from_tools(live_tools)
    arguments = {"resource_id": "alpha", "level": 2}
    argument_digest = hashlib.sha256(
        json.dumps(
            arguments,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    report = PlanVerifier(snapshot).verify(
        plan,
        available_evidence=[
            evidence_record(
                "input.confirmed",
                source=EvidenceSource.INPUT,
                value=True,
                status=EvidenceStatus.OBSERVED,
                producer_kind="confirm",
                schema_digest=snapshot.digest,
                authorizations=(
                    ActionAuthorization(
                        operation="update_resource",
                        argument_digest=argument_digest,
                    ),
                ),
            )
        ],
    )

    assert report.accepted
    assert "confirmation_provenance_weak" not in issue_codes(report)


def test_recompiled_critical_action_rejects_false_or_unobserved_input(
    live_tools: list[dict[str, Any]],
) -> None:
    plan = PlanIR(
        plan_id="invalid-carried-confirmation",
        evidence_inputs=("input.confirmed",),
        nodes=(
            ActNode(
                id="apply",
                operation="update_resource",
                arguments={"resource_id": "alpha", "level": 2},
                success_evidence_key="resource.updated",
            ),
            RespondNode(
                id="finish",
                depends_on=("apply",),
                text="The update completed.",
                outcome=ResponseOutcome.COMPLETED,
                requires_evidence=("resource.updated",),
            ),
        ),
    )
    verifier = PlanVerifier(CapabilitySnapshot.from_tools(live_tools))

    false_report = verifier.verify(
        plan,
        available_evidence=[
            evidence_record(
                "input.confirmed",
                source=EvidenceSource.INPUT,
                value=False,
                status=EvidenceStatus.FAILURE,
                producer_kind="confirm",
            )
        ],
    )
    success_status_report = verifier.verify(
        plan,
        available_evidence=[
            evidence_record(
                "input.confirmed",
                source=EvidenceSource.INPUT,
                value=True,
                status=EvidenceStatus.SUCCESS,
                producer_kind="confirm",
            )
        ],
    )

    assert "critical_action_without_confirmation" in issue_codes(false_report)
    assert "critical_action_without_confirmation" in issue_codes(success_status_report)


def test_carried_confirmation_is_scoped_to_operation_arguments_and_snapshot(
    live_tools: list[dict[str, Any]],
) -> None:
    plan = PlanIR(
        plan_id="scoped-carried-confirmation",
        evidence_inputs=("input.confirmed",),
        nodes=(
            ActNode(
                id="apply",
                operation="update_resource",
                arguments={"resource_id": "alpha", "level": 2},
                success_evidence_key="resource.updated",
            ),
            RespondNode(
                id="finish",
                depends_on=("apply",),
                text="The update completed.",
                outcome=ResponseOutcome.COMPLETED,
                requires_evidence=("resource.updated",),
            ),
        ),
    )
    snapshot = CapabilitySnapshot.from_tools(live_tools)

    def digest(arguments: dict[str, Any]) -> str:
        return hashlib.sha256(
            json.dumps(
                arguments,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()

    current_digest = digest({"resource_id": "alpha", "level": 2})
    matching_authorization = (
        ActionAuthorization(
            operation="update_resource",
            argument_digest=current_digest,
        ),
    )
    invalid_records = (
        evidence_record(
            "input.confirmed",
            source=EvidenceSource.INPUT,
            value=True,
            status=EvidenceStatus.OBSERVED,
            producer_kind="ask",
            schema_digest=snapshot.digest,
            authorizations=matching_authorization,
        ),
        evidence_record(
            "input.confirmed",
            source=EvidenceSource.INPUT,
            value=True,
            status=EvidenceStatus.OBSERVED,
            producer_kind="confirm",
            schema_digest=snapshot.digest,
            authorizations=(
                ActionAuthorization(
                    operation="different_operation",
                    argument_digest=current_digest,
                ),
            ),
        ),
        evidence_record(
            "input.confirmed",
            source=EvidenceSource.INPUT,
            value=True,
            status=EvidenceStatus.OBSERVED,
            producer_kind="confirm",
            schema_digest=snapshot.digest,
            authorizations=(
                ActionAuthorization(
                    operation="update_resource",
                    argument_digest=digest({"resource_id": "alpha", "level": 1}),
                ),
            ),
        ),
        evidence_record(
            "input.confirmed",
            source=EvidenceSource.INPUT,
            value=True,
            status=EvidenceStatus.OBSERVED,
            producer_kind="confirm",
            schema_digest="stale-capability-snapshot",
            authorizations=matching_authorization,
        ),
    )

    for record in invalid_records:
        report = PlanVerifier(snapshot).verify(
            plan,
            available_evidence=[record],
        )
        assert "critical_action_without_confirmation" in issue_codes(report)


def test_confirmation_required_operation_cannot_be_downgraded_to_observe(
    live_tools: list[dict[str, Any]],
) -> None:
    plan = PlanIR(
        plan_id="kind-downgrade",
        nodes=(
            ObserveNode(
                id="misclassified",
                operation="update_resource",
                arguments={"resource_id": "alpha", "level": 2},
                success_evidence_key="resource.updated",
            ),
            RespondNode(
                id="finish",
                depends_on=("misclassified",),
                text="Here is the result.",
                outcome=ResponseOutcome.ANSWERED,
                requires_evidence=("resource.updated",),
            ),
        ),
    )

    report = PlanVerifier(CapabilitySnapshot.from_tools(live_tools)).verify(plan)

    mismatch = next(
        issue for issue in report.issues if issue.code == "operation_kind_mismatch"
    )
    assert mismatch.node_id == "misclassified"
    assert not report.accepted


def test_explicit_observation_effect_cannot_be_upgraded_to_action() -> None:
    plan = PlanIR(
        plan_id="kind-upgrade",
        nodes=(
            ActNode(
                id="misclassified",
                operation="inspect_resource",
                arguments={},
                success_evidence_key="resource.inspected",
            ),
            RespondNode(
                id="finish",
                depends_on=("misclassified",),
                text="Inspection completed.",
                outcome=ResponseOutcome.COMPLETED,
                requires_evidence=("resource.inspected",),
            ),
        ),
    )
    snapshot = CapabilitySnapshot.from_tools(
        [tool("inspect_resource")],
        operation_effects={"inspect_resource": "observe"},
    )

    report = PlanVerifier(snapshot).verify(plan)

    assert "operation_kind_mismatch" in issue_codes(report)


def test_observation_with_trusted_read_effect_can_ground_answer() -> None:
    plan = PlanIR(
        plan_id="observation-only",
        nodes=(
            ObserveNode(
                id="inspect",
                operation="inspect_resource",
                arguments={},
                success_evidence_key="resource.inspected",
            ),
            RespondNode(
                id="finish",
                depends_on=("inspect",),
                text="Here is the observed information.",
                outcome=ResponseOutcome.ANSWERED,
                requires_evidence=("resource.inspected",),
            ),
        ),
    )
    snapshot = CapabilitySnapshot.from_tools(
        [tool("inspect_resource")],
        operation_effects={"inspect_resource": "observe"},
    )

    report = PlanVerifier(snapshot).verify(plan)

    assert report.accepted


def test_explicit_critical_policy_does_not_depend_on_operation_names() -> None:
    snapshot = CapabilitySnapshot.from_tools(
        [tool("opaque_operation")],
        critical_operations={"opaque_operation"},
    )

    capability = snapshot.get("opaque_operation")
    assert capability is not None
    assert capability.requires_confirmation is True


def test_invalid_raw_plan_returns_structured_ir_issues(
    live_tools: list[dict[str, Any]],
) -> None:
    report = PlanVerifier(CapabilitySnapshot.from_tools(live_tools)).verify(
        {"plan_id": "bad-plan", "nodes": []}
    )

    assert not report.accepted
    assert report.plan is None
    assert report.plan_id == "bad-plan"
    assert issue_codes(report) == {"invalid_plan_ir"}
    assert report.issues[0].stage == "ir"


def test_snapshot_issue_makes_otherwise_valid_plan_fail_closed(
    live_tools: list[dict[str, Any]],
) -> None:
    malformed = {"type": "function", "function": {"name": "extra", "parameters": []}}
    snapshot = CapabilitySnapshot.from_tools([*live_tools, malformed])

    report = PlanVerifier(snapshot).verify(safe_plan())

    assert not report.accepted
    assert "invalid_capability_schema" in issue_codes(report)


def evidence_record(
    key: str,
    *,
    source: EvidenceSource = EvidenceSource.EXTERNAL,
    event_id: str = "event-1",
    value: Any = None,
    status: EvidenceStatus = EvidenceStatus.SUCCESS,
    producer_kind: Literal["observe", "ask", "confirm", "act"] | None = None,
    schema_digest: str | None = None,
    authorizations: tuple[ActionAuthorization, ...] = (),
) -> EvidenceRecord:
    return EvidenceRecord(
        key=key,
        value={"acknowledged": True} if value is None else value,
        source=source,
        status=status,
        event_id=event_id,
        call_id="call-1",
        node_id="prior-node",
        producer_kind=producer_kind,
        operation="prior-operation" if source == EvidenceSource.EXTERNAL else None,
        schema_digest=schema_digest,
        authorizations=authorizations,
    )


def test_evidence_availability_only_trusts_external_records_for_completion() -> None:
    availability = EvidenceAvailability.from_records(
        [
            evidence_record("external.success"),
            evidence_record(
                "external.failure",
                status=EvidenceStatus.FAILURE,
            ),
            evidence_record("input.value", source=EvidenceSource.INPUT),
        ]
    )

    assert availability.keys == frozenset(
        {"external.success", "external.failure", "input.value"}
    )
    assert availability.successful_external_keys == frozenset({"external.success"})
    assert availability.failure_keys == frozenset({"external.failure"})


def test_raw_evidence_keys_require_explicit_trusted_parameter() -> None:
    without_key = EvidenceAvailability.from_records([])
    with_trusted_key = EvidenceAvailability.from_records(
        [], trusted_external_keys={"prior.success"}
    )

    assert "prior.success" not in without_key.keys
    assert "prior.success" in with_trusted_key.keys
    assert "prior.success" not in with_trusted_key.record_keys
    assert "prior.success" not in with_trusted_key.successful_external_keys


def test_verifier_preserves_carried_forward_completion_obligations(
    live_tools: list[dict[str, Any]],
) -> None:
    report = PlanVerifier(CapabilitySnapshot.from_tools(live_tools)).verify(
        safe_plan(),
        available_evidence=[evidence_record("prior.success", producer_kind="act")],
        required_completion_evidence={"prior.success"},
    )

    assert not report.accepted
    issue = next(
        issue
        for issue in report.issues
        if issue.code == "required_completion_evidence_missing"
    )
    assert issue.node_id == "finish"
    assert issue.instance_path == ("prior.success",)


def test_carried_action_obligation_cannot_be_bypassed_with_answered_outcome() -> None:
    prior = evidence_record("prior.success", producer_kind="act")
    plan = PlanIR(
        plan_id="wrong-terminal-outcome",
        evidence_inputs=("prior.success",),
        nodes=(
            RespondNode(
                id="finish",
                text="The request is done.",
                outcome=ResponseOutcome.ANSWERED,
                requires_evidence=("prior.success",),
            ),
        ),
    )

    report = PlanVerifier(CapabilitySnapshot.from_tools([])).verify(
        plan,
        available_evidence=[prior],
        required_completion_evidence={"prior.success"},
    )

    assert not report.accepted
    assert "completion_obligation_wrong_outcome" in issue_codes(report)


def test_conflicting_evidence_provenance_fails_closed(
    live_tools: list[dict[str, Any]],
) -> None:
    report = PlanVerifier(CapabilitySnapshot.from_tools(live_tools)).verify(
        safe_plan(),
        available_evidence=[
            evidence_record("same.key", event_id="event-1"),
            evidence_record("same.key", event_id="event-2"),
        ],
    )

    assert "conflicting_evidence_provenance" in issue_codes(report)


def carried_completion_plan() -> PlanIR:
    return PlanIR(
        plan_id="carried-completion",
        evidence_inputs=("prior.success",),
        nodes=(
            RespondNode(
                id="finish",
                text="The prior operation completed.",
                outcome=ResponseOutcome.COMPLETED,
                requires_evidence=("prior.success",),
            ),
        ),
    )


def test_completed_carried_evidence_requires_successful_external_record() -> None:
    verifier = PlanVerifier(CapabilitySnapshot.from_tools([]))

    accepted = verifier.verify(
        carried_completion_plan(),
        available_evidence=[evidence_record("prior.success", producer_kind="act")],
        required_completion_evidence={"prior.success"},
    )
    input_backed = verifier.verify(
        carried_completion_plan(),
        available_evidence=[
            evidence_record(
                "prior.success",
                source=EvidenceSource.INPUT,
                value=True,
            )
        ],
    )
    missing = verifier.verify(carried_completion_plan())

    assert accepted.accepted
    assert "completion_evidence_not_action" in issue_codes(input_backed)
    assert "evidence_input_unavailable" in issue_codes(missing)
    assert "completion_evidence_unavailable" in issue_codes(missing)


def test_completed_rejects_current_input_producer_as_success_evidence() -> None:
    plan = PlanIR(
        plan_id="mixed-evidence",
        nodes=(
            AskNode(
                id="ask",
                prompt="Which value?",
                evidence_key="input.value",
            ),
            ActNode(
                id="apply",
                depends_on=("ask",),
                operation="apply_update",
                arguments={"enabled": True},
                success_evidence_key="update.applied",
            ),
            RespondNode(
                id="finish",
                depends_on=("apply",),
                text="The update completed.",
                outcome=ResponseOutcome.COMPLETED,
                requires_evidence=("input.value", "update.applied"),
            ),
        ),
    )
    snapshot = CapabilitySnapshot.from_tools(
        [
            tool(
                "apply_update",
                properties={"enabled": {"type": "boolean"}},
                required=["enabled"],
            )
        ]
    )

    report = PlanVerifier(snapshot).verify(plan)

    assert "completion_evidence_not_external" in issue_codes(report)


def test_trusted_raw_key_does_not_satisfy_declared_evidence_input() -> None:
    report = PlanVerifier(CapabilitySnapshot.from_tools([])).verify(
        carried_completion_plan(),
        trusted_evidence_keys={"prior.success"},
    )

    assert not report.accepted
    assert "evidence_input_unavailable" in issue_codes(report)
    assert "completion_evidence_unavailable" in issue_codes(report)


def declined_plan() -> PlanIR:
    return PlanIR(
        plan_id="declined-plan",
        evidence_inputs=("input.declined",),
        nodes=(
            RespondNode(
                id="finish",
                text="I did not perform the operation.",
                outcome=ResponseOutcome.DECLINED,
                requires_evidence=("input.declined",),
            ),
        ),
    )


def test_declined_requires_available_false_input_evidence() -> None:
    verifier = PlanVerifier(CapabilitySnapshot.from_tools([]))

    false_input = verifier.verify(
        declined_plan(),
        available_evidence=[
            evidence_record(
                "input.declined",
                source=EvidenceSource.INPUT,
                value=False,
                status=EvidenceStatus.FAILURE,
                producer_kind="confirm",
            )
        ],
    )
    true_input = verifier.verify(
        declined_plan(),
        available_evidence=[
            evidence_record(
                "input.declined",
                source=EvidenceSource.INPUT,
                value=True,
                status=EvidenceStatus.OBSERVED,
                producer_kind="confirm",
            )
        ],
    )

    assert false_input.accepted
    assert "declined_without_false_input" in issue_codes(true_input)


def test_fail_safe_is_allowed_with_explicit_provenance_warning() -> None:
    plan = PlanIR(
        plan_id="fail-safe-plan",
        nodes=(
            RespondNode(
                id="finish",
                text="I stopped safely after the failure.",
                outcome=ResponseOutcome.FAIL_SAFE,
            ),
        ),
    )

    report = PlanVerifier(CapabilitySnapshot.from_tools([])).verify(plan)

    assert report.accepted
    warning = next(
        issue
        for issue in report.issues
        if issue.code == "fail_safe_failure_provenance_unavailable"
    )
    assert warning.severity == "warning"


def test_fail_safe_accepts_cited_failure_record_without_warning() -> None:
    plan = PlanIR(
        plan_id="grounded-fail-safe",
        evidence_inputs=("operation.failure",),
        nodes=(
            RespondNode(
                id="finish",
                text="I stopped safely after the failure.",
                outcome=ResponseOutcome.FAIL_SAFE,
                requires_evidence=("operation.failure",),
            ),
        ),
    )

    report = PlanVerifier(CapabilitySnapshot.from_tools([])).verify(
        plan,
        available_evidence=[
            evidence_record(
                "operation.failure",
                status=EvidenceStatus.FAILURE,
            )
        ],
    )

    assert report.accepted
    assert "fail_safe_failure_provenance_unavailable" not in issue_codes(report)


def test_completed_rejects_external_failure_record() -> None:
    report = PlanVerifier(CapabilitySnapshot.from_tools([])).verify(
        carried_completion_plan(),
        available_evidence=[
            evidence_record(
                "prior.success",
                status=EvidenceStatus.FAILURE,
            )
        ],
    )

    assert not report.accepted
    assert "completion_evidence_not_action" in issue_codes(report)


def test_completed_rejects_carried_observation_as_action_success() -> None:
    report = PlanVerifier(CapabilitySnapshot.from_tools([])).verify(
        carried_completion_plan(),
        available_evidence=[
            evidence_record(
                "prior.success",
                producer_kind="observe",
            )
        ],
        required_completion_evidence={"prior.success"},
    )

    assert not report.accepted
    assert "completion_evidence_not_action" in issue_codes(report)


def test_existing_ledger_key_rejects_new_producer_before_execution(
    live_tools: list[dict[str, Any]],
) -> None:
    existing = evidence_record(
        "resource.updated",
        producer_kind="act",
    )

    report = PlanVerifier(CapabilitySnapshot.from_tools(live_tools)).verify(
        safe_plan(),
        available_evidence=[existing],
    )

    assert not report.accepted
    conflict = next(
        issue for issue in report.issues if issue.code == "evidence_key_reuse"
    )
    assert conflict.stage == "evidence"
    assert conflict.instance_path == ("resource.updated",)


def test_answered_and_refused_outcomes_do_not_require_external_success() -> None:
    answer = PlanIR(
        plan_id="answer-plan",
        evidence_inputs=("input.fact",),
        nodes=(
            RespondNode(
                id="finish",
                text="Here is the answer.",
                outcome=ResponseOutcome.ANSWERED,
                requires_evidence=("input.fact",),
            ),
        ),
    )
    refusal = PlanIR(
        plan_id="refusal-plan",
        nodes=(
            RespondNode(
                id="finish",
                text="That capability is unavailable.",
                outcome=ResponseOutcome.REFUSED,
            ),
        ),
    )
    verifier = PlanVerifier(CapabilitySnapshot.from_tools([]))

    answer_report = verifier.verify(
        answer,
        available_evidence=[
            evidence_record(
                "input.fact",
                source=EvidenceSource.INPUT,
                value="known",
            )
        ],
    )
    refusal_report = verifier.verify(refusal)

    assert answer_report.accepted
    assert refusal_report.accepted
