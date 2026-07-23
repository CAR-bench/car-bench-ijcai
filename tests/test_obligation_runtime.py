from collections.abc import Iterator
import hashlib
from typing import Any, Literal

import pytest
from pydantic import ValidationError

from carbench_agent_core.evidence_ledger import (
    ActionAuthorization,
    ConfirmationEvent,
    EventDisposition,
    EvidenceSource,
    EvidenceStatus,
    ExternalResultEvent,
    ExternalResultStatus,
    InputEvent,
)
from carbench_agent_core.obligation_runtime import (
    AskAction,
    ConfirmAction,
    EmissionRejectedError,
    ExternalCallAction,
    NodeExecutionStatus,
    ObligationRuntime,
    PlanRunStatus,
    RespondAction,
)
from carbench_agent_core.plan_ir import (
    ActNode,
    AskNode,
    ConfirmNode,
    ObserveNode,
    PlanIR,
    RespondNode,
    ResponseOutcome,
)


def call_ids(*values: str):
    iterator: Iterator[str] = iter(values)
    return iterator.__next__


def allow_external_emission(
    kind: Literal["observe", "act"],
    operation: str,
    arguments: dict[str, Any],
) -> None:
    assert kind in {"observe", "act"}
    assert operation
    assert isinstance(arguments, dict)


def guarded_step(runtime: ObligationRuntime):
    return runtime.step(emission_guard=allow_external_emission)


EMPTY_ARGUMENT_DIGEST = hashlib.sha256(b"{}").hexdigest()


def full_plan() -> PlanIR:
    return PlanIR(
        plan_id="runtime-example",
        nodes=(
            ObserveNode(
                id="observe-context",
                operation="read.context",
                success_evidence_key="context.observed",
            ),
            AskNode(
                id="ask-choice",
                depends_on=("observe-context",),
                prompt="Which option should be used?",
                evidence_key="input.choice",
            ),
            ConfirmNode(
                id="confirm-change",
                depends_on=("ask-choice",),
                prompt="Should I apply the change?",
                evidence_key="input.confirmed",
                authorizes=("apply-change",),
            ),
            ActNode(
                id="apply-change",
                depends_on=("confirm-change",),
                operation="write.change",
                arguments={"mode": "selected"},
                success_evidence_key="change.applied",
            ),
            RespondNode(
                id="complete",
                depends_on=("apply-change",),
                text="The requested change is complete.",
                outcome=ResponseOutcome.COMPLETED,
                requires_evidence=("change.applied",),
            ),
        ),
    )


def test_runtime_executes_all_node_kinds_with_provenance() -> None:
    runtime = ObligationRuntime(
        full_plan(),
        context_id="context-1",
        capability_digest="schema-v1",
        call_id_factory=call_ids(
            "call-observe", "call-ask", "call-confirm", "call-act"
        ),
    )

    observe = guarded_step(runtime)
    assert isinstance(observe, ExternalCallAction)
    assert observe.kind == "observe"
    assert observe.node_id == "observe-context"
    assert observe.call_id == "call-observe"
    assert observe.plan_id == "runtime-example"
    assert observe.operation == "read.context"
    assert observe.arguments == {}
    assert len(observe.argument_digest) == 64
    assert observe.capability_digest == "schema-v1"
    assert guarded_step(runtime) is None
    assert runtime.status == PlanRunStatus.WAITING

    observed = runtime.apply_event(
        ExternalResultEvent(
            event_id="event-observe",
            call_id="call-observe",
            external_call_id="evaluator-observe",
            context_id="context-1",
            plan_id="runtime-example",
            node_id="observe-context",
            operation="read.context",
            argument_digest=observe.argument_digest,
            schema_digest="schema-v1",
            status=ExternalResultStatus.SUCCESS,
            payload={"state": "ready"},
        )
    )
    assert observed.disposition == EventDisposition.APPLIED

    ask = guarded_step(runtime)
    assert isinstance(ask, AskAction)
    assert ask.call_id == "call-ask"
    runtime.apply_event(
        InputEvent(
            event_id="event-input",
            call_id="call-ask",
            value="option-a",
        )
    )

    confirm = guarded_step(runtime)
    assert isinstance(confirm, ConfirmAction)
    assert confirm.call_id == "call-confirm"
    runtime.apply_event(
        ConfirmationEvent(
            event_id="event-confirm",
            call_id="call-confirm",
            confirmed=True,
        )
    )

    act = guarded_step(runtime)
    assert isinstance(act, ExternalCallAction)
    assert act.kind == "act"
    assert act.node_id == "apply-change"
    assert act.call_id == "call-act"
    assert act.plan_id == "runtime-example"
    assert act.operation == "write.change"
    assert act.arguments == {"mode": "selected"}
    assert len(act.argument_digest) == 64
    assert act.capability_digest == "schema-v1"
    confirmation_evidence = runtime.ledger.get("input.confirmed")
    assert confirmation_evidence is not None
    assert confirmation_evidence.context_id == "context-1"
    assert confirmation_evidence.plan_id == "runtime-example"
    assert confirmation_evidence.schema_digest == "schema-v1"
    assert confirmation_evidence.authorizations == (
        ActionAuthorization(
            operation="write.change",
            argument_digest=act.argument_digest,
        ),
    )
    runtime.apply_event(
        ExternalResultEvent(
            event_id="event-act",
            call_id="call-act",
            external_call_id="evaluator-act",
            context_id="context-1",
            plan_id="runtime-example",
            node_id="apply-change",
            operation="write.change",
            argument_digest=act.argument_digest,
            schema_digest="schema-v1",
            status=ExternalResultStatus.SUCCESS,
            payload={"applied": True},
        )
    )

    response = guarded_step(runtime)
    assert response == RespondAction(
        node_id="complete",
        plan_id="runtime-example",
        text="The requested change is complete.",
        outcome=ResponseOutcome.COMPLETED,
        evidence_refs=("event-act",),
    )
    assert runtime.status == PlanRunStatus.COMPLETED
    assert guarded_step(runtime) is None

    evidence = runtime.ledger.get("change.applied")
    assert evidence is not None
    assert evidence.event_id == "event-act"
    assert evidence.call_id == "call-act"
    assert evidence.node_id == "apply-change"
    assert evidence.producer_kind == "act"
    assert evidence.operation == "write.change"
    assert evidence.status == EvidenceStatus.SUCCESS
    assert evidence.external_call_id == "evaluator-act"
    assert evidence.context_id == "context-1"
    assert evidence.plan_id == "runtime-example"
    assert evidence.argument_digest == act.argument_digest
    assert evidence.schema_digest == "schema-v1"


def test_failure_never_satisfies_action_or_emits_completion() -> None:
    plan = PlanIR(
        plan_id="failure-example",
        nodes=(
            ActNode(
                id="apply",
                operation="write.value",
                success_evidence_key="value.applied",
            ),
            RespondNode(
                id="complete",
                depends_on=("apply",),
                text="Complete.",
                outcome=ResponseOutcome.COMPLETED,
                requires_evidence=("value.applied",),
            ),
        ),
    )
    runtime = ObligationRuntime(
        plan,
        context_id="context-failure",
        capability_digest="schema-v1",
        call_id_factory=call_ids("call-act"),
    )
    action = guarded_step(runtime)
    assert isinstance(action, ExternalCallAction)

    event = ExternalResultEvent(
        event_id="event-failure",
        call_id="call-act",
        external_call_id="evaluator-call-failure",
        context_id="context-failure",
        plan_id="failure-example",
        node_id="apply",
        operation="write.value",
        argument_digest=action.argument_digest,
        schema_digest="schema-v1",
        status=ExternalResultStatus.FAILURE,
        payload={"accepted": False},
        error={"code": "rejected"},
    )
    result = runtime.apply_event(event)

    assert result.disposition == EventDisposition.FAILURE
    assert runtime.status == PlanRunStatus.BLOCKED
    assert runtime.node_status("apply") == NodeExecutionStatus.FAILED
    assert not runtime.ledger.has("value.applied")
    failure = runtime.ledger.get("value.applied.failure")
    assert failure is not None
    assert failure.value == {
        "payload": {"accepted": False},
        "error": {"code": "rejected"},
    }
    assert failure.source == EvidenceSource.EXTERNAL
    assert failure.status == EvidenceStatus.FAILURE
    assert failure.event_id == "event-failure"
    assert failure.call_id == "call-act"
    assert failure.external_call_id == "evaluator-call-failure"
    assert failure.context_id == "context-failure"
    assert failure.plan_id == "failure-example"
    assert failure.node_id == "apply"
    assert failure.producer_kind == "act"
    assert failure.operation == "write.value"
    assert failure.argument_digest == action.argument_digest
    assert failure.schema_digest == "schema-v1"
    assert guarded_step(runtime) is None

    replay = runtime.apply_event(event)
    assert replay.disposition == EventDisposition.DUPLICATE
    assert runtime.status == PlanRunStatus.BLOCKED


def test_out_of_order_and_wrong_call_results_are_idempotently_ignored() -> None:
    plan = PlanIR(
        plan_id="ordering-example",
        nodes=(
            ObserveNode(
                id="observe",
                operation="read.value",
                success_evidence_key="value.observed",
            ),
            RespondNode(
                id="complete",
                depends_on=("observe",),
                text="Complete.",
                outcome=ResponseOutcome.ANSWERED,
                requires_evidence=("value.observed",),
            ),
        ),
    )
    runtime = ObligationRuntime(
        plan,
        context_id="context-ordering",
        capability_digest="schema-v1",
        call_id_factory=call_ids("call-observe"),
    )

    early = ExternalResultEvent(
        event_id="event-early",
        call_id="call-observe",
        external_call_id="evaluator-early",
        context_id="context-ordering",
        plan_id="ordering-example",
        node_id="observe",
        operation="read.value",
        argument_digest=EMPTY_ARGUMENT_DIGEST,
        schema_digest="schema-v1",
        status=ExternalResultStatus.SUCCESS,
        payload={"value": 3},
    )
    assert runtime.apply_event(early).disposition == EventDisposition.OUT_OF_ORDER
    assert not runtime.ledger.has("value.observed")

    observe = guarded_step(runtime)
    assert isinstance(observe, ExternalCallAction)
    assert runtime.apply_event(early).disposition == EventDisposition.DUPLICATE
    assert runtime.status == PlanRunStatus.WAITING

    wrong_call = ExternalResultEvent(
        event_id="event-wrong-call",
        call_id="call-other",
        external_call_id="evaluator-wrong-call",
        context_id="context-ordering",
        plan_id="ordering-example",
        node_id="observe",
        operation="read.value",
        argument_digest=EMPTY_ARGUMENT_DIGEST,
        schema_digest="schema-v1",
        status=ExternalResultStatus.SUCCESS,
        payload={"value": 4},
    )
    assert runtime.apply_event(wrong_call).disposition == EventDisposition.OUT_OF_ORDER
    assert runtime.status == PlanRunStatus.WAITING

    correct = ExternalResultEvent(
        event_id="event-correct",
        call_id="call-observe",
        external_call_id="evaluator-observe",
        context_id="context-ordering",
        plan_id="ordering-example",
        node_id="observe",
        operation="read.value",
        argument_digest=observe.argument_digest,
        schema_digest="schema-v1",
        status=ExternalResultStatus.SUCCESS,
        payload={"value": 5},
    )
    assert runtime.apply_event(correct).disposition == EventDisposition.APPLIED
    assert runtime.ledger.get("value.observed").value == {"value": 5}
    assert isinstance(guarded_step(runtime), RespondAction)


def test_same_call_with_wrong_event_type_does_not_consume_pending_action() -> None:
    plan = PlanIR(
        plan_id="type-correlation-example",
        nodes=(
            AskNode(
                id="ask",
                prompt="Which value?",
                evidence_key="input.value",
            ),
            ActNode(
                id="apply",
                depends_on=("ask",),
                operation="write.value",
                success_evidence_key="value.applied",
            ),
            RespondNode(
                id="complete",
                depends_on=("apply",),
                text="Complete.",
                outcome=ResponseOutcome.COMPLETED,
                requires_evidence=("value.applied",),
            ),
        ),
    )
    runtime = ObligationRuntime(
        plan,
        call_id_factory=call_ids("call-ask", "call-act"),
    )
    assert isinstance(guarded_step(runtime), AskAction)

    wrong_type = ExternalResultEvent(
        event_id="event-wrong-type",
        call_id="call-ask",
        external_call_id="evaluator-wrong-type",
        context_id="context-type",
        plan_id="type-correlation-example",
        node_id="ask",
        operation="not-an-external-node",
        argument_digest=EMPTY_ARGUMENT_DIGEST,
        schema_digest="schema-type",
        status=ExternalResultStatus.SUCCESS,
        payload="value-a",
    )
    assert runtime.apply_event(wrong_type).disposition == EventDisposition.OUT_OF_ORDER
    assert runtime.pending_action is not None
    assert runtime.pending_action.call_id == "call-ask"

    accepted = runtime.apply_event(
        InputEvent(
            event_id="event-input",
            call_id="call-ask",
            value="value-a",
        )
    )
    assert accepted.disposition == EventDisposition.APPLIED
    assert isinstance(guarded_step(runtime), ExternalCallAction)


def test_declined_confirmation_blocks_descendants() -> None:
    plan = PlanIR(
        plan_id="confirmation-example",
        nodes=(
            ConfirmNode(
                id="confirm",
                prompt="Should I continue?",
                evidence_key="input.confirmed",
                authorizes=("apply",),
            ),
            ActNode(
                id="apply",
                depends_on=("confirm",),
                operation="write.value",
                success_evidence_key="value.applied",
            ),
            RespondNode(
                id="complete",
                depends_on=("apply",),
                text="Complete.",
                outcome=ResponseOutcome.COMPLETED,
                requires_evidence=("value.applied",),
            ),
        ),
    )
    runtime = ObligationRuntime(plan, call_id_factory=call_ids("call-confirm"))
    assert isinstance(guarded_step(runtime), ConfirmAction)

    rejected = runtime.apply_event(
        ConfirmationEvent(
            event_id="event-decline",
            call_id="call-confirm",
            confirmed=False,
        )
    )

    assert rejected.disposition == EventDisposition.FAILURE
    assert runtime.status == PlanRunStatus.BLOCKED
    assert runtime.node_status("apply") == NodeExecutionStatus.PENDING
    assert guarded_step(runtime) is None


def test_runtime_rejects_reused_call_ids() -> None:
    runtime = ObligationRuntime(
        full_plan(),
        call_id_factory=lambda: "call-reused",
    )
    observe = guarded_step(runtime)
    assert isinstance(observe, ExternalCallAction)
    runtime.apply_event(
        ExternalResultEvent(
            event_id="event-observe",
            call_id="call-reused",
            external_call_id="evaluator-observe",
            context_id="context-reuse",
            plan_id="runtime-example",
            node_id="observe-context",
            operation="read.context",
            argument_digest=observe.argument_digest,
            schema_digest="schema-reuse",
            status=ExternalResultStatus.SUCCESS,
            payload={"state": "ready"},
        )
    )

    try:
        guarded_step(runtime)
    except ValueError as exc:
        assert "reused call ID" in str(exc)
    else:
        raise AssertionError("reusing a call ID should fail")


@pytest.mark.parametrize(
    ("field", "wrong_value"),
    (
        ("context_id", "other-context"),
        ("operation", "write.other"),
        ("node_id", "other-node"),
        ("argument_digest", "0" * 64),
        ("schema_digest", "schema-v2"),
        ("plan_id", "other-plan"),
    ),
)
def test_external_result_requires_exact_operation_and_plan_correlation(
    field: str,
    wrong_value: str,
) -> None:
    plan = PlanIR(
        plan_id="correlation-example",
        nodes=(
            ActNode(
                id="apply",
                operation="write.value",
                arguments={"nested": {"value": 7}},
                success_evidence_key="value.applied",
            ),
            RespondNode(
                id="complete",
                depends_on=("apply",),
                text="Complete.",
                outcome=ResponseOutcome.COMPLETED,
                requires_evidence=("value.applied",),
            ),
        ),
    )
    runtime = ObligationRuntime(
        plan,
        context_id="context-correlation",
        capability_digest="schema-v1",
        call_id_factory=call_ids("call-act"),
    )
    action = guarded_step(runtime)
    assert isinstance(action, ExternalCallAction)
    provenance = {
        "external_call_id": "evaluator-act",
        "context_id": "context-correlation",
        "plan_id": "correlation-example",
        "node_id": "apply",
        "operation": "write.value",
        "argument_digest": action.argument_digest,
        "schema_digest": "schema-v1",
    }
    provenance[field] = wrong_value

    rejected = runtime.apply_event(
        ExternalResultEvent(
            event_id=f"event-wrong-{field}",
            call_id="call-act",
            status=ExternalResultStatus.SUCCESS,
            payload={"applied": True},
            **provenance,
        )
    )

    assert rejected.disposition == EventDisposition.OUT_OF_ORDER
    assert runtime.pending_action is not None
    assert runtime.pending_action.call_id == "call-act"
    assert runtime.node_status("apply") == NodeExecutionStatus.WAITING
    assert not runtime.ledger.has("value.applied")

    accepted = runtime.apply_event(
        ExternalResultEvent(
            event_id=f"event-correct-after-{field}",
            call_id="call-act",
            status=ExternalResultStatus.SUCCESS,
            payload={"applied": True},
            external_call_id="evaluator-act",
            context_id="context-correlation",
            plan_id="correlation-example",
            node_id="apply",
            operation="write.value",
            argument_digest=action.argument_digest,
            schema_digest="schema-v1",
        )
    )
    assert accepted.disposition == EventDisposition.APPLIED
    assert runtime.node_status("apply") == NodeExecutionStatus.SATISFIED


@pytest.mark.parametrize(
    "missing_field",
    (
        "external_call_id",
        "context_id",
        "plan_id",
        "node_id",
        "operation",
        "argument_digest",
        "schema_digest",
    ),
)
def test_external_result_contract_requires_complete_provenance(
    missing_field: str,
) -> None:
    payload: dict[str, Any] = {
        "event_id": "event-complete-provenance",
        "call_id": "call-internal",
        "external_call_id": "call-evaluator",
        "context_id": "context-1",
        "plan_id": "plan-1",
        "node_id": "act-1",
        "operation": "write.value",
        "argument_digest": EMPTY_ARGUMENT_DIGEST,
        "schema_digest": "schema-v1",
        "status": ExternalResultStatus.SUCCESS,
        "payload": {"applied": True},
    }
    payload.pop(missing_field)

    with pytest.raises(ValidationError):
        ExternalResultEvent.model_validate(payload)


def _single_action_plan() -> PlanIR:
    return PlanIR(
        plan_id="emission-example",
        nodes=(
            ActNode(
                id="apply",
                operation="write.value",
                arguments={"value": 7},
                success_evidence_key="value.applied",
            ),
            RespondNode(
                id="complete",
                depends_on=("apply",),
                text="Complete.",
                outcome=ResponseOutcome.COMPLETED,
                requires_evidence=("value.applied",),
            ),
        ),
    )


def test_external_emission_without_live_guard_fails_closed() -> None:
    runtime = ObligationRuntime(
        _single_action_plan(),
        call_id_factory=call_ids("call-rejected"),
    )

    with pytest.raises(EmissionRejectedError, match="requires a live capability"):
        runtime.step()

    assert runtime.pending_action is None
    assert runtime.node_status("apply") == NodeExecutionStatus.PENDING


def test_emission_guard_rejects_arguments_after_schema_change() -> None:
    runtime = ObligationRuntime(
        _single_action_plan(),
        call_id_factory=call_ids("call-rejected"),
    )

    def changed_schema_guard(
        kind: Literal["observe", "act"],
        operation: str,
        arguments: dict[str, Any],
    ) -> None:
        assert kind == "act"
        assert operation == "write.value"
        if not isinstance(arguments.get("value"), str):
            raise ValueError("live schema now requires a string")

    with pytest.raises(EmissionRejectedError, match="live capability"):
        runtime.step(emission_guard=changed_schema_guard)

    assert runtime.status == PlanRunStatus.ACTIVE
    assert runtime.pending_action is None
    assert runtime.node_status("apply") == NodeExecutionStatus.PENDING


def test_emission_guard_rejects_revoked_capability() -> None:
    runtime = ObligationRuntime(
        _single_action_plan(),
        call_id_factory=call_ids("call-rejected"),
    )

    def revoked_capability_guard(
        kind: Literal["observe", "act"],
        operation: str,
        arguments: dict[str, Any],
    ) -> None:
        del kind, arguments
        raise LookupError(f"operation {operation!r} is no longer available")

    with pytest.raises(EmissionRejectedError, match="live capability"):
        runtime.step(emission_guard=revoked_capability_guard)

    assert runtime.status == PlanRunStatus.ACTIVE
    assert runtime.pending_action is None
    assert runtime.node_status("apply") == NodeExecutionStatus.PENDING


def test_runtime_isolates_nested_arguments_from_the_caller_owned_plan() -> None:
    original_plan = PlanIR(
        plan_id="nested-isolation-example",
        nodes=(
            ActNode(
                id="apply",
                operation="write.value",
                arguments={"nested": {"items": [{"value": 7}]}},
                success_evidence_key="value.applied",
            ),
            RespondNode(
                id="complete",
                depends_on=("apply",),
                text="Complete.",
                outcome=ResponseOutcome.COMPLETED,
                requires_evidence=("value.applied",),
            ),
        ),
    )
    runtime = ObligationRuntime(
        original_plan,
        call_id_factory=call_ids("call-isolated"),
    )

    # Frozen Pydantic models do not recursively freeze dictionaries.  A caller
    # retaining the original plan must nevertheless be unable to alter what
    # the runtime emits after construction.
    original_node = original_plan.node_by_id["apply"]
    assert isinstance(original_node, ActNode)
    original_node.arguments["nested"]["items"][0]["value"] = 99

    action = guarded_step(runtime)

    assert isinstance(action, ExternalCallAction)
    assert action.arguments == {"nested": {"items": [{"value": 7}]}}


def test_runtime_detects_nested_mutation_before_the_emission_guard_runs() -> None:
    runtime = ObligationRuntime(
        PlanIR(
            plan_id="nested-tamper-example",
            nodes=(
                ActNode(
                    id="apply",
                    operation="write.value",
                    arguments={"nested": {"items": [{"value": 7}]}},
                    success_evidence_key="value.applied",
                ),
                RespondNode(
                    id="complete",
                    depends_on=("apply",),
                    text="Complete.",
                    outcome=ResponseOutcome.COMPLETED,
                    requires_evidence=("value.applied",),
                ),
            ),
        ),
        call_id_factory=call_ids("call-must-not-be-issued"),
    )
    runtime_node = runtime.plan.node_by_id["apply"]
    assert isinstance(runtime_node, ActNode)
    runtime_node.arguments["nested"]["items"][0]["value"] = 99
    guard_was_called = False

    def recording_guard(
        kind: Literal["observe", "act"],
        operation: str,
        arguments: dict[str, Any],
    ) -> None:
        nonlocal guard_was_called
        del kind, operation, arguments
        guard_was_called = True

    with pytest.raises(EmissionRejectedError, match="verified plan changed"):
        runtime.step(emission_guard=recording_guard)

    assert not guard_was_called
    assert runtime.pending_action is None
    assert runtime.node_status("apply") == NodeExecutionStatus.PENDING
