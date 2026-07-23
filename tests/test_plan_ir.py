from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from carbench_agent_core.plan_ir import (
    ActNode,
    AskNode,
    ConfirmNode,
    ObserveNode,
    PlanIR,
    RespondNode,
    ResponseOutcome,
)


def completed_plan() -> PlanIR:
    return PlanIR(
        plan_id="example-plan",
        nodes=(
            ObserveNode(
                id="observe-context",
                operation="read.context",
                success_evidence_key="context.observed",
            ),
            AskNode(
                id="ask-choice",
                prompt="Which option should be used?",
                evidence_key="input.choice",
            ),
            ConfirmNode(
                id="confirm-change",
                depends_on=("observe-context", "ask-choice"),
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
                id="terminal",
                depends_on=("apply-change",),
                text="The requested change is complete.",
                outcome=ResponseOutcome.COMPLETED,
                requires_evidence=("change.applied",),
            ),
        ),
    )


def test_plan_accepts_typed_completed_dag_and_returns_stable_order() -> None:
    plan = completed_plan()

    assert plan.topological_order() == (
        "observe-context",
        "ask-choice",
        "confirm-change",
        "apply-change",
        "terminal",
    )
    assert plan.terminal.outcome == ResponseOutcome.COMPLETED
    assert set(plan.node_by_id) == {
        "observe-context",
        "ask-choice",
        "confirm-change",
        "apply-change",
        "terminal",
    }


def test_completed_outcome_accepts_carried_evidence_input() -> None:
    plan = PlanIR(
        plan_id="carried-completion",
        evidence_inputs=("prior.action.success",),
        nodes=(
            RespondNode(
                id="terminal",
                text="The prior action is complete.",
                outcome=ResponseOutcome.COMPLETED,
                requires_evidence=("prior.action.success",),
            ),
        ),
    )

    assert plan.evidence_inputs == ("prior.action.success",)


@pytest.mark.parametrize(
    "terminal",
    [
        RespondNode(
            id="terminal",
            text="I cannot perform that operation.",
            outcome=ResponseOutcome.REFUSED,
        ),
        RespondNode(
            id="terminal",
            text="The request was declined.",
            outcome=ResponseOutcome.DECLINED,
            requires_evidence=("input.declined",),
        ),
        RespondNode(
            id="terminal",
            text="I stopped safely after the failure.",
            outcome=ResponseOutcome.FAIL_SAFE,
        ),
    ],
)
def test_plan_accepts_non_effect_terminal_outcomes(terminal: RespondNode) -> None:
    evidence_inputs = (
        ("input.declined",) if terminal.outcome == ResponseOutcome.DECLINED else ()
    )

    plan = PlanIR(
        plan_id=f"{terminal.outcome.value}-plan",
        evidence_inputs=evidence_inputs,
        nodes=(terminal,),
    )

    assert plan.terminal.outcome == terminal.outcome


def test_answered_outcome_may_cite_observation_and_input_evidence() -> None:
    plan = PlanIR(
        plan_id="answer-plan",
        evidence_inputs=("input.scope",),
        nodes=(
            ObserveNode(
                id="inspect",
                operation="read.resource",
                success_evidence_key="resource.observed",
            ),
            RespondNode(
                id="terminal",
                depends_on=("inspect",),
                text="Here is the requested information.",
                outcome=ResponseOutcome.ANSWERED,
                requires_evidence=("resource.observed", "input.scope"),
            ),
        ),
    )

    assert plan.terminal.requires_evidence == (
        "resource.observed",
        "input.scope",
    )


def test_answered_outcome_cannot_drop_observation_evidence() -> None:
    with pytest.raises(ValidationError, match="every observation success key"):
        PlanIR(
            plan_id="ungrounded-answer",
            nodes=(
                ObserveNode(
                    id="inspect",
                    operation="read.resource",
                    success_evidence_key="resource.observed",
                ),
                RespondNode(
                    id="terminal",
                    depends_on=("inspect",),
                    text="Here is the requested information.",
                    outcome=ResponseOutcome.ANSWERED,
                ),
            ),
        )


@pytest.mark.parametrize(
    ("nodes", "message"),
    [
        (
            (
                ActNode(
                    id="same",
                    operation="write.first",
                    success_evidence_key="first.done",
                ),
                ActNode(
                    id="same",
                    operation="write.second",
                    success_evidence_key="second.done",
                ),
                RespondNode(
                    id="terminal",
                    depends_on=("same",),
                    text="Complete.",
                    outcome=ResponseOutcome.COMPLETED,
                    requires_evidence=("first.done",),
                ),
            ),
            "node IDs must be unique",
        ),
        (
            (
                ActNode(
                    id="apply",
                    depends_on=("missing",),
                    operation="write.value",
                    success_evidence_key="value.applied",
                ),
                RespondNode(
                    id="terminal",
                    depends_on=("apply",),
                    text="Complete.",
                    outcome=ResponseOutcome.COMPLETED,
                    requires_evidence=("value.applied",),
                ),
            ),
            "unknown nodes",
        ),
        (
            (
                ObserveNode(
                    id="first",
                    depends_on=("second",),
                    operation="read.first",
                    success_evidence_key="first.observed",
                ),
                ActNode(
                    id="second",
                    depends_on=("first",),
                    operation="write.second",
                    success_evidence_key="second.applied",
                ),
                RespondNode(
                    id="terminal",
                    depends_on=("second",),
                    text="Complete.",
                    outcome=ResponseOutcome.COMPLETED,
                    requires_evidence=("second.applied",),
                ),
            ),
            "acyclic graph",
        ),
    ],
)
def test_plan_rejects_invalid_graphs(nodes: tuple[Any, ...], message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        PlanIR(plan_id="invalid-plan", nodes=nodes)


def test_plan_requires_exactly_one_terminal_response() -> None:
    with pytest.raises(ValidationError, match="exactly one response"):
        PlanIR(
            plan_id="missing-response",
            nodes=(
                ObserveNode(
                    id="inspect",
                    operation="read.value",
                    success_evidence_key="value.observed",
                ),
            ),
        )

    with pytest.raises(ValidationError, match="exactly one response"):
        PlanIR(
            plan_id="multiple-responses",
            nodes=(
                RespondNode(
                    id="first",
                    text="First.",
                    outcome=ResponseOutcome.ANSWERED,
                ),
                RespondNode(
                    id="second",
                    text="Second.",
                    outcome=ResponseOutcome.REFUSED,
                ),
            ),
        )


def test_plan_rejects_response_with_dependents_and_unreachable_nodes() -> None:
    with pytest.raises(ValidationError, match="response must be terminal"):
        PlanIR(
            plan_id="nonterminal-response",
            nodes=(
                RespondNode(
                    id="response",
                    text="Answer.",
                    outcome=ResponseOutcome.ANSWERED,
                ),
                AskNode(
                    id="later",
                    depends_on=("response",),
                    prompt="More?",
                    evidence_key="input.more",
                ),
            ),
        )

    with pytest.raises(ValidationError, match="every plan node must lead"):
        PlanIR(
            plan_id="unreachable-node",
            nodes=(
                ObserveNode(
                    id="orphan",
                    operation="read.orphan",
                    success_evidence_key="orphan.observed",
                ),
                RespondNode(
                    id="terminal",
                    text="Answer.",
                    outcome=ResponseOutcome.ANSWERED,
                ),
            ),
        )


def test_plan_rejects_unknown_duplicate_and_reproduced_evidence() -> None:
    with pytest.raises(ValidationError, match="no producer or input"):
        PlanIR(
            plan_id="unknown-evidence",
            nodes=(
                RespondNode(
                    id="terminal",
                    text="Answer.",
                    outcome=ResponseOutcome.ANSWERED,
                    requires_evidence=("missing",),
                ),
            ),
        )

    with pytest.raises(ValidationError, match="evidence inputs must be unique"):
        PlanIR(
            plan_id="duplicate-input",
            evidence_inputs=("prior", "prior"),
            nodes=(
                RespondNode(
                    id="terminal",
                    text="Answer.",
                    outcome=ResponseOutcome.ANSWERED,
                ),
            ),
        )

    with pytest.raises(ValidationError, match="cannot be reproduced"):
        PlanIR(
            plan_id="reproduced-input",
            evidence_inputs=("shared",),
            nodes=(
                ObserveNode(
                    id="inspect",
                    operation="read.value",
                    success_evidence_key="shared",
                ),
                RespondNode(
                    id="terminal",
                    depends_on=("inspect",),
                    text="Answer.",
                    outcome=ResponseOutcome.ANSWERED,
                    requires_evidence=("shared",),
                ),
            ),
        )


def test_plan_rejects_duplicate_evidence_producers_and_requirements() -> None:
    with pytest.raises(ValidationError, match="multiple producers"):
        PlanIR(
            plan_id="duplicate-producer",
            nodes=(
                ObserveNode(
                    id="observe",
                    operation="read.value",
                    success_evidence_key="shared",
                ),
                AskNode(
                    id="ask",
                    depends_on=("observe",),
                    prompt="Which value?",
                    evidence_key="shared",
                ),
                RespondNode(
                    id="terminal",
                    depends_on=("ask",),
                    text="Answer.",
                    outcome=ResponseOutcome.ANSWERED,
                    requires_evidence=("shared",),
                ),
            ),
        )

    with pytest.raises(ValidationError, match="requirements must be unique"):
        PlanIR(
            plan_id="duplicate-requirement",
            evidence_inputs=("prior",),
            nodes=(
                RespondNode(
                    id="terminal",
                    text="Answer.",
                    outcome=ResponseOutcome.ANSWERED,
                    requires_evidence=("prior", "prior"),
                ),
            ),
        )


@pytest.mark.parametrize(
    ("confirmation", "nodes", "message"),
    [
        (
            ConfirmNode(
                id="confirm",
                prompt="Proceed?",
                evidence_key="input.confirmed",
                authorizes=("ask",),
            ),
            (
                AskNode(
                    id="ask",
                    depends_on=("confirm",),
                    prompt="Which value?",
                    evidence_key="input.value",
                ),
                RespondNode(
                    id="terminal",
                    depends_on=("ask",),
                    text="Answered.",
                    outcome=ResponseOutcome.ANSWERED,
                    requires_evidence=("input.value",),
                ),
            ),
            "must authorize an action node",
        ),
        (
            ConfirmNode(
                id="confirm",
                prompt="Proceed?",
                evidence_key="input.confirmed",
                authorizes=("apply",),
            ),
            (
                ActNode(
                    id="apply",
                    operation="write.value",
                    success_evidence_key="value.applied",
                ),
                RespondNode(
                    id="terminal",
                    depends_on=("confirm", "apply"),
                    text="Complete.",
                    outcome=ResponseOutcome.COMPLETED,
                    requires_evidence=("value.applied",),
                ),
            ),
            "must depend on confirmation",
        ),
    ],
)
def test_confirmation_authorization_targets_exact_dependent_actions(
    confirmation: ConfirmNode,
    nodes: tuple[Any, ...],
    message: str,
) -> None:
    with pytest.raises(ValidationError, match=message):
        PlanIR(
            plan_id="invalid-confirmation-scope",
            nodes=(confirmation, *nodes),
        )


def test_confirmation_rejects_duplicate_authorization_targets() -> None:
    with pytest.raises(ValidationError, match="duplicate authorization targets"):
        PlanIR(
            plan_id="duplicate-confirmation-target",
            nodes=(
                ConfirmNode(
                    id="confirm",
                    prompt="Proceed?",
                    evidence_key="input.confirmed",
                    authorizes=("apply", "apply"),
                ),
                ActNode(
                    id="apply",
                    depends_on=("confirm",),
                    operation="write.value",
                    success_evidence_key="value.applied",
                ),
                RespondNode(
                    id="terminal",
                    depends_on=("apply",),
                    text="Complete.",
                    outcome=ResponseOutcome.COMPLETED,
                    requires_evidence=("value.applied",),
                ),
            ),
        )


def test_completed_outcome_must_cite_every_action() -> None:
    with pytest.raises(ValidationError, match="every action success key"):
        PlanIR(
            plan_id="missing-action-proof",
            nodes=(
                ActNode(
                    id="first",
                    operation="write.first",
                    success_evidence_key="first.done",
                ),
                ActNode(
                    id="second",
                    depends_on=("first",),
                    operation="write.second",
                    success_evidence_key="second.done",
                ),
                RespondNode(
                    id="terminal",
                    depends_on=("second",),
                    text="Complete.",
                    outcome=ResponseOutcome.COMPLETED,
                    requires_evidence=("second.done",),
                ),
            ),
        )


@pytest.mark.parametrize(
    ("outcome", "external_kind", "message"),
    [
        (ResponseOutcome.ANSWERED, "act", "answered outcome cannot contain actions"),
        (ResponseOutcome.REFUSED, "observe", "cannot contain external calls"),
        (ResponseOutcome.DECLINED, "act", "declined outcome cannot contain actions"),
        (ResponseOutcome.FAIL_SAFE, "act", "fail_safe outcome cannot contain actions"),
    ],
)
def test_non_effect_outcomes_reject_incompatible_external_nodes(
    outcome: ResponseOutcome,
    external_kind: str,
    message: str,
) -> None:
    external = (
        ActNode(
            id="external",
            operation="write.value",
            success_evidence_key="external.done",
        )
        if external_kind == "act"
        else ObserveNode(
            id="external",
            operation="read.value",
            success_evidence_key="external.done",
        )
    )
    with pytest.raises(ValidationError, match=message):
        PlanIR(
            plan_id="invalid-outcome",
            nodes=(
                external,
                RespondNode(
                    id="terminal",
                    depends_on=("external",),
                    text="Terminal.",
                    outcome=outcome,
                    requires_evidence=("external.done",),
                ),
            ),
        )


def test_identifiers_reject_whitespace_only_values() -> None:
    with pytest.raises(ValidationError):
        PlanIR(
            plan_id="   ",
            nodes=(
                RespondNode(
                    id="terminal",
                    text="Answer.",
                    outcome=ResponseOutcome.ANSWERED,
                ),
            ),
        )

    with pytest.raises(ValidationError):
        ObserveNode(
            id="\t",
            operation="read.value",
            success_evidence_key="value.observed",
        )

    with pytest.raises(ValidationError):
        RespondNode(
            id="terminal",
            depends_on=("  ",),
            text="Answer.",
            outcome=ResponseOutcome.ANSWERED,
        )
