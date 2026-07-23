from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import pytest

from carbench_agent_core.evidence_ledger import (
    EvidenceRecord,
    EvidenceSource,
    EvidenceStatus,
)
from carbench_agent_core.semantic_compiler import (
    CompilationRequest,
    CompilerBackendError,
    CompilerInputError,
    CompilerTokenUsage,
    ConversationEvent,
    ModelCandidate,
    PlanRepairExhaustedError,
    SEMANTIC_COMPILER_INSTRUCTIONS,
    SEMANTIC_REVIEW_INSTRUCTION,
    SemanticCompiler,
    SemanticCompilerLimits,
)
from track_2_agent_under_test_cerebras.cerebras_client import (
    DEFAULT_CEREBRAS_API_BASE,
    TokenUsage,
)
from track_2_agent_under_test_cerebras.plan_compiler_backend import (
    CerebrasCompilerSettings,
    CerebrasStructuredPlanBackend,
    CompilerConfigurationError,
    create_cerebras_semantic_compiler,
)


def capability(
    name: str = "update_resource",
    *,
    properties: dict[str, Any] | None = None,
    required: list[str] | None = None,
    description: str = "Access a synthetic resource.",
) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "x-pact-effect": "act",
            "parameters": {
                "type": "object",
                "properties": properties
                or {"resource_id": {"type": "string", "minLength": 1}},
                "required": required or ["resource_id"],
                "additionalProperties": False,
            },
        },
    }


def plan_json(
    *,
    operation: str = "update_resource",
    arguments: dict[str, Any] | None = None,
    plan_id: str = "request-1",
    evidence_inputs: list[str] | None = None,
    requires_evidence: list[str] | None = None,
) -> str:
    return json.dumps(
        {
            "plan_id": plan_id,
            "evidence_inputs": evidence_inputs or [],
            "nodes": [
                {
                    "id": "apply",
                    "kind": "act",
                    "depends_on": [],
                    "operation": operation,
                    "arguments_json": json.dumps(arguments or {"resource_id": "alpha"}),
                    "success_evidence_key": "resource.updated",
                },
                {
                    "id": "respond",
                    "kind": "respond",
                    "depends_on": ["apply"],
                    "text": "The requested update is complete.",
                    "outcome": "completed",
                    "requires_evidence": requires_evidence or ["resource.updated"],
                },
            ],
        }
    )


class SequenceBackend:
    def __init__(self, outcomes: list[ModelCandidate | BaseException]) -> None:
        self.outcomes = list(outcomes)
        self.calls: list[dict[str, Any]] = []

    def generate(
        self,
        *,
        messages: list[dict[str, str]],
        response_schema: dict[str, Any],
        response_schema_name: str,
    ) -> ModelCandidate:
        self.calls.append(
            {
                "messages": messages,
                "response_schema": response_schema,
                "response_schema_name": response_schema_name,
            }
        )
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def request(
    *,
    evidence: list[EvidenceRecord] | None = None,
    required_completion_evidence: set[str] | None = None,
) -> CompilationRequest:
    return CompilationRequest.from_live_tools(
        plan_id="request-1",
        trusted_policy="Use only authorized capabilities and report truthfully.",
        goal="Update synthetic resource alpha.",
        current_user_event="Please update resource alpha.",
        conversation=(ConversationEvent(role="user", content="Earlier context."),),
        tools=[capability()],
        evidence=evidence or [],
        required_completion_evidence=required_completion_evidence or set(),
    )


def response_plan_json(
    *,
    outcome: str,
    text: str,
    evidence_inputs: list[str] | None = None,
    requires_evidence: list[str] | None = None,
    plan_id: str = "request-1",
) -> str:
    return json.dumps(
        {
            "plan_id": plan_id,
            "evidence_inputs": evidence_inputs or [],
            "nodes": [
                {
                    "id": "respond",
                    "kind": "respond",
                    "depends_on": [],
                    "text": text,
                    "outcome": outcome,
                    "requires_evidence": requires_evidence or [],
                }
            ],
        }
    )


def test_compiler_accepts_strict_plan_and_records_content_digests() -> None:
    backend = SequenceBackend(
        [
            ModelCandidate(
                text=plan_json(),
                model="synthetic-model",
                duration_ms=12.5,
                cost=0.02,
                quota_wait_ms=4.5,
                usage=CompilerTokenUsage(
                    prompt_tokens=100,
                    completion_tokens=30,
                    thinking_tokens=20,
                ),
            )
        ]
    )

    result = SemanticCompiler(backend).compile(request())

    assert result.plan.plan_id == "request-1"
    assert result.plan.topological_order() == ("apply", "respond")
    assert len(result.attempts) == 1
    assert result.attempts[0].accepted is True
    assert result.attempts[0].was_repair is False
    assert result.usage == CompilerTokenUsage(
        prompt_tokens=100,
        completion_tokens=30,
        thinking_tokens=20,
    )
    assert result.cost == 0.02
    assert result.quota_wait_ms == 4.5
    assert len(result.request_sha256) == 64
    assert len(result.capability_sha256) == 64

    call = backend.calls[0]
    assert call["response_schema_name"] == "obligation_plan"
    assert call["response_schema"]["properties"]["plan_id"]["enum"] == ["request-1"]
    node_schema = call["response_schema"]["properties"]["nodes"]["items"]
    variants = node_schema["anyOf"]
    act_schema = next(
        item for item in variants if item["properties"]["kind"]["enum"] == ["act"]
    )
    respond_schema = next(
        item for item in variants if item["properties"]["kind"]["enum"] == ["respond"]
    )
    assert act_schema["properties"]["operation"] == {"$ref": "#/$defs/operation"}
    assert call["response_schema"]["$defs"]["operation"]["enum"] == ["update_resource"]
    assert respond_schema["properties"]["outcome"]["enum"] == [
        "completed",
        "answered",
        "refused",
        "declined",
        "fail_safe",
    ]
    assert "completion" not in json.dumps(call["response_schema"])
    assert "requires_success_evidence" not in json.dumps(call["response_schema"])
    assert len(json.dumps(call["response_schema"])) < 5000
    assert "maxItems" not in json.dumps(call["response_schema"])


def test_provider_wire_schema_is_forward_compatible_with_strict_v2() -> None:
    backend = SequenceBackend([ModelCandidate(text=plan_json())])
    SemanticCompiler(backend).compile(request())
    schema = backend.calls[0]["response_schema"]

    assert len(json.dumps(schema, separators=(",", ":"))) <= 5000

    def inspect(value: Any) -> None:
        if isinstance(value, dict):
            if value.get("type") == "object":
                assert value.get("additionalProperties") is False
                assert set(value.get("required", [])) == set(
                    value.get("properties", {})
                )
            assert not ({"minItems", "maxItems", "pattern", "format"} & value.keys())
            for child in value.values():
                inspect(child)
        elif isinstance(value, list):
            for child in value:
                inspect(child)

    inspect(schema)


def test_compiler_monotonically_closes_completed_action_evidence() -> None:
    payload = json.loads(plan_json(requires_evidence=["resource.updated"]))
    payload["nodes"].insert(
        0,
        {
            "id": "prepare",
            "kind": "act",
            "depends_on": [],
            "operation": "update_resource",
            "arguments_json": json.dumps({"resource_id": "prerequisite"}),
            "success_evidence_key": "prerequisite.updated",
        },
    )
    payload["nodes"][1]["depends_on"] = ["prepare"]
    backend = SequenceBackend([ModelCandidate(text=json.dumps(payload))])

    result = SemanticCompiler(backend).compile(request())

    assert len(backend.calls) == 1
    assert result.plan.terminal.requires_evidence == (
        "prerequisite.updated",
        "resource.updated",
    )


def test_large_live_surface_keeps_provider_schema_under_hard_limit() -> None:
    tools = [
        capability(f"synthetic_capability_{index:03d}_{'x' * 24}")
        for index in range(100)
    ]
    compile_request = CompilationRequest.from_live_tools(
        plan_id="request-1",
        trusted_policy="Answer safely.",
        goal="Return an informational response.",
        tools=tools,
    )
    backend = SequenceBackend(
        [
            ModelCandidate(
                text=response_plan_json(
                    outcome="answered",
                    text="The response is available.",
                )
            )
        ]
    )

    SemanticCompiler(backend).compile(compile_request)

    schema = backend.calls[0]["response_schema"]
    assert len(json.dumps(schema, separators=(",", ":"))) <= 5000
    assert schema["$defs"]["operation"] == {"type": "string"}


def test_compiler_performs_exactly_one_structured_repair() -> None:
    backend = SequenceBackend(
        [
            ModelCandidate(text=plan_json(operation="invented_operation")),
            ModelCandidate(text=plan_json()),
        ]
    )

    result = SemanticCompiler(backend).compile(request())

    assert len(backend.calls) == 2
    assert [attempt.accepted for attempt in result.attempts] == [False, True]
    assert result.attempts[0].issues[0].code == "operation_unavailable"
    assert result.attempts[1].was_repair is True
    repair = json.loads(backend.calls[1]["messages"][-1]["content"])
    assert repair["verification_issues"][0]["code"] == ("operation_unavailable")
    assert "invented_operation" not in repair["verification_issues"][0]["message"]


def test_action_plan_receives_one_fresh_semantic_review() -> None:
    backend = SequenceBackend(
        [
            ModelCandidate(text=plan_json(arguments={"resource_id": "draft"})),
            ModelCandidate(text=plan_json(arguments={"resource_id": "reviewed"})),
        ]
    )

    result = SemanticCompiler(backend, semantic_review=True).compile(request())

    assert len(backend.calls) == 2
    assert [attempt.accepted for attempt in result.attempts] == [True, True]
    assert [attempt.was_repair for attempt in result.attempts] == [False, False]
    assert result.plan.node_by_id["apply"].arguments == {
        "resource_id": "reviewed"
    }
    assert backend.calls[1]["messages"][-1] == {
        "role": "user",
        "content": SEMANTIC_REVIEW_INSTRUCTION,
    }
    reviewed_input = json.loads(backend.calls[1]["messages"][-2]["content"])
    assert reviewed_input["plan_id"] == "request-1"
    assert json.loads(reviewed_input["nodes"][0]["arguments_json"]) == {
        "resource_id": "draft"
    }


def test_failed_semantic_review_fails_closed_instead_of_using_draft() -> None:
    backend = SequenceBackend(
        [
            ModelCandidate(text=plan_json()),
            ModelCandidate(text=plan_json(operation="invented_operation")),
        ]
    )

    with pytest.raises(PlanRepairExhaustedError) as captured:
        SemanticCompiler(backend, semantic_review=True).compile(request())

    assert len(backend.calls) == 2
    assert [attempt.accepted for attempt in captured.value.attempts] == [True, False]
    assert {issue.code for issue in captured.value.issues} == {
        "operation_unavailable"
    }


def test_truncated_semantic_review_fails_closed_and_preserves_all_usage() -> None:
    backend = SequenceBackend(
        [
            ModelCandidate(
                text=plan_json(),
                usage=CompilerTokenUsage(
                    prompt_tokens=10,
                    completion_tokens=4,
                    thinking_tokens=3,
                ),
            ),
            ModelCandidate(
                text="",
                finish_reason="length",
                usage=CompilerTokenUsage(
                    prompt_tokens=12,
                    completion_tokens=6,
                    thinking_tokens=5,
                ),
            ),
        ]
    )

    with pytest.raises(PlanRepairExhaustedError) as captured:
        SemanticCompiler(backend, semantic_review=True).compile(request())

    assert [item.phase for item in captured.value.attempts] == [
        "proposal",
        "audit",
    ]
    assert captured.value.issues[0].code == "provider_output_truncated"
    assert captured.value.usage == CompilerTokenUsage(
        prompt_tokens=22,
        completion_tokens=10,
        thinking_tokens=8,
    )


def test_semantic_review_backend_failure_preserves_proposal_usage() -> None:
    backend = SequenceBackend(
        [
            ModelCandidate(
                text=plan_json(),
                cost=0.2,
                quota_wait_ms=15,
                usage=CompilerTokenUsage(
                    prompt_tokens=9,
                    completion_tokens=4,
                    thinking_tokens=2,
                ),
            ),
            RuntimeError("review provider unavailable"),
        ]
    )

    with pytest.raises(CompilerBackendError) as captured:
        SemanticCompiler(backend, semantic_review=True).compile(request())

    assert len(backend.calls) == 2
    assert [item.phase for item in captured.value.attempts] == ["proposal"]
    assert captured.value.cost == 0.2
    assert captured.value.quota_wait_ms == 15
    assert captured.value.usage == CompilerTokenUsage(
        prompt_tokens=9,
        completion_tokens=4,
        thinking_tokens=2,
    )


def test_structural_repair_then_semantic_review_uses_at_most_three_calls() -> None:
    backend = SequenceBackend(
        [
            ModelCandidate(text=plan_json(operation="invented_operation")),
            ModelCandidate(text=plan_json(arguments={"resource_id": "repaired"})),
            ModelCandidate(text=plan_json(arguments={"resource_id": "reviewed"})),
        ]
    )

    result = SemanticCompiler(backend, semantic_review=True).compile(request())

    assert len(backend.calls) == 3
    assert [attempt.was_repair for attempt in result.attempts] == [False, True, False]
    assert result.plan.node_by_id["apply"].arguments == {
        "resource_id": "reviewed"
    }


def test_non_action_plan_does_not_spend_semantic_review_call() -> None:
    backend = SequenceBackend(
        [
            ModelCandidate(
                text=response_plan_json(
                    outcome="answered",
                    text="Here is the requested information.",
                )
            )
        ]
    )

    result = SemanticCompiler(backend, semantic_review=True).compile(request())

    assert len(backend.calls) == 1
    assert result.plan.terminal.outcome.value == "answered"


def test_completed_proof_closure_filters_non_action_model_references() -> None:
    candidate = plan_json(
        requires_evidence=["interaction.confirmed", "resource.updated"]
    )

    result = SemanticCompiler(SequenceBackend([ModelCandidate(text=candidate)])).compile(
        request()
    )

    assert result.plan.terminal.requires_evidence == ("resource.updated",)


def test_compiler_materializes_explicit_confirmation_authorization_edge() -> None:
    payload = json.loads(plan_json())
    payload["nodes"].insert(
        0,
        {
            "id": "authorize",
            "kind": "confirm",
            "depends_on": [],
            "prompt": "Proceed with the exact update?",
            "evidence_key": "resource.authorized",
            "authorizes": ["apply"],
        },
    )
    payload["nodes"][1]["depends_on"] = []

    result = SemanticCompiler(
        SequenceBackend([ModelCandidate(text=json.dumps(payload))])
    ).compile(request())

    assert result.plan.node_by_id["apply"].depends_on == ("authorize",)


def test_compiler_stops_after_one_failed_repair() -> None:
    backend = SequenceBackend(
        [ModelCandidate(text="not-json"), ModelCandidate(text="still-not-json")]
    )

    with pytest.raises(PlanRepairExhaustedError) as captured:
        SemanticCompiler(backend).compile(request())

    assert len(backend.calls) == 2
    assert len(captured.value.attempts) == 2
    assert captured.value.issues[0].code == "invalid_json"
    assert "not-json" not in str(captured.value)


def test_provider_length_truncation_is_typed_and_not_retried() -> None:
    backend = SequenceBackend(
        [
            ModelCandidate(
                text="",
                finish_reason="length",
                usage=CompilerTokenUsage(
                    prompt_tokens=100,
                    completion_tokens=64,
                    thinking_tokens=63,
                ),
            ),
            ModelCandidate(text=plan_json()),
        ]
    )

    with pytest.raises(PlanRepairExhaustedError) as captured:
        SemanticCompiler(backend).compile(request())

    assert len(backend.calls) == 1
    assert len(captured.value.attempts) == 1
    assert captured.value.attempts[0].finish_reason == "length"
    assert captured.value.issues[0].code == "provider_output_truncated"
    assert captured.value.usage.thinking_tokens == 63


def test_compiler_rejects_schema_invalid_operation_arguments() -> None:
    backend = SequenceBackend(
        [
            ModelCandidate(text=plan_json(arguments={"resource_id": ""})),
            ModelCandidate(text=plan_json(arguments={"resource_id": ""})),
        ]
    )

    with pytest.raises(PlanRepairExhaustedError) as captured:
        SemanticCompiler(backend).compile(request())

    assert captured.value.issues[0].code == "arguments_invalid_format"
    assert captured.value.issues[0].path == (
        "nodes",
        "apply",
        "resource_id",
    )
    assert "alpha" not in captured.value.issues[0].message


@pytest.mark.parametrize(
    "candidate",
    [
        '{"plan_id":"one","plan_id":"two","nodes":[]}',
        '{"plan_id":"one","nodes":[],"score":NaN}',
        "[]",
        "```json\n{}\n```",
    ],
)
def test_compiler_rejects_non_strict_json(candidate: str) -> None:
    backend = SequenceBackend([ModelCandidate(text=candidate)])

    with pytest.raises(PlanRepairExhaustedError) as captured:
        SemanticCompiler(
            backend,
            limits=SemanticCompilerLimits(max_repair_attempts=0),
        ).compile(request())

    assert captured.value.issues[0].code == "invalid_json"


def test_compiler_backend_failure_is_typed_and_not_repaired() -> None:
    backend = SequenceBackend([RuntimeError("provider unavailable")])

    with pytest.raises(CompilerBackendError) as captured:
        SemanticCompiler(backend).compile(request())

    assert len(backend.calls) == 1
    assert isinstance(captured.value.__cause__, RuntimeError)


def test_backend_failure_after_repair_request_preserves_prior_usage() -> None:
    backend = SequenceBackend(
        [
            ModelCandidate(
                text="not-json",
                cost=0.4,
                quota_wait_ms=75,
                usage=CompilerTokenUsage(
                    prompt_tokens=11,
                    completion_tokens=3,
                    thinking_tokens=2,
                ),
            ),
            RuntimeError("provider unavailable"),
        ]
    )

    with pytest.raises(CompilerBackendError) as captured:
        SemanticCompiler(backend).compile(request())

    assert len(captured.value.attempts) == 1
    prior = captured.value.attempts[0]
    assert prior.cost == 0.4
    assert prior.quota_wait_ms == 75
    assert prior.usage.prompt_tokens == 11
    assert captured.value.cost == 0.4
    assert captured.value.quota_wait_ms == 75
    assert captured.value.usage == CompilerTokenUsage(
        prompt_tokens=11,
        completion_tokens=3,
        thinking_tokens=2,
    )


def test_request_carries_immutable_evidence_provenance_into_prompt() -> None:
    record = EvidenceRecord(
        key="prior.observation",
        value={"version": 3},
        source=EvidenceSource.EXTERNAL,
        event_id="event-7",
        call_id="call-4",
        node_id="observe-prior",
        producer_kind="observe",
        operation="update_resource",
    )
    backend = SequenceBackend([ModelCandidate(text=plan_json())])

    SemanticCompiler(backend).compile(request(evidence=[record]))

    messages = backend.calls[0]["messages"]
    policy = json.loads(messages[1]["content"])
    payload = json.loads(messages[2]["content"])
    assert policy == {
        "trusted_policy": "Use only authorized capabilities and report truthfully."
    }
    assert payload["goal"] == "Update synthetic resource alpha."
    assert payload["current_user_event"] == "Please update resource alpha."
    assert payload["recent_conversation"] == [
        {"content": "Earlier context.", "name": None, "role": "user"}
    ]
    assert "trusted_policy" not in payload
    assert payload["live_capabilities"][0]["effect"] == "act"
    assert payload["existing_evidence"] == [
        {
            "key": "prior.observation",
            "operation": "update_resource",
            "argument_digest": None,
            "authorizations": [],
            "context_id": None,
            "external_call_id": None,
            "plan_id": None,
            "schema_digest": None,
            "producer_call_id": "call-4",
            "producer_event_id": "event-7",
            "producer_node_id": "observe-prior",
            "producer_kind": "observe",
            "source": "external",
            "status": "success",
            "value": {"version": 3},
        }
    ]


def test_compiler_preserves_carried_completion_evidence_obligations() -> None:
    prior = EvidenceRecord(
        key="prior.success",
        value={"acknowledged": True},
        source=EvidenceSource.EXTERNAL,
        status=EvidenceStatus.SUCCESS,
        event_id="event-prior",
        call_id="call-prior",
        node_id="act-prior",
        producer_kind="act",
        operation="update_resource",
    )
    compile_request = request(
        evidence=[prior],
        required_completion_evidence={"prior.success"},
    )
    valid = plan_json(
        evidence_inputs=["prior.success"],
        requires_evidence=["resource.updated", "prior.success"],
    )

    result = SemanticCompiler(SequenceBackend([ModelCandidate(text=valid)])).compile(
        compile_request
    )

    assert result.plan.evidence_inputs == ("prior.success",)
    assert result.plan.terminal.requires_evidence == (
        "resource.updated",
        "prior.success",
    )

    locally_closed = SemanticCompiler(
        SequenceBackend([ModelCandidate(text=plan_json())]),
        limits=SemanticCompilerLimits(max_repair_attempts=0),
    ).compile(compile_request)
    assert locally_closed.plan.evidence_inputs == ("prior.success",)
    assert locally_closed.plan.terminal.requires_evidence == (
        "resource.updated",
        "prior.success",
    )


def test_compiler_cannot_relabel_carried_action_obligation_as_answered() -> None:
    prior = EvidenceRecord(
        key="prior.success",
        value={"acknowledged": True},
        source=EvidenceSource.EXTERNAL,
        status=EvidenceStatus.SUCCESS,
        event_id="event-prior",
        call_id="call-prior",
        node_id="act-prior",
        producer_kind="act",
        operation="update_resource",
    )
    compile_request = request(
        evidence=[prior],
        required_completion_evidence={"prior.success"},
    )
    bypass = response_plan_json(
        outcome="answered",
        text="The request is done.",
        evidence_inputs=["prior.success"],
        requires_evidence=["prior.success"],
    )

    with pytest.raises(PlanRepairExhaustedError) as captured:
        SemanticCompiler(
            SequenceBackend([ModelCandidate(text=bypass)]),
            limits=SemanticCompilerLimits(max_repair_attempts=0),
        ).compile(compile_request)

    assert "completion_obligation_wrong_outcome" in {
        issue.code for issue in captured.value.issues
    }


def test_compiler_rejects_existing_ledger_key_as_new_action_producer() -> None:
    existing = EvidenceRecord(
        key="resource.updated",
        value={"acknowledged": True},
        source=EvidenceSource.EXTERNAL,
        status=EvidenceStatus.SUCCESS,
        event_id="event-existing",
        call_id="call-existing",
        node_id="act-existing",
        producer_kind="act",
        operation="update_resource",
    )

    with pytest.raises(PlanRepairExhaustedError) as captured:
        SemanticCompiler(
            SequenceBackend([ModelCandidate(text=plan_json())]),
            limits=SemanticCompilerLimits(max_repair_attempts=0),
        ).compile(request(evidence=[existing]))

    assert "evidence_key_reuse" in {issue.code for issue in captured.value.issues}


def test_failure_evidence_status_survives_prompt_and_verifier_reconstruction() -> None:
    failure = EvidenceRecord(
        key="operation.failure",
        value={"reason": "unavailable"},
        source=EvidenceSource.EXTERNAL,
        status=EvidenceStatus.FAILURE,
        event_id="event-failure",
        call_id="call-failure",
        node_id="act-failed",
        operation="update_resource",
    )
    compile_request = CompilationRequest.from_live_tools(
        plan_id="request-1",
        trusted_policy="Report external failures truthfully.",
        goal="Handle the failed operation safely.",
        tools=[],
        evidence=[failure],
    )
    backend = SequenceBackend(
        [
            ModelCandidate(
                text=response_plan_json(
                    outcome="fail_safe",
                    text="The operation could not be completed safely.",
                    evidence_inputs=["operation.failure"],
                    requires_evidence=["operation.failure"],
                )
            )
        ]
    )

    result = SemanticCompiler(backend).compile(compile_request)

    assert result.plan.terminal.outcome.value == "fail_safe"
    assert compile_request.as_evidence_records()[0].status == EvidenceStatus.FAILURE
    payload = json.loads(backend.calls[0]["messages"][2]["content"])
    assert payload["existing_evidence"][0]["status"] == "failure"


def test_declined_outcome_requires_available_negative_input_evidence() -> None:
    declined = EvidenceRecord(
        key="input.confirmed",
        value=False,
        source=EvidenceSource.INPUT,
        status=EvidenceStatus.FAILURE,
        event_id="event-declined",
        call_id="call-confirm",
        node_id="confirm",
        producer_kind="confirm",
    )
    compile_request = CompilationRequest.from_live_tools(
        plan_id="request-1",
        trusted_policy="Honor explicit user decisions.",
        goal="Respect the declined change.",
        tools=[],
        evidence=[declined],
    )
    candidate = response_plan_json(
        outcome="declined",
        text="Okay, I will not make that change.",
        evidence_inputs=["input.confirmed"],
        requires_evidence=["input.confirmed"],
    )

    result = SemanticCompiler(
        SequenceBackend([ModelCandidate(text=candidate)])
    ).compile(compile_request)

    assert result.plan.terminal.outcome.value == "declined"


@pytest.mark.parametrize("outcome", ["answered", "refused"])
def test_no_tool_terminal_outcomes_compile_without_external_nodes(
    outcome: str,
) -> None:
    compile_request = CompilationRequest.from_live_tools(
        plan_id="request-1",
        trusted_policy="Answer safely and truthfully.",
        goal="Provide a response without external operations.",
        tools=[],
    )
    backend = SequenceBackend(
        [
            ModelCandidate(
                text=response_plan_json(
                    outcome=outcome,
                    text="Here is the safe response.",
                )
            )
        ]
    )

    result = SemanticCompiler(backend).compile(compile_request)

    assert result.plan.terminal.outcome.value == outcome
    variants = backend.calls[0]["response_schema"]["properties"]["nodes"]["items"][
        "anyOf"
    ]
    kinds = {item["properties"]["kind"]["enum"][0] for item in variants}
    assert kinds == {"ask", "confirm", "respond"}


def test_request_rejects_duplicate_or_invalid_live_contracts() -> None:
    with pytest.raises(CompilerInputError, match="duplicated"):
        CompilationRequest.from_live_tools(
            plan_id="p",
            trusted_policy="Follow policy.",
            goal="Do something.",
            tools=[capability(), capability()],
        )

    invalid = capability()
    invalid["function"]["parameters"] = {"type": "not-a-json-type"}
    with pytest.raises(CompilerInputError, match="invalid parameter schema"):
        CompilationRequest.from_live_tools(
            plan_id="p",
            trusted_policy="Follow policy.",
            goal="Do something.",
            tools=[invalid],
        )

    non_function = capability()
    non_function["type"] = "resource"
    with pytest.raises(CompilerInputError, match="not a function tool"):
        CompilationRequest.from_live_tools(
            plan_id="p",
            trusted_policy="Follow policy.",
            goal="Do something.",
            tools=[non_function],
        )


def test_compiler_enforces_trusted_capability_effect_classification() -> None:
    mismatched_tool = capability()
    mismatched_tool["function"]["x-pact-effect"] = "observe"
    compile_request = CompilationRequest.from_live_tools(
        plan_id="request-1",
        trusted_policy="Follow policy.",
        goal="Test a typed effect contract.",
        tools=[mismatched_tool],
    )
    backend = SequenceBackend([ModelCandidate(text=plan_json())])

    with pytest.raises(PlanRepairExhaustedError) as captured:
        SemanticCompiler(
            backend,
            limits=SemanticCompilerLimits(max_repair_attempts=0),
        ).compile(compile_request)

    assert "operation_kind_mismatch" in {issue.code for issue in captured.value.issues}


def test_compiler_enforces_input_and_output_resource_bounds() -> None:
    compiler = SemanticCompiler(
        SequenceBackend([ModelCandidate(text=plan_json())]),
        limits=SemanticCompilerLimits(max_goal_chars=4),
    )

    with pytest.raises(CompilerInputError, match="goal exceeds"):
        compiler.compile(request())

    backend = SequenceBackend([ModelCandidate(text="{" + " " * 100 + "}")])
    with pytest.raises(PlanRepairExhaustedError) as captured:
        SemanticCompiler(
            backend,
            limits=SemanticCompilerLimits(
                max_repair_attempts=0,
                max_candidate_chars=16,
            ),
        ).compile(request())
    assert captured.value.issues[0].code == "candidate_too_large"


def test_system_contract_is_domain_independent_and_receding_horizon() -> None:
    normalized = SEMANTIC_COMPILER_INSTRUCTIONS.lower()

    assert "only the first ready action" in normalized
    assert "already-verified immutable suffix" in normalized
    assert "semantic replan boundaries" in normalized
    assert "provenance obligation" in normalized
    assert "never copy a number" in normalized
    assert "external failure" in normalized
    for outcome in ("completed", "answered", "refused", "declined", "fail_safe"):
        assert outcome in normalized
    for forbidden in ("car-bench", "sunroof", "navigation", "task id"):
        assert forbidden not in normalized


def test_compile_request_projects_anchored_confirmation_marker_as_contract() -> None:
    compile_request = CompilationRequest.from_live_tools(
        plan_id="request-1",
        trusted_policy="Honor live capability contracts.",
        goal="Apply a synthetic update.",
        tools=[
            capability(
                description="REQUIRES_CONFIRMATION: apply a synthetic update."
            )
        ],
    )

    projected = compile_request.capabilities[0]
    assert projected.requires_confirmation is True
    assert projected.effect == "act"


def compiler_settings() -> CerebrasCompilerSettings:
    return CerebrasCompilerSettings.from_env({})


def test_cerebras_compiler_defaults_reserve_reasoning_and_review_budget() -> None:
    settings = compiler_settings()

    assert settings.reasoning_effort == "medium"
    assert settings.max_completion_tokens == 8192
    assert settings.semantic_review is True


def test_archived_planner_environment_cannot_override_pact_defaults() -> None:
    settings = CerebrasCompilerSettings.from_env(
        {
            "TRACK2_PLANNER_MODEL": "archived-model",
            "TRACK2_PLANNER_REASONING_EFFORT": "high",
            "TRACK2_PLANNER_MAX_COMPLETION_TOKENS": "1",
            "TRACK2_CEREBRAS_API_BASE": "https://archived.invalid",
        }
    )

    assert settings.model == "gpt-oss-120b"
    assert settings.reasoning_effort == "medium"
    assert settings.max_completion_tokens == 8192
    assert settings.semantic_review is True
    assert settings.api_base == DEFAULT_CEREBRAS_API_BASE


def test_cerebras_compiler_settings_are_fully_environment_configurable() -> None:
    settings = CerebrasCompilerSettings.from_env(
        {
            "PACT_COMPILER_MODEL": "model-from-env",
            "PACT_COMPILER_CEREBRAS_API_BASE": "https://provider.invalid",
            "PACT_COMPILER_SERVICE_TIER": "priority",
            "PACT_COMPILER_MAX_COMPLETION_TOKENS": "2048",
            "PACT_COMPILER_TEMPERATURE": "0.25",
            "PACT_COMPILER_REASONING_EFFORT": "medium",
            "PACT_COMPILER_SEMANTIC_REVIEW": "false",
            "PACT_COMPILER_MAX_REPAIR_ATTEMPTS": "0",
            "PACT_COMPILER_MAX_NODES": "7",
            "PACT_COMPILER_MAX_DEPENDENCY_DEPTH": "6",
            "PACT_COMPILER_MAX_POLICY_CHARS": "800",
            "PACT_COMPILER_MAX_GOAL_CHARS": "1000",
            "PACT_COMPILER_MAX_USER_EVENT_CHARS": "900",
            "PACT_COMPILER_MAX_CONVERSATION_MESSAGES": "5",
            "PACT_COMPILER_MAX_CONVERSATION_CHARS": "1500",
            "PACT_COMPILER_MAX_CONTEXT_CHARS": "2000",
            "PACT_COMPILER_MAX_CANDIDATE_CHARS": "3000",
            "PACT_COMPILER_MAX_EVIDENCE_RECORDS": "9",
        }
    )

    assert settings == CerebrasCompilerSettings(
        model="model-from-env",
        api_base="https://provider.invalid",
        service_tier="priority",
        max_completion_tokens=2048,
        temperature=0.25,
        reasoning_effort="medium",
        semantic_review=False,
        max_repair_attempts=0,
        max_nodes=7,
        max_dependency_depth=6,
        max_policy_chars=800,
        max_goal_chars=1000,
        max_user_event_chars=900,
        max_conversation_messages=5,
        max_conversation_chars=1500,
        max_context_chars=2000,
        max_candidate_chars=3000,
        max_evidence_records=9,
    )


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("PACT_COMPILER_MAX_COMPLETION_TOKENS", "0"),
        ("PACT_COMPILER_TEMPERATURE", "3.0"),
        ("PACT_COMPILER_REASONING_EFFORT", "extreme"),
        ("PACT_COMPILER_SEMANTIC_REVIEW", "sometimes"),
        ("PACT_COMPILER_MAX_REPAIR_ATTEMPTS", "2"),
        ("PACT_COMPILER_MAX_NODES", "not-an-int"),
    ],
)
def test_cerebras_compiler_settings_reject_invalid_env(
    name: str,
    value: str,
) -> None:
    with pytest.raises(CompilerConfigurationError):
        CerebrasCompilerSettings.from_env({name: value})


@pytest.mark.parametrize(
    "name",
    ["PACT_COMPILER_MAX_NODES", "PACT_COMPILER_TEMPERATURE"],
)
def test_compiler_settings_do_not_echo_invalid_environment_values(
    name: str,
) -> None:
    secret = "SENSITIVE_INVALID_CONFIG_VALUE"

    with pytest.raises(CompilerConfigurationError) as caught:
        CerebrasCompilerSettings.from_env({name: secret})

    assert secret not in str(caught.value)
    assert caught.value.__cause__ is None


class FakeCerebrasClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def generate(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return SimpleNamespace(
            text=plan_json(),
            model="resolved-model",
            finish_reason="stop",
            duration_ms=18.0,
            cost=0.03,
            quota_wait_ms=125.0,
            token_usage=TokenUsage(
                input_tokens=40,
                output_tokens=12,
                reasoning_output_tokens=5,
            ),
        )


def test_cerebras_backend_passes_only_environment_settings_to_sdk_client() -> None:
    fake = FakeCerebrasClient()
    settings = replace(
        compiler_settings(),
        model="configured-model",
        api_base="https://configured.invalid",
        service_tier="configured-tier",
        max_completion_tokens=1536,
        temperature=0.1,
        reasoning_effort="low",
    )
    backend = CerebrasStructuredPlanBackend(settings=settings, client=fake)

    candidate = backend.generate(
        messages=[{"role": "user", "content": "structured input"}],
        response_schema={"type": "object"},
        response_schema_name="test_schema",
    )

    assert fake.calls == [
        {
            "model": "configured-model",
            "messages": [{"role": "user", "content": "structured input"}],
            "response_schema": {"type": "object"},
            "response_schema_name": "test_schema",
            "max_completion_tokens": 1536,
            "temperature": 0.1,
            "reasoning_effort": "low",
        }
    ]
    assert candidate.usage == CompilerTokenUsage(
        prompt_tokens=40,
        completion_tokens=7,
        thinking_tokens=5,
    )
    assert candidate.cost == 0.03
    assert candidate.quota_wait_ms == 125.0
    assert candidate.finish_reason == "stop"


def test_cerebras_factory_applies_env_derived_verification_limits() -> None:
    fake = FakeCerebrasClient()
    settings = replace(
        compiler_settings(),
        max_repair_attempts=0,
        max_nodes=1,
    )
    compiler = create_cerebras_semantic_compiler(
        settings=settings,
        client=fake,
    )

    with pytest.raises(PlanRepairExhaustedError) as captured:
        compiler.compile(request())

    assert len(fake.calls) == 1
    assert captured.value.issues[0].code == "node_limit_exceeded"
