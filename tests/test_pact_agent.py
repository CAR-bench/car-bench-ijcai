"""End-to-end A2A contract tests for the final PACT executor.

The semantic compiler is replaced with a deterministic fake so these tests
exercise the trusted adapter/runtime boundary without a network dependency.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import threading
from collections.abc import Callable
from typing import Any

import pytest
from agentbeats.sync_client import create_message_with_parts
from google.protobuf.json_format import MessageToDict

from a2a.helpers.proto_helpers import new_data_part, new_text_part
from carbench_agent_core.evidence_ledger import (
    EvidenceSource,
    EvidenceStatus,
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
from carbench_agent_core.plan_verifier import CapabilitySnapshot, PlanVerifier
from carbench_agent_core.semantic_compiler import (
    CompilationAttempt,
    CompilationIssue,
    CompilationRequest,
    CompilationResult,
    CompilerBackendError,
    CompilerTokenUsage,
    PlanRepairExhaustedError,
)
from track_2_agent_under_test_cerebras import pact_agent as pact_module
from track_2_agent_under_test_cerebras.pact_agent import (
    PACTAgentExecutor,
    _input_event_id,
)
from turn_metrics import (
    AVG_LLM_CALL_TIME_MS,
    COMPLETION_TOKENS,
    COST,
    MODEL,
    NUM_LLM_CALLS,
    NUM_PASSES,
    PROMPT_TOKENS,
    QUOTA_WAIT_TIME_MS,
    THINKING_TOKENS,
    extract_turn_metrics,
)


class FakeRequestContext:
    def __init__(self, message: Any, *, context_id: str) -> None:
        self.message = message
        self.context_id = context_id

    def get_user_input(self) -> str:
        return ""


class FakeEventQueue:
    def __init__(self) -> None:
        self.events: list[Any] = []

    async def enqueue_event(self, event: Any) -> None:
        self.events.append(event)


class RecordingBoundLogger:
    def __init__(self) -> None:
        self.records: list[tuple[str, str, dict[str, Any]]] = []

    def bind(self, **fields: Any) -> "RecordingBoundLogger":
        return self

    def warning(self, message: str, **fields: Any) -> None:
        self.records.append(("warning", message, fields))

    def error(self, message: str, **fields: Any) -> None:
        self.records.append(("error", message, fields))


Outcome = CompilationResult | BaseException
OutcomeFactory = Callable[[CompilationRequest], Outcome]


class FakeCompiler:
    """Thread-safe-enough queued compiler fake for ``asyncio.to_thread``."""

    def __init__(self, *outcomes: Outcome | OutcomeFactory) -> None:
        self.outcomes = list(outcomes)
        self.requests: list[CompilationRequest] = []

    def compile(self, request: CompilationRequest) -> CompilationResult:
        self.requests.append(request)
        if not self.outcomes:
            raise AssertionError("unexpected compiler invocation")
        queued = self.outcomes.pop(0)
        outcome = queued(request) if callable(queued) else queued
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def tool(
    name: str,
    *,
    effect: str = "unknown",
    properties: dict[str, Any] | None = None,
    required: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": f"Contract for {name}",
            "parameters": {
                "type": "object",
                "properties": properties or {},
                "required": required or [],
                "additionalProperties": False,
            },
            "x-pact-effect": effect,
        },
    }


def attempt(
    number: int = 1,
    *,
    accepted: bool = True,
    repair: bool = False,
    prompt_tokens: int = 11,
    completion_tokens: int = 7,
    thinking_tokens: int = 3,
    duration_ms: float = 20.0,
    cost: float = 0.01,
    quota_wait_ms: float = 2.0,
) -> CompilationAttempt:
    issues = ()
    if not accepted:
        issues = (
            CompilationIssue(
                code="candidate_rejected",
                message="Candidate failed local verification.",
            ),
        )
    return CompilationAttempt(
        attempt=number,
        was_repair=repair,
        accepted=accepted,
        output_sha256=f"{number:064x}",
        model="test-model",
        duration_ms=duration_ms,
        cost=cost,
        quota_wait_ms=quota_wait_ms,
        usage=CompilerTokenUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            thinking_tokens=thinking_tokens,
        ),
        issues=issues,
    )


def result_for(
    request: CompilationRequest,
    plan: PlanIR,
    *,
    attempts: tuple[CompilationAttempt, ...] | None = None,
) -> CompilationResult:
    snapshot = CapabilitySnapshot.from_tools(request.as_tool_declarations())
    assert snapshot.is_valid
    return CompilationResult(
        plan=plan,
        attempts=attempts or (attempt(),),
        request_sha256="a" * 64,
        capability_sha256=snapshot.digest,
    )


def response_plan(
    request: CompilationRequest,
    *,
    text: str,
    evidence_inputs: tuple[str, ...] = (),
    requires_evidence: tuple[str, ...] = (),
    outcome: ResponseOutcome = ResponseOutcome.ANSWERED,
) -> PlanIR:
    return PlanIR(
        plan_id=request.plan_id,
        evidence_inputs=evidence_inputs,
        nodes=(
            RespondNode(
                id="respond",
                text=text,
                outcome=outcome,
                requires_evidence=requires_evidence,
            ),
        ),
    )


def act_plan(request: CompilationRequest) -> PlanIR:
    return PlanIR(
        plan_id=request.plan_id,
        nodes=(
            ActNode(
                id="act",
                operation="set_mode",
                arguments={"mode": "eco"},
                success_evidence_key="mode.changed",
            ),
            RespondNode(
                id="respond",
                depends_on=("act",),
                text="Mode changed.",
                outcome=ResponseOutcome.COMPLETED,
                requires_evidence=("mode.changed",),
            ),
        ),
    )


def observe_plan(request: CompilationRequest) -> PlanIR:
    return PlanIR(
        plan_id=request.plan_id,
        nodes=(
            ObserveNode(
                id="observe",
                operation="read_temperature",
                arguments={},
                success_evidence_key="temperature.reading",
            ),
            RespondNode(
                id="respond",
                depends_on=("observe",),
                text="The temperature is available.",
                outcome=ResponseOutcome.ANSWERED,
                requires_evidence=("temperature.reading",),
            ),
        ),
    )


def confirmation_plan(request: CompilationRequest) -> PlanIR:
    return PlanIR(
        plan_id=request.plan_id,
        nodes=(
            ConfirmNode(
                id="confirm",
                prompt="Proceed with the change?",
                evidence_key="change.approved",
                authorizes=("act",),
            ),
            ActNode(
                id="act",
                depends_on=("confirm",),
                operation="set_mode",
                arguments={"mode": "eco"},
                success_evidence_key="mode.changed",
            ),
            RespondNode(
                id="respond",
                depends_on=("act",),
                text="Mode changed.",
                outcome=ResponseOutcome.COMPLETED,
                requires_evidence=("mode.changed",),
            ),
        ),
    )


def ask_plan(request: CompilationRequest) -> PlanIR:
    return PlanIR(
        plan_id=request.plan_id,
        nodes=(
            AskNode(
                id="ask",
                prompt=f"Which value should I use for {request.goal}?",
                evidence_key="requested.value",
            ),
            RespondNode(
                id="respond",
                depends_on=("ask",),
                text="Value received.",
                outcome=ResponseOutcome.ANSWERED,
                requires_evidence=("requested.value",),
            ),
        ),
    )


def compile_plan(
    builder: Callable[[CompilationRequest], PlanIR],
    *,
    attempts: tuple[CompilationAttempt, ...] | None = None,
) -> OutcomeFactory:
    def factory(request: CompilationRequest) -> CompilationResult:
        return result_for(request, builder(request), attempts=attempts)

    return factory


def compile_response(text_value: str) -> OutcomeFactory:
    def factory(request: CompilationRequest) -> CompilationResult:
        return result_for(request, response_plan(request, text=text_value))

    return factory


def inbound_message(
    *,
    context_id: str,
    text: str | None = None,
    tools: list[dict[str, Any]] | None = None,
    tool_results: list[dict[str, Any]] | None = None,
    message_id: str | None = None,
) -> Any:
    parts: list[Any] = []
    if text is not None:
        parts.append(new_text_part(text))
    data: dict[str, Any] = {}
    if tools is not None:
        data["tools"] = tools
    if tool_results is not None:
        data["tool_results"] = tool_results
    if data:
        parts.append(new_data_part(data))
    message = create_message_with_parts(parts=parts, context_id=context_id)
    if message_id is not None:
        message.message_id = message_id
    return message


def run_turn(
    executor: PACTAgentExecutor,
    *,
    context_id: str,
    text: str | None = None,
    tools: list[dict[str, Any]] | None = None,
    tool_results: list[dict[str, Any]] | None = None,
    message_id: str | None = None,
) -> Any:
    queue = FakeEventQueue()
    message = inbound_message(
        context_id=context_id,
        text=text,
        tools=tools,
        tool_results=tool_results,
        message_id=message_id,
    )
    asyncio.run(
        executor.execute(
            FakeRequestContext(message, context_id=context_id),
            queue,
        )
    )
    assert len(queue.events) == 1
    return queue.events[0]


def tool_call(response: Any) -> dict[str, Any]:
    assert len(response.parts) == 1
    assert response.parts[0].WhichOneof("content") == "data"
    payload = MessageToDict(response.parts[0].data)
    assert len(payload["tool_calls"]) == 1
    return payload["tool_calls"][0]


def text(response: Any) -> str:
    assert len(response.parts) == 1
    assert response.parts[0].WhichOneof("content") == "text"
    return response.parts[0].text


def success_result(
    operation: str,
    *,
    call_id: str = "evaluator-call-1",
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    content = {"status": "SUCCESS"}
    if payload:
        content.update(payload)
    return {
        "tool_name": operation,
        "tool_call_id": call_id,
        "content": content,
    }


def test_act_success_is_grounded_and_reports_accumulated_turn_metrics() -> None:
    compiler = FakeCompiler(compile_plan(act_plan))
    executor = PACTAgentExecutor(compiler=compiler, model="test-model")
    live_tools = [
        tool(
            "set_mode",
            effect="act",
            properties={"mode": {"type": "string", "enum": ["eco"]}},
            required=["mode"],
        )
    ]

    emitted = run_turn(
        executor,
        context_id="ctx-act",
        text="System: Only make requested changes.\n\nUser: Set eco mode.",
        tools=live_tools,
    )
    assert tool_call(emitted) == {
        "tool_name": "set_mode",
        "arguments": {"mode": "eco"},
    }

    completed = run_turn(
        executor,
        context_id="ctx-act",
        tool_results=[success_result("set_mode", payload={"mode": "eco"})],
    )

    assert text(completed) == "Mode changed."
    metrics = extract_turn_metrics(completed.metadata)
    assert metrics[PROMPT_TOKENS] == 11
    assert metrics[COMPLETION_TOKENS] == 7
    assert metrics[THINKING_TOKENS] == 3
    assert metrics[NUM_LLM_CALLS] == 1
    assert metrics[NUM_PASSES] == 1
    evidence = executor.context_states["ctx-act"].ledger.get("mode.changed")
    assert evidence is not None
    assert evidence.source == EvidenceSource.EXTERNAL
    assert evidence.status == EvidenceStatus.SUCCESS
    assert evidence.operation == "set_mode"
    assert evidence.external_call_id == "evaluator-call-1"
    assert evidence.context_id == "ctx-act"
    assert evidence.plan_id == compiler.requests[0].plan_id
    assert evidence.node_id == "act"
    assert evidence.producer_kind == "act"
    assert (
        evidence.argument_digest
        == hashlib.sha256(
            json.dumps(
                {"mode": "eco"},
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
    )
    assert (
        evidence.schema_digest
        == CapabilitySnapshot.from_tools(
            compiler.requests[0].as_tool_declarations()
        ).digest
    )


def test_observation_result_forces_receding_horizon_recompilation() -> None:
    def answer_from_evidence(request: CompilationRequest) -> CompilationResult:
        assert request.current_user_event == ""
        assert len(request.evidence) == 1
        observation = request.evidence[0]
        assert observation.key == "temperature.reading"
        assert observation.source == EvidenceSource.EXTERNAL
        assert observation.status == EvidenceStatus.SUCCESS
        assert observation.producer_kind == "observe"
        plan = response_plan(
            request,
            text="It is 21 C.",
            evidence_inputs=(observation.key,),
            requires_evidence=(observation.key,),
        )
        return result_for(request, plan)

    compiler = FakeCompiler(compile_plan(observe_plan), answer_from_evidence)
    executor = PACTAgentExecutor(compiler=compiler, model="test-model")

    emitted = run_turn(
        executor,
        context_id="ctx-observe",
        text="What is the temperature?",
        tools=[tool("read_temperature", effect="observe")],
    )
    assert tool_call(emitted)["tool_name"] == "read_temperature"

    answered = run_turn(
        executor,
        context_id="ctx-observe",
        tool_results=[
            success_result("read_temperature", payload={"temperature_c": 21})
        ],
    )

    assert text(answered) == "It is 21 C."
    assert len(compiler.requests) == 2
    metrics = extract_turn_metrics(answered.metadata)
    assert metrics[NUM_LLM_CALLS] == 2
    assert metrics[PROMPT_TOKENS] == 22
    assert metrics[COMPLETION_TOKENS] == 14
    assert metrics[THINKING_TOKENS] == 6


def test_explicit_tool_failure_is_an_immediate_fail_safe_terminal() -> None:
    compiler = FakeCompiler(compile_plan(act_plan))
    executor = PACTAgentExecutor(compiler=compiler, model="test-model")
    run_turn(
        executor,
        context_id="ctx-failure",
        text="Set eco mode.",
        tools=[
            tool(
                "set_mode",
                effect="act",
                properties={"mode": {"type": "string"}},
                required=["mode"],
            )
        ],
    )

    failed = run_turn(
        executor,
        context_id="ctx-failure",
        tool_results=[
            {
                "tool_name": "set_mode",
                "tool_call_id": "evaluator-call-failed",
                "content": {
                    "status": "FAILURE",
                    "error": "device unavailable",
                },
            }
        ],
    )

    assert text(failed) == "I couldn't complete that action."
    assert len(compiler.requests) == 1
    state = executor.context_states["ctx-failure"]
    assert state.runtime is None
    assert not state.goal
    failure = state.ledger.get("mode.changed.failure")
    assert failure is not None
    assert failure.status == EvidenceStatus.FAILURE
    assert not state.ledger.has("mode.changed")


@pytest.mark.parametrize(
    "bad_results",
    [
        [success_result("wrong_operation")],
        [success_result("set_mode"), success_result("set_mode", call_id="two")],
        [
            {
                "tool_name": "set_mode",
                "tool_call_id": "malformed",
                "content": {"mode": "eco"},
            }
        ],
        [
            {
                "tool_name": "set_mode",
                "tool_call_id": "not-json",
                "content": "not JSON",
            }
        ],
    ],
    ids=("wrong-operation", "multiple-results", "missing-status", "malformed-json"),
)
def test_untrusted_tool_result_contract_violations_fail_closed(
    bad_results: list[dict[str, Any]],
) -> None:
    compiler = FakeCompiler(compile_plan(act_plan))
    executor = PACTAgentExecutor(compiler=compiler, model="test-model")
    run_turn(
        executor,
        context_id="ctx-bad-result",
        text="Set eco mode.",
        tools=[
            tool(
                "set_mode",
                effect="act",
                properties={"mode": {"type": "string"}},
                required=["mode"],
            )
        ],
    )

    response = run_turn(
        executor,
        context_id="ctx-bad-result",
        tool_results=bad_results,
    )

    assert text(response) == "I couldn't complete that request safely."
    assert len(compiler.requests) == 1
    state = executor.context_states["ctx-bad-result"]
    assert state.runtime is None
    assert not state.ledger.has("mode.changed")


def test_explicit_empty_tools_revoke_the_previous_capability_snapshot() -> None:
    def plain_answer(label: str) -> OutcomeFactory:
        def factory(request: CompilationRequest) -> CompilationResult:
            plan = response_plan(request, text=label)
            return result_for(request, plan)

        return factory

    compiler = FakeCompiler(plain_answer("first"), plain_answer("second"))
    executor = PACTAgentExecutor(compiler=compiler, model="test-model")
    run_turn(
        executor,
        context_id="ctx-revoke",
        text="First question.",
        tools=[tool("legacy_read", effect="observe")],
    )
    assert [item.name for item in compiler.requests[0].capabilities] == ["legacy_read"]

    response = run_turn(
        executor,
        context_id="ctx-revoke",
        text="Second question.",
        tools=[],
    )

    assert text(response) == "second"
    assert compiler.requests[1].capabilities == ()
    assert executor.context_states["ctx-revoke"].tools == []


def test_confirmation_yes_continues_the_verified_plan_without_recompiling() -> None:
    compiler = FakeCompiler(compile_plan(confirmation_plan))
    executor = PACTAgentExecutor(compiler=compiler, model="test-model")
    live_tools = [
        tool(
            "set_mode",
            effect="act",
            properties={"mode": {"type": "string"}},
            required=["mode"],
        )
    ]
    prompt = run_turn(
        executor,
        context_id="ctx-confirm-yes",
        text="Change the mode.",
        tools=live_tools,
    )
    assert text(prompt) == "Proceed with the change?"

    ambiguous = run_turn(
        executor,
        context_id="ctx-confirm-yes",
        text="perhaps",
    )
    assert text(ambiguous) == "Please answer yes or no."
    assert len(compiler.requests) == 1
    original_runtime = executor.context_states["ctx-confirm-yes"].runtime
    assert original_runtime is not None
    original_plan_id = original_runtime.plan.plan_id

    accepted = run_turn(
        executor,
        context_id="ctx-confirm-yes",
        text="yes, proceed with eco mode please",
    )
    assert tool_call(accepted) == {
        "tool_name": "set_mode",
        "arguments": {"mode": "eco"},
    }
    assert len(compiler.requests) == 1
    state = executor.context_states["ctx-confirm-yes"]
    assert state.runtime is original_runtime
    assert state.runtime.plan.plan_id == original_plan_id

    confirmation = state.ledger.get("change.approved")
    assert confirmation is not None
    assert confirmation.value is True
    assert confirmation.source == EvidenceSource.INPUT
    assert confirmation.status == EvidenceStatus.OBSERVED
    assert confirmation.producer_kind == "confirm"
    assert confirmation.context_id == "ctx-confirm-yes"
    assert confirmation.plan_id == original_plan_id
    assert (
        confirmation.schema_digest
        == CapabilitySnapshot.from_tools(
            compiler.requests[0].as_tool_declarations()
        ).digest
    )
    assert len(confirmation.authorizations) == 1
    authorization = confirmation.authorizations[0]
    assert authorization.operation == "set_mode"
    assert (
        authorization.argument_digest
        == hashlib.sha256(
            json.dumps(
                {"mode": "eco"},
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
    )

    completed = run_turn(
        executor,
        context_id="ctx-confirm-yes",
        tool_results=[
            success_result(
                "set_mode",
                call_id="evaluator-confirmed-set-mode",
                payload={"mode": "eco"},
            )
        ],
    )
    assert text(completed) == "Mode changed."
    assert len(compiler.requests) == 1


@pytest.mark.parametrize(
    "revision",
    (
        "Yes, but use sport instead.",
        "Yes, make it 75% please.",
        "Yes, Alice.",
        "Yes, sport.",
        "Yes, off.",
    ),
)
def test_qualified_confirmation_replans_without_authorizing_old_arguments(
    revision: str,
) -> None:
    compiler = FakeCompiler(
        compile_plan(confirmation_plan),
        compile_response("I treated that as a revised request."),
    )
    executor = PACTAgentExecutor(compiler=compiler, model="test-model")
    live_tools = [
        tool(
            "set_mode",
            effect="act",
            properties={"mode": {"type": "string"}},
            required=["mode"],
        )
    ]
    run_turn(
        executor,
        context_id="ctx-confirm-revision",
        text="Change the mode.",
        tools=live_tools,
    )

    revised = run_turn(
        executor,
        context_id="ctx-confirm-revision",
        text=revision,
    )

    assert text(revised) == "I treated that as a revised request."
    assert len(compiler.requests) == 2
    assert compiler.requests[1].current_user_event == revision
    assert (
        executor.context_states["ctx-confirm-revision"].ledger.get("change.approved")
        is None
    )


def test_confirmation_may_restate_the_exact_numeric_scope() -> None:
    def numeric_confirmation_plan(request: CompilationRequest) -> PlanIR:
        return PlanIR(
            plan_id=request.plan_id,
            nodes=(
                ConfirmNode(
                    id="confirm",
                    prompt="Proceed with level 100?",
                    evidence_key="level.approved",
                    authorizes=("act",),
                ),
                ActNode(
                    id="act",
                    depends_on=("confirm",),
                    operation="set_level",
                    arguments={"level": 100},
                    success_evidence_key="level.changed",
                ),
                RespondNode(
                    id="respond",
                    depends_on=("act",),
                    text="Level changed.",
                    outcome=ResponseOutcome.COMPLETED,
                    requires_evidence=("level.changed",),
                ),
            ),
        )

    compiler = FakeCompiler(compile_plan(numeric_confirmation_plan))
    executor = PACTAgentExecutor(compiler=compiler, model="test-model")
    run_turn(
        executor,
        context_id="ctx-confirm-exact-number",
        text="Set the level.",
        tools=[
            tool(
                "set_level",
                effect="act",
                properties={"level": {"type": "integer"}},
                required=["level"],
            )
        ],
    )

    accepted = run_turn(
        executor,
        context_id="ctx-confirm-exact-number",
        text="Yes, go ahead at 100%.",
    )

    assert tool_call(accepted) == {
        "tool_name": "set_level",
        "arguments": {"level": 100.0},
    }
    assert len(compiler.requests) == 1


def test_confirmation_may_restate_exact_extent_and_operation_scope() -> None:
    def extent_confirmation_plan(request: CompilationRequest) -> PlanIR:
        return PlanIR(
            plan_id=request.plan_id,
            nodes=(
                ConfirmNode(
                    id="confirm",
                    prompt="Proceed with the full adjustment?",
                    evidence_key="level.approved",
                    authorizes=("act",),
                ),
                ActNode(
                    id="act",
                    depends_on=("confirm",),
                    operation="open_close_panel",
                    arguments={"percentage": 100},
                    success_evidence_key="panel.changed",
                ),
                RespondNode(
                    id="respond",
                    depends_on=("act",),
                    text="Panel changed.",
                    outcome=ResponseOutcome.COMPLETED,
                    requires_evidence=("panel.changed",),
                ),
            ),
        )

    compiler = FakeCompiler(compile_plan(extent_confirmation_plan))
    executor = PACTAgentExecutor(compiler=compiler, model="test-model")
    run_turn(
        executor,
        context_id="ctx-confirm-exact-extent",
        text="Adjust the panel.",
        tools=[
            tool(
                "open_close_panel",
                effect="act",
                properties={"percentage": {"type": "integer"}},
                required=["percentage"],
            )
        ],
    )

    accepted = run_turn(
        executor,
        context_id="ctx-confirm-exact-extent",
        text="Yeah, that's fine. Go ahead and open the panel all the way.",
    )

    assert tool_call(accepted) == {
        "tool_name": "open_close_panel",
        "arguments": {"percentage": 100.0},
    }
    assert len(compiler.requests) == 1


def completed_precondition_confirmation_plan(request: CompilationRequest) -> PlanIR:
    return PlanIR(
        plan_id=request.plan_id,
        nodes=(
            ActNode(
                id="prepare",
                operation="open_close_protectivecover",
                arguments={"percentage": 100},
                success_evidence_key="cover.opened",
            ),
            ConfirmNode(
                id="confirm",
                depends_on=("prepare",),
                prompt="Proceed with opening the panel halfway?",
                evidence_key="panel.approved",
                authorizes=("open",),
            ),
            ActNode(
                id="open",
                depends_on=("confirm",),
                operation="open_close_accesspanel",
                arguments={"percentage": 50},
                success_evidence_key="panel.opened",
            ),
            RespondNode(
                id="respond",
                depends_on=("open",),
                text="Panel opened.",
                outcome=ResponseOutcome.COMPLETED,
                requires_evidence=("cover.opened", "panel.opened"),
            ),
        ),
    )


def _prepare_completed_precondition_confirmation(
    executor: PACTAgentExecutor,
    *,
    context_id: str,
) -> None:
    first = run_turn(
        executor,
        context_id=context_id,
        text="Open the panel halfway.",
        tools=[
            tool(
                "open_close_protectivecover",
                effect="act",
                properties={"percentage": {"type": "integer"}},
                required=["percentage"],
            ),
            tool(
                "open_close_accesspanel",
                effect="act",
                properties={"percentage": {"type": "integer"}},
                required=["percentage"],
            ),
        ],
    )
    assert tool_call(first)["tool_name"] == "open_close_protectivecover"
    confirmation = run_turn(
        executor,
        context_id=context_id,
        tool_results=[
            success_result(
                "open_close_protectivecover",
                call_id="evaluator-cover",
                payload={"percentage": 100},
            )
        ],
    )
    assert text(confirmation) == "Proceed with opening the panel halfway?"


def test_confirmation_allows_exact_restatement_of_completed_precondition() -> None:
    compiler = FakeCompiler(compile_plan(completed_precondition_confirmation_plan))
    executor = PACTAgentExecutor(compiler=compiler, model="test-model")
    _prepare_completed_precondition_confirmation(
        executor,
        context_id="ctx-confirm-completed-precondition",
    )

    accepted = run_turn(
        executor,
        context_id="ctx-confirm-completed-precondition",
        text=(
            "Yes, that's right. And if the cover needs to open first, "
            "please open it all the way."
        ),
    )

    assert tool_call(accepted) == {
        "tool_name": "open_close_accesspanel",
        "arguments": {"percentage": 50.0},
    }
    assert len(compiler.requests) == 1


def test_confirmation_rejects_new_extent_for_authorized_action_even_with_completed_scope() -> None:
    compiler = FakeCompiler(
        compile_plan(completed_precondition_confirmation_plan),
        compile_response("I treated that as a revised request."),
    )
    executor = PACTAgentExecutor(compiler=compiler, model="test-model")
    _prepare_completed_precondition_confirmation(
        executor,
        context_id="ctx-confirm-authorized-mismatch",
    )

    revised = run_turn(
        executor,
        context_id="ctx-confirm-authorized-mismatch",
        text=(
            "Yes, open the panel all the way; the cover should stay all the way open."
        ),
    )

    assert text(revised) == "I treated that as a revised request."
    assert len(compiler.requests) == 2
    assert (
        executor.context_states["ctx-confirm-authorized-mismatch"].ledger.get(
            "panel.approved"
        )
        is None
    )


def test_confirmation_does_not_cross_a_changed_capability_digest() -> None:
    compiler = FakeCompiler(
        compile_plan(confirmation_plan),
        compile_response("The capability contract changed."),
    )
    executor = PACTAgentExecutor(compiler=compiler, model="test-model")
    original_tool = tool(
        "set_mode",
        effect="act",
        properties={"mode": {"type": "string"}},
        required=["mode"],
    )
    run_turn(
        executor,
        context_id="ctx-confirm-contract-change",
        text="Change the mode.",
        tools=[original_tool],
    )
    changed_tool = json.loads(json.dumps(original_tool))
    changed_tool["function"]["description"] = "A materially changed contract"

    response = run_turn(
        executor,
        context_id="ctx-confirm-contract-change",
        text="yes",
        tools=[changed_tool],
    )

    assert text(response) == "The capability contract changed."
    assert len(compiler.requests) == 2
    assert (
        executor.context_states["ctx-confirm-contract-change"].ledger.get(
            "change.approved"
        )
        is None
    )


def test_confirmation_no_declines_without_recompiling_or_calling_a_tool() -> None:
    compiler = FakeCompiler(compile_plan(confirmation_plan))
    executor = PACTAgentExecutor(compiler=compiler, model="test-model")
    run_turn(
        executor,
        context_id="ctx-confirm-no",
        text="Change the mode.",
        tools=[
            tool(
                "set_mode",
                effect="act",
                properties={"mode": {"type": "string"}},
                required=["mode"],
            )
        ],
    )

    declined = run_turn(
        executor,
        context_id="ctx-confirm-no",
        text="No, cancel it.",
    )

    assert text(declined) == "Understood. I won't proceed."
    assert len(compiler.requests) == 1
    state = executor.context_states["ctx-confirm-no"]
    decision = state.ledger.get("change.approved")
    assert decision is not None
    assert decision.value is False
    assert decision.status == EvidenceStatus.FAILURE
    assert state.runtime is None


def test_decline_after_completed_precondition_discloses_partial_effect() -> None:
    compiler = FakeCompiler(compile_plan(completed_precondition_confirmation_plan))
    executor = PACTAgentExecutor(compiler=compiler, model="test-model")
    context_id = "ctx-confirm-no-after-precondition"
    _prepare_completed_precondition_confirmation(
        executor,
        context_id=context_id,
    )

    declined = run_turn(
        executor,
        context_id=context_id,
        text="No, stop there.",
    )

    assert text(declined) == (
        "I completed part of the request, and I won't proceed with the remaining "
        "action."
    )
    assert len(compiler.requests) == 1
    state = executor.context_states[context_id]
    completed = state.ledger.get("cover.opened")
    decision = state.ledger.get("panel.approved")
    assert completed is not None
    assert completed.status == EvidenceStatus.SUCCESS
    assert completed.producer_kind == "act"
    assert decision is not None
    assert decision.value is False
    assert decision.status == EvidenceStatus.FAILURE
    assert state.runtime is None
    assert state.goal == ""
    assert state.required_completion_evidence == set()


def test_context_state_is_isolated_and_cancel_removes_only_the_target() -> None:
    compiler = FakeCompiler(compile_plan(ask_plan), compile_plan(ask_plan))
    executor = PACTAgentExecutor(compiler=compiler, model="test-model")

    first = run_turn(executor, context_id="ctx-a", text="alpha")
    second = run_turn(executor, context_id="ctx-b", text="beta")
    assert text(first) == "Which value should I use for alpha?"
    assert text(second) == "Which value should I use for beta?"
    assert executor.context_states["ctx-a"].goal == "alpha"
    assert executor.context_states["ctx-b"].goal == "beta"
    assert executor.context_states["ctx-a"].runtime is not (
        executor.context_states["ctx-b"].runtime
    )

    cancel_queue = FakeEventQueue()
    cancel_message = inbound_message(context_id="ctx-a", text="cancel")
    asyncio.run(
        executor.cancel(
            FakeRequestContext(cancel_message, context_id="ctx-a"),
            cancel_queue,
        )
    )

    assert "ctx-a" not in executor.context_states
    assert "ctx-b" in executor.context_states
    assert executor.context_states["ctx-b"].goal == "beta"
    assert cancel_queue.events == []


def test_cancel_cannot_split_one_context_across_concurrent_locks() -> None:
    started = threading.Event()
    release = threading.Event()

    def blocking_plan(request: CompilationRequest) -> CompilationResult:
        started.set()
        assert release.wait(timeout=2)
        return result_for(request, ask_plan(request))

    compiler = FakeCompiler(blocking_plan, compile_plan(ask_plan))
    executor = PACTAgentExecutor(compiler=compiler, model="test-model")

    async def exercise() -> None:
        first_queue = FakeEventQueue()
        second_queue = FakeEventQueue()
        cancel_queue = FakeEventQueue()
        first_context = FakeRequestContext(
            inbound_message(context_id="ctx-race", text="first"),
            context_id="ctx-race",
        )
        second_context = FakeRequestContext(
            inbound_message(context_id="ctx-race", text="second"),
            context_id="ctx-race",
        )

        first = asyncio.create_task(executor.execute(first_context, first_queue))
        assert await asyncio.to_thread(started.wait, 1)
        cancel = asyncio.create_task(executor.cancel(first_context, cancel_queue))
        second = asyncio.create_task(executor.execute(second_context, second_queue))
        await asyncio.sleep(0.02)

        # The first compiler call is blocked. A replacement lock would let the
        # second call enter concurrently after cancellation removed the old one.
        assert len(compiler.requests) == 1
        release.set()
        await asyncio.gather(first, cancel, second)

        assert len(first_queue.events) == 1
        assert len(second_queue.events) == 1
        assert cancel_queue.events == []

    asyncio.run(exercise())


def test_repair_attempt_metrics_are_aggregated_on_the_terminal_response() -> None:
    attempts = (
        attempt(
            1,
            accepted=False,
            prompt_tokens=10,
            completion_tokens=4,
            thinking_tokens=2,
            duration_ms=30,
            cost=0.02,
            quota_wait_ms=5,
        ),
        attempt(
            2,
            accepted=True,
            repair=True,
            prompt_tokens=14,
            completion_tokens=6,
            thinking_tokens=5,
            duration_ms=50,
            cost=0.03,
            quota_wait_ms=7,
        ),
    )

    def repaired(request: CompilationRequest) -> PlanIR:
        return response_plan(request, text="Verified after repair.")

    compiler = FakeCompiler(compile_plan(repaired, attempts=attempts))
    response = run_turn(
        PACTAgentExecutor(compiler=compiler, model="fallback-model"),
        context_id="ctx-repair",
        text="Answer safely.",
    )

    assert text(response) == "Verified after repair."
    metrics = extract_turn_metrics(response.metadata)
    assert metrics[PROMPT_TOKENS] == 24
    assert metrics[COMPLETION_TOKENS] == 10
    assert metrics[THINKING_TOKENS] == 7
    assert metrics[NUM_LLM_CALLS] == 2
    assert metrics[NUM_PASSES] == 2
    assert metrics[AVG_LLM_CALL_TIME_MS] == 40
    assert metrics[QUOTA_WAIT_TIME_MS] == 12
    assert metrics[COST] == pytest.approx(0.05)
    assert metrics[MODEL] == "test-model"


def test_compiler_error_is_fail_safe_and_preserves_attempt_metrics() -> None:
    failed_attempt = attempt(
        accepted=False,
        prompt_tokens=19,
        completion_tokens=8,
        thinking_tokens=6,
        duration_ms=25,
    )
    compiler = FakeCompiler(
        CompilerBackendError("provider failed", attempts=(failed_attempt,))
    )

    response = run_turn(
        PACTAgentExecutor(compiler=compiler, model="fallback-model"),
        context_id="ctx-compiler-error",
        text="Answer safely.",
    )

    assert text(response) == "I couldn't complete that request safely."
    metrics = extract_turn_metrics(response.metadata)
    assert metrics[PROMPT_TOKENS] == 19
    assert metrics[COMPLETION_TOKENS] == 8
    assert metrics[THINKING_TOKENS] == 6
    assert metrics[NUM_LLM_CALLS] == 1
    assert metrics[NUM_PASSES] == 1


def test_plan_rejection_logs_only_fixed_issue_codes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "SENSITIVE_MODEL_ECHO_DO_NOT_LOG"
    rejected_attempt = attempt(1, accepted=False)
    compiler = FakeCompiler(
        PlanRepairExhaustedError(
            attempts=(rejected_attempt,),
            issues=(
                CompilationIssue(
                    code="invalid_plan",
                    path=("nodes", secret),
                    message=f"candidate echoed {secret}",
                ),
            ),
        )
    )
    test_logger = RecordingBoundLogger()
    monkeypatch.setattr(pact_module, "logger", test_logger)
    executor = PACTAgentExecutor(compiler=compiler, model="test-model")

    response = run_turn(executor, context_id="ctx-redacted-log", text="request")

    assert text(response) == "I couldn't complete that request safely."
    serialized = json.dumps(test_logger.records, sort_keys=True)
    assert "invalid_plan" in serialized
    assert secret not in serialized


def test_confirmation_authorization_is_consumed_before_the_next_horizon() -> None:
    critical_set_mode = tool(
        "set_mode",
        effect="act",
        properties={"mode": {"type": "string", "enum": ["eco"]}},
        required=["mode"],
    )
    critical_set_mode["function"]["x-pact-requires-confirmation"] = True
    live_tools = [
        critical_set_mode,
        tool("read_mode", effect="observe"),
    ]

    def confirmation_then_act_and_observe(request: CompilationRequest) -> PlanIR:
        return PlanIR(
            plan_id=request.plan_id,
            nodes=(
                ConfirmNode(
                    id="confirm",
                    prompt="Proceed with the change?",
                    evidence_key="change.approved",
                    authorizes=("act",),
                ),
                ActNode(
                    id="act",
                    depends_on=("confirm",),
                    operation="set_mode",
                    arguments={"mode": "eco"},
                    success_evidence_key="mode.changed",
                ),
                ObserveNode(
                    id="observe",
                    depends_on=("act",),
                    operation="read_mode",
                    success_evidence_key="mode.observed",
                ),
                RespondNode(
                    id="respond",
                    depends_on=("observe",),
                    text="Mode changed and verified.",
                    outcome=ResponseOutcome.COMPLETED,
                    requires_evidence=("mode.changed", "mode.observed"),
                ),
            ),
        )

    def verify_consumed_scope(request: CompilationRequest) -> CompilationResult:
        confirmation = next(
            item for item in request.evidence if item.key == "change.approved"
        )
        assert confirmation.authorizations == ()

        # Reusing the same operation+arguments without a fresh confirmation is
        # rejected by the verifier once the adapter projects consumed scopes
        # out of the next compilation horizon.
        replay = PlanIR(
            plan_id="replay-attempt",
            evidence_inputs=(confirmation.key,),
            nodes=(
                ActNode(
                    id="act-again",
                    operation="set_mode",
                    arguments={"mode": "eco"},
                    success_evidence_key="mode.changed.again",
                ),
                RespondNode(
                    id="respond",
                    depends_on=("act-again",),
                    text="Changed again.",
                    outcome=ResponseOutcome.COMPLETED,
                    requires_evidence=("mode.changed.again",),
                ),
            ),
        )
        snapshot = CapabilitySnapshot.from_tools(request.as_tool_declarations())
        report = PlanVerifier(snapshot).verify(
            replay,
            available_evidence=request.as_evidence_records(),
        )
        assert not report.accepted
        assert "critical_action_without_confirmation" in {
            issue.code for issue in report.issues
        }

        return result_for(
            request,
            response_plan(
                request,
                text="Mode changed and verified.",
                evidence_inputs=("mode.changed", "mode.observed"),
                requires_evidence=("mode.changed", "mode.observed"),
                outcome=ResponseOutcome.COMPLETED,
            ),
        )

    compiler = FakeCompiler(
        compile_plan(confirmation_then_act_and_observe),
        verify_consumed_scope,
    )
    executor = PACTAgentExecutor(compiler=compiler, model="test-model")

    prompt = run_turn(
        executor,
        context_id="ctx-one-shot-confirmation",
        text="Change and verify the mode.",
        tools=live_tools,
    )
    assert text(prompt) == "Proceed with the change?"

    act = run_turn(
        executor,
        context_id="ctx-one-shot-confirmation",
        text="yes",
    )
    assert tool_call(act)["tool_name"] == "set_mode"

    observe = run_turn(
        executor,
        context_id="ctx-one-shot-confirmation",
        tool_results=[
            success_result(
                "set_mode",
                call_id="evaluator-set-mode",
                payload={"mode": "eco"},
            )
        ],
    )
    assert tool_call(observe)["tool_name"] == "read_mode"

    completed = run_turn(
        executor,
        context_id="ctx-one-shot-confirmation",
        tool_results=[
            success_result(
                "read_mode",
                call_id="evaluator-read-mode",
                payload={"mode": "eco"},
            )
        ],
    )

    assert text(completed) == "Mode changed and verified."
    assert len(compiler.requests) == 2
    persisted_confirmation = executor.context_states[
        "ctx-one-shot-confirmation"
    ].ledger.get("change.approved")
    assert persisted_confirmation is not None
    assert len(persisted_confirmation.authorizations) == 1


def test_user_text_does_not_clear_an_outstanding_external_obligation() -> None:
    compiler = FakeCompiler(compile_plan(act_plan))
    executor = PACTAgentExecutor(compiler=compiler, model="test-model")
    live_tools = [
        tool(
            "set_mode",
            effect="act",
            properties={"mode": {"type": "string"}},
            required=["mode"],
        )
    ]
    run_turn(
        executor,
        context_id="ctx-out-of-order-text",
        text="Set eco mode.",
        tools=live_tools,
    )
    state = executor.context_states["ctx-out-of-order-text"]
    runtime = state.runtime
    assert runtime is not None
    pending = runtime.pending_action
    assert pending is not None
    original_goal = state.goal
    original_evidence = state.ledger.evidence

    waiting = run_turn(
        executor,
        context_id="ctx-out-of-order-text",
        text="Is it finished yet?",
        message_id="message-while-operation-pending",
    )

    assert text(waiting) == "I'm still waiting for the operation result."
    assert state.runtime is runtime
    assert runtime.pending_action == pending
    assert state.ledger.evidence == original_evidence
    assert state.goal == original_goal
    assert len(compiler.requests) == 1

    completed = run_turn(
        executor,
        context_id="ctx-out-of-order-text",
        tool_results=[
            success_result(
                "set_mode",
                call_id="evaluator-after-user-text",
                payload={"mode": "eco"},
            )
        ],
    )

    assert text(completed) == "Mode changed."
    evidence = state.ledger.get("mode.changed")
    assert evidence is not None
    assert evidence.external_call_id == "evaluator-after-user-text"


def test_second_sequential_action_failure_reports_partial_completion() -> None:
    def two_action_plan(request: CompilationRequest) -> PlanIR:
        return PlanIR(
            plan_id=request.plan_id,
            nodes=(
                ActNode(
                    id="first",
                    operation="write_step",
                    arguments={"step": 1},
                    success_evidence_key="first.applied",
                ),
                ActNode(
                    id="second",
                    depends_on=("first",),
                    operation="write_step",
                    arguments={"step": 2},
                    success_evidence_key="second.applied",
                ),
                RespondNode(
                    id="respond",
                    depends_on=("second",),
                    text="Both steps are complete.",
                    outcome=ResponseOutcome.COMPLETED,
                    requires_evidence=("first.applied", "second.applied"),
                ),
            ),
        )

    compiler = FakeCompiler(compile_plan(two_action_plan))
    executor = PACTAgentExecutor(compiler=compiler, model="test-model")
    live_tools = [
        tool(
            "write_step",
            effect="act",
            properties={"step": {"type": "integer", "enum": [1, 2]}},
            required=["step"],
        )
    ]

    first = run_turn(
        executor,
        context_id="ctx-partial-failure",
        text="Apply both steps.",
        tools=live_tools,
    )
    assert tool_call(first) == {
        "tool_name": "write_step",
        "arguments": {"step": 1},
    }

    second = run_turn(
        executor,
        context_id="ctx-partial-failure",
        tool_results=[
            success_result(
                "write_step",
                call_id="evaluator-first-step",
                payload={"step": 1},
            )
        ],
    )
    assert tool_call(second) == {
        "tool_name": "write_step",
        "arguments": {"step": 2},
    }

    failed = run_turn(
        executor,
        context_id="ctx-partial-failure",
        tool_results=[
            {
                "tool_name": "write_step",
                "tool_call_id": "evaluator-second-step",
                "content": {
                    "status": "FAILURE",
                    "error": "second step rejected",
                },
            }
        ],
    )

    assert (
        text(failed)
        == "I completed part of the request, but couldn't complete the remaining action."
    )
    state = executor.context_states["ctx-partial-failure"]
    first_evidence = state.ledger.get("first.applied")
    assert first_evidence is not None
    assert first_evidence.status == EvidenceStatus.SUCCESS
    assert first_evidence.external_call_id == "evaluator-first-step"
    assert not state.ledger.has("second.applied")
    second_failure = state.ledger.get("second.applied.failure")
    assert second_failure is not None
    assert second_failure.status == EvidenceStatus.FAILURE
    assert second_failure.external_call_id == "evaluator-second-step"

    metrics = extract_turn_metrics(failed.metadata)
    assert metrics[PROMPT_TOKENS] == 11
    assert metrics[COMPLETION_TOKENS] == 7
    assert metrics[THINKING_TOKENS] == 3
    assert metrics[NUM_LLM_CALLS] == 1
    assert metrics[NUM_PASSES] == 1


@pytest.mark.parametrize(
    ("builder", "evidence_key", "answer", "context_id"),
    (
        (ask_plan, "requested.value", "forty two", "ctx-event-id-ask"),
        (confirmation_plan, "change.approved", "yes", "ctx-event-id-confirm"),
    ),
    ids=("ask", "confirm"),
)
def test_a2a_message_id_derives_deterministic_input_event_ids(
    builder: Callable[[CompilationRequest], PlanIR],
    evidence_key: str,
    answer: str,
    context_id: str,
) -> None:
    def answer_from_input(request: CompilationRequest) -> CompilationResult:
        evidence = next(item for item in request.evidence if item.key == evidence_key)
        return result_for(
            request,
            response_plan(
                request,
                text="Input recorded.",
                evidence_inputs=(evidence.key,),
                requires_evidence=(evidence.key,),
            ),
        )

    compiler = FakeCompiler(
        compile_plan(builder),
        *(() if builder is confirmation_plan else (answer_from_input,)),
    )
    executor = PACTAgentExecutor(compiler=compiler, model="test-model")
    live_tools = (
        [
            tool(
                "set_mode",
                effect="act",
                properties={"mode": {"type": "string"}},
                required=["mode"],
            )
        ]
        if builder is confirmation_plan
        else None
    )
    run_turn(
        executor,
        context_id=context_id,
        text="Collect a correlated input.",
        tools=live_tools,
        message_id="initial-a2a-message",
    )
    state = executor.context_states[context_id]
    assert state.runtime is not None
    pending = state.runtime.pending_action
    assert pending is not None
    assert pending.kind in {"ask", "confirm"}

    response = run_turn(
        executor,
        context_id=context_id,
        text=answer,
        message_id="stable-a2a-answer-message",
    )

    if builder is confirmation_plan:
        assert tool_call(response) == {
            "tool_name": "set_mode",
            "arguments": {"mode": "eco"},
        }
        assert len(compiler.requests) == 1
    else:
        assert text(response) == "Input recorded."
        assert len(compiler.requests) == 2
    record = state.ledger.get(evidence_key)
    assert record is not None
    expected = _input_event_id(
        context_id,
        "stable-a2a-answer-message",
        pending.call_id,
    )
    assert record.event_id == expected
    assert (
        _input_event_id(
            context_id,
            "stable-a2a-answer-message",
            pending.call_id,
        )
        == expected
    )
    assert (
        _input_event_id(
            context_id,
            "different-a2a-answer-message",
            pending.call_id,
        )
        != expected
    )
