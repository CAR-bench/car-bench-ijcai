"""A2A adapter for the PACT contract-guided obligation runtime.

The adapter deliberately contains protocol plumbing, not domain workflows.  A
model compiles the trusted policy and current event into a typed plan; local
verification and a deterministic runtime own every externally visible action.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from typing import Any, Protocol, Sequence
from uuid import uuid4

from a2a.helpers.proto_helpers import new_data_part, new_message, new_text_part
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.types import Role
from google.protobuf.json_format import MessageToDict  # type: ignore[import-untyped]

from carbench_agent_core.evidence_ledger import (
    ConfirmationEvent,
    EventDisposition,
    EvidenceLedger,
    EvidenceRecord,
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
    ObligationRuntime,
    RespondAction,
    RuntimeAction,
)
from carbench_agent_core.plan_ir import ActNode, ConfirmNode
from carbench_agent_core.plan_verifier import CapabilitySnapshot
from carbench_agent_core.response_renderer import clean_user_content
from carbench_agent_core.semantic_compiler import (
    CompilationAttempt,
    CompilationRequest,
    CompilationResult,
    CompilerBackendError,
    ConversationEvent,
    PlanRepairExhaustedError,
    SemanticCompilationError,
)
from logging_utils import configure_logger
from tool_call_types import ToolCall, ToolCallsData
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
    TURN_METRICS_KEY,
)

from .plan_compiler_backend import create_cerebras_semantic_compiler


logger = configure_logger(role="agent_under_test", context="pact")

_SAFE_FAILURE_TEXT = "I couldn't complete that request safely."
_EXECUTION_FAILURE_TEXT = "I couldn't complete that action."
_DECLINED_TEXT = "Understood. I won't proceed."
_PARTIAL_DECLINED_TEXT = (
    "I completed part of the request, and I won't proceed with the remaining action."
)
_CONFIRMATION_RETRY_TEXT = "Please answer yes or no."
_WAITING_FOR_RESULT_TEXT = "I'm still waiting for the operation result."
_PARTIAL_FAILURE_TEXT = (
    "I completed part of the request, but couldn't complete the remaining action."
)
_CONFIRMATION_FILLER_WORDS = frozenset(
    {
        "a",
        "ahead",
        "an",
        "and",
        "as",
        "at",
        "certainly",
        "confirm",
        "confirmed",
        "continue",
        "do",
        "fine",
        "first",
        "go",
        "good",
        "i",
        "if",
        "it",
        "just",
        "need",
        "needs",
        "now",
        "okay",
        "ok",
        "please",
        "proceed",
        "requested",
        "right",
        "sounds",
        "still",
        "sure",
        "thanks",
        "thank",
        "the",
        "this",
        "that's",
        "to",
        "with",
        "want",
        "yeah",
        "yep",
        "yes",
        "you",
    }
)
_BOOLEAN_SCOPE_WORDS = frozenset(
    {"disable", "disabled", "enable", "enabled", "false", "off", "on", "true"}
)


class SemanticCompilerProtocol(Protocol):
    """Narrow dependency boundary used by the A2A adapter and its tests."""

    def compile(self, request: CompilationRequest) -> CompilationResult:
        """Return a locally verified plan or raise a typed compiler error."""


@dataclass
class _MetricAccumulator:
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    thinking_tokens: int = 0
    cost: float = 0.0
    num_calls: int = 0
    max_passes: int = 1
    quota_wait_ms: float = 0.0
    total_llm_time_ms: float = 0.0

    def add_attempts(self, attempts: Sequence[CompilationAttempt]) -> None:
        if not attempts:
            return
        self.max_passes = max(self.max_passes, len(attempts))
        for attempt in attempts:
            self.num_calls += 1
            self.prompt_tokens += attempt.usage.prompt_tokens
            self.completion_tokens += attempt.usage.completion_tokens
            self.thinking_tokens += attempt.usage.thinking_tokens
            self.cost += attempt.cost
            self.quota_wait_ms += attempt.quota_wait_ms
            self.total_llm_time_ms += attempt.duration_ms
            if attempt.model:
                self.model = attempt.model

    def public(self) -> dict[str, Any]:
        average_ms = (
            round(self.total_llm_time_ms / self.num_calls, 1) if self.num_calls else 0.0
        )
        return {
            PROMPT_TOKENS: self.prompt_tokens,
            COMPLETION_TOKENS: self.completion_tokens,
            THINKING_TOKENS: self.thinking_tokens,
            COST: self.cost,
            MODEL: self.model,
            NUM_LLM_CALLS: self.num_calls,
            AVG_LLM_CALL_TIME_MS: average_ms,
            NUM_PASSES: self.max_passes,
            QUOTA_WAIT_TIME_MS: round(self.quota_wait_ms, 1),
        }


@dataclass
class _ContextState:
    context_id: str
    model: str
    trusted_policy: str = ""
    goal: str = ""
    tools: list[dict[str, Any]] = field(default_factory=list)
    conversation: list[ConversationEvent] = field(default_factory=list)
    ledger: EvidenceLedger = field(default_factory=EvidenceLedger)
    runtime: ObligationRuntime | None = None
    required_completion_evidence: set[str] = field(default_factory=set)
    consumed_authorizations: set[tuple[str, str, str]] = field(default_factory=set)
    plan_generation: int = 0
    metrics: _MetricAccumulator = field(init=False)

    def __post_init__(self) -> None:
        self.metrics = _MetricAccumulator(model=self.model)

    def reset_metrics(self) -> None:
        self.metrics = _MetricAccumulator(model=self.model)

    def begin_goal(self, goal: str) -> None:
        self.goal = goal
        self.ledger = EvidenceLedger()
        self.runtime = None
        self.required_completion_evidence.clear()
        self.consumed_authorizations.clear()

    def next_plan_id(self) -> str:
        self.plan_generation += 1
        context_digest = hashlib.sha256(self.context_id.encode("utf-8")).hexdigest()[
            :10
        ]
        return f"plan-{context_digest}-{self.plan_generation}"


@dataclass(frozen=True)
class _Inbound:
    message_id: str
    trusted_policy: str | None
    user_text: str | None
    tools: list[dict[str, Any]] | None
    tool_results: list[dict[str, Any]] | None


class PACTAgentExecutor(AgentExecutor):
    """Context-isolated PACT executor for the CAR-bench A2A contract."""

    def __init__(
        self,
        *,
        compiler: SemanticCompilerProtocol | None = None,
        model: str | None = None,
    ) -> None:
        self.model = model or os.getenv("PACT_COMPILER_MODEL") or "gpt-oss-120b"
        self.compiler = compiler or create_cerebras_semantic_compiler(
            logger=logger.bind(context="compiler")
        )
        self._states: dict[str, _ContextState] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    @property
    def context_states(self) -> dict[str, _ContextState]:
        """Read-only-by-convention state view used by contract tests."""

        return self._states

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        context_id = context.context_id or f"invalid-context-{uuid4().hex}"
        lock = self._locks.setdefault(context_id, asyncio.Lock())
        async with lock:
            # Construct state only after acquiring the stable per-context lock.
            # Cancellation may clear state, but it never replaces this lock, so
            # an old in-flight turn and a newly arriving turn cannot execute in
            # parallel under two different locks for the same context ID.
            state = self._states.setdefault(
                context_id,
                _ContextState(context_id=context_id, model=self.model),
            )
            try:
                inbound = self._parse_inbound(context)
                response = await self._advance(state, inbound)
            except Exception as exc:  # final protocol fail-safe
                logger.bind(context=f"ctx:{context_id[:8]}").error(
                    "PACT turn failed",
                    error_type=type(exc).__name__,
                )
                response = self._safe_terminal(state, _SAFE_FAILURE_TEXT)
            await event_queue.enqueue_event(response)

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        context_id = context.context_id
        if context_id is None:
            return
        lock = self._locks.setdefault(context_id, asyncio.Lock())
        async with lock:
            self._states.pop(context_id, None)

    async def _advance(self, state: _ContextState, inbound: _Inbound):
        # The evaluator supplies the trusted policy only on the first turn.
        # Never let a later user utterance that mimics the transport envelope
        # replace that already-established trust root.
        if inbound.trusted_policy is not None and not state.trusted_policy:
            state.trusted_policy = inbound.trusted_policy
        if inbound.tools is not None:
            state.tools = inbound.tools

        if inbound.tool_results is not None:
            return await self._handle_tool_results(state, inbound.tool_results)
        if inbound.user_text is not None:
            return await self._handle_user_text(
                state,
                inbound.user_text,
                message_id=inbound.message_id,
            )
        return self._safe_terminal(state, _SAFE_FAILURE_TEXT)

    async def _handle_user_text(
        self,
        state: _ContextState,
        text: str,
        *,
        message_id: str,
    ):
        normalized_text = clean_user_content(text)
        if not normalized_text:
            return self._safe_terminal(state, _SAFE_FAILURE_TEXT)
        state.conversation.append(
            ConversationEvent(role="user", content=normalized_text)
        )

        pending = state.runtime.pending_action if state.runtime is not None else None
        if pending is not None and pending.kind in {"observe", "act"}:
            # An emitted external operation has an unresolved outcome.  Never
            # discard that obligation merely because a user message arrived
            # out of protocol order.
            return self._text_message(state, _WAITING_FOR_RESULT_TEXT)
        if pending is None or pending.kind not in {"ask", "confirm"}:
            state.begin_goal(normalized_text)
            return await self._compile_and_emit(
                state,
                current_user_event=normalized_text,
            )

        if pending.kind == "ask":
            runtime = state.runtime
            if runtime is None:  # pragma: no cover - narrowed by pending above
                return self._safe_terminal(state, _SAFE_FAILURE_TEXT)
            application = runtime.apply_event(
                InputEvent(
                    event_id=_input_event_id(
                        state.context_id,
                        message_id,
                        pending.call_id,
                    ),
                    call_id=pending.call_id,
                    value=normalized_text,
                )
            )
            if application.disposition != EventDisposition.APPLIED:
                return self._safe_terminal(state, _SAFE_FAILURE_TEXT)
            state.runtime = None
            return await self._compile_and_emit(
                state,
                current_user_event=normalized_text,
            )

        confirmation = _parse_explicit_confirmation(normalized_text)
        if confirmation is None:
            return self._text_message(state, _CONFIRMATION_RETRY_TEXT)
        runtime = state.runtime
        if runtime is None:  # pragma: no cover - narrowed by pending above
            return self._safe_terminal(state, _SAFE_FAILURE_TEXT)
        if confirmation and _confirmation_revises_scope(
            normalized_text,
            runtime=runtime,
            pending_node_id=pending.node_id,
        ):
            # A qualified "yes" is a new semantic request, not authorization
            # for the immutable operation/arguments shown by this Confirm.
            # Abandon the old horizon without manufacturing confirmation
            # evidence; a new critical action must establish a fresh scope.
            state.runtime = None
            return await self._compile_and_emit(
                state,
                current_user_event=normalized_text,
            )
        snapshot = CapabilitySnapshot.from_tools(state.tools)
        if not snapshot.is_valid or snapshot.digest != runtime.capability_digest:
            # Confirmation is bound to the exact capability snapshot.  A
            # changed contract is a replan boundary and cannot inherit the old
            # scope.
            state.runtime = None
            return await self._compile_and_emit(
                state,
                current_user_event=normalized_text,
            )
        application = runtime.apply_event(
            ConfirmationEvent(
                event_id=_input_event_id(
                    state.context_id,
                    message_id,
                    pending.call_id,
                ),
                call_id=pending.call_id,
                confirmed=confirmation,
            )
        )
        if confirmation is False:
            response_text = (
                _PARTIAL_DECLINED_TEXT
                if state.required_completion_evidence
                else _DECLINED_TEXT
            )
            state.runtime = None
            state.goal = ""
            state.required_completion_evidence.clear()
            return self._text_message(state, response_text)
        if application.disposition != EventDisposition.APPLIED:
            return self._safe_terminal(state, _SAFE_FAILURE_TEXT)
        # Confirmation is a control event: it only releases an exact Act
        # suffix that has already passed PlanIR and live-contract validation.
        # Continuing the same immutable plan avoids asking the stochastic
        # compiler to reconstruct an already verified authorization.
        try:
            action = self._step_runtime(state, runtime)
        except EmissionRejectedError:
            return self._safe_terminal(state, _SAFE_FAILURE_TEXT)
        if action is None:
            return self._safe_terminal(state, _SAFE_FAILURE_TEXT)
        return self._action_message(state, action)

    async def _handle_tool_results(
        self,
        state: _ContextState,
        tool_results: list[dict[str, Any]],
    ):
        runtime = state.runtime
        pending = runtime.pending_action if runtime is not None else None
        if (
            runtime is None
            or pending is None
            or pending.kind not in {"observe", "act"}
            or len(tool_results) != 1
        ):
            return self._safe_terminal(state, _SAFE_FAILURE_TEXT)

        result = tool_results[0]
        operation = _tool_result_field(result, "tool_name", "toolName", "name")
        external_call_id = _tool_result_field(
            result,
            "tool_call_id",
            "toolCallId",
        )
        if operation != pending.operation or not external_call_id:
            return self._safe_terminal(state, _SAFE_FAILURE_TEXT)
        if pending.argument_digest is None or pending.capability_digest is None:
            return self._safe_terminal(state, _SAFE_FAILURE_TEXT)

        envelope, status, error = _parse_result_envelope(result.get("content"))
        state.conversation.append(
            ConversationEvent(role="tool", name=operation, content=envelope)
        )
        event = ExternalResultEvent(
            event_id=_correlated_event_id(
                state.context_id,
                pending.call_id,
                external_call_id,
                envelope,
            ),
            call_id=pending.call_id,
            external_call_id=external_call_id,
            context_id=state.context_id,
            plan_id=runtime.plan.plan_id,
            node_id=pending.node_id,
            operation=pending.operation,
            argument_digest=pending.argument_digest,
            schema_digest=pending.capability_digest,
            status=status,
            payload=envelope,
            error=error,
        )
        pending_kind = pending.kind
        application = runtime.apply_event(event)
        if application.disposition == EventDisposition.FAILURE:
            return self._safe_terminal(state, _EXECUTION_FAILURE_TEXT)
        if application.disposition != EventDisposition.APPLIED:
            return self._safe_terminal(state, _SAFE_FAILURE_TEXT)

        if pending_kind == "observe":
            state.runtime = None
            return await self._compile_and_emit(state, current_user_event="")

        completed_node = runtime.plan.node_by_id[pending.node_id]
        if isinstance(completed_node, ActNode):
            state.required_completion_evidence.add(completed_node.success_evidence_key)
            self._consume_authorization(
                state,
                completed_node,
                capability_digest=runtime.capability_digest,
            )
        try:
            action = self._step_runtime(state, runtime)
        except EmissionRejectedError:
            return self._safe_terminal(state, _SAFE_FAILURE_TEXT)
        if action is None:
            return self._safe_terminal(state, _SAFE_FAILURE_TEXT)
        return self._action_message(state, action)

    async def _compile_and_emit(
        self,
        state: _ContextState,
        *,
        current_user_event: str,
    ):
        if not state.goal:
            return self._safe_terminal(state, _SAFE_FAILURE_TEXT)
        trusted_policy = state.trusted_policy or (
            "Honor the user's request, use only live capabilities, and report "
            "external outcomes truthfully."
        )
        try:
            request = CompilationRequest.from_live_tools(
                plan_id=state.next_plan_id(),
                trusted_policy=trusted_policy,
                goal=state.goal,
                current_user_event=current_user_event,
                conversation=tuple(state.conversation[-32:]),
                tools=state.tools,
                evidence=self._available_evidence(state),
                required_completion_evidence=(state.required_completion_evidence),
            )
            result = await asyncio.to_thread(self.compiler.compile, request)
        except PlanRepairExhaustedError as exc:
            state.metrics.add_attempts(exc.attempts)
            issue_codes = [issue.code for issue in exc.issues]
            issue_summary = ",".join(issue_codes) or "unknown"
            logger.bind(context=f"ctx:{state.context_id[:8]}").warning(
                f"Semantic plan rejected after bounded repair: {issue_summary}",
                attempt_count=len(exc.attempts),
                issue_codes=issue_codes,
            )
            return self._safe_terminal(state, _SAFE_FAILURE_TEXT)
        except CompilerBackendError as exc:
            state.metrics.add_attempts(exc.attempts)
            logger.bind(context=f"ctx:{state.context_id[:8]}").warning(
                "Semantic compiler backend failed",
                attempt_count=len(exc.attempts),
            )
            return self._safe_terminal(state, _SAFE_FAILURE_TEXT)
        except SemanticCompilationError as exc:
            logger.bind(context=f"ctx:{state.context_id[:8]}").warning(
                "Semantic compilation input rejected",
                error_type=type(exc).__name__,
            )
            return self._safe_terminal(state, _SAFE_FAILURE_TEXT)

        state.metrics.add_attempts(result.attempts)
        runtime = ObligationRuntime(
            result.plan,
            context_id=state.context_id,
            capability_digest=result.capability_sha256,
            ledger=state.ledger,
        )
        state.runtime = runtime
        try:
            action = self._step_runtime(state, runtime)
        except EmissionRejectedError:
            return self._safe_terminal(state, _SAFE_FAILURE_TEXT)
        if action is None:
            return self._safe_terminal(state, _SAFE_FAILURE_TEXT)
        return self._action_message(state, action)

    def _step_runtime(
        self,
        state: _ContextState,
        runtime: ObligationRuntime,
    ) -> RuntimeAction | None:
        snapshot = CapabilitySnapshot.from_tools(state.tools)

        def guard(kind: str, operation: str, arguments: dict[str, Any]) -> None:
            if not snapshot.is_valid or snapshot.digest != runtime.capability_digest:
                raise ValueError("live capability snapshot changed")
            capability = snapshot.get(operation)
            if capability is None:
                raise ValueError("operation is no longer available")
            if capability.effect != "unknown" and capability.effect != kind:
                raise ValueError("operation effect does not match plan node")
            issue = snapshot.tool_index().validation_issue(operation, arguments)
            if issue is not None:
                raise ValueError(f"operation arguments rejected: {issue.code}")

        return runtime.step(emission_guard=guard)

    @staticmethod
    def _available_evidence(state: _ContextState) -> tuple[EvidenceRecord, ...]:
        """Return ledger records with already-consumed auth scopes removed."""

        projected = []
        for record in state.ledger.evidence:
            remaining = tuple(
                authorization
                for authorization in record.authorizations
                if (
                    record.key,
                    authorization.operation,
                    authorization.argument_digest,
                )
                not in state.consumed_authorizations
            )
            projected.append(record.model_copy(update={"authorizations": remaining}))
        return tuple(projected)

    @staticmethod
    def _consume_authorization(
        state: _ContextState,
        node: ActNode,
        *,
        capability_digest: str | None,
    ) -> None:
        if capability_digest is None:
            return
        argument_digest = hashlib.sha256(
            json.dumps(
                node.arguments,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()
        for record in state.ledger.evidence:
            if record.producer_kind != "confirm":
                continue
            if record.schema_digest != capability_digest:
                continue
            if any(
                authorization.operation == node.operation
                and authorization.argument_digest == argument_digest
                for authorization in record.authorizations
            ):
                state.consumed_authorizations.add(
                    (record.key, node.operation, argument_digest)
                )

    def _action_message(self, state: _ContextState, action: RuntimeAction):
        if isinstance(action, ExternalCallAction):
            state.conversation.append(
                ConversationEvent(
                    role="assistant",
                    content={
                        "kind": action.kind,
                        "operation": action.operation,
                        "arguments": action.arguments,
                        "plan_id": action.plan_id,
                    },
                )
            )
            payload = ToolCallsData(
                tool_calls=[
                    ToolCall(
                        tool_name=action.operation,
                        arguments=action.arguments,
                    )
                ]
            ).model_dump()
            return new_message(
                parts=[new_data_part(payload)],
                context_id=state.context_id,
                role=Role.ROLE_AGENT,
            )

        if isinstance(action, (AskAction, ConfirmAction)):
            return self._text_message(state, action.prompt)

        if not isinstance(action, RespondAction):
            raise TypeError(f"unsupported runtime action: {type(action).__name__}")
        state.runtime = None
        state.goal = ""
        state.required_completion_evidence.clear()
        return self._text_message(state, action.text)

    def _safe_terminal(self, state: _ContextState, text: str):
        if state.required_completion_evidence:
            text = _PARTIAL_FAILURE_TEXT
        state.runtime = None
        state.goal = ""
        state.required_completion_evidence.clear()
        return self._text_message(state, text)

    def _text_message(self, state: _ContextState, text: str):
        content = clean_user_content(text) or _SAFE_FAILURE_TEXT
        state.conversation.append(ConversationEvent(role="assistant", content=content))
        response = new_message(
            parts=[new_text_part(content)],
            context_id=state.context_id,
            role=Role.ROLE_AGENT,
        )
        response.metadata.update({TURN_METRICS_KEY: state.metrics.public()})
        state.reset_metrics()
        return response

    @staticmethod
    def _parse_inbound(context: RequestContext) -> _Inbound:
        text_parts: list[str] = []
        tools: list[dict[str, Any]] | None = None
        tool_results: list[dict[str, Any]] | None = None

        if context.message is None:
            raise ValueError("A2A request has no message")

        for part in context.message.parts:
            content_type = part.WhichOneof("content")
            if content_type == "text":
                text_parts.append(part.text)
                continue
            if content_type != "data":
                continue
            data = MessageToDict(part.data)
            if "tools" in data:
                raw_tools = data["tools"]
                if not isinstance(raw_tools, list):
                    raise ValueError("tools must be a list")
                tools = [item for item in raw_tools if isinstance(item, dict)]
                if len(tools) != len(raw_tools):
                    raise ValueError("every tool declaration must be an object")
            if "tool_results" in data:
                raw_results = data["tool_results"]
                if not isinstance(raw_results, list):
                    raise ValueError("tool_results must be a list")
                tool_results = [item for item in raw_results if isinstance(item, dict)]
                if len(tool_results) != len(raw_results):
                    raise ValueError("every tool result must be an object")

        combined_text = "\n".join(text_parts).strip()
        if not combined_text and tool_results is None:
            combined_text = (context.get_user_input() or "").strip()

        trusted_policy: str | None = None
        user_text: str | None = combined_text or None
        if combined_text.startswith("System:") and "\n\nUser:" in combined_text:
            policy_part, user_part = combined_text.split("\n\nUser:", 1)
            trusted_policy = policy_part.removeprefix("System:").strip()
            user_text = user_part.strip() or None

        return _Inbound(
            message_id=context.message.message_id or f"message-{uuid4().hex}",
            trusted_policy=trusted_policy,
            user_text=user_text,
            tools=tools,
            tool_results=tool_results,
        )


def _parse_explicit_confirmation(text: str) -> bool | None:
    normalized = re.sub(r"[^a-z0-9']+", " ", text.casefold()).strip()
    negative = re.search(
        r"\b(no|nope|cancel|stop|decline|don't|do not|not now|never mind)\b",
        normalized,
    )
    positive = re.search(
        r"\b(yes|yeah|yep|confirm|confirmed|proceed|continue|go ahead|do it)\b",
        normalized,
    )
    if negative and positive:
        return None
    if negative:
        return False
    if positive:
        return True
    return None


def _confirmation_revises_scope(
    text: str,
    *,
    runtime: ObligationRuntime,
    pending_node_id: str,
) -> bool:
    """Return whether a positive reply changes the authorized operation scope.

    The accepted plan contains static operation arguments.  A qualified reply
    must therefore trigger semantic recompilation instead of silently applying
    the old values.  This check is deliberately domain-agnostic: it recognizes
    generic revision language and lexical, boolean, or numeric values absent
    from the exact descendant Act arguments.
    """

    node = runtime.plan.node_by_id.get(pending_node_id)
    if not isinstance(node, ConfirmNode):
        return True

    normalized = re.sub(r"[^a-z0-9.%+\-']+", " ", text.casefold()).strip()
    if re.search(
        r"\b(but|instead|rather|except|change|modify|different|also|plus|"
        r"switch|adjust|set|use)\b",
        normalized,
    ):
        return True

    authorized_targets = [
        target
        for target_id in node.authorizes
        for target in (runtime.plan.node_by_id.get(target_id),)
        if isinstance(target, ActNode)
    ]
    authorized_arguments = [target.arguments for target in authorized_targets]
    authorized_numbers = {
        canonical
        for arguments in authorized_arguments
        for value in _leaf_values(arguments)
        for token in _numeric_tokens(value)
        if (canonical := _canonical_number(token)) is not None
    }
    mentioned_numbers = {
        canonical
        for token in re.findall(r"(?<![a-z0-9])-?\d+(?:\.\d+)?", normalized)
        if (canonical := _canonical_number(token)) is not None
    }
    qualitative_numbers, qualitative_words = _qualitative_number_scope(normalized)
    mentioned_numbers.update(qualitative_numbers)

    completed_scopes = [
        (
            _operation_words(record.operation),
            _distinctive_operation_words(record.operation),
            {
                canonical
                for value in _leaf_values(record.value)
                for token in _numeric_tokens(value)
                if (canonical := _canonical_number(token)) is not None
            },
        )
        for record in runtime.ledger.evidence
        if record.producer_kind == "act"
        and record.status == EvidenceStatus.SUCCESS
        and record.operation is not None
    ]
    consumed_completed_numbers: set[Decimal] = set()
    for clause in re.split(r"(?:[.;!?]+|\b(?:and|but)\b)", normalized):
        clause_words = set(re.findall(r"[a-z]+(?:'[a-z]+)?", clause))
        if not clause_words:
            continue
        clause_numbers = {
            canonical
            for token in re.findall(r"(?<![a-z0-9])-?\d+(?:\.\d+)?", clause)
            if (canonical := _canonical_number(token)) is not None
        }
        clause_qualitative_numbers, _ = _qualitative_number_scope(clause)
        clause_numbers.update(clause_qualitative_numbers)
        matching_authorized = [
            target
            for target in authorized_targets
            if _scope_words_overlap(
                clause_words,
                _distinctive_operation_words(target.operation),
            )
        ]
        if matching_authorized:
            allowed = {
                canonical
                for target in matching_authorized
                for value in _leaf_values(target.arguments)
                for token in _numeric_tokens(value)
                if (canonical := _canonical_number(token)) is not None
            }
            if clause_numbers.difference(allowed):
                return True
            continue
        matching_completed = [
            scope
            for scope in completed_scopes
            if _scope_words_overlap(clause_words, scope[1])
        ]
        if matching_completed:
            allowed = {
                number for scope in matching_completed for number in scope[2]
            }
            if clause_numbers.difference(allowed):
                return True
            consumed_completed_numbers.update(clause_numbers)

    if mentioned_numbers.difference(
        authorized_numbers | consumed_completed_numbers
    ):
        return True

    authorized_words = {
        word
        for arguments in authorized_arguments
        for word in _argument_words(arguments)
    }
    authorized_words.update(
        word
        for target in authorized_targets
        for word in re.findall(r"[a-z]+", target.operation.casefold())
    )
    authorized_words.update(
        word for scope in completed_scopes for word in scope[0]
    )
    authorized_boolean_words = {
        word
        for arguments in authorized_arguments
        for value in _leaf_values(arguments)
        if isinstance(value, bool)
        for word in (
            {"enable", "enabled", "on", "true"}
            if value
            else {"disable", "disabled", "false", "off"}
        )
    }
    mentioned_words = set(re.findall(r"[a-z]+(?:'[a-z]+)?", normalized))
    operation_scope_words = {
        word
        for target in authorized_targets
        for word in _distinctive_operation_words(target.operation)
    } | {word for scope in completed_scopes for word in scope[1]}
    authorized_words.update(
        word
        for word in mentioned_words
        if _scope_words_overlap({word}, operation_scope_words)
    )
    mentioned_boolean_words = mentioned_words & _BOOLEAN_SCOPE_WORDS
    if mentioned_boolean_words.difference(authorized_boolean_words):
        return True
    unexplained_words = mentioned_words.difference(
        _CONFIRMATION_FILLER_WORDS
        | authorized_words
        | authorized_boolean_words
        | qualitative_words
    )
    return bool(unexplained_words)


def _leaf_values(value: Any):
    if isinstance(value, dict):
        for nested in value.values():
            yield from _leaf_values(nested)
        return
    if isinstance(value, list):
        for nested in value:
            yield from _leaf_values(nested)
        return
    yield value


def _argument_words(value: Any):
    if isinstance(value, dict):
        for key, nested in value.items():
            yield from re.findall(r"[a-z]+", str(key).casefold())
            yield from _argument_words(nested)
        return
    if isinstance(value, list):
        for nested in value:
            yield from _argument_words(nested)
        return
    if isinstance(value, str):
        yield from re.findall(r"[a-z]+(?:'[a-z]+)?", value.casefold())


def _numeric_tokens(value: Any):
    if isinstance(value, bool):
        return ()
    if isinstance(value, (int, float)):
        return (value,)
    if isinstance(value, str):
        return tuple(re.findall(r"(?<![a-z0-9])-?\d+(?:\.\d+)?", value.casefold()))
    return ()


def _qualitative_number_scope(text: str) -> tuple[set[Decimal], set[str]]:
    """Map a small domain-neutral set of explicit extent phrases to numbers."""

    numbers: set[Decimal] = set()
    words: set[str] = set()
    extent_patterns = (
        (r"\b(all the way|fully|full|maximum|max)\b", Decimal(100)),
        (r"\b(half way|halfway|half)\b", Decimal(50)),
        (r"\b(zero|minimum|min)\b", Decimal(0)),
    )
    for pattern, value in extent_patterns:
        for match in re.finditer(pattern, text):
            numbers.add(value)
            words.update(re.findall(r"[a-z]+", match.group(0)))
    return numbers, words


def _operation_words(operation: str) -> set[str]:
    return set(re.findall(r"[a-z]+", operation.casefold()))


def _distinctive_operation_words(operation: str) -> set[str]:
    words = _operation_words(operation)
    generic = {
        "act",
        "apply",
        "change",
        "close",
        "create",
        "delete",
        "execute",
        "get",
        "open",
        "read",
        "run",
        "send",
        "set",
        "tool",
        "update",
        "write",
    }
    distinctive = words.difference(generic)
    return distinctive or words


def _scope_words_overlap(left: set[str], right: set[str]) -> bool:
    """Match exact or informative compound fragments across a scope clause."""

    return any(
        first == second
        or (
            min(len(first), len(second)) >= 4
            and (first in second or second in first)
        )
        for first in left
        for second in right
    )


def _canonical_number(value: Any) -> Decimal | None:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return None
    try:
        return Decimal(str(value)).normalize()
    except InvalidOperation:
        return None


def _parse_result_envelope(
    content: Any,
) -> tuple[dict[str, Any], ExternalResultStatus, Any]:
    if isinstance(content, dict):
        envelope = _json_object_copy(content)
    elif isinstance(content, str):
        envelope = _strict_json_object(content)
    else:
        raise ValueError("tool result content must be a JSON object")

    raw_status = envelope.get("status")
    status = raw_status.upper() if isinstance(raw_status, str) else ""
    if status == "SUCCESS":
        return envelope, ExternalResultStatus.SUCCESS, None
    if status == "FAILURE":
        return envelope, ExternalResultStatus.FAILURE, envelope.get("error")
    raise ValueError("tool result has no explicit SUCCESS or FAILURE status")


def _strict_json_object(text: str) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON number {value!r}")

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r}")
            result[key] = value
        return result

    parsed = json.loads(
        text,
        parse_constant=reject_constant,
        object_pairs_hook=unique_object,
    )
    if not isinstance(parsed, dict):
        raise ValueError("tool result must decode to an object")
    return parsed


def _json_object_copy(value: dict[str, Any]) -> dict[str, Any]:
    return _strict_json_object(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    )


def _tool_result_field(result: dict[str, Any], *names: str) -> str:
    for name in names:
        value = result.get(name)
        if isinstance(value, str) and value:
            return value
    return ""


def _correlated_event_id(
    context_id: str,
    internal_call_id: str,
    external_call_id: str,
    envelope: dict[str, Any],
) -> str:
    payload = json.dumps(
        [context_id, internal_call_id, external_call_id, envelope],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return f"event-{hashlib.sha256(payload.encode('utf-8')).hexdigest()}"


def _input_event_id(context_id: str, message_id: str, call_id: str) -> str:
    payload = json.dumps(
        [context_id, message_id, call_id],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return f"event-{hashlib.sha256(payload.encode('utf-8')).hexdigest()}"


__all__ = ["PACTAgentExecutor", "SemanticCompilerProtocol"]
