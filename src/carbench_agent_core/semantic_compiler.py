"""Untrusted-model semantic compilation into verified obligation plans.

The model used by this module is a semantic translator, not an executor.  Its
output crosses a trust boundary: strict JSON decoding, :class:`PlanIR`
validation, live-capability validation, and resource bounds are all enforced
locally before a plan can reach the deterministic runtime.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from typing import Annotated, Any, Literal, Protocol, Sequence, TypeAlias

from jsonschema import Draft202012Validator  # type: ignore[import-untyped]
from jsonschema.exceptions import SchemaError  # type: ignore[import-untyped]
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .evidence_ledger import (
    ActionAuthorization,
    EvidenceRecord,
    EvidenceSource,
    EvidenceStatus,
)
from .plan_ir import (
    ActNode,
    AskNode,
    ConfirmNode,
    ObserveNode,
    PlanIR,
    RespondNode,
    ResponseOutcome,
)
from .plan_verifier import (
    CapabilitySnapshot,
    CapabilityEffect,
    PlanVerificationIssue,
    PlanVerifier,
    VerificationPolicy,
    description_requires_confirmation,
)


SEMANTIC_COMPILER_INSTRUCTIONS = """You are an untrusted semantic compiler.
Translate the supplied user intent into one finite receding-horizon obligation
plan. The trusted runtime submits only the first ready action and records the
result as evidence. Exact confirmations and successful actions advance the
already-verified immutable suffix; Ask answers, observations, and capability
changes are semantic replan boundaries.
Do not execute operations and do not claim that an operation has succeeded.
Return only JSON matching the requested schema.

Compilation contract:
- Use only operation names present in live_capabilities, exactly as supplied.
- Encode operation arguments in arguments_json as one strict JSON object
  string satisfying the associated live parameter schema. Never emit
  placeholders or fabricate observations.
- Treat every required argument as a provenance obligation. Its value must be
  unambiguously grounded in a user clause about that same operation/field, a
  trusted schema default, or successful existing evidence. Never copy a number,
  name, entity, target, or option from a sibling operation merely because it is
  nearby in the conversation.
- If an argument is absent or ambiguous, follow the trusted policy's
  disambiguation order. When internal resolution is required, first use a live
  read-only capability whose contract describes preferences, defaults, history,
  or current state; otherwise ask the user. Never guess a required value.
- Use observe only for read-only acquisition and act only for state changes.
- Use ask when a required value is absent. Use confirm when an explicit user
  decision is an unmet precondition for a consequential state change. Every
  confirm node must list the exact downstream act node IDs it authorizes.
- Dependencies must encode every prerequisite. Keep the graph minimal.
- Before proposing any act, perform a fresh policy audit: identify the intended
  operation, scan the entire trusted policy for every rule that can constrain
  it, and encode all required observations, confirmations, ordering rules, and
  automatic companion actions as ancestors. If a policy condition is unknown,
  resolve it with observe or ask before act; never assume the condition is safe.
- Re-run that full policy audit after every semantic replan boundary. Existing
  observations are evidence, not permission to skip unrelated policy checks.
- Assign a unique evidence key to each ask, confirm, observe, and act producer.
- Emit exactly one terminal respond node and classify its outcome precisely:
  completed for a finished state change, answered for an informational answer,
  refused when policy or the live capability surface prohibits the request,
  declined after a negative user confirmation, and fail_safe after an external
  failure prevents truthful completion.
- completed must cite every action-success key plus every carried key listed in
  required_completion_evidence. answered must not contain actions and should
  cite observation evidence whenever the answer depends on external state.
  refused must not call external operations. declined must cite the available
  negative-confirmation evidence. fail_safe must cite failure evidence.
- External failure never counts as completed success.
- Existing evidence is immutable context. Do not alter its value or provenance.
- Keep user-facing prompts and responses concise and truthful.
- Each wire node contains only fields defined for its kind. evidence_inputs may
  contain only exact keys from existing_evidence.

The trusted compiler owns the terminal proof closure: completed responses cite
exactly current action-success keys plus carried completion obligations, while
answered responses include every current observation-success key. Still emit
semantically relevant references; model-authored interaction evidence can never
substitute for successful external action evidence.
"""

SEMANTIC_REVIEW_INSTRUCTION = """Perform an independent semantic safety review
of the previous complete candidate against the entire trusted policy, original
goal, current event, live capabilities, and immutable evidence. Return only one
complete corrected JSON plan, never a verdict or explanation.

Review every intended external action from scratch. Find every applicable
policy condition, required observation, confirmation, ordering constraint,
automatic companion action, and unavailable-capability restriction. Unknown
conditions must be resolved before action. Audit each argument's provenance for
the same operation and field; do not inherit a value from a sibling operation.
Preserve the user's requested goal and values. Refuse only when the policy or
live capability surface actually prohibits fulfillment. The returned plan is
untrusted and will be decoded and verified again locally.
"""

MAX_PROVIDER_SCHEMA_CHARS = 5_000


class SemanticCompilationError(RuntimeError):
    """Base class for typed semantic-compilation failures."""


class CompilerInputError(SemanticCompilationError):
    """The trusted caller supplied an invalid or over-sized compile request."""


class CompilerBackendError(SemanticCompilationError):
    """The configured model backend failed before returning a candidate."""

    def __init__(
        self,
        message: str,
        *,
        attempts: tuple["CompilationAttempt", ...] = (),
    ) -> None:
        self.attempts = attempts
        super().__init__(message)

    @property
    def usage(self) -> "CompilerTokenUsage":
        return _aggregate_usage(self.attempts)

    @property
    def cost(self) -> float:
        return sum(item.cost for item in self.attempts)

    @property
    def quota_wait_ms(self) -> float:
        return sum(item.quota_wait_ms for item in self.attempts)


class PlanRepairExhaustedError(SemanticCompilationError):
    """Every bounded candidate attempt failed local verification."""

    def __init__(
        self,
        *,
        attempts: tuple["CompilationAttempt", ...],
        issues: tuple["CompilationIssue", ...],
    ) -> None:
        self.attempts = attempts
        self.issues = issues
        issue_codes = ", ".join(issue.code for issue in issues) or "unknown"
        super().__init__(
            f"semantic plan verification failed after {len(attempts)} "
            f"attempt(s): {issue_codes}"
        )

    @property
    def usage(self) -> "CompilerTokenUsage":
        return _aggregate_usage(self.attempts)

    @property
    def cost(self) -> float:
        return sum(item.cost for item in self.attempts)

    @property
    def quota_wait_ms(self) -> float:
        return sum(item.quota_wait_ms for item in self.attempts)


class Capability(BaseModel):
    """One operation from the current, revocable capability surface."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1)
    description: str = ""
    parameters: dict[str, Any]
    requires_confirmation: bool = False
    effect: CapabilityEffect = "unknown"


class ConversationEvent(BaseModel):
    """One bounded, untrusted recent-conversation event."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    role: Literal["user", "assistant", "tool"]
    content: Any
    name: str | None = None


class EvidenceSnapshot(BaseModel):
    """Prompt-safe projection of immutable execution evidence."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    key: str = Field(min_length=1)
    value: Any
    source: EvidenceSource
    status: EvidenceStatus
    producer_event_id: str = Field(min_length=1)
    producer_call_id: str = Field(min_length=1)
    producer_node_id: str = Field(min_length=1)
    producer_kind: Literal["observe", "ask", "confirm", "act"] | None = None
    operation: str | None = None
    external_call_id: str | None = None
    context_id: str | None = None
    plan_id: str | None = None
    argument_digest: str | None = None
    schema_digest: str | None = None
    authorizations: tuple[ActionAuthorization, ...] = ()

    @classmethod
    def from_record(cls, record: EvidenceRecord) -> "EvidenceSnapshot":
        """Create a compiler projection without discarding provenance."""

        return cls(
            key=record.key,
            value=record.value,
            source=record.source,
            status=record.status,
            producer_event_id=record.event_id,
            producer_call_id=record.call_id,
            producer_node_id=record.node_id,
            producer_kind=record.producer_kind,
            operation=record.operation,
            external_call_id=record.external_call_id,
            context_id=record.context_id,
            plan_id=record.plan_id,
            argument_digest=record.argument_digest,
            schema_digest=record.schema_digest,
            authorizations=record.authorizations,
        )


class CompilationRequest(BaseModel):
    """Bounded semantic input supplied to the untrusted compiler model."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    plan_id: str = Field(min_length=1, max_length=128)
    trusted_policy: str = Field(min_length=1)
    goal: str = Field(min_length=1)
    current_user_event: str = ""
    conversation: tuple[ConversationEvent, ...] = ()
    capabilities: tuple[Capability, ...]
    evidence: tuple[EvidenceSnapshot, ...] = ()
    required_completion_evidence: tuple[str, ...] = ()

    @classmethod
    def from_live_tools(
        cls,
        *,
        plan_id: str,
        trusted_policy: str,
        goal: str,
        current_user_event: str = "",
        conversation: Sequence[ConversationEvent | dict[str, Any]] = (),
        tools: Sequence[dict[str, Any]],
        evidence: Sequence[EvidenceRecord | EvidenceSnapshot] = (),
        critical_operations: Collection[str] = (),
        operation_effects: Mapping[str, CapabilityEffect] | None = None,
        required_completion_evidence: Collection[str] = (),
    ) -> "CompilationRequest":
        """Normalize OpenAI-shaped live tool declarations into capabilities."""

        capabilities: list[Capability] = []
        names: set[str] = set()
        critical = frozenset(critical_operations)
        effects = operation_effects or {}
        for index, tool in enumerate(tools):
            if not isinstance(tool, dict) or tool.get("type") != "function":
                raise CompilerInputError(
                    f"live capability at index {index} is not a function tool"
                )
            function = tool.get("function") if isinstance(tool, dict) else None
            if not isinstance(function, dict):
                raise CompilerInputError(
                    f"live capability at index {index} has no function object"
                )
            name = function.get("name")
            if not isinstance(name, str) or not name.strip():
                raise CompilerInputError(
                    f"live capability at index {index} has no valid name"
                )
            if name in names:
                raise CompilerInputError(f"live capability name {name!r} is duplicated")
            names.add(name)
            description = function.get("description", "")
            if not isinstance(description, str):
                raise CompilerInputError(
                    f"live capability {name!r} has a non-string description"
                )
            if "parameters" not in function:
                raise CompilerInputError(
                    f"live capability {name!r} has no parameter schema"
                )
            parameters = function["parameters"]
            if not isinstance(parameters, dict):
                raise CompilerInputError(
                    f"live capability {name!r} has no JSON object schema"
                )
            try:
                Draft202012Validator.check_schema(parameters)
            except SchemaError as exc:
                raise CompilerInputError(
                    f"live capability {name!r} has an invalid parameter schema"
                ) from exc
            requires_confirmation = (
                name in critical
                or function.get("x-pact-requires-confirmation") is True
                or description_requires_confirmation(description)
            )
            raw_effect = effects.get(name, function.get("x-pact-effect", "unknown"))
            if raw_effect not in {"unknown", "observe", "act"}:
                raise CompilerInputError(
                    f"live capability {name!r} has an invalid effect classification"
                )
            effect: CapabilityEffect = raw_effect
            if requires_confirmation and effect == "observe":
                raise CompilerInputError(
                    f"live capability {name!r} has a conflicting effect classification"
                )
            if requires_confirmation:
                effect = "act"
            capabilities.append(
                Capability(
                    name=name,
                    description=description,
                    parameters=parameters,
                    requires_confirmation=requires_confirmation,
                    effect=effect,
                )
            )

        unknown_effects = set(effects) - names
        unknown_critical = set(critical) - names
        if unknown_effects or unknown_critical:
            raise CompilerInputError(
                "effect and critical classifications must reference live capabilities"
            )

        evidence_snapshots = tuple(
            item
            if isinstance(item, EvidenceSnapshot)
            else EvidenceSnapshot.from_record(item)
            for item in evidence
        )
        try:
            return cls(
                plan_id=plan_id,
                trusted_policy=trusted_policy,
                goal=goal,
                current_user_event=current_user_event,
                conversation=tuple(
                    item
                    if isinstance(item, ConversationEvent)
                    else ConversationEvent.model_validate(item)
                    for item in conversation
                ),
                capabilities=tuple(capabilities),
                evidence=evidence_snapshots,
                required_completion_evidence=tuple(
                    sorted(set(required_completion_evidence))
                ),
            )
        except ValidationError as exc:
            raise CompilerInputError("invalid semantic compilation request") from exc

    def as_policy_payload(self) -> dict[str, Any]:
        """Return policy in its own trusted system-message partition."""

        return {"trusted_policy": self.trusted_policy}

    def as_prompt_payload(self) -> dict[str, Any]:
        """Return untrusted task state separately from trusted policy."""

        return {
            "requested_plan_id": self.plan_id,
            "goal": self.goal,
            "current_user_event": self.current_user_event,
            "recent_conversation": [
                event.model_dump(mode="json") for event in self.conversation
            ],
            "live_capabilities": [
                capability.model_dump(mode="json") for capability in self.capabilities
            ],
            "existing_evidence": [
                item.model_dump(mode="json") for item in self.evidence
            ],
            "required_completion_evidence": list(self.required_completion_evidence),
        }

    def as_tool_declarations(self) -> list[dict[str, Any]]:
        """Reconstruct the declaration shape consumed by :class:`ToolIndex`."""

        return [
            {
                "type": "function",
                "function": {
                    "name": capability.name,
                    "description": capability.description,
                    "parameters": capability.parameters,
                    "x-pact-requires-confirmation": (capability.requires_confirmation),
                    "x-pact-effect": capability.effect,
                },
            }
            for capability in self.capabilities
        ]

    def as_evidence_records(self) -> tuple[EvidenceRecord, ...]:
        """Reconstruct records for the trusted verifier, preserving provenance."""

        return tuple(
            EvidenceRecord(
                key=item.key,
                value=item.value,
                source=item.source,
                status=item.status,
                event_id=item.producer_event_id,
                call_id=item.producer_call_id,
                node_id=item.producer_node_id,
                producer_kind=item.producer_kind,
                operation=item.operation,
                external_call_id=item.external_call_id,
                context_id=item.context_id,
                plan_id=item.plan_id,
                argument_digest=item.argument_digest,
                schema_digest=item.schema_digest,
                authorizations=item.authorizations,
            )
            for item in self.evidence
        )


class CompilerTokenUsage(BaseModel):
    """Provider-neutral usage attached to one compiler call."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)
    thinking_tokens: int = Field(default=0, ge=0)


class ModelCandidate(BaseModel):
    """Provider-neutral result returned by a structured model backend."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    text: str
    model: str | None = None
    finish_reason: str | None = None
    duration_ms: float = Field(default=0.0, ge=0.0)
    cost: float = Field(default=0.0, ge=0.0)
    quota_wait_ms: float = Field(default=0.0, ge=0.0)
    usage: CompilerTokenUsage = Field(default_factory=CompilerTokenUsage)


class _WireNodeBase(BaseModel):
    """Fields common to compact provider-safe node variants."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    depends_on: tuple[str, ...]


class _WireObserveNode(_WireNodeBase):
    kind: Literal["observe"]
    operation: str
    arguments_json: str
    success_evidence_key: str


class _WireAskNode(_WireNodeBase):
    kind: Literal["ask"]
    prompt: str
    evidence_key: str


class _WireConfirmNode(_WireNodeBase):
    kind: Literal["confirm"]
    prompt: str
    evidence_key: str
    authorizes: tuple[str, ...]


class _WireActNode(_WireNodeBase):
    kind: Literal["act"]
    operation: str
    arguments_json: str
    success_evidence_key: str


class _WireRespondNode(_WireNodeBase):
    kind: Literal["respond"]
    text: str
    outcome: ResponseOutcome
    requires_evidence: tuple[str, ...]


_WireNode: TypeAlias = Annotated[
    _WireObserveNode
    | _WireAskNode
    | _WireConfirmNode
    | _WireActNode
    | _WireRespondNode,
    Field(discriminator="kind"),
]


class _WirePlan(BaseModel):
    """Cerebras strict-output envelope independent of dynamic argument shapes."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    plan_id: str
    evidence_inputs: tuple[str, ...]
    nodes: tuple[_WireNode, ...]


class StructuredPlanBackend(Protocol):
    """Minimal boundary required from an inference provider adapter."""

    def generate(
        self,
        *,
        messages: list[dict[str, str]],
        response_schema: dict[str, Any],
        response_schema_name: str,
    ) -> ModelCandidate:
        """Return one candidate without interpreting or executing it."""


class CompilationIssue(BaseModel):
    """Non-sensitive verifier feedback suitable for one bounded repair."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    code: str = Field(min_length=1)
    path: tuple[str | int, ...] = ()
    message: str = Field(min_length=1)


class CompilationAttempt(BaseModel):
    """Audit metadata for a model candidate; raw output is not retained."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    attempt: int = Field(ge=1)
    phase: Literal["proposal", "repair", "audit"] = "proposal"
    was_repair: bool
    accepted: bool
    output_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    model: str | None = None
    finish_reason: str | None = None
    duration_ms: float = Field(default=0.0, ge=0.0)
    cost: float = Field(default=0.0, ge=0.0)
    quota_wait_ms: float = Field(default=0.0, ge=0.0)
    usage: CompilerTokenUsage = Field(default_factory=CompilerTokenUsage)
    issues: tuple[CompilationIssue, ...] = ()


def _aggregate_usage(
    attempts: Sequence[CompilationAttempt],
) -> CompilerTokenUsage:
    return CompilerTokenUsage(
        prompt_tokens=sum(item.usage.prompt_tokens for item in attempts),
        completion_tokens=sum(item.usage.completion_tokens for item in attempts),
        thinking_tokens=sum(item.usage.thinking_tokens for item in attempts),
    )


class CompilationResult(BaseModel):
    """A verified plan plus privacy-preserving inference audit metadata."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    plan: PlanIR
    attempts: tuple[CompilationAttempt, ...]
    request_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    capability_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @property
    def usage(self) -> CompilerTokenUsage:
        """Aggregate usage across the initial call and optional repair."""

        return _aggregate_usage(self.attempts)

    @property
    def cost(self) -> float:
        """Aggregate provider-reported cost across all compiler calls."""

        return sum(item.cost for item in self.attempts)

    @property
    def quota_wait_ms(self) -> float:
        """Aggregate provider quota waits across all compiler calls."""

        return sum(item.quota_wait_ms for item in self.attempts)


@dataclass(frozen=True)
class SemanticCompilerLimits:
    """Hard bounds applied before and after untrusted inference."""

    max_repair_attempts: int = 1
    max_nodes: int = 20
    max_dependency_depth: int = 12
    max_policy_chars: int = 64_000
    max_goal_chars: int = 12_000
    max_user_event_chars: int = 12_000
    max_conversation_messages: int = 32
    max_conversation_chars: int = 64_000
    max_context_chars: int = 96_000
    max_candidate_chars: int = 64_000
    max_evidence_records: int = 128

    def __post_init__(self) -> None:
        if self.max_repair_attempts not in {0, 1}:
            raise ValueError("max_repair_attempts must be zero or one")
        for field_name in (
            "max_nodes",
            "max_dependency_depth",
            "max_policy_chars",
            "max_goal_chars",
            "max_user_event_chars",
            "max_conversation_messages",
            "max_conversation_chars",
            "max_context_chars",
            "max_candidate_chars",
            "max_evidence_records",
        ):
            if getattr(self, field_name) <= 0:
                raise ValueError(f"{field_name} must be positive")


class SemanticCompiler:
    """Compile intent into a locally verified, benchmark-agnostic PlanIR."""

    def __init__(
        self,
        backend: StructuredPlanBackend,
        *,
        limits: SemanticCompilerLimits | None = None,
        semantic_review: bool = False,
    ) -> None:
        self._backend = backend
        self._limits = limits or SemanticCompilerLimits()
        self._semantic_review = semantic_review

    def compile(self, request: CompilationRequest) -> CompilationResult:
        """Compile once, perform at most one repair, and never execute output."""

        policy_json = _canonical_json(request.as_policy_payload())
        payload_json = _canonical_json(request.as_prompt_payload())
        request_json = _canonical_json(request.model_dump(mode="json"))
        self._validate_request_bounds(
            request,
            policy_json=policy_json,
            payload_json=payload_json,
        )
        response_schema = _wire_plan_response_schema(
            plan_id=request.plan_id,
            operation_names=tuple(item.name for item in request.capabilities),
        )
        base_messages = [
            {"role": "system", "content": SEMANTIC_COMPILER_INSTRUCTIONS},
            {"role": "system", "content": policy_json},
            {"role": "user", "content": payload_json},
        ]
        messages = base_messages
        snapshot = CapabilitySnapshot.from_tools(request.as_tool_declarations())
        verifier = PlanVerifier(
            snapshot,
            policy=VerificationPolicy(
                max_nodes=self._limits.max_nodes,
                max_dependency_depth=self._limits.max_dependency_depth,
            ),
        )
        attempts: list[CompilationAttempt] = []
        final_issues: tuple[CompilationIssue, ...] = ()
        accepted_plan: PlanIR | None = None

        for attempt_index in range(self._limits.max_repair_attempts + 1):
            try:
                candidate = self._backend.generate(
                    messages=messages,
                    response_schema=response_schema,
                    response_schema_name="obligation_plan",
                )
            except CompilerBackendError as exc:
                if not exc.attempts:
                    exc.attempts = tuple(attempts)
                raise
            except SemanticCompilationError:
                raise
            except Exception as exc:
                raise CompilerBackendError(
                    f"semantic compiler backend failed on attempt {attempt_index + 1}",
                    attempts=tuple(attempts),
                ) from exc

            issues: tuple[CompilationIssue, ...]
            plan: PlanIR | None
            if candidate.finish_reason == "length":
                issues = (
                    CompilationIssue(
                        code="provider_output_truncated",
                        message=(
                            "The provider exhausted its output budget before "
                            "producing a complete plan."
                        ),
                    ),
                )
                plan = None
            else:
                issues, plan = self._decode_and_verify(
                    candidate.text,
                    request=request,
                    verifier=verifier,
                )
            accepted = plan is not None and not issues
            attempt = CompilationAttempt(
                attempt=attempt_index + 1,
                phase="repair" if attempt_index > 0 else "proposal",
                was_repair=attempt_index > 0,
                accepted=accepted,
                output_sha256=_sha256(candidate.text),
                model=candidate.model,
                finish_reason=candidate.finish_reason,
                duration_ms=candidate.duration_ms,
                cost=candidate.cost,
                quota_wait_ms=candidate.quota_wait_ms,
                usage=candidate.usage,
                issues=issues,
            )
            attempts.append(attempt)
            if accepted and plan is not None:
                accepted_plan = plan
                break

            final_issues = issues
            if any(issue.code == "provider_output_truncated" for issue in issues):
                # Repeating an identical call cannot repair a provider-side
                # output-budget exhaustion and only burns the Track 2 call and
                # token budget. Configuration can be adjusted on the next turn.
                break
            if attempt_index < self._limits.max_repair_attempts:
                messages = [
                    *messages,
                    {"role": "assistant", "content": candidate.text},
                    {
                        "role": "user",
                        "content": _repair_prompt(issues),
                    },
                ]

        if accepted_plan is None:
            raise PlanRepairExhaustedError(
                attempts=tuple(attempts),
                issues=final_issues,
            )

        if self._semantic_review and any(
            isinstance(node, ActNode) for node in accepted_plan.nodes
        ):
            review_messages = [
                *base_messages,
                {
                    "role": "assistant",
                    "content": _plan_to_wire_json(accepted_plan),
                },
                {"role": "user", "content": SEMANTIC_REVIEW_INSTRUCTION},
            ]
            try:
                candidate = self._backend.generate(
                    messages=review_messages,
                    response_schema=response_schema,
                    response_schema_name="obligation_plan",
                )
            except CompilerBackendError as exc:
                if not exc.attempts:
                    exc.attempts = tuple(attempts)
                raise
            except SemanticCompilationError:
                raise
            except Exception as exc:
                raise CompilerBackendError(
                    "semantic compiler backend failed during semantic review",
                    attempts=tuple(attempts),
                ) from exc

            review_issues: tuple[CompilationIssue, ...]
            if candidate.finish_reason == "length":
                review_issues = (
                    CompilationIssue(
                        code="provider_output_truncated",
                        message=(
                            "The provider exhausted its output budget before "
                            "producing a complete reviewed plan."
                        ),
                    ),
                )
                reviewed_plan = None
            else:
                review_issues, reviewed_plan = self._decode_and_verify(
                    candidate.text,
                    request=request,
                    verifier=verifier,
                )
            review_accepted = reviewed_plan is not None and not review_issues
            attempts.append(
                CompilationAttempt(
                    attempt=len(attempts) + 1,
                    phase="audit",
                    was_repair=False,
                    accepted=review_accepted,
                    output_sha256=_sha256(candidate.text),
                    model=candidate.model,
                    finish_reason=candidate.finish_reason,
                    duration_ms=candidate.duration_ms,
                    cost=candidate.cost,
                    quota_wait_ms=candidate.quota_wait_ms,
                    usage=candidate.usage,
                    issues=review_issues,
                )
            )
            if not review_accepted or reviewed_plan is None:
                raise PlanRepairExhaustedError(
                    attempts=tuple(attempts),
                    issues=review_issues,
                )
            accepted_plan = reviewed_plan

        return CompilationResult(
            plan=accepted_plan,
            attempts=tuple(attempts),
            request_sha256=_sha256(request_json),
            capability_sha256=snapshot.digest,
        )

    def _validate_request_bounds(
        self,
        request: CompilationRequest,
        *,
        policy_json: str,
        payload_json: str,
    ) -> None:
        if len(request.trusted_policy) > self._limits.max_policy_chars:
            raise CompilerInputError("trusted policy exceeds compiler size limit")
        if len(request.goal) > self._limits.max_goal_chars:
            raise CompilerInputError("goal exceeds compiler size limit")
        if len(request.current_user_event) > self._limits.max_user_event_chars:
            raise CompilerInputError("current user event exceeds compiler size limit")
        if len(request.conversation) > self._limits.max_conversation_messages:
            raise CompilerInputError(
                "recent conversation exceeds compiler message limit"
            )
        conversation_json = _canonical_json(
            [item.model_dump(mode="json") for item in request.conversation]
        )
        if len(conversation_json) > self._limits.max_conversation_chars:
            raise CompilerInputError("recent conversation exceeds compiler size limit")
        if len(request.evidence) > self._limits.max_evidence_records:
            raise CompilerInputError("evidence set exceeds compiler record limit")
        if len(policy_json) + len(payload_json) > self._limits.max_context_chars:
            raise CompilerInputError("compiler context exceeds size limit")

    def _decode_and_verify(
        self,
        candidate_text: str,
        *,
        request: CompilationRequest,
        verifier: PlanVerifier,
    ) -> tuple[tuple[CompilationIssue, ...], PlanIR | None]:
        if len(candidate_text) > self._limits.max_candidate_chars:
            return (
                CompilationIssue(
                    code="candidate_too_large",
                    message="Return a smaller plan within the output size bound.",
                ),
            ), None

        try:
            payload = _strict_json_object(candidate_text)
        except (json.JSONDecodeError, ValueError) as exc:
            return (
                CompilationIssue(
                    code="invalid_json",
                    message=f"Return exactly one strict JSON object: {_safe_error(exc)}",
                ),
            ), None

        try:
            wire = _WirePlan.model_validate_json(
                _canonical_json(payload),
                strict=True,
            )
            plan = PlanIR.model_validate_json(
                _canonical_json(
                    _wire_to_plan_payload(
                        wire,
                        required_completion_evidence=(
                            request.required_completion_evidence
                        ),
                    )
                ),
                strict=True,
            )
        except (ValidationError, ValueError, TypeError) as exc:
            return _issues_from_validation_error(exc), None

        issues = list(
            self._verify_plan_id(
                plan,
                requested_plan_id=request.plan_id,
            )
        )
        report = verifier.verify(
            plan,
            available_evidence=request.as_evidence_records(),
            required_completion_evidence=(request.required_completion_evidence),
        )
        issues.extend(
            _issues_from_verifier(
                tuple(issue for issue in report.issues if issue.severity == "error")
            )
        )
        return tuple(issues), plan if not issues else None

    @staticmethod
    def _verify_plan_id(
        plan: PlanIR,
        *,
        requested_plan_id: str,
    ) -> tuple[CompilationIssue, ...]:
        issues: list[CompilationIssue] = []
        if plan.plan_id != requested_plan_id:
            issues.append(
                CompilationIssue(
                    code="plan_id_mismatch",
                    path=("plan_id",),
                    message="Use requested_plan_id exactly.",
                )
            )
        return tuple(issues)


def _wire_plan_response_schema(
    *,
    plan_id: str,
    operation_names: tuple[str, ...],
) -> dict[str, Any]:
    """Build a compact schema supported by Cerebras strict output v2.

    Dynamic operation arguments cannot be represented as a free-form object in
    strict mode because every object must set ``additionalProperties: false``.
    The wire format therefore carries one JSON object string, which is decoded
    and checked against the live Draft 2020-12 schema before PlanIR creation.
    """

    common = {
        "id": {"type": "string"},
        "depends_on": {"type": "array", "items": {"type": "string"}},
    }

    def node_schema(
        kind: str,
        fields: Mapping[str, Any],
    ) -> dict[str, Any]:
        properties = {
            **common,
            "kind": {"type": "string", "enum": [kind]},
            **fields,
        }
        return {
            "type": "object",
            "properties": properties,
            "required": list(properties),
            "additionalProperties": False,
        }

    operation_fields = {
        "operation": {"$ref": "#/$defs/operation"},
        "arguments_json": {"type": "string"},
        "success_evidence_key": {"type": "string"},
    }
    ask_fields = {
        "prompt": {"type": "string"},
        "evidence_key": {"type": "string"},
    }
    confirm_fields = {
        **ask_fields,
        "authorizes": {
            "type": "array",
            "items": {"type": "string"},
        },
    }
    response_fields = {
        "text": {"type": "string"},
        "outcome": {
            "type": "string",
            "enum": [outcome.value for outcome in ResponseOutcome],
        },
        "requires_evidence": {
            "type": "array",
            "items": {"type": "string"},
        },
    }
    node_variants = [
        node_schema("ask", ask_fields),
        node_schema("confirm", confirm_fields),
        node_schema("respond", response_fields),
    ]
    if operation_names:
        node_variants = [
            node_schema("observe", operation_fields),
            *node_variants[:2],
            node_schema("act", operation_fields),
            node_variants[2],
        ]

    schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "plan_id": {"type": "string", "enum": [plan_id]},
            "evidence_inputs": {
                "type": "array",
                "items": {"type": "string"},
            },
            "nodes": {
                "type": "array",
                "items": {
                    "anyOf": node_variants,
                },
            },
        },
        "required": ["plan_id", "evidence_inputs", "nodes"],
        "additionalProperties": False,
    }
    if operation_names:
        schema["$defs"] = {
            "operation": {
                "type": "string",
                "enum": list(operation_names),
            }
        }
        if len(_canonical_json(schema)) > MAX_PROVIDER_SCHEMA_CHARS:
            # Structure remains provider-constrained; exact operation
            # membership is still enforced by the trusted live verifier.
            schema["$defs"]["operation"] = {"type": "string"}
    if len(_canonical_json(schema)) > MAX_PROVIDER_SCHEMA_CHARS:
        raise CompilerInputError("provider response schema exceeds size limit")
    return schema


def _wire_to_plan_payload(
    wire: _WirePlan,
    *,
    required_completion_evidence: Collection[str] = (),
) -> dict[str, Any]:
    """Convert the uniform provider envelope into the discriminated PlanIR."""

    if len(set(wire.evidence_inputs)) != len(wire.evidence_inputs):
        raise ValueError("evidence_inputs must be unique")

    nodes: list[dict[str, Any]] = []
    for node in wire.nodes:
        common = {
            "id": node.id,
            "kind": node.kind,
            "depends_on": list(node.depends_on),
        }
        if isinstance(node, (_WireObserveNode, _WireActNode)):
            arguments = _strict_json_object(node.arguments_json)
            if not node.operation or not node.success_evidence_key:
                raise ValueError(
                    "operation nodes require operation and success_evidence_key"
                )
            nodes.append(
                {
                    **common,
                    "operation": node.operation,
                    "arguments": arguments,
                    "success_evidence_key": node.success_evidence_key,
                }
            )
            continue

        if isinstance(node, (_WireAskNode, _WireConfirmNode)):
            if not node.prompt or not node.evidence_key:
                raise ValueError("interaction nodes require prompt and evidence_key")
            interaction = {
                **common,
                "prompt": node.prompt,
                "evidence_key": node.evidence_key,
            }
            if isinstance(node, _WireConfirmNode):
                interaction["authorizes"] = list(node.authorizes)
            nodes.append(interaction)
            continue

        if not isinstance(node, _WireRespondNode):  # pragma: no cover
            raise TypeError("unknown wire node variant")
        if not node.text:
            raise ValueError("respond nodes require text")
        nodes.append(
            {
                **common,
                "text": node.text,
                "outcome": node.outcome.value,
                "requires_evidence": list(node.requires_evidence),
            }
        )

    # ``authorizes`` is already an explicit model-authored semantic relation.
    # Materialize its missing sequencing edge locally so authorization can only
    # delay an action, never release or add one. Unknown/non-Act targets and
    # cycles remain verifier errors.
    node_by_id = {
        node_payload["id"]: node_payload for node_payload in nodes
    }
    if len(node_by_id) == len(nodes):
        for node_payload in nodes:
            if node_payload["kind"] != "confirm":
                continue
            for target_id in node_payload["authorizes"]:
                target = node_by_id.get(target_id)
                if target is None or target["kind"] != "act":
                    continue
                target["depends_on"] = list(
                    dict.fromkeys(
                        [*target["depends_on"], node_payload["id"]]
                    )
                )

    # Do not trust the model to define completion proof semantics.  The local
    # kernel derives the exact action-success closure and carried obligations,
    # filtering interaction or observation keys that cannot prove completion.
    # PlanIR and PlanVerifier still verify every producer and provenance record.
    action_success_keys = [
        node["success_evidence_key"] for node in nodes if node["kind"] == "act"
    ]
    observation_success_keys = [
        node["success_evidence_key"] for node in nodes if node["kind"] == "observe"
    ]
    for node_payload in nodes:
        if node_payload["kind"] != "respond":
            continue
        induced: list[str] = []
        if node_payload["outcome"] == ResponseOutcome.COMPLETED.value:
            node_payload["requires_evidence"] = list(
                dict.fromkeys(
                    [
                        *action_success_keys,
                        *sorted(required_completion_evidence),
                    ]
                )
            )
            continue
        elif node_payload["outcome"] == ResponseOutcome.ANSWERED.value:
            induced = observation_success_keys
        node_payload["requires_evidence"] = list(
            dict.fromkeys([*node_payload["requires_evidence"], *induced])
        )

    payload: dict[str, Any] = {
        "plan_id": wire.plan_id,
        "nodes": nodes,
    }
    payload["evidence_inputs"] = list(
        dict.fromkeys(
            [*wire.evidence_inputs, *sorted(required_completion_evidence)]
        )
    )
    return payload


def _strict_json_object(text: str) -> dict[str, Any]:
    """Decode RFC-style JSON while rejecting duplicate keys and non-finite values."""

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite number {value!r} is not valid")

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate object key {key!r}")
            result[key] = value
        return result

    payload = json.loads(
        text,
        parse_constant=reject_constant,
        object_pairs_hook=unique_object,
    )
    if not isinstance(payload, dict):
        raise ValueError("top-level value must be an object")
    return payload


def _plan_to_wire_json(plan: PlanIR) -> str:
    """Serialize a verified PlanIR back into the compact provider wire shape."""

    nodes: list[dict[str, Any]] = []
    for node in plan.nodes:
        common = {
            "id": node.id,
            "kind": node.kind,
            "depends_on": list(node.depends_on),
        }
        if isinstance(node, (ObserveNode, ActNode)):
            nodes.append(
                {
                    **common,
                    "operation": node.operation,
                    "arguments_json": _canonical_json(node.arguments),
                    "success_evidence_key": node.success_evidence_key,
                }
            )
        elif isinstance(node, AskNode):
            nodes.append(
                {
                    **common,
                    "prompt": node.prompt,
                    "evidence_key": node.evidence_key,
                }
            )
        elif isinstance(node, ConfirmNode):
            nodes.append(
                {
                    **common,
                    "prompt": node.prompt,
                    "evidence_key": node.evidence_key,
                    "authorizes": list(node.authorizes),
                }
            )
        elif isinstance(node, RespondNode):
            nodes.append(
                {
                    **common,
                    "text": node.text,
                    "outcome": node.outcome.value,
                    "requires_evidence": list(node.requires_evidence),
                }
            )
        else:  # pragma: no cover - PlanIR is a closed discriminated union
            raise TypeError(f"unsupported plan node: {type(node).__name__}")
    return _canonical_json(
        {
            "plan_id": plan.plan_id,
            "evidence_inputs": list(plan.evidence_inputs),
            "nodes": nodes,
        }
    )


def _issues_from_validation_error(
    error: ValidationError | ValueError | TypeError,
) -> tuple[CompilationIssue, ...]:
    if not isinstance(error, ValidationError):
        return (
            CompilationIssue(
                code="invalid_plan",
                message=f"Return a valid obligation plan: {_safe_error(error)}",
            ),
        )
    issues: list[CompilationIssue] = []
    for detail in error.errors(include_input=False, include_url=False)[:12]:
        issue_type = str(detail.get("type") or "validation_error")
        location = tuple(detail.get("loc") or ())
        message = str(detail.get("msg") or "Satisfy the PlanIR contract.")
        normalized_message = message.casefold()
        invariant_code = next(
            (
                code
                for fragment, code in (
                    (
                        "must depend on confirmation",
                        "confirmation_dependency_missing",
                    ),
                    ("every plan node must lead", "node_not_terminal_ancestor"),
                    ("response must be terminal", "response_not_terminal"),
                    ("must form an acyclic graph", "dependency_cycle"),
                    ("must contain exactly one response", "response_count"),
                )
                if fragment in normalized_message
            ),
            None,
        )
        issues.append(
            CompilationIssue(
                code=(
                    f"invalid_plan:{invariant_code}"
                    if invariant_code is not None
                    else f"invalid_plan:{issue_type}"
                ),
                path=location,
                message=message,
            )
        )
    return tuple(issues) or (
        CompilationIssue(
            code="invalid_plan",
            message="Return a valid obligation plan.",
        ),
    )


def _issues_from_verifier(
    issues: tuple[PlanVerificationIssue, ...],
) -> tuple[CompilationIssue, ...]:
    """Project trusted verifier findings into value-redacted repair feedback."""

    return tuple(
        CompilationIssue(
            code=issue.code,
            path=(
                *(("nodes", issue.node_id) if issue.node_id is not None else ()),
                *issue.instance_path,
            ),
            message=issue.message,
        )
        for issue in issues
    )


def _repair_prompt(issues: tuple[CompilationIssue, ...]) -> str:
    payload = {
        "repair_instruction": (
            "Repair the previous candidate once. Return only the complete corrected "
            "JSON object; do not explain the changes."
        ),
        "verification_issues": [issue.model_dump(mode="json") for issue in issues],
    }
    return _canonical_json(payload)


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise CompilerInputError("compiler input is not finite JSON data") from exc


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _safe_error(error: BaseException) -> str:
    text = str(error).replace("\n", " ").strip()
    return text[:240] or error.__class__.__name__


__all__ = [
    "Capability",
    "CompilationAttempt",
    "CompilationIssue",
    "CompilationRequest",
    "CompilationResult",
    "CompilerBackendError",
    "CompilerInputError",
    "CompilerTokenUsage",
    "ConversationEvent",
    "EvidenceSnapshot",
    "ModelCandidate",
    "PlanRepairExhaustedError",
    "SEMANTIC_COMPILER_INSTRUCTIONS",
    "SEMANTIC_REVIEW_INSTRUCTION",
    "SemanticCompilationError",
    "SemanticCompiler",
    "SemanticCompilerLimits",
    "StructuredPlanBackend",
]
