"""Deterministic runtime for validated obligation-plan DAGs."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Callable, Literal, TypeAlias
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

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
    RuntimeEvent,
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


class NodeExecutionStatus(StrEnum):
    PENDING = "pending"
    WAITING = "waiting"
    SATISFIED = "satisfied"
    FAILED = "failed"


class PlanRunStatus(StrEnum):
    ACTIVE = "active"
    WAITING = "waiting"
    BLOCKED = "blocked"
    COMPLETED = "completed"


class ExternalCallAction(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["observe", "act"]
    node_id: str = Field(min_length=1)
    call_id: str = Field(min_length=1)
    plan_id: str = Field(min_length=1)
    operation: str = Field(min_length=1)
    arguments: dict[str, Any] = Field(default_factory=dict)
    argument_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    capability_digest: str | None = None


class AskAction(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["ask"] = "ask"
    node_id: str = Field(min_length=1)
    call_id: str = Field(min_length=1)
    plan_id: str = Field(min_length=1)
    prompt: str = Field(min_length=1)


class ConfirmAction(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["confirm"] = "confirm"
    node_id: str = Field(min_length=1)
    call_id: str = Field(min_length=1)
    plan_id: str = Field(min_length=1)
    prompt: str = Field(min_length=1)


class RespondAction(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["respond"] = "respond"
    node_id: str = Field(min_length=1)
    plan_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    outcome: ResponseOutcome
    evidence_refs: tuple[str, ...] = ()


RuntimeAction: TypeAlias = (
    ExternalCallAction | AskAction | ConfirmAction | RespondAction
)
EmissionGuard: TypeAlias = Callable[
    [Literal["observe", "act"], str, dict[str, Any]],
    None,
]


class EmissionRejectedError(RuntimeError):
    """The live capability contract rejected an operation at emission time."""


@dataclass(frozen=True)
class PendingAction:
    """The single emitted action currently awaiting a correlated event."""

    node_id: str
    call_id: str
    kind: Literal["observe", "ask", "confirm", "act"]
    operation: str | None = None
    argument_digest: str | None = None
    capability_digest: str | None = None


@dataclass(frozen=True)
class EventApplication:
    """Public audit result of applying one runtime event."""

    disposition: EventDisposition
    event_id: str
    call_id: str
    node_id: str | None


CallIdFactory: TypeAlias = Callable[[], str]


@dataclass
class ObligationRuntime:
    """Execute one immutable plan with exactly-once event semantics."""

    plan: PlanIR
    context_id: str | None = None
    capability_digest: str | None = None
    call_id_factory: CallIdFactory = field(
        default=lambda: f"call_{uuid4().hex}", repr=False
    )
    ledger: EvidenceLedger = field(default_factory=EvidenceLedger)
    _node_status: dict[str, NodeExecutionStatus] = field(init=False, repr=False)
    _pending: PendingAction | None = field(default=None, init=False, repr=False)
    _issued_call_ids: set[str] = field(default_factory=set, init=False, repr=False)
    _plan_digest: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        # Pydantic's frozen models are not recursively immutable.  Isolate the
        # plan from caller-owned dicts, then retain a digest so any later
        # mutation through ``runtime.plan`` is detected before emission.
        self.plan = PlanIR.model_validate_json(self.plan.model_dump_json())
        self._plan_digest = _json_digest(self.plan.model_dump(mode="json"))
        self._node_status = {
            node.id: NodeExecutionStatus.PENDING for node in self.plan.nodes
        }

    @property
    def pending_action(self) -> PendingAction | None:
        return self._pending

    @property
    def status(self) -> PlanRunStatus:
        statuses = tuple(self._node_status.values())
        if any(status == NodeExecutionStatus.FAILED for status in statuses):
            return PlanRunStatus.BLOCKED
        if all(status == NodeExecutionStatus.SATISFIED for status in statuses):
            return PlanRunStatus.COMPLETED
        if self._pending is not None:
            return PlanRunStatus.WAITING
        return PlanRunStatus.ACTIVE

    def node_status(self, node_id: str) -> NodeExecutionStatus:
        return self._node_status[node_id]

    def step(
        self,
        *,
        emission_guard: EmissionGuard | None = None,
    ) -> RuntimeAction | None:
        """Emit the next ready action once, or return ``None`` while not runnable."""

        if _json_digest(self.plan.model_dump(mode="json")) != self._plan_digest:
            raise EmissionRejectedError("the verified plan changed before emission")

        if self.status in {
            PlanRunStatus.WAITING,
            PlanRunStatus.BLOCKED,
            PlanRunStatus.COMPLETED,
        }:
            return None

        for node_id in self.plan.topological_order():
            if self._node_status[node_id] != NodeExecutionStatus.PENDING:
                continue
            node = self.plan.node_by_id[node_id]
            if not self._dependencies_satisfied(node):
                continue
            return self._emit(node, emission_guard=emission_guard)
        return None

    def apply_event(self, event: RuntimeEvent) -> EventApplication:
        """Apply a correlated event without replaying or reordering effects."""

        pending = self._pending
        preliminary = self.ledger.register_event(
            event,
            disposition=(
                EventDisposition.APPLIED
                if pending is not None and event.call_id == pending.call_id
                else EventDisposition.OUT_OF_ORDER
            ),
            node_id=(
                pending.node_id
                if pending is not None and event.call_id == pending.call_id
                else None
            ),
        )
        if not preliminary.is_new:
            return EventApplication(
                disposition=preliminary.record.disposition,
                event_id=event.event_id,
                call_id=event.call_id,
                node_id=preliminary.record.node_id,
            )

        if pending is None or event.call_id != pending.call_id:
            return EventApplication(
                disposition=EventDisposition.OUT_OF_ORDER,
                event_id=event.event_id,
                call_id=event.call_id,
                node_id=None,
            )

        node = self.plan.node_by_id[pending.node_id]
        if not _event_matches_pending(
            event,
            node,
            pending,
            self.plan.plan_id,
            self.context_id,
        ):
            # The event is consumed as out of order, but the legitimate pending
            # action remains available for a correctly typed event.
            self._replace_event_disposition(event, EventDisposition.OUT_OF_ORDER)
            return EventApplication(
                disposition=EventDisposition.OUT_OF_ORDER,
                event_id=event.event_id,
                call_id=event.call_id,
                node_id=pending.node_id,
            )

        if isinstance(event, ExternalResultEvent):
            if not isinstance(node, (ObserveNode, ActNode)):
                raise TypeError("external results require an external-call node")
            if event.status == ExternalResultStatus.FAILURE:
                evidence_key = _evidence_key(node)
                if evidence_key is None:
                    raise RuntimeError("external node has no evidence key")
                self.ledger.add(
                    EvidenceRecord(
                        key=f"{evidence_key}.failure",
                        value={"payload": event.payload, "error": event.error},
                        source=EvidenceSource.EXTERNAL,
                        status=EvidenceStatus.FAILURE,
                        event_id=event.event_id,
                        call_id=event.call_id,
                        node_id=pending.node_id,
                        producer_kind=node.kind,
                        operation=pending.operation,
                        external_call_id=event.external_call_id,
                        context_id=event.context_id,
                        plan_id=event.plan_id,
                        argument_digest=event.argument_digest,
                        schema_digest=event.schema_digest,
                    )
                )
                self._node_status[pending.node_id] = NodeExecutionStatus.FAILED
                self._pending = None
                self._replace_event_disposition(event, EventDisposition.FAILURE)
                return EventApplication(
                    disposition=EventDisposition.FAILURE,
                    event_id=event.event_id,
                    call_id=event.call_id,
                    node_id=pending.node_id,
                )
            evidence_key = _evidence_key(node)
            assert evidence_key is not None
            self.ledger.add(
                EvidenceRecord(
                    key=evidence_key,
                    value=event.payload,
                    source=EvidenceSource.EXTERNAL,
                    status=EvidenceStatus.SUCCESS,
                    event_id=event.event_id,
                    call_id=event.call_id,
                    node_id=pending.node_id,
                    producer_kind=node.kind,
                    operation=pending.operation,
                    external_call_id=event.external_call_id,
                    context_id=event.context_id,
                    plan_id=event.plan_id,
                    argument_digest=event.argument_digest,
                    schema_digest=event.schema_digest,
                )
            )
        elif isinstance(event, InputEvent):
            if not isinstance(node, AskNode):
                raise TypeError("input events require an ask node")
            self.ledger.add(
                EvidenceRecord(
                    key=node.evidence_key,
                    value=event.value,
                    source=EvidenceSource.INPUT,
                    status=EvidenceStatus.OBSERVED,
                    event_id=event.event_id,
                    call_id=event.call_id,
                    node_id=pending.node_id,
                    producer_kind=node.kind,
                )
            )
        else:
            if not isinstance(event, ConfirmationEvent):  # pragma: no cover
                raise TypeError("unsupported runtime event type")
            if not isinstance(node, ConfirmNode):
                raise TypeError("confirmation events require a confirm node")
            self.ledger.add(
                EvidenceRecord(
                    key=node.evidence_key,
                    value=event.confirmed,
                    source=EvidenceSource.INPUT,
                    status=(
                        EvidenceStatus.OBSERVED
                        if event.confirmed
                        else EvidenceStatus.FAILURE
                    ),
                    event_id=event.event_id,
                    call_id=event.call_id,
                    node_id=pending.node_id,
                    producer_kind=node.kind,
                    context_id=self.context_id,
                    plan_id=self.plan.plan_id,
                    schema_digest=self.capability_digest,
                    authorizations=tuple(
                        ActionAuthorization(
                            operation=authorized.operation,
                            argument_digest=_json_digest(authorized.arguments),
                        )
                        for target_id in node.authorizes
                        for authorized in (self.plan.node_by_id[target_id],)
                        if isinstance(authorized, ActNode)
                    ),
                )
            )
            if not event.confirmed:
                self._node_status[pending.node_id] = NodeExecutionStatus.FAILED
                self._pending = None
                self._replace_event_disposition(event, EventDisposition.FAILURE)
                return EventApplication(
                    disposition=EventDisposition.FAILURE,
                    event_id=event.event_id,
                    call_id=event.call_id,
                    node_id=pending.node_id,
                )

        self._node_status[pending.node_id] = NodeExecutionStatus.SATISFIED
        self._pending = None
        return EventApplication(
            disposition=EventDisposition.APPLIED,
            event_id=event.event_id,
            call_id=event.call_id,
            node_id=pending.node_id,
        )

    def _dependencies_satisfied(self, node: PlanNode) -> bool:
        return all(
            self._node_status[dependency] == NodeExecutionStatus.SATISFIED
            for dependency in node.depends_on
        )

    def _emit(
        self,
        node: PlanNode,
        *,
        emission_guard: EmissionGuard | None,
    ) -> RuntimeAction:
        if isinstance(node, RespondNode):
            evidence_ref_list: list[str] = []
            evidence_records: list[EvidenceRecord] = []
            for key in node.requires_evidence:
                evidence = self.ledger.get(key)
                if evidence is not None:
                    evidence_records.append(evidence)
                    evidence_ref_list.append(evidence.event_id)
            evidence_refs = tuple(evidence_ref_list)
            if len(evidence_refs) != len(node.requires_evidence):
                raise RuntimeError(
                    "terminal response is missing required evidence records"
                )
            if node.outcome == ResponseOutcome.COMPLETED and any(
                evidence.source != EvidenceSource.EXTERNAL
                or evidence.status != EvidenceStatus.SUCCESS
                for evidence in evidence_records
            ):
                raise RuntimeError(
                    "completed response requires successful external evidence"
                )
            if node.outcome == ResponseOutcome.ANSWERED and any(
                evidence.status == EvidenceStatus.FAILURE
                for evidence in evidence_records
            ):
                raise RuntimeError("answered response cannot cite failed evidence")
            if node.outcome == ResponseOutcome.DECLINED and not any(
                evidence.source == EvidenceSource.INPUT
                and evidence.status == EvidenceStatus.FAILURE
                and evidence.producer_kind == "confirm"
                and evidence.value is False
                for evidence in evidence_records
            ):
                raise RuntimeError(
                    "declined response requires negative-confirmation evidence"
                )
            if (
                node.outcome == ResponseOutcome.FAIL_SAFE
                and evidence_records
                and not any(
                    evidence.status == EvidenceStatus.FAILURE
                    for evidence in evidence_records
                )
            ):
                raise RuntimeError("fail-safe response cites no failure evidence")
            self._node_status[node.id] = NodeExecutionStatus.SATISFIED
            return RespondAction(
                node_id=node.id,
                plan_id=self.plan.plan_id,
                text=node.text,
                outcome=node.outcome,
                evidence_refs=evidence_refs,
            )

        call_id = self.call_id_factory()
        if not call_id:
            raise ValueError("call_id_factory must return a non-empty call ID")
        if call_id in self._issued_call_ids:
            raise ValueError(f"call_id_factory reused call ID {call_id!r}")
        self._issued_call_ids.add(call_id)
        if isinstance(node, ObserveNode):
            arguments = _json_copy(node.arguments)
            _guard_external_emission(
                "observe", node.operation, arguments, emission_guard
            )
            argument_digest = _json_digest(arguments)
            action: RuntimeAction = ExternalCallAction(
                kind="observe",
                node_id=node.id,
                call_id=call_id,
                plan_id=self.plan.plan_id,
                operation=node.operation,
                arguments=arguments,
                argument_digest=argument_digest,
                capability_digest=self.capability_digest,
            )
            operation = node.operation
        elif isinstance(node, ActNode):
            arguments = _json_copy(node.arguments)
            _guard_external_emission("act", node.operation, arguments, emission_guard)
            argument_digest = _json_digest(arguments)
            action = ExternalCallAction(
                kind="act",
                node_id=node.id,
                call_id=call_id,
                plan_id=self.plan.plan_id,
                operation=node.operation,
                arguments=arguments,
                argument_digest=argument_digest,
                capability_digest=self.capability_digest,
            )
            operation = node.operation
        elif isinstance(node, AskNode):
            action = AskAction(
                node_id=node.id,
                call_id=call_id,
                plan_id=self.plan.plan_id,
                prompt=node.prompt,
            )
            operation = None
            argument_digest = None
        else:
            if not isinstance(node, ConfirmNode):
                raise TypeError(f"unsupported plan node type: {type(node).__name__}")
            action = ConfirmAction(
                node_id=node.id,
                call_id=call_id,
                plan_id=self.plan.plan_id,
                prompt=node.prompt,
            )
            operation = None
            argument_digest = None

        self._node_status[node.id] = NodeExecutionStatus.WAITING
        self._pending = PendingAction(
            node_id=node.id,
            call_id=call_id,
            kind=node.kind,
            operation=operation,
            argument_digest=argument_digest,
            capability_digest=self.capability_digest,
        )
        return action

    def _replace_event_disposition(
        self, event: RuntimeEvent, disposition: EventDisposition
    ) -> None:
        """Update the just-created audit record without exposing mutable models."""

        self.ledger.update_event_disposition(event.event_id, disposition)


def _event_matches_pending(
    event: RuntimeEvent,
    node: PlanNode,
    pending: PendingAction,
    plan_id: str,
    context_id: str | None,
) -> bool:
    if isinstance(event, ExternalResultEvent):
        if not isinstance(node, (ObserveNode, ActNode)):
            return False
        if not event.external_call_id:
            return False
        required_matches = (
            event.node_id == pending.node_id,
            event.operation == pending.operation,
            event.argument_digest == pending.argument_digest,
            event.plan_id == plan_id,
        )
        if not all(required_matches):
            return False
        if (
            pending.capability_digest is not None
            and event.schema_digest != pending.capability_digest
        ):
            return False
        return context_id is None or event.context_id == context_id
    if isinstance(event, InputEvent):
        return isinstance(node, AskNode)
    return isinstance(node, ConfirmNode)


def _evidence_key(node: PlanNode) -> str | None:
    if isinstance(node, (ObserveNode, ActNode)):
        return node.success_evidence_key
    if isinstance(node, (AskNode, ConfirmNode)):
        return node.evidence_key
    return None


def _json_digest(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _json_copy(value: dict[str, Any]) -> dict[str, Any]:
    serialized = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    parsed = json.loads(serialized)
    if not isinstance(parsed, dict):  # pragma: no cover - input is typed as object
        raise TypeError("operation arguments must be a JSON object")
    return parsed


def _guard_external_emission(
    kind: Literal["observe", "act"],
    operation: str,
    arguments: dict[str, Any],
    guard: EmissionGuard | None,
) -> None:
    if guard is None:
        raise EmissionRejectedError(
            "external emission requires a live capability guard"
        )
    try:
        guard(kind, operation, arguments)
    except Exception as exc:
        raise EmissionRejectedError(
            "live capability validation rejected the external operation"
        ) from exc
