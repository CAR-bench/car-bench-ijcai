"""Provenance-preserving evidence storage for obligation execution.

The ledger is deliberately domain agnostic.  It records immutable facts and the
events that produced them, but it does not decide which facts a plan requires.
"""

from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field


class ExternalResultStatus(StrEnum):
    """Outcome reported by an external operation."""

    SUCCESS = "success"
    FAILURE = "failure"


class EvidenceSource(StrEnum):
    """Origin of an evidence record."""

    EXTERNAL = "external"
    INPUT = "input"


class EvidenceStatus(StrEnum):
    """Whether an evidence record proves success or records a failed outcome."""

    SUCCESS = "success"
    FAILURE = "failure"
    OBSERVED = "observed"


class ActionAuthorization(BaseModel):
    """One exact external action covered by a user confirmation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    operation: str = Field(min_length=1)
    argument_digest: str = Field(pattern=r"^[0-9a-f]{64}$")


class EventDisposition(StrEnum):
    """How a runtime handled an incoming event."""

    APPLIED = "applied"
    DUPLICATE = "duplicate"
    OUT_OF_ORDER = "out_of_order"
    FAILURE = "failure"
    CONFLICT = "conflict"


class ExternalResultEvent(BaseModel):
    """A result correlated with a previously emitted external call."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    event_id: str = Field(min_length=1)
    call_id: str = Field(min_length=1)
    external_call_id: str = Field(min_length=1)
    context_id: str = Field(min_length=1)
    plan_id: str = Field(min_length=1)
    node_id: str = Field(min_length=1)
    operation: str = Field(min_length=1)
    argument_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    schema_digest: str = Field(min_length=1)
    status: ExternalResultStatus
    payload: Any = None
    error: Any = None


class InputEvent(BaseModel):
    """A value supplied in response to an ask action."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    event_id: str = Field(min_length=1)
    call_id: str = Field(min_length=1)
    value: Any


class ConfirmationEvent(BaseModel):
    """A boolean decision supplied in response to a confirmation action."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    event_id: str = Field(min_length=1)
    call_id: str = Field(min_length=1)
    confirmed: bool


RuntimeEvent: TypeAlias = ExternalResultEvent | InputEvent | ConfirmationEvent


class EvidenceRecord(BaseModel):
    """An immutable fact together with its complete production provenance."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    key: str = Field(min_length=1)
    value: Any
    source: EvidenceSource
    status: EvidenceStatus = EvidenceStatus.SUCCESS
    event_id: str = Field(min_length=1)
    call_id: str = Field(min_length=1)
    node_id: str = Field(min_length=1)
    producer_kind: Literal["observe", "ask", "confirm", "act"] | None = None
    operation: str | None = None
    external_call_id: str | None = None
    context_id: str | None = None
    plan_id: str | None = None
    argument_digest: str | None = None
    schema_digest: str | None = None
    authorizations: tuple[ActionAuthorization, ...] = ()


class ProcessedEvent(BaseModel):
    """Audit record used to make event handling idempotent."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    event_id: str = Field(min_length=1)
    call_id: str = Field(min_length=1)
    fingerprint: str = Field(min_length=1)
    disposition: EventDisposition
    node_id: str | None = None


@dataclass(frozen=True)
class EventRegistration:
    """Result of registering an event in the idempotency index."""

    is_new: bool
    record: ProcessedEvent


@dataclass
class EvidenceLedger:
    """Append-only evidence and event-disposition ledger.

    Evidence keys have a single producer within one plan run.  Re-applying the
    same record is harmless, while attempting to overwrite a key with different
    provenance is rejected.
    """

    _evidence: dict[str, EvidenceRecord] = field(default_factory=dict)
    _events: dict[str, ProcessedEvent] = field(default_factory=dict)

    def register_event(
        self,
        event: RuntimeEvent,
        *,
        disposition: EventDisposition,
        node_id: str | None = None,
    ) -> EventRegistration:
        """Register an event exactly once and return its audit disposition.

        An identical event ID and payload is reported as ``DUPLICATE`` on later
        delivery.  Reusing an event ID for different content is reported as a
        ``CONFLICT`` and never changes previously recorded evidence.
        """

        fingerprint = _event_fingerprint(event)
        existing = self._events.get(event.event_id)
        if existing is not None:
            repeated_disposition = (
                EventDisposition.DUPLICATE
                if existing.fingerprint == fingerprint
                else EventDisposition.CONFLICT
            )
            return EventRegistration(
                is_new=False,
                record=ProcessedEvent(
                    event_id=event.event_id,
                    call_id=event.call_id,
                    fingerprint=fingerprint,
                    disposition=repeated_disposition,
                    node_id=existing.node_id,
                ),
            )

        record = ProcessedEvent(
            event_id=event.event_id,
            call_id=event.call_id,
            fingerprint=fingerprint,
            disposition=disposition,
            node_id=node_id,
        )
        self._events[event.event_id] = record
        return EventRegistration(is_new=True, record=record)

    def add(self, record: EvidenceRecord) -> None:
        """Add one successful evidence record without allowing replacement.

        Pydantic's ``frozen`` setting is intentionally not treated as deep
        immutability: JSON objects nested inside a model remain mutable.  The
        ledger therefore stores an isolated JSON round-trip and returns fresh
        copies from its read APIs.  Mutating either the event payload supplied
        by an adapter or a value returned by :meth:`get` cannot rewrite history.
        """

        isolated = _isolated_model(record)
        existing = self._evidence.get(isolated.key)
        if existing is None:
            self._evidence[isolated.key] = isolated
            return
        if existing != isolated:
            raise ValueError(
                f"evidence key {isolated.key!r} already has different provenance"
            )

    def get(self, key: str) -> EvidenceRecord | None:
        record = self._evidence.get(key)
        return _isolated_model(record) if record is not None else None

    def has(self, key: str) -> bool:
        return key in self._evidence

    def event(self, event_id: str) -> ProcessedEvent | None:
        return self._events.get(event_id)

    def update_event_disposition(
        self, event_id: str, disposition: EventDisposition
    ) -> ProcessedEvent:
        """Replace only the disposition of an already registered audit event."""

        existing = self._events.get(event_id)
        if existing is None:
            raise KeyError(f"event {event_id!r} has not been registered")
        updated = existing.model_copy(update={"disposition": disposition})
        self._events[event_id] = updated
        return updated

    @property
    def evidence(self) -> tuple[EvidenceRecord, ...]:
        return tuple(_isolated_model(record) for record in self._evidence.values())

    @property
    def processed_events(self) -> tuple[ProcessedEvent, ...]:
        return tuple(self._events.values())


def _event_fingerprint(event: RuntimeEvent) -> str:
    """Return a deterministic representation suitable for replay detection."""

    payload = json.dumps(
        event.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _isolated_model(record: EvidenceRecord) -> EvidenceRecord:
    """Return a JSON-isolated copy of an evidence record."""

    return EvidenceRecord.model_validate_json(record.model_dump_json())
