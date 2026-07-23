from carbench_agent_core.evidence_ledger import (
    EventDisposition,
    EvidenceLedger,
    EvidenceRecord,
    EvidenceSource,
    EvidenceStatus,
    ExternalResultEvent,
    ExternalResultStatus,
)


def external_event(
    *,
    event_id: str = "event-1",
    call_id: str = "call-1",
    payload: object = None,
) -> ExternalResultEvent:
    return ExternalResultEvent(
        event_id=event_id,
        call_id=call_id,
        external_call_id="evaluator-call-1",
        context_id="context-1",
        plan_id="plan-1",
        node_id="observe-value",
        operation="read.value",
        argument_digest="a" * 64,
        schema_digest="schema-1",
        status=ExternalResultStatus.SUCCESS,
        payload=payload,
    )


def test_ledger_preserves_event_and_call_provenance() -> None:
    ledger = EvidenceLedger()
    event = external_event(payload={"value": 7})

    registration = ledger.register_event(
        event,
        disposition=EventDisposition.APPLIED,
        node_id="observe-value",
    )
    ledger.add(
        EvidenceRecord(
            key="observed.value",
            value=event.payload,
            source=EvidenceSource.EXTERNAL,
            status=EvidenceStatus.SUCCESS,
            event_id=event.event_id,
            call_id=event.call_id,
            node_id="observe-value",
            producer_kind="observe",
            operation="read.value",
            external_call_id="evaluator-call-1",
            context_id="context-1",
            plan_id="plan-1",
            argument_digest="a" * 64,
            schema_digest="schema-1",
        )
    )

    assert registration.is_new
    record = ledger.get("observed.value")
    assert record is not None
    assert record.event_id == "event-1"
    assert record.call_id == "call-1"
    assert record.node_id == "observe-value"
    assert record.producer_kind == "observe"
    assert record.operation == "read.value"
    assert record.source == EvidenceSource.EXTERNAL
    assert record.status == EvidenceStatus.SUCCESS
    assert record.external_call_id == "evaluator-call-1"
    assert record.context_id == "context-1"
    assert record.plan_id == "plan-1"
    assert record.argument_digest == "a" * 64
    assert record.schema_digest == "schema-1"


def test_ledger_reports_identical_replay_as_duplicate() -> None:
    ledger = EvidenceLedger()
    event = external_event(payload={"value": 7})
    ledger.register_event(event, disposition=EventDisposition.APPLIED)

    replay = ledger.register_event(event, disposition=EventDisposition.APPLIED)

    assert not replay.is_new
    assert replay.record.disposition == EventDisposition.DUPLICATE
    assert len(ledger.processed_events) == 1


def test_ledger_rejects_event_id_reuse_with_different_content() -> None:
    ledger = EvidenceLedger()
    ledger.register_event(
        external_event(payload={"value": 7}),
        disposition=EventDisposition.APPLIED,
    )

    conflict = ledger.register_event(
        external_event(payload={"value": 8}),
        disposition=EventDisposition.APPLIED,
    )

    assert not conflict.is_new
    assert conflict.record.disposition == EventDisposition.CONFLICT
    assert len(ledger.processed_events) == 1


def test_ledger_does_not_overwrite_evidence_provenance() -> None:
    ledger = EvidenceLedger()
    first = EvidenceRecord(
        key="observed.value",
        value=7,
        source=EvidenceSource.EXTERNAL,
        event_id="event-1",
        call_id="call-1",
        node_id="observe-value",
    )
    ledger.add(first)
    ledger.add(first)

    different = first.model_copy(update={"event_id": "event-2"})
    try:
        ledger.add(different)
    except ValueError as exc:
        assert "different provenance" in str(exc)
    else:
        raise AssertionError("overwriting evidence should fail")


def test_ledger_deeply_isolates_input_payload_and_returned_values() -> None:
    ledger = EvidenceLedger()
    payload = {
        "outer": {
            "items": [
                {"name": "original", "metadata": {"verified": True}},
            ]
        }
    }
    ledger.add(
        EvidenceRecord(
            key="observed.deep",
            value=payload,
            source=EvidenceSource.EXTERNAL,
            status=EvidenceStatus.SUCCESS,
            event_id="event-deep",
            call_id="call-deep",
            node_id="observe-deep",
            producer_kind="observe",
            operation="read.deep",
        )
    )

    payload["outer"]["items"][0]["name"] = "mutated-before-read"
    first_read = ledger.get("observed.deep")
    assert first_read is not None
    assert first_read.value["outer"]["items"][0] == {
        "name": "original",
        "metadata": {"verified": True},
    }

    first_read.value["outer"]["items"][0]["name"] = "mutated-return-value"
    first_read.value["outer"]["items"][0]["metadata"]["verified"] = False
    second_read = ledger.get("observed.deep")
    assert second_read is not None
    assert second_read.value["outer"]["items"][0] == {
        "name": "original",
        "metadata": {"verified": True},
    }

    evidence_view = ledger.evidence
    evidence_view[0].value["outer"]["items"].append({"name": "injected"})
    final_read = ledger.get("observed.deep")
    assert final_read is not None
    assert final_read.value == {
        "outer": {
            "items": [
                {"name": "original", "metadata": {"verified": True}},
            ]
        }
    }
