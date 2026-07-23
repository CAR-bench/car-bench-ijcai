"""Typed, domain-agnostic intermediate representation for obligation plans."""

from __future__ import annotations

from collections import deque
from enum import StrEnum
from typing import Annotated, Any, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator


NonBlankStr: TypeAlias = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1),
]


class ResponseOutcome(StrEnum):
    """Semantic class of the plan's unique terminal response."""

    COMPLETED = "completed"
    ANSWERED = "answered"
    REFUSED = "refused"
    DECLINED = "declined"
    FAIL_SAFE = "fail_safe"


class _NodeBase(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: NonBlankStr
    depends_on: tuple[NonBlankStr, ...] = ()


class ObserveNode(_NodeBase):
    """Acquire a fact through a read-only external operation."""

    kind: Literal["observe"] = "observe"
    operation: NonBlankStr
    arguments: dict[str, Any] = Field(default_factory=dict)
    success_evidence_key: NonBlankStr


class AskNode(_NodeBase):
    """Ask for a missing value and record the correlated input."""

    kind: Literal["ask"] = "ask"
    prompt: NonBlankStr
    evidence_key: NonBlankStr


class ConfirmNode(_NodeBase):
    """Request an explicit boolean confirmation."""

    kind: Literal["confirm"] = "confirm"
    prompt: NonBlankStr
    evidence_key: NonBlankStr
    authorizes: tuple[NonBlankStr, ...] = Field(min_length=1)


class ActNode(_NodeBase):
    """Execute a state-changing external operation."""

    kind: Literal["act"] = "act"
    operation: NonBlankStr
    arguments: dict[str, Any] = Field(default_factory=dict)
    success_evidence_key: NonBlankStr


class RespondNode(_NodeBase):
    """Emit the plan's unique typed terminal response."""

    kind: Literal["respond"] = "respond"
    text: NonBlankStr
    outcome: ResponseOutcome
    requires_evidence: tuple[NonBlankStr, ...] = ()


PlanNode: TypeAlias = Annotated[
    ObserveNode | AskNode | ConfirmNode | ActNode | RespondNode,
    Field(discriminator="kind"),
]


class PlanIR(BaseModel):
    """Validated DAG ending in one typed response.

    ``evidence_inputs`` declares immutable records carried into this horizon.
    The IR validates references, but only :class:`PlanVerifier` may decide
    whether those records actually exist and have suitable provenance.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    plan_id: NonBlankStr
    evidence_inputs: tuple[NonBlankStr, ...] = ()
    nodes: tuple[PlanNode, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_plan(self) -> "PlanIR":
        if len(set(self.evidence_inputs)) != len(self.evidence_inputs):
            raise ValueError("evidence inputs must be unique")

        node_by_id = {node.id: node for node in self.nodes}
        if len(node_by_id) != len(self.nodes):
            raise ValueError("plan node IDs must be unique")

        for node in self.nodes:
            if len(set(node.depends_on)) != len(node.depends_on):
                raise ValueError(f"node {node.id!r} has duplicate dependencies")
            missing = set(node.depends_on) - set(node_by_id)
            if missing:
                raise ValueError(
                    f"node {node.id!r} depends on unknown nodes: {sorted(missing)}"
                )
            if node.id in node.depends_on:
                raise ValueError(f"node {node.id!r} cannot depend on itself")

        order = _topological_order(self.nodes)
        if len(order) != len(self.nodes):
            raise ValueError("plan dependencies must form an acyclic graph")

        for node in self.nodes:
            if not isinstance(node, ConfirmNode):
                continue
            if len(set(node.authorizes)) != len(node.authorizes):
                raise ValueError(
                    f"confirmation {node.id!r} has duplicate authorization targets"
                )
            for target_id in node.authorizes:
                target = node_by_id.get(target_id)
                if not isinstance(target, ActNode):
                    raise ValueError(
                        f"confirmation {node.id!r} must authorize an action node"
                    )
                if node.id not in _ancestors(target_id, node_by_id):
                    raise ValueError(
                        f"authorized action {target_id!r} must depend on confirmation "
                        f"{node.id!r}"
                    )

        responses = [node for node in self.nodes if isinstance(node, RespondNode)]
        if len(responses) != 1:
            raise ValueError("plan must contain exactly one response")
        terminal = responses[0]

        dependents = _dependents(self.nodes)
        if dependents[terminal.id]:
            raise ValueError("the response must be terminal")

        ancestors = _ancestors(terminal.id, node_by_id)
        unreachable = set(node_by_id) - ancestors - {terminal.id}
        if unreachable:
            raise ValueError(
                "every plan node must lead to the terminal response; "
                f"unreachable nodes: {sorted(unreachable)}"
            )

        if len(set(terminal.requires_evidence)) != len(terminal.requires_evidence):
            raise ValueError("terminal evidence requirements must be unique")

        producer_by_key: dict[str, str] = {}
        for node in self.nodes:
            evidence_key = _produced_evidence_key(node)
            if evidence_key is None:
                continue
            if evidence_key in producer_by_key:
                raise ValueError(
                    f"evidence key {evidence_key!r} has multiple producers"
                )
            producer_by_key[evidence_key] = node.id

        overlap = set(self.evidence_inputs) & set(producer_by_key)
        if overlap:
            raise ValueError(
                "evidence inputs cannot be reproduced in the same plan: "
                f"{sorted(overlap)}"
            )

        known_evidence = set(producer_by_key) | set(self.evidence_inputs)
        missing_evidence = set(terminal.requires_evidence) - known_evidence
        if missing_evidence:
            raise ValueError(
                "terminal response requires evidence with no producer or input: "
                f"{sorted(missing_evidence)}"
            )

        for key in terminal.requires_evidence:
            producer_id = producer_by_key.get(key)
            if producer_id is not None and producer_id not in ancestors:
                raise ValueError(
                    f"evidence producer {producer_id!r} is not an ancestor of response"
                )

        self._validate_outcome(terminal)
        return self

    def _validate_outcome(self, terminal: RespondNode) -> None:
        acts = tuple(node for node in self.nodes if isinstance(node, ActNode))
        observations = tuple(
            node for node in self.nodes if isinstance(node, ObserveNode)
        )

        if terminal.outcome == ResponseOutcome.COMPLETED:
            act_evidence = {node.success_evidence_key for node in acts}
            if not acts and not (
                set(terminal.requires_evidence) & set(self.evidence_inputs)
            ):
                raise ValueError(
                    "completed outcome requires an action or carried evidence input"
                )
            missing_actions = act_evidence - set(terminal.requires_evidence)
            if missing_actions:
                raise ValueError(
                    "completed outcome must cite every action success key: "
                    f"{sorted(missing_actions)}"
                )
            if not terminal.requires_evidence:
                raise ValueError("completed outcome must require evidence")
            return

        if terminal.outcome == ResponseOutcome.ANSWERED:
            if acts:
                raise ValueError("answered outcome cannot contain actions")
            missing_observations = {
                node.success_evidence_key for node in observations
            } - set(terminal.requires_evidence)
            if missing_observations:
                raise ValueError(
                    "answered outcome must cite every observation success key: "
                    f"{sorted(missing_observations)}"
                )

        if terminal.outcome == ResponseOutcome.REFUSED and (acts or observations):
            raise ValueError("refused outcome cannot contain external calls")

        if (
            terminal.outcome
            in {
                ResponseOutcome.DECLINED,
                ResponseOutcome.FAIL_SAFE,
            }
            and acts
        ):
            raise ValueError(f"{terminal.outcome.value} outcome cannot contain actions")

    @property
    def node_by_id(self) -> dict[str, PlanNode]:
        return {node.id: node for node in self.nodes}

    @property
    def terminal(self) -> RespondNode:
        """Return the response guaranteed unique by validation."""

        return next(node for node in self.nodes if isinstance(node, RespondNode))

    def topological_order(self) -> tuple[str, ...]:
        return _topological_order(self.nodes)


def _produced_evidence_key(node: PlanNode) -> str | None:
    if isinstance(node, (ObserveNode, ActNode)):
        return node.success_evidence_key
    if isinstance(node, (AskNode, ConfirmNode)):
        return node.evidence_key
    return None


def _dependents(nodes: tuple[PlanNode, ...]) -> dict[str, set[str]]:
    result: dict[str, set[str]] = {node.id: set() for node in nodes}
    for node in nodes:
        for dependency in node.depends_on:
            result[dependency].add(node.id)
    return result


def _topological_order(nodes: tuple[PlanNode, ...]) -> tuple[str, ...]:
    """Stable Kahn ordering, preserving declaration order among ready nodes."""

    by_id = {node.id: node for node in nodes}
    position = {node.id: index for index, node in enumerate(nodes)}
    indegree = {node.id: len(node.depends_on) for node in nodes}
    dependents = _dependents(nodes)
    ready = deque(node.id for node in nodes if indegree[node.id] == 0)
    ordered: list[str] = []

    while ready:
        node_id = ready.popleft()
        ordered.append(node_id)
        newly_ready: list[str] = []
        for dependent in dependents[node_id]:
            indegree[dependent] -= 1
            if indegree[dependent] == 0:
                newly_ready.append(dependent)
        for dependent in sorted(newly_ready, key=position.__getitem__):
            ready.append(dependent)

    return tuple(node_id for node_id in ordered if node_id in by_id)


def _ancestors(node_id: str, node_by_id: dict[str, PlanNode]) -> set[str]:
    result: set[str] = set()
    frontier = list(node_by_id[node_id].depends_on)
    while frontier:
        ancestor = frontier.pop()
        if ancestor in result:
            continue
        result.add(ancestor)
        frontier.extend(node_by_id[ancestor].depends_on)
    return result


__all__ = [
    "ActNode",
    "AskNode",
    "ConfirmNode",
    "ObserveNode",
    "PlanIR",
    "PlanNode",
    "RespondNode",
    "ResponseOutcome",
]
