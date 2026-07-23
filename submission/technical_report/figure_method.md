# PACT V2 method figure brief

PACT is a domain-agnostic, contract-guided tool agent. The model is an
untrusted semantic compiler, while a small deterministic kernel is the trusted
execution boundary.

Inputs arrive over A2A as three visually separated objects: trusted policy,
the latest user/tool event, and the current live function contracts (tool name,
description, and Draft 2020-12 argument schema). A Cerebras-hosted semantic
compiler produces a compact typed obligation-plan DAG. It may make one initial
structured-output call and at most one bounded repair call. An action-bearing
candidate then receives one same-model audit and full deterministic
reverification. Thus the longest semantic path is three serial model calls
with fan-out one. Each provider request may be retried at most three times
within 600 seconds of scheduled retry wait. Track 2 accounting distinguishes
the maximum three completed model calls from at most twelve HTTP attempts per
horizon; a rate-limited attempt returns no completion or tokens. Exported quota
wait excludes waits from an invocation that exhausts its retry budget and
raises. The graph has five node kinds: Observe, Ask, Confirm, Act, and Respond.
Confirm nodes explicitly name their downstream Act targets.

The candidate crosses into a shaded Trusted PACT Kernel. First, the static
PlanVerifier checks DAG structure and bounds, live capability membership, the
complete JSON Schema, typed terminal outcome, evidence-key freshness,
outstanding effect obligations, and confirmation scope. Effect/criticality
checks are conditional on trusted machine-readable metadata; unknown effects
are not inferred from operation names. A rejected graph produces one
value-redacted repair report, then fails safely if still invalid.

An accepted graph enters the deterministic ObligationRuntime. The runtime
emits exactly one ready benchmark-visible action. Immediately before any
external call, it rebuilds the live capability snapshot, compares the snapshot
and immutable plan digests, and revalidates the exact arguments. Ask waits for
a correlated A2A user event; Observe and Act wait for one strictly correlated
evaluator result. A qualified positive Confirm authorizes only its named Act.

An append-only EvidenceLedger records success, failure, or observed input with
event ID, internal and evaluator call IDs, context, plan, node, operation,
argument digest, capability digest, and producer kind. Positive confirmation
is a one-shot authorization scoped to the exact operation, argument digest,
and capability snapshot. Failed, malformed, stale, replayed, or mismatched
events never satisfy success obligations. Observe and free-form Ask responses
are semantic data events that trigger receding-horizon recompilation. An exact
positive Confirm and a matched successful Act are control events that advance
the verified suffix without another model call. Successful Acts remain durable
completion obligations. A Completed terminal is emitted only with successful
external Act evidence. Partial failure is reported without erasing already
completed effects.

The right side shows either one A2A tool call or user-facing text. Final text
responses carry accumulated turn_metrics: prompt_tokens, completion_tokens,
thinking_tokens, number of calls/passes, cost, latency, and quota wait. A dashed
feedback arrow carries evaluator results back to the ledger and next horizon.

Visual style: clean IJCAI paper architecture diagram, horizontal 16:9 layout,
white background, Times/serif-like labels, navy for inputs/compiler, teal for
the trusted kernel, amber for verification gates, green for evidence, red only
for rejection/fail-safe. Clearly separate the untrusted model region from the
trusted deterministic kernel. Avoid automotive icons, task examples, and
benchmark-specific tool names.
