# PACT: contract-verified receding-horizon obligations

Status: final Track 2 submission architecture. The production server imports
`PACTAgentExecutor`; the release Dockerfile copies only the PACT runtime
dependency closure. "V2" is only the historical design-iteration label in this
file name: there is one submitted PACT agent and no V1 compatibility path in
the release image.

## 1. Design objective

PACT separates semantic interpretation from execution authority:

- A Cerebras-hosted LLM is an **untrusted semantic compiler**. It proposes a
  small typed obligation plan but cannot execute a tool or certify success.
- A deterministic local boundary performs strict decoding, PlanIR validation,
  complete live JSON Schema validation, evidence checks, and bounded execution.
- The A2A adapter exposes at most one outstanding external operation and lets
  only a correlated evaluator-returned success create external evidence. Ask
  and Confirm replies are separately typed as input evidence.

The implementation is domain-independent. The reachable PACT kernel contains
no benchmark task IDs, memorized public utterances, operation-name allowlists,
or domain workflow branches. Task semantics enter only through the trusted
policy envelope, current user event, live capability contracts, and returned
evidence.

PACT does **not** prove that an LLM understood a natural-language policy or tool
description. Its unconditional checks begin at strict wire decoding and cover
typed structure, live capability contracts, local state transitions, and
evidence provenance. Effect and confirmation claims are conditional on trusted
capability metadata or the narrowly parsed contract marker when either exists.

## 2. Reachable production components

| Component | File | Responsibility |
|---|---|---|
| A2A entry point | `track_2_agent_under_test_cerebras/server.py` | Loads one environment snapshot and installs `PACTAgentExecutor` |
| Protocol adapter | `track_2_agent_under_test_cerebras/pact_agent.py` | Context isolation, A2A parsing/rendering, single-result correlation, receding horizons, metrics |
| Cerebras backend | `track_2_agent_under_test_cerebras/plan_compiler_backend.py` | Environment-configured structured completion and trusted local limits |
| Provider client | `track_2_agent_under_test_cerebras/cerebras_client.py` | Cerebras SDK call, proactive pacing, bounded 429 retry/wait, usage extraction |
| Semantic compiler | `carbench_agent_core/semantic_compiler.py` | Strict compact wire schema, local decode, one bounded repair, independent action audit |
| Typed IR | `carbench_agent_core/plan_ir.py` | Frozen obligation DAG and typed terminal outcomes |
| Static verifier | `carbench_agent_core/plan_verifier.py` | Live capability snapshot, resource/safety/evidence checks |
| Runtime | `carbench_agent_core/obligation_runtime.py` | Exactly-one pending action and deterministic node transitions |
| Evidence ledger | `carbench_agent_core/evidence_ledger.py` | Append-only, replay-aware, provenance-preserving evidence |
| Schema guard | `carbench_agent_core/tool_index.py` | Draft 2020-12 argument validation and structured issues |

The source checkout retains older controller and planner/executor code only as
archived comparison material. The final Dockerfile neither copies nor imports
those modules, and the submission workflow has no alternate legacy target.

## 3. Trust and threat model

### Trusted inputs

The adapter treats the first transport envelope of the form
`System: ...\n\nUser: ...` as the policy trust root for a context. A later user
message cannot replace an already established policy. The current A2A tool list
and tool-result envelope are evaluator inputs; `tools = []` is an explicit
capability revocation rather than permission to reuse a cached list.

### Untrusted inputs

The following remain untrusted until locally checked:

- model-generated wire JSON and every operation argument;
- user text, including text that resembles a system envelope after context
  initialization;
- free-form tool-description semantics, except the explicitly anchored trusted
  `REQUIRES_CONFIRMATION` contract marker described in Section 7;
- malformed, duplicate, stale, or out-of-order result envelopes; and
- nested mutable values supplied to Pydantic models.

### Trusted computing boundary

The trusted boundary includes strict JSON decoding, Pydantic PlanIR validation,
the Draft 2020-12 validator, capability/evidence verification, runtime state,
the ledger, the A2A adapter's single-pending discipline, and response metrics.

It assumes the evaluator obeys the A2A turn contract: a result delivered for
the sole outstanding request actually corresponds to that request. The current
outbound tool-call data does not carry an agent-selected nonce, so this
assumption cannot be cryptographically established by the agent.

## 4. Formal state model

For A2A context `q` at turn `t`, define:

```text
S[q,t] = (P, g, H, T, E, G, p, O, C, M)
```

- `P`: established policy text;
- `g`: current user goal;
- `H`: bounded recent conversation;
- `T`: current live tool declarations;
- `E`: immutable evidence ledger;
- `G`: accepted PlanIR for the current horizon, if any;
- `p`: at most one pending runtime action;
- `O`: successful action-evidence keys that a later completion must cite;
- `C`: consumed confirmation scopes; and
- `M`: inference usage accumulated until the next text response.

The semantic compiler proposes a plan:

```text
C_theta(P, g, H, T, view(E), O) -> WirePlan
```

The local acceptance pipeline is a partial function:

```text
proposal
  -> decode_strict(WirePlan)
  -> compact wire model
  -> frozen PlanIR
  -> verify(PlanIR, snapshot(T), E, O)
  -> accepted proposal
     | value-redacted rejection -> [one repair -> decode -> verify]
  -> [if accepted plan has Act: independent audit -> decode -> verify]
  -> AcceptedPlan | typed rejection
```

The deterministic kernel advances one node:

```text
K(AcceptedPlan, E, fresh_snapshot(T))
  -> one A2A action | WAIT | terminal response | fail closed
```

## 5. Untrusted semantic compiler

### 5.1 Prompt partitioning

The compiler receives three messages:

1. a fixed domain-independent compilation contract;
2. the trusted policy serialized in its own system partition; and
3. untrusted goal, current event, bounded conversation, live capabilities,
   typed evidence, and outstanding completion obligations.

The compiler is explicitly instructed not to execute operations or claim that
an external operation succeeded. Existing evidence includes source, status,
event/call/node provenance, operation and digest fields, and exact
authorizations rather than a prose-only summary.

For an accepted proposal containing an Act, a separate review prompt asks the
model to reconstruct the complete plan after independently auditing the entire
trusted policy, action prerequisites, argument provenance, ordering, and
capability availability. This is a stochastic semantic cross-check, not a
formal proof. Its returned plan crosses the same untrusted boundary as the
proposal.

### 5.2 Cerebras-compatible compact wire schema

Cerebras strict structured output requires every object to set
`additionalProperties: false` and limits the response schema to 5,000
characters. Tool argument objects are dynamic and cannot safely be represented
as free-form strict-schema objects. PACT therefore uses a uniform wire node:

```json
{
  "id": "n1",
  "kind": "observe",
  "depends_on": [],
  "operation": "live_operation_name",
  "arguments_json": "{\"key\":\"value\"}",
  "success_evidence_key": "observation.1"
}
```

The provider schema:

- has a discriminated variant for each node kind;
- forbids extra fields on every object;
- constrains terminal outcomes;
- constrains operation names by enum while the schema remains at or below 5,000
  characters; and
- falls back to a string operation only if a large live inventory would exceed
  that provider limit, while the local verifier still enforces exact live
  membership.

The local decoder rejects duplicate JSON keys, non-finite numbers, a non-object
top level, wrong variant fields, non-object `arguments_json`, and any candidate
over the configured byte/character bound. The decoded argument object then
passes through the full live Draft 2020-12 schema.

### 5.3 Bounded proposal, repair, and action audit

The initial candidate is decoded and verified locally. On rejection, the
compiler may receive one repair request containing:

- the rejected candidate already present in conversation context;
- value-redacted issue code and structural path; and
- a request for one complete corrected JSON object.

`max_repair_attempts` is restricted to `0` or `1`. A second invalid proposal
raises `PlanRepairExhaustedError`; it is never executed. Attempt records retain
hashes and metrics, not raw model output.

With the submitted `PACT_COMPILER_SEMANTIC_REVIEW=true`, an accepted plan that
contains at least one Act receives exactly one additional action-audit call.
The audit sees the fixed compilation contract, trusted policy, original input
payload, and a canonical rendering of the accepted plan; it must return one
complete replacement plan. That replacement is strictly decoded and fully
verified again. An invalid, truncated, or unavailable audit fails closed: PACT
does not fall back to executing the pre-audit candidate.

Consequently one compilation has at most three successful model completions:
proposal, optional verification-guided repair, and action audit. Plans without
an Act skip the audit. The repair and audit are different controls: repair
addresses a locally identified structural/contract rejection, whereas audit
re-examines semantic coverage from scratch.

Provider termination is classified before repair or audit acceptance. A
`finish_reason=length` response is a typed `provider_output_truncated` failure:
its usage is retained, but a proposal-side repair call is skipped because the
provider supplied no complete candidate to correct. Empty content is likewise
a typed invalid candidate and can never enter `PlanIR` or the runtime.

## 6. Typed PlanIR

`PlanIR` is a frozen, non-executable DAG. It allows only these node variants:

- `observe(operation, arguments, success_evidence_key)`;
- `ask(prompt, evidence_key)`;
- `confirm(prompt, evidence_key, authorizes)`;
- `act(operation, arguments, success_evidence_key)`; and
- `respond(text, outcome, requires_evidence)`.

All variants forbid extra fields. Arguments are JSON constants; the IR has no
callbacks, template evaluator, arbitrary expressions, imports, regular
expressions generated as code, or evidence-binding language. Dynamic values
are resolved at semantic data boundaries by compiling a fresh horizon after
relevant evidence arrives; the runtime never evaluates model-generated
expressions.

### Structural conditions

PlanIR construction rejects:

- blank or duplicate node IDs;
- unknown, repeated, or self dependencies;
- dependency cycles;
- multiple or missing response nodes;
- a nonterminal response;
- a node that does not lead to the terminal response;
- duplicate evidence producers or reuse of an input key in the same horizon;
- response evidence with no producer/input; and
- confirmation targets that are not dependent action nodes.

The verifier additionally enforces configured node and dependency-depth bounds.
The production defaults are 20 nodes and depth 12.

During wire lowering, an explicit model-authored `Confirm.authorizes` relation
also induces the missing Confirm-to-Act sequencing edge, if needed. This only
strengthens ordering for a target the model already declared; it never invents
an operation, argument, confirmation target, or permission. Unknown targets,
non-Act targets, and cycles remain verifier errors.

### Typed terminal outcomes

| Outcome | Structural/evidence condition |
|---|---|
| `completed` | Has a current action or carried action evidence; its evidence list is locally replaced by the exact current-plus-carried action-success closure, and every cited record must be successful external Act evidence |
| `answered` | Contains no action and cites all observation results used in the horizon; cited evidence cannot be failed |
| `refused` | Contains no external operation |
| `declined` | Contains no action and requires an available false confirmation record |
| `fail_safe` | Contains no action; compiler contract requests failure evidence when available, but the emergency adapter path may respond without it |

The last row is intentional: the system must retain a truthful non-completion
escape path when strict decoding, provider access, or protocol parsing fails
before a typed failure record can exist.

## 7. Live capability verification

`CapabilitySnapshot.from_tools` canonicalizes each live function declaration
into:

```text
(operation, description, parameter_schema, schema_digest,
 requires_confirmation, effect)
```

It validates parameter schemas with `Draft202012Validator.check_schema`,
rejects malformed or duplicate declarations, and hashes the normalized
inventory. `ToolIndex` then validates arguments using a cached full Draft
2020-12 validator and `FormatChecker`. Validation includes nested objects,
arrays, type/enum/const, ranges, `multipleOf`, length/pattern/format,
combinators, and `additionalProperties` according to the supplied schema.

No operation is classified from its name. The trusted metadata surface is:

- `x-pact-effect = "observe" | "act" | "unknown"`; and
- `x-pact-requires-confirmation = true`; or
- an anchored `REQUIRES_CONFIRMATION` marker at the beginning of the trusted
  function description (allowing leading whitespace and a delimiter after the
  marker).

The description marker is a contract adapter for the evaluator-provided live
function declaration. A mention in the middle of a description, a longer
identifier sharing the prefix, user text, and model output do not activate it.
The original description plus the derived confirmation/effect fields are part
of the normalized capability digest, so a marker change invalidates an
accepted horizon.

A confirmation-required capability is forced to effect `act`. Conflicting or
invalid metadata fails closed. If effect is `unknown`, schema and membership
checks still hold, but PACT cannot guarantee whether the model's `observe` or
`act` label reflects the operation's real-world effect.

Every external emission is rechecked against a newly built snapshot. It must:

1. be a valid snapshot;
2. have the exact digest accepted at compilation;
3. still contain the operation;
4. agree with any known effect label; and
5. accept a deep JSON copy of the emitted arguments.

This closes the plan-verification-to-emission gap for capability revocation,
schema mutation, and caller-owned argument mutation.

## 8. Evidence ledger and runtime

### 8.1 Evidence records

An `EvidenceRecord` contains:

- unique evidence key and JSON-isolated value;
- source: `external` or `input`;
- status: `success`, `failure`, or `observed`;
- producer event, internal call, and node IDs;
- producer kind;
- optional operation and evaluator call ID;
- optional context, plan, argument digest, and schema digest; and
- zero or more exact `ActionAuthorization` scopes.

The ledger is append-only by evidence key. Re-adding an identical record is
idempotent; a different record for the same key is rejected. Because frozen
Pydantic objects are not recursively immutable, the ledger JSON-round-trips
records when storing and reading them. Mutating either an inbound event payload
or a returned value cannot rewrite stored evidence.

### 8.2 Event idempotency

Every runtime event is registered with a canonical SHA-256 fingerprint. A
repeated event ID with identical content is `duplicate`; reuse with different
content is `conflict`. Both leave prior evidence unchanged. An event for no
pending call or the wrong internal call is `out_of_order`.

### 8.3 One pending obligation

Node state is one of `pending`, `waiting`, `satisfied`, or `failed`. The runtime
walks the stable topological order and selects the first pending node whose
dependencies are satisfied. On Ask, Confirm, Observe, or Act it allocates a
unique internal call ID, marks that node waiting, and emits exactly one action.
While any node is waiting, `step()` returns no second action.

An external result can satisfy a pending Observe/Act only after matching:

```text
internal call ID
context ID
plan ID
node ID
operation
argument digest
capability/schema digest
non-empty evaluator call ID
explicit SUCCESS status
```

Explicit `FAILURE` creates `<success_key>.failure` evidence, marks the node
failed, and blocks the plan. Malformed or status-less result envelopes are
rejected by the adapter and cause safe termination rather than success.

## 9. Control events, data events, and receding horizons

PACT separates events that release an already verified transition from events
that introduce new semantic data:

- **Control events:** an exact positive confirmation and a successful Act
  result advance the same immutable accepted PlanIR suffix. They do not ask the
  stochastic compiler to reconstruct an authorization or an already verified
  next action.
- **Data events:** an Ask answer and a successful Observe result enter the
  ledger and compile a fresh horizon because later arguments or policy choices
  may depend on their content.
- **Contract events:** a live capability digest change detected while
  releasing a pending confirmation abandons the old horizon and recompiles.
  A mismatch detected immediately before external-call emission rejects the
  call and terminates safely. No action is released against a changed schema.
- **Scope revisions:** a qualified positive reply that changes an operation,
  argument, or value is a new semantic request. PACT discards the old horizon
  without creating authorization evidence and recompiles.

Negative confirmation is a control event that terminates as `declined` without
recompiling; if an earlier action in the same request already succeeded, the
adapter explicitly reports partial completion before acknowledging that the
remaining action will not proceed. Ambiguous confirmation text elicits an
explicit yes/no retry. This split is monotone: control events can only advance
or stop an accepted graph, while all new decision-bearing data crosses the
compiler and verifier boundary.

Every successful Act adds its success key to
`required_completion_evidence`. The compiler receives that set, and the
trusted wire-lowering step replaces a `completed` response's model-authored
evidence list with exactly:

```text
{success key of every Act in the current PlanIR}
union required_completion_evidence
```

The verifier then requires every member to resolve to successful external Act
evidence. Ask, Confirm, Observe, input-only, failed, and missing records cannot
enter this completion proof. If later work fails, the adapter reports partial
failure rather than forgetting that an effect already occurred. For an
`answered` response, lowering additionally includes all current observation
success keys, but those keys never prove `completed`.

## 10. Scoped one-shot confirmation

Within one plan, a Confirm node lists exact descendant Act node IDs. When the
user explicitly confirms, the ledger stores an authorization for each target:

```text
(schema_digest, operation, sha256(canonical_arguments))
```

A future horizon may use a prior confirmation only when all three fields match
and the plan declares that confirmation record as an evidence input. The
adapter removes the matching scope from the compiler-visible projection after
the authorized Act succeeds. A confirmation therefore cannot authorize a
different operation, different arguments, changed schema, or a second use.

Confirmation parsing is deliberately narrow. Unambiguous positive/negative
phrases become booleans; mixed or ambiguous text produces a yes/no retry rather
than an action. An exact positive decision releases the verified suffix only
while the capability digest is unchanged. A qualified positive decision that
revises scope is replanned without applying the old authorization. A negative
decision is recorded as failure-status input and terminates with a
non-completion acknowledgement.

This guarantee applies only when a live capability is explicitly marked
confirmation-required through the anchored trusted-description marker or the
explicit capability extension. PACT does not infer consequentiality from an
operation name, user message, or unconstrained prose semantics.

## 11. A2A correlation boundary

The A2A adapter accepts exactly one result for exactly one outstanding external
obligation. Before constructing the trusted runtime event, it requires:

- one result object, not a batch;
- the pending operation name; and
- a nonempty evaluator-assigned `tool_call_id`.

It then binds that result to the pending runtime's plan/node/argument/schema
metadata and uses the evaluator ID plus content in the event fingerprint.

The current outbound `ToolCallsData` lets the agent send operation and
arguments but not an agent-chosen call nonce that the evaluator must echo. As a
result, PACT cannot independently prove that a returned evaluator ID belongs to
the exact outbound call or distinguish a replay of an identical call by a
misbehaving evaluator. The single-pending rule removes local ambiguity but does
not remove this protocol trust assumption. Reports and proofs must preserve
this distinction.

## 12. Inference and rate-limit bounds

Each semantic compilation performs:

```text
1 proposal + at most 1 verification-guided repair
           + at most 1 independent action audit
```

Therefore successful semantic completions per compile request are at most
three, below the Track 2 limit of five. The action audit occurs only when the
accepted proposal contains an Act and semantic review is enabled. Every audit
candidate is decoded and verified from scratch; audit failure never authorizes
the earlier proposal.

The following independent limits are enforced locally and configurable:

- policy, goal, current-event, conversation, and total context characters;
- conversation message and evidence-record counts;
- candidate size;
- PlanIR node count and dependency depth; and
- maximum completion tokens.

Provider rate-limit retry is a separate bounded loop. The Cerebras client has:

- finite `PACT_CEREBRAS_MAX_RATE_LIMIT_RETRIES`;
- finite `PACT_CEREBRAS_MAX_RATE_LIMIT_WAIT_SECONDS`;
- proactive token pacing using recent quota headers;
- reactive `Retry-After`/reset-aware scheduling with bounded queue backoff; and
- `CerebrasRateLimitBudgetExceededError` when the next retry or wait would
  exceed a configured budget.

No provider path waits or retries indefinitely. Rate-limit retries are
transport recovery for a completion that did not succeed; they do not create
or accept additional semantic candidates.

The submitted defaults are `gpt-oss-120b`, `medium` reasoning effort, an
8,192-token completion ceiling, and semantic review enabled. The corresponding
environment variables are `PACT_COMPILER_MODEL`,
`PACT_COMPILER_REASONING_EFFORT`,
`PACT_COMPILER_MAX_COMPLETION_TOKENS`, and
`PACT_COMPILER_SEMANTIC_REVIEW`. All provider routes and inference controls
remain environment-configurable. Archived `TRACK2_PLANNER_*` and
`TRACK2_EXECUTOR_*` variables are not compatibility aliases for PACT and cannot
silently change the submitted runtime.

## 13. A2A metrics

Every A2A text response includes `metadata.turn_metrics`, including asks,
confirmation prompts, typed terminal responses, refusal, and safe-error text.
Tool-call responses defer metrics so usage accumulates across the external
operation loop until the next text response.

Reported fields include:

- `prompt_tokens`;
- `completion_tokens`;
- `thinking_tokens`;
- `num_llm_calls`;
- `avg_llm_call_time_ms`;
- `num_passes`;
- `quota_wait_time_ms`;
- model and provider-reported cost.

The three token counters are disjoint. Cerebras reports reasoning tokens inside
its provider `output_tokens`; PACT maps them to A2A as:

```text
prompt_tokens     = input_tokens
thinking_tokens   = reasoning_output_tokens
completion_tokens = max(0, output_tokens - reasoning_output_tokens)
```

Thus evaluator aggregation does not double-count reasoning. Proposal, rejected
repair, and action-audit attempts all retain their disjoint usage. If a typed
compiler error exposes completed attempts, the safe terminal response reports
those metrics before the context accumulator resets.

## 14. Enforced invariants

These statements assume the production adapter is the only emission path, the
hash function and JSON validator behave correctly, and the live schemas are
valid.

### I1. Structural plan validity

Every executed plan is a bounded DAG with unique IDs/evidence producers,
exactly one terminal response, and no node outside the response ancestry.

### I2. Capability confinement

Every emitted external operation belongs to the current digest-matched live
snapshot, and its fully materialized arguments validate against that
operation's current Draft 2020-12 schema.

### I3. Single outstanding effect

The runtime emits no second Ask, Confirm, Observe, or Act while one action is
waiting for its compatible event.

### I4. Evidence non-escalation

A failed, malformed, duplicate, conflicting, out-of-order, wrong-kind, or
wrong-provenance event cannot create successful evidence or satisfy the pending
node.

### I5. Grounded completion

Before PlanIR construction, a typed `completed` response's evidence list is
replaced by the exact union of every current Act success key and every carried
successful-action obligation. The verifier accepts those keys only as
successful external Act evidence. Input-only, confirmation, observation,
failure, or missing evidence cannot discharge completion.

### I6. Conditional scoped confirmation

For a capability explicitly marked confirmation-required, an Act has either a
current authorizing Confirm ancestor or an unconsumed true confirmation record
matching schema digest, operation, and argument digest.

### I7. Context isolation

Policy, goal, tools, ledger, runtime, obligations, authorization consumption,
and metrics are keyed by A2A `context_id`; cancellation removes the context
state while retaining its stable lock object so a concurrent turn cannot enter
through a replacement lock.

### I8. Bounded semantic inference

One compile request has at most three successful completion calls: proposal,
optional repair, and optional action audit. Provider rate-limit retries and
cumulative wait are separately finite.

## 15. Failure policy

PACT fails closed at each boundary:

| Failure | Visible behavior |
|---|---|
| Invalid/missing policy event or empty goal | Generic safe text |
| Invalid/removed capability or schema | No call; safe text |
| Invalid wire/PlanIR | One bounded repair, then safe text |
| Invalid/truncated/unavailable action audit | Do not execute the pre-audit plan; safe text |
| Provider/configuration/rate-limit budget error | Safe text with available attempt metrics |
| Malformed/multiple/unmatched tool result | No success evidence; safe text |
| Explicit tool failure | Failure evidence; no completion claim |
| Failure after an earlier successful Act | Partial-failure text |
| Ambiguous confirmation | Ask for an explicit yes/no answer |
| User message while an external result is pending | Preserve obligation and report waiting |

Generic failure text intentionally reveals neither verifier internals nor
provider exception data.

## 16. Reproducible engineering evidence

Release evidence is taken only from the final PACT configuration and exact
candidate image. Deterministic coverage exercises graph
cycles/limits/reachability, strict wire shape and size, repair and audit
failure, nested Draft 2020-12 constraints, description-marker parsing,
capability revocation, deep mutation, external failure,
replay/conflict/out-of-order events, confirmation scope and one-shot
consumption, exact completion closure, control/data event transitions, context
isolation, disjoint token metrics, and finite rate-limit budgets. Generic
synthetic capability contracts exercise the same reachable kernel without
task-name or domain-workflow branches.

Test counts, image digests, and public-run measurements belong in the release
record and technical report only after the final commands complete. Historical
legacy-agent results and pre-audit PACT runs are not evidence for the submitted
configuration. No hidden-set result is claimed.

## 17. Release checklist

Before final submission:

1. Rebase/verify compatibility with the latest official main branch.
2. Run the complete unit suite, Ruff, and mypy.
3. Repeat the completed public-format A2A smoke against the exact candidate
   image.
4. Build the final image for `linux/amd64` from the selective Dockerfile.
5. Scan source, image layers, and scenario TOML for secrets.
6. Publish the GHCR package publicly and pin its immutable digest.
7. Validate the exact digest with README option C.
8. Propagate the same digest to the submission scenario and form.
9. Re-run the four-page report build and update only dated test evidence that
   changed; do not add unverified task scores.

## 18. Known limitations

- Natural-language policy and goal semantics remain model-produced and are not
  formally verified.
- Unknown effect metadata cannot establish Observe-versus-Act correctness or a
  confirmation requirement.
- Exact evaluator result origin relies on the A2A single-pending contract
  because outbound agent nonces are unavailable.
- The ledger is process-local and is deleted on cancellation; it is not a
  durable distributed audit log.
- SHA-256 provenance is content addressing, not a signature or remote
  attestation.
- The independent action audit is still model-generated; it improves semantic
  cross-checking but is not a formal natural-language proof.
- One repair and one action audit bound compute but may trade away recovery on
  semantically hard requests.
- The engineering suite establishes mechanisms, not benchmark task quality;
  final competitiveness must be measured separately without changing the
  kernel toward benchmark-specific branches.
