# Track 2 PACT Cerebras Harness

The final Track 2 submission has one production architecture: **PACT**
(Policy-Aware Contract-guided Tool-use). It uses direct Cerebras-hosted
`gpt-oss` inference through the Cerebras Python SDK, while keeping all execution
authority in a typed local verifier and deterministic obligation runtime. The
canonical setup, validation, and submission instructions are in the
[PACT agent README](../src/track_2_agent_under_test_cerebras/README.md).

> **Submission source of truth.** Use
> [`src/track_2_agent_under_test_cerebras/`](../src/track_2_agent_under_test_cerebras/),
> its PACT scenarios, and the `PACT_*` environment variables documented below.
> The earlier direct next-action controller and planner/executor variant are
> **archived historical implementations only**. They are not final submission
> candidates, are not included in the release image, and their `TRACK2_*`
> settings are intentionally ignored by PACT.

## Final Agent Map

| Status | Agent | Package | Local Scenario | Internal Strategy |
| --- | --- | --- | --- | --- |
| **Final submission** | PACT | [`src/track_2_agent_under_test_cerebras/`](../src/track_2_agent_under_test_cerebras/) | [`scenarios/track_2_agent_under_test_cerebras/local_smoke.toml`](../scenarios/track_2_agent_under_test_cerebras/local_smoke.toml) | Cerebras is an untrusted semantic compiler; a strict decoder, typed PlanIR verifier, deterministic runtime, and evidence ledger control execution. |
| Archived historical | Planner/executor V1 | `src/track_2_agent_under_test_cerebras_planner/` | Not a release scenario | Retained only for source-history comparison; do not configure or submit it. |

The old rule-controller implementation once shared the final package path, but
the package entry point now instantiates `PACTAgentExecutor`. Old descriptions
of that package as a direct next-action executor are therefore historical, not
descriptions of the reachable release runtime.

The retained GHCR package and submission-directory names may still contain the
word `direct` for compatibility. Those names do not select the archived
controller; the release Dockerfile and server entry point select PACT.

## Final Model and Runtime Configuration

Use the `PACT_*` namespace for every submission-relevant model, provider, and
budget setting:

```env
CEREBRAS_API_KEY=...
PACT_COMPILER_MODEL=gpt-oss-120b
PACT_COMPILER_CEREBRAS_API_BASE=https://api.cerebras.ai
PACT_COMPILER_SERVICE_TIER=
PACT_COMPILER_REASONING_EFFORT=medium
PACT_COMPILER_MAX_COMPLETION_TOKENS=8192
PACT_COMPILER_SEMANTIC_REVIEW=true
PACT_COMPILER_TEMPERATURE=
PACT_COMPILER_MAX_REPAIR_ATTEMPTS=1
PACT_CEREBRAS_MAX_RATE_LIMIT_RETRIES=3
PACT_CEREBRAS_MAX_RATE_LIMIT_WAIT_SECONDS=600
```

Leave `PACT_COMPILER_TEMPERATURE` unset to use the provider default. The final
defaults use medium reasoning, an 8,192-token completion ceiling, semantic
review for action-bearing plans, and at most one verifier-guided repair. See
the [PACT configuration table](../src/track_2_agent_under_test_cerebras/README.md#configuration)
for the complete set of provider and verifier bounds.

Do not use `TRACK2_EXECUTOR_*`, `TRACK2_PLANNER_*`, or `TRACK2_CEREBRAS_*` in a
new scenario. Those names belong to archived historical implementations and do
not configure PACT. This fail-closed separation prevents a stale V1 environment
from silently changing the submitted model, reasoning effort, or retry budget.

## PACT Harness Pattern

PACT separates probabilistic interpretation from deterministic execution:

```text
A2A event and live tool contracts
  -> Cerebras strict-schema semantic compilation
  -> compact wire-plan decoding
  -> typed PlanIR construction and local verification
  -> optional independent semantic audit for action-bearing plans
  -> deterministic obligation runtime
  -> one text response or one externally visible tool call
  -> correlated result evidence and verified continuation
```

The model cannot call tools directly. The verifier checks live operation
membership, complete argument schemas, dependency structure, confirmation
contracts, resource bounds, and terminal evidence obligations. The runtime
allows at most one pending external action and accepts its result only when the
operation and argument/capability fingerprints match. Exact confirmations and
successful actions advance an already verified immutable suffix; observations,
free-form answers, revised confirmations, and capability changes trigger
bounded replanning.

Each compilation uses one proposal and at most one local-verification-guided
repair. A locally valid plan containing an action receives one independently
prompted full-policy and argument-provenance audit, after which its output is
strictly decoded and verified again. The worst semantic path is three
sequential completions, below Track 2's five-call limit. Answer-only and
refusal-only plans skip the audit.

## Cerebras Development Logistics

Free personal Cerebras accounts can have strict rate limits, so begin with the
smallest smoke scenario. PACT bounds both retry count and cumulative wait time
with `PACT_CEREBRAS_MAX_RATE_LIMIT_RETRIES` and
`PACT_CEREBRAS_MAX_RATE_LIMIT_WAIT_SECONDS`. It uses provider reset hints when
available and fails closed when the configured budget is exhausted.

Known Cerebras `queue_exceeded` and `token_quota_exceeded` responses produce
diagnostic JSON reports under `CAR_BENCH_CEREBRAS_RATE_LIMIT_REPORT_DIR`, then
`CAR_BENCH_RATE_LIMIT_REPORT_DIR`, or
`/tmp/car-bench-rate-limit-reports` by default. These reports support debugging
and reproducibility; they are not fabricated timing or scoring metadata.
`avg_llm_call_time_ms` includes successful provider calls only, while aggregate
token usage includes all successful internal completions used for the turn.

## Track 2 Accounting

Track 2 permits up to five sequential LLM calls per baseline LLM step, allows
parallel calls inside a step, and limits average task usage to 500k aggregate
input, reasoning, and output tokens. PACT reports all successful internal calls
through the existing A2A `turn_metrics` fields:

- `prompt_tokens`
- `completion_tokens`
- `thinking_tokens`

The three fields are disjoint. Because Cerebras includes reasoning tokens in
provider output tokens, PACT reports visible completion as
`max(0, output_tokens - reasoning_output_tokens)` and reports reasoning only as
`thinking_tokens`. Sequential-call structure belongs in the technical-report
architecture diagram rather than a new A2A metadata field.

## Archived Historical Patterns

The following patterns explain repository history only. They must not be used
to configure, validate, describe, or submit the final Track 2 agent.

### Archived: Direct Next-Action Controller

The original baseline made one Cerebras call per assistant step and forwarded a
schema-constrained next action after lightweight parsing. It predated PACT's
typed PlanIR verification, obligation runtime, evidence ledger, and independent
action audit. References to a “direct executor,” `TRACK2_EXECUTOR_*`, or
`TRACK2_TEMPERATURE` describe this archived historical design.

### Archived: High-Effort Planner Plus Medium-Effort Executor

The V1 planner/executor generated private guidance with one model call and then
used another model call for the benchmark-visible action. It lived under
`src/track_2_agent_under_test_cerebras_planner/` and used
`TRACK2_PLANNER_*` settings. That directory is retained for comparison only; it
has no final publish target and is excluded from the PACT release image.

For current commands and the exact submission configuration, return to the
[PACT agent README](../src/track_2_agent_under_test_cerebras/README.md).
