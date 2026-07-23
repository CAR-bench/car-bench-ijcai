# CAR-bench IJCAI Track 2 — Final Submission Form Answers

This file follows the submission form in field order. Copy the answer blocks into
the form. Do not replace any `${...}` expression with a real credential.

## Pre-submission checklist

- [x] The agent is compatible with the latest state of `main` in
  `CAR-bench/car-bench-ijcai` used for this release.
- [x] Final A2A responses report `turn_metrics`, including `prompt_tokens`,
  `completion_tokens`, and `thinking_tokens` where available.
- [x] README option C, GHCR Image Validation completed against the exact public
  digest; `a2a-client` exited with code 0.
- [x] The hidden-test format and A2A/scenario contract require no separate agent
  configuration beyond the supplied hidden-set scenario.
- [x] No secrets are included in the image reference, source files, or Scenario
  TOML.

## Public GHCR agent image

Copy:

```text
ghcr.io/puyanguvic/car-bench-track-2-direct@sha256:173fd630e1691b40af55e731e82023c1df9cfcae52d3b84420c1d307eccea6d1
```

## Agent Image checks

- [x] I confirm this GHCR package is public and accessible to organizers.
- [x] I confirm the image platform is `linux/amd64`.

## Scenario TOML

Copy the full block exactly:

```toml
# CAR-bench Track 2 final submission: PACT with direct Cerebras inference.
# The image must be public, linux/amd64, and pinned by its GHCR registry digest.
# Do not put secret values in this file.

[evaluator]
image = "ghcr.io/car-bench/car-bench-evaluator:latest"

[evaluator.env]
GEMINI_API_KEY = "${GEMINI_API_KEY:?Set GEMINI_API_KEY}"
LOGURU_LEVEL = "${LOGURU_LEVEL:-INFO}"

[agent_under_test]
image = "ghcr.io/puyanguvic/car-bench-track-2-direct@sha256:173fd630e1691b40af55e731e82023c1df9cfcae52d3b84420c1d307eccea6d1"

[agent_under_test.env]
CEREBRAS_API_KEY = "${CEREBRAS_API_KEY:?Set CEREBRAS_API_KEY}"
PACT_COMPILER_MODEL = "${PACT_COMPILER_MODEL:-gpt-oss-120b}"
PACT_COMPILER_CEREBRAS_API_BASE = "${PACT_COMPILER_CEREBRAS_API_BASE:-https://api.cerebras.ai}"
PACT_COMPILER_SERVICE_TIER = "${PACT_COMPILER_SERVICE_TIER:-}"
PACT_COMPILER_REASONING_EFFORT = "${PACT_COMPILER_REASONING_EFFORT:-medium}"
PACT_COMPILER_MAX_COMPLETION_TOKENS = "${PACT_COMPILER_MAX_COMPLETION_TOKENS:-8192}"
PACT_COMPILER_SEMANTIC_REVIEW = "${PACT_COMPILER_SEMANTIC_REVIEW:-true}"
PACT_COMPILER_TEMPERATURE = "${PACT_COMPILER_TEMPERATURE:-}"
PACT_COMPILER_MAX_REPAIR_ATTEMPTS = "${PACT_COMPILER_MAX_REPAIR_ATTEMPTS:-1}"
PACT_CEREBRAS_MAX_RATE_LIMIT_RETRIES = "${PACT_CEREBRAS_MAX_RATE_LIMIT_RETRIES:-3}"
PACT_CEREBRAS_MAX_RATE_LIMIT_WAIT_SECONDS = "${PACT_CEREBRAS_MAX_RATE_LIMIT_WAIT_SECONDS:-600}"
LOGURU_LEVEL = "${LOGURU_LEVEL:-INFO}"

[config]
num_trials = 3
task_split = "hidden"
tasks_base_num_tasks = -1
tasks_hallucination_num_tasks = -1
tasks_disambiguation_num_tasks = -1
max_steps = 50
```

## Scenario TOML checks

- [x] I confirm the Scenario TOML uses the official evaluator agent Docker
  image.
- [x] I confirm the `[config]` section is set as in the hidden-test template.
- [x] I confirm all model names, provider routes, API bases, service tiers,
  reasoning settings, output limits, and retry controls needed by the agent are
  configurable through clearly named environment variables where applicable.

## Source code

Copy:

```text
https://github.com/puyanguvic/car-bench-ijcai/tree/3255c9320bfa36e3fe0c4647ffc401b78e7259a6
```

Immutable source commit:
`3255c9320bfa36e3fe0c4647ffc401b78e7259a6`.

## Short agent description

Copy (166 words):

```text
PACT is a policy-aware, contract-guided Track 2 agent that separates probabilistic interpretation from deterministic execution. A Cerebras-hosted gpt-oss-120b semantic compiler emits a compact strict-schema plan; action-bearing plans receive a second full-policy and argument-provenance audit. Model outputs cannot execute tools. PACT decodes each candidate into a typed PlanIR DAG and locally verifies live-operation membership, complete Draft 2020-12 argument schemas, dependencies, resource bounds, anchored confirmation contracts, and terminal evidence obligations. A deterministic runtime emits at most one externally visible action and correlates every result with operation, argument, and capability-snapshot digests. Exact confirmations and successful actions advance the verified immutable suffix without recompilation; observations, revised confirmations, and free-form answers trigger bounded replanning. A capability change causes confirmation-time replanning or emission-time fail-closed handling. Successful effects remain completion obligations in an append-only evidence ledger, preventing later horizons from forgetting prior actions. Malformed, stale, revoked, or unverifiable proposals fail closed. The release kernel contains no task IDs or automotive workflow branches, and final A2A responses report disjoint prompt, completion, and thinking-token usage.
```

## LLM inference setup

Copy:

```text
Cerebras API + direct Cerebras Python SDK (cerebras-cloud-sdk), using the Cerebras-hosted gpt-oss-120b model with strict JSON-schema chat completions. Each compilation uses one proposal, at most one optional verification-guided repair, and—when the plan contains an action—one separate second-pass semantic and argument-provenance audit: at most three sequential model completions. The submitted defaults are medium reasoning, an 8,192-token completion limit, and semantic review enabled. The model, API base, service tier, reasoning effort, output limit, temperature, repair/review controls, and bounded rate-limit retry/wait budgets are configurable through Scenario TOML environment variables.
```

## Special runtime requirements

Copy:

```text
No GPU or persistent storage is required. The unprivileged linux/amd64 container requires outbound HTTPS access to the configured Cerebras API base and a runtime CEREBRAS_API_KEY.
```

## Validation Checks

- [x] I confirm the submitted agent is compatible with the latest `main` branch
  of `https://github.com/CAR-bench/car-bench-ijcai` used for this release.
- [x] I confirm the submitted GHCR image was validated with README option C:
  GHCR Image Validation. The exact digest completed with client exit code 0.
- [x] I confirm the agent sends A2A `turn_metrics` on final responses where
  available, including `prompt_tokens`, `completion_tokens`, and
  `thinking_tokens`.
- [x] I confirm the submitted scenario does not contain secret values.
- [x] I confirm the Docker image and uploaded source do not contain API keys,
  tokens, passwords, or private credentials.
- [x] I confirm the agent does not inspect hidden evaluator state, hidden tasks,
  answer keys, or CAR-bench internals during evaluation.

## Technical Report Awareness

- [x] We are aware that a four-page technical report is required separately by
  July 26, 2026, 23:59 AoE, and that submissions without a report will not be
  considered for awards.

## Final Notes

Copy:

```text
Please run the exact digest-pinned image above. PACT makes at most three sequential model completions per compilation and bounds provider retries and cumulative rate-limit waiting. It deliberately fails closed on malformed plans, stale capabilities, revoked confirmation, or an unavailable inference route. No benchmark task IDs or domain-specific workflow branches are present in the release kernel.
```

## Final action before pressing Submit

README option C is complete. Before pressing Submit, compare the image digest
and Scenario TOML once more with `OPTION_C_VALIDATION.md`, then copy the form
answers exactly without substituting any secret values.
