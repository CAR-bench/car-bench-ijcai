# PACT Track 2 Agent

This is the final Track 2 runtime. PACT (Policy-Aware Contract-guided Tool-use)
uses direct Cerebras inference through the Cerebras Python SDK, but does not
execute model-proposed calls immediately. It compiles a request into a typed
plan, verifies that plan against the live A2A tool contracts and policy
obligations, and only then emits one externally visible action.

## Runtime Contract

- The A2A server entry point is `server.py`; it instantiates
  `PACTAgentExecutor`, not the legacy direct executor.
- State, evidence, pending actions, and metrics are isolated by `context_id`.
- Tool capabilities are taken only from the current evaluator message. An
  explicit empty tool list revokes the previous capability snapshot.
- At most one tool call is outstanding. Results are correlated with the
  pending operation and argument/schema fingerprints before they become
  evidence.
- Consequential actions require the plan's confirmation obligations to be
  satisfied. Completion claims require successful external evidence.
- Every final A2A text response reports aggregate `turn_metrics`, including
  `prompt_tokens`, `completion_tokens`, and `thinking_tokens`.
- Each compilation uses one proposal and at most one verification-guided
  repair. A locally valid plan containing an action receives one independent
  full-policy and argument-provenance audit. The worst case is three sequential
  completions, below Track 2's five-call limit; answer/refusal-only plans skip
  the audit.
- Exact confirmations and successful actions advance the already-verified
  immutable suffix without another model call. Ask answers, observations, and
  revised confirmations are replan boundaries. A capability change detected
  while releasing a confirmation triggers a fresh compile; one detected at
  external-call emission rejects the call and fails closed.

Provider output uses a compact Cerebras strict JSON schema. The richer plan is
decoded and checked locally, so malformed or unverifiable output fails closed
instead of being forwarded to the evaluator.

## Configuration

Put development secrets in an untracked `.env` file. Do not add them to source,
Docker build arguments, image layers, or a submitted scenario TOML.

```bash
GEMINI_API_KEY=...
CEREBRAS_API_KEY=...
PACT_COMPILER_MODEL=gpt-oss-120b
PACT_COMPILER_CEREBRAS_API_BASE=https://api.cerebras.ai
PACT_COMPILER_REASONING_EFFORT=medium
PACT_COMPILER_MAX_COMPLETION_TOKENS=8192
PACT_COMPILER_SEMANTIC_REVIEW=true
```

The final submission template exposes the following runtime settings:

| Variable | Default | Purpose |
| --- | --- | --- |
| `CEREBRAS_API_KEY` | required | Cerebras API key expected by the direct Cerebras SDK. |
| `PACT_COMPILER_MODEL` | `gpt-oss-120b` | Cerebras-hosted compiler model. A legacy `cerebras/` prefix is normalized by the client. |
| `PACT_COMPILER_CEREBRAS_API_BASE` | `https://api.cerebras.ai` | Configurable provider/API route. |
| `PACT_COMPILER_SERVICE_TIER` | unset | Optional Cerebras service tier. |
| `PACT_COMPILER_REASONING_EFFORT` | `medium` | `low`, `medium`, or `high`; the submitted setting gives the policy/argument audit a moderate reasoning budget. |
| `PACT_COMPILER_MAX_COMPLETION_TOKENS` | `8192` | Per compiler-call completion-token cap, including provider reasoning tokens. |
| `PACT_COMPILER_SEMANTIC_REVIEW` | `true` | Run one independently prompted, locally reverified audit for action-bearing plans. |
| `PACT_COMPILER_TEMPERATURE` | unset | Optional sampling temperature in `[0, 2]`; unset uses the provider default. |
| `PACT_COMPILER_MAX_REPAIR_ATTEMPTS` | `1` | Local verification repair budget; only `0` or `1` is accepted. |
| `PACT_CEREBRAS_MAX_RATE_LIMIT_RETRIES` | `3` | Maximum number of provider rate-limit retries per compiler call. |
| `PACT_CEREBRAS_MAX_RATE_LIMIT_WAIT_SECONDS` | `600` | Cumulative provider rate-limit wait budget per compiler call. |
| `LOGURU_LEVEL` | `INFO` | Runtime log level. |

The local verifier limits are also configurable as
`PACT_COMPILER_MAX_NODES`, `PACT_COMPILER_MAX_DEPENDENCY_DEPTH`,
`PACT_COMPILER_MAX_POLICY_CHARS`, `PACT_COMPILER_MAX_GOAL_CHARS`,
`PACT_COMPILER_MAX_USER_EVENT_CHARS`,
`PACT_COMPILER_MAX_CONVERSATION_MESSAGES`,
`PACT_COMPILER_MAX_CONVERSATION_CHARS`, `PACT_COMPILER_MAX_CONTEXT_CHARS`,
`PACT_COMPILER_MAX_CANDIDATE_CHARS`, and
`PACT_COMPILER_MAX_EVIDENCE_RECORDS`. Their defaults are defined and validated
in `plan_compiler_backend.py`.

PACT intentionally ignores archived `TRACK2_PLANNER_*`, `TRACK2_EXECUTOR_*`,
and `TRACK2_CEREBRAS_*` variables. This prevents a stale V1 environment from
silently changing the submitted compiler model, reasoning effort, or budgets.

## Local Validation

Install the Track 2 and evaluator dependencies, then run the smallest public
scenario first:

```bash
uv sync --frozen --extra track-2-agent --extra car-bench-evaluator
uv run car-bench-run scenarios/track_2_agent_under_test_cerebras/local_smoke.toml --show-logs
```

Validate the local release image with the official evaluator image:

```bash
uv run python generate_compose.py \
  --scenario scenarios/track_2_agent_under_test_cerebras/local_docker_smoke.toml
docker compose --env-file .env \
  -f scenarios/track_2_agent_under_test_cerebras/docker-compose.yml \
  up --abort-on-container-exit --exit-code-from a2a-client
```

## Option C: GHCR Image Validation

Build and publish the exact `linux/amd64` release candidate. The repository's
manual `Publish Track 2 Release Candidate to GHCR` workflow is preferred because
its summary records the immutable digest. For a manual build:

```bash
PACT_IMAGE=ghcr.io/puyanguvic/car-bench-track-2-direct
PACT_TAG=track2-pact-rc
docker buildx build --platform linux/amd64 \
  --provenance=false \
  -f src/track_2_agent_under_test_cerebras/Dockerfile.track-2-agent-under-test-cerebras \
  -t "${PACT_IMAGE}:${PACT_TAG}" --push .
docker buildx imagetools inspect "${PACT_IMAGE}:${PACT_TAG}"
```

Set the GHCR package visibility to **Public**. Copy the digest reported by the
push or workflow; do not submit the mutable tag. Put that exact digest-pinned
image and the `PACT_*` environment block from
`submission/track_2_direct/scenario.toml.template` into
`scenarios/track_2_agent_under_test_cerebras/ghcr_smoke.toml`, then run:

```bash
uv run python generate_compose.py \
  --scenario scenarios/track_2_agent_under_test_cerebras/ghcr_smoke.toml
docker compose --env-file .env \
  -f scenarios/track_2_agent_under_test_cerebras/docker-compose.yml \
  up --abort-on-container-exit --exit-code-from a2a-client
```

This is the README option C required by the submission checklist: it exercises
the public, digest-pinned image rather than a local build. Retain the command
output as validation evidence.

## Final Scenario

After Option C succeeds, render the hidden-set scenario with the same digest:

```bash
uv run python scripts/prepare_track2_submission.py \
  --variant direct \
  --image-digest "${PACT_DIGEST}" \
  --output dist/track2-direct/scenario.toml
```

`PACT_DIGEST` must be the actual `sha256:` digest from the published image. The
renderer rejects mutable image tags, non-hidden test configuration, incorrect
task counts, and literal environment values. Inspect the generated file before
copy-pasting it into the organizer form; it must not contain a secret.

## Read More

- [Main README](../../README.md): official A2A workflow and validation modes.
- [Submission package](../../submission/README.md): final release checklist.
- [PACT design](../../docs/pact-v2-design.md): architecture and invariants.
- [Harnessing guide](../../docs/agent-under-test-harnessing.md): allowed
  evaluator boundary.
