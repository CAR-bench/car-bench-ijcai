# Track 2 Cerebras Planner/Executor Agent Under Test

This package is a Track 2 planner/executor template. A private Cerebras
`gpt-oss` planner creates compact internal guidance after a user turn, and a
Cerebras-hosted `gpt-oss` executor returns the benchmark-visible next action
through the normal A2A interface.

By default, both roles use `gpt-oss-120b`. The planner runs with
`TRACK2_PLANNER_REASONING_EFFORT=high`; the executor runs with
`TRACK2_EXECUTOR_REASONING_EFFORT=medium`.

## What This Agent Demonstrates

- Keeps the public A2A boundary identical to Track 1 and the direct Track 2
  Cerebras template.
- Uses `TRACK2_PLANNER_MODEL`, default `gpt-oss-120b`, for private plan
  creation.
- Uses `TRACK2_EXECUTOR_MODEL`, default `gpt-oss-120b`, as the
  Cerebras-hosted `gpt-oss` model for final next-action execution.
- Reuses the private plan across tool-result continuation turns until the
  executor can answer the user.
- Aggregates planner and executor token usage and call counts in
  `turn_metrics`.

## Configuration

Set the required keys in `.env`:

```bash
GEMINI_API_KEY=...
CEREBRAS_API_KEY=...
TRACK2_PLANNER_MODEL=gpt-oss-120b
TRACK2_PLANNER_REASONING_EFFORT=high
TRACK2_PLANNER_MAX_COMPLETION_TOKENS=4096
TRACK2_EXECUTOR_MODEL=gpt-oss-120b
TRACK2_EXECUTOR_REASONING_EFFORT=medium
```

Important environment variables:

| Variable | Default | Purpose |
| --- | --- | --- |
| `CEREBRAS_API_KEY` | required | Cerebras API key used by planner and executor calls. |
| `TRACK2_CEREBRAS_API_BASE` | `https://api.cerebras.ai` | Cerebras API base shared by planner and executor. Keep the official endpoint for final Track 2 evaluation unless organizers instruct otherwise. |
| `TRACK2_PLANNER_MODEL` | `gpt-oss-120b` | Cerebras-hosted `gpt-oss` planner model. Old `cerebras/...` prefixes are accepted and stripped for compatibility. |
| `TRACK2_PLANNER_REASONING_EFFORT` | `high` | Cerebras `gpt-oss` reasoning effort for planner calls. Supported values are `low`, `medium`, and `high`. |
| `TRACK2_PLANNER_TEMPERATURE` | unset | Optional planner temperature. Leave unset to use the provider default. |
| `TRACK2_PLANNER_MAX_COMPLETION_TOKENS` | `4096` | Completion-token cap for planner calls. |
| `TRACK2_EXECUTOR_MODEL` | `gpt-oss-120b` | Cerebras-hosted `gpt-oss` executor model. Old `cerebras/...` prefixes are accepted and stripped for compatibility. |
| `TRACK2_EXECUTOR_REASONING_EFFORT` | `medium` | Cerebras `gpt-oss` reasoning effort for executor calls. |
| `TRACK2_MAX_COMPLETION_TOKENS` | `1024` | Completion-token cap for executor calls. |
| `TRACK2_TEMPERATURE` | unset | Optional executor temperature. Leave unset to use the provider default. |
| `TRACK2_PROMPT_HISTORY_MAX_MESSAGES` | `24` | Maximum transcript messages retained for planner and executor calls; the system policy and initial user request are always preserved. |
| `TRACK2_CEREBRAS_SERVICE_TIER` | unset | Optional service tier, for example `default`, `priority`, `auto`, or `flex`. |
| `TRACK2_CEREBRAS_QUEUE_BACKOFF_SECONDS` | `60` | Nominal first local pause after a Cerebras executor `queue_exceeded` 429. |
| `TRACK2_CEREBRAS_QUEUE_BACKOFF_INITIAL_JITTER_RATIO` | `0.1` | First queue retry jitter ratio; default gives roughly 54-66 seconds. |
| `TRACK2_CEREBRAS_QUEUE_BACKOFF_SECOND_MIN_SECONDS` | `90` | Minimum second queue retry wait. |
| `TRACK2_CEREBRAS_QUEUE_BACKOFF_SECOND_MAX_SECONDS` | `120` | Maximum second queue retry wait. |
| `TRACK2_CEREBRAS_QUEUE_BACKOFF_CAP_MIN_SECONDS` | `180` | Minimum third-and-later queue retry wait. |
| `TRACK2_CEREBRAS_QUEUE_BACKOFF_CAP_MAX_SECONDS` | `300` | Maximum third-and-later queue retry wait. |
| `TRACK2_CEREBRAS_RATE_LIMIT_RETRY_BUFFER_SECONDS` | `1` | Safety buffer added to provider reset or retry headers before retrying a Cerebras executor 429. |
| `TRACK2_CEREBRAS_PROACTIVE_TOKEN_PACING` | `true` | Wait before a planner or executor request when the previous successful response proves the estimated request exceeds the remaining token-minute quota. |
| `TRACK2_CEREBRAS_TOKEN_QUOTA_WINDOW_SECONDS` | `60` | Fallback token-minute window when Cerebras omits its reset header. |
| `CAR_BENCH_CEREBRAS_RATE_LIMIT_REPORT_DIR` | `/tmp/car-bench-rate-limit-reports` | Directory for Cerebras rate-limit JSON reports. Falls back to `CAR_BENCH_RATE_LIMIT_REPORT_DIR` when set. |
| `TRACK2_LLM_MALFORMED_RETRIES` | `1` | Retry budget for malformed planner or executor JSON (0–3; the cap preserves the Track 2 five-sequential-call limit). |

Planner/executor runs may consume two provider calls for one benchmark-visible
assistant step. Both clients proactively wait when their previous successful
rate-limit headers prove the next estimated request would exceed the remaining
token-minute quota, and still record any provider-visible 429s in logs/reports.

## Run

Local smoke:

```bash
uv run car-bench-run scenarios/track_2_agent_under_test_cerebras_planner/local_smoke.toml --show-logs
```

Docker smoke:

```bash
uv run python generate_compose.py --scenario scenarios/track_2_agent_under_test_cerebras_planner/local_docker_smoke.toml
docker compose --env-file .env -f scenarios/track_2_agent_under_test_cerebras_planner/docker-compose.yml up --abort-on-container-exit
```

## Notes

The planner output is private harness state. It is never returned as a
CAR-bench tool call. If the executor needs benchmark-visible planning behavior,
it may call the supplied `planning_tool` like any other available CAR-bench
tool.

Both planner and executor use the same Cerebras SDK rate-limit handling as the
direct template. Final time-budget and quota-wait accounting details will be
announced before the official evaluation.
