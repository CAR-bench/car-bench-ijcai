# Track 2 Cerebras Agent Under Test

This package is the direct Track 2 Cerebras Fast-Reasoning starter agent for
CAR-bench A2A evaluation. It calls Cerebras-hosted `gpt-oss` models directly
through the Cerebras Python SDK and returns the same benchmark-visible A2A text
responses or tool calls as the Track 1 template.

The baseline image uses `gpt-oss-120b`. It omits temperature, reasoning effort,
and service tier so Cerebras applies its provider defaults (currently `medium`
reasoning for `gpt-oss-120b`). It intentionally sets a 2,048-token completion
ceiling as a rate-limit guardrail; that ceiling is an organizer choice, not a
Cerebras provider default. Participants may replace prompting, validation, or
harnessing strategy as long as the external A2A contract stays unchanged.

## What This Agent Demonstrates

- Parses evaluator messages into policy/user text, tool definitions, and tool
  results.
- Maintains conversation history per `context_id`.
- Calls the Cerebras SDK `chat.completions.with_raw_response.create(...)`.
- Uses Cerebras structured JSON schema output for a strict next-action object.
- Logs successful-response `x-ratelimit-*` headers for visibility.
- Writes JSON rate-limit reports for Cerebras 429s, including provider headers
  and error body.

## Configuration

Set the evaluator key and Cerebras key in `.env`:

```bash
GEMINI_API_KEY=...
CEREBRAS_API_KEY=...
```

Important environment variables:

| Variable | Default | Purpose |
| --- | --- | --- |
| `CEREBRAS_API_KEY` | required | Cerebras API key used by the SDK. |
| `TRACK2_EXECUTOR_MODEL` | `gpt-oss-120b` | Cerebras-hosted `gpt-oss` executor model. Old `cerebras/...` prefixes are accepted and stripped for compatibility. |
| `TRACK2_EXECUTOR_REASONING_EFFORT` | unset | Optional Cerebras `gpt-oss` reasoning effort override. When unset, Cerebras currently defaults to `medium`. |
| `TRACK2_CEREBRAS_SERVICE_TIER` | unset | Optional Cerebras service tier, for example `default`, `priority`, `auto`, or `flex`. |
| `TRACK2_MAX_COMPLETION_TOKENS` | `2048` in the baseline image; `1024` in the bare server | Completion-token cap for executor calls. Keep the image default for comparable baseline runs. |
| `TRACK2_TEMPERATURE` | unset | Optional executor temperature. Leave unset to use the provider default. |
| `TRACK2_CEREBRAS_QUEUE_BACKOFF_SECONDS` | `60` | Nominal first local pause after a provider `queue_exceeded` 429. |
| `TRACK2_CEREBRAS_QUEUE_BACKOFF_INITIAL_JITTER_RATIO` | `0.1` | First queue retry jitter ratio; default gives roughly 54-66 seconds. |
| `TRACK2_CEREBRAS_QUEUE_BACKOFF_SECOND_MIN_SECONDS` | `90` | Minimum second queue retry wait. |
| `TRACK2_CEREBRAS_QUEUE_BACKOFF_SECOND_MAX_SECONDS` | `120` | Maximum second queue retry wait. |
| `TRACK2_CEREBRAS_QUEUE_BACKOFF_CAP_MIN_SECONDS` | `180` | Minimum third-and-later queue retry wait. |
| `TRACK2_CEREBRAS_QUEUE_BACKOFF_CAP_MAX_SECONDS` | `300` | Maximum third-and-later queue retry wait. |
| `TRACK2_CEREBRAS_RATE_LIMIT_RETRY_BUFFER_SECONDS` | `1` | Safety buffer added to provider reset or retry headers before retrying a 429. |
| `CAR_BENCH_CEREBRAS_RATE_LIMIT_REPORT_DIR` | `/tmp/car-bench-rate-limit-reports` | Directory for Cerebras rate-limit JSON reports. Falls back to `CAR_BENCH_RATE_LIMIT_REPORT_DIR` when set. |
| `TRACK2_LLM_MALFORMED_RETRIES` | `1` | Retry budget for malformed next-action JSON. |

## Rate Limits And Development Windows

During normal development, participants are expected to use the Cerebras public
tier, where rate limits can be strict. Cerebras includes the requested maximum
completion length in its preflight token estimate, so use smaller smoke
scenarios first, keep `TRACK2_MAX_COMPLETION_TOKENS` as low as the task allows,
and schedule larger public-tier runs externally instead of launching many jobs
at once.

The reference client does not proactively throttle based on local request/token
limits or previous successful responses. It sends the request and only waits
reactively after a provider-visible Cerebras 429.
The reference client records successful provider-call timings and keeps failed
429 attempts in logs/reports.

When a Cerebras 429 is observed, the client writes a JSON report by default to
`/tmp/car-bench-rate-limit-reports`. The report includes session start time,
wall time until the limit, wall time since the previous limit/retry marker,
successful-call token usage, estimated attempted request tokens, the current
failed call shape, and the raw provider payload. A provider
`queue_exceeded` error respects `retry-after` if Cerebras sends one; otherwise
it applies jittered local backoff: roughly 60 seconds on the first queue retry,
90-120 seconds on the second, and 180-300 seconds on later queue retries. A
quota error with `x-ratelimit-reset-tokens-minute` applies that reset hint plus
the configured buffer before retrying. If Cerebras omits that token-reset
header, the client logs the missing header and falls back to
`x-ratelimit-reset-requests-day` or `retry-after` when present.
Final time-budget and quota-wait accounting details will be announced before
the official evaluation.

Terminal logs show the attempted request estimate, previous successful
rate-limit headers when available, and on 429 the wait duration, resume time,
wait reason, source header, reset-header values, and any expected reset header
that Cerebras did not provide.

Organizers will provide a few test windows with elevated rate limits and
priority tier access so participants can test harness behavior at higher speed
and with less throttling. Participants may also self-host the open-source models
used by the Cerebras `gpt-oss` executor during development, then validate
speed-sensitive behavior during those windows.

Codex Pro plans are still provided for selected Track 2 teams to accelerate
harness engineering and development. They are not the submitted-agent runtime
for this template. Plans are allocated by June 15.

## Run

Local smoke:

```bash
uv run car-bench-run scenarios/track_2_agent_under_test_cerebras/local_smoke.toml --show-logs
```

Docker smoke:

```bash
uv run python generate_compose.py --scenario scenarios/track_2_agent_under_test_cerebras/local_docker_smoke.toml
docker compose --env-file .env -f scenarios/track_2_agent_under_test_cerebras/docker-compose.yml up --abort-on-container-exit
```

## Read More

- [Main README](../../README.md): setup, validation modes, and submission shape.
- [Development guide](../../docs/development-guide.md): detailed A2A turn
  contract.
- [Harnessing guide](../../docs/agent-under-test-harnessing.md): allowed
  harness boundaries.
- [Track 2 harness patterns](../../docs/cerebras-harness-patterns.md): direct
  Cerebras, planner/executor, and rate-limit development guidance.
