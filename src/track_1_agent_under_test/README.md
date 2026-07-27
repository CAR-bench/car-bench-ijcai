# Track 1 Agent Under Test

This package is the minimal Track 1 starter agent for the Open Track. It shows
the complete A2A boundary with the CAR-bench evaluator while leaving the model,
provider, prompting strategy, and internal architecture up to you.

## What This Agent Demonstrates

- Parses evaluator messages into policy/user text, tool definitions, and tool
  results.
- Maintains conversation history per `context_id`.
- Calls a LiteLLM-compatible model using the configured `AGENT_LLM`.
- Returns either a user-facing text Part, a data Part with `{"tool_calls": [...]}`,
  or both.
- Never executes CAR-bench tools directly; the evaluator executes tool calls and
  returns tool results on the next turn.

## Turn Contract

The high-level contract is:

| Turn situation | Evaluator sends | Agent returns |
| --- | --- | --- |
| First task turn | text Part with `System: ... User: ...`, data Part with `{"tools": [...]}` | text Part and/or tool-call data Part |
| After agent tool calls | data Part with `{"tool_results": [...]}` | text Part and/or more tool-call data |
| After agent text response | next simulated user text Part | text Part and/or tool-call data |

For exact schemas and helper functions, read the
[development guide](../../docs/development-guide.md), especially:

- [Inbound messages](../../docs/development-guide.md#inbound-messages--what-your-agent-receives)
- [Outbound messages](../../docs/development-guide.md#outbound-messages--what-your-agent-should-return)
- [Agent executor contract](../../docs/development-guide.md#agent-executor-contract)

## Configuration

The Docker baseline supports both Chat Completions (`AGENT_API_MODE=chat`, the
default) and Responses (`AGENT_API_MODE=responses`) through LiteLLM. Chat mode
sends `reasoning_effort="none"` for GPT-5.6 Sol function-tool requests.
Responses mode preserves reasoning items across tool turns and can leave
reasoning effort unset so GPT-5.6 Sol applies its provider default. By default,
Responses requests use portable stateless replay (`store=false` with encrypted
reasoning content) instead of relying on provider-side response-item storage.

Set the evaluator and OpenAI keys in `.env`:

```bash
GEMINI_API_KEY=...
OPENAI_API_KEY=...
```

The non-Docker starter remains provider-neutral. To use another model locally,
set its LiteLLM model string and provider key, for example:

```bash
GEMINI_API_KEY=...
AGENT_LLM=anthropic/claude-haiku-4-5-20251001
ANTHROPIC_API_KEY=...
```

`AGENT_LLM` can be any LiteLLM-compatible model string if you keep this starter
implementation. If you replace the model client, keep the A2A input/output
contract unchanged.

`AGENT_TEMPERATURE` and `AGENT_REASONING_EFFORT` are optional overrides. Leave
them unset for the baseline. `AGENT_REASONING_EFFORT` is sent only when thinking
mode is explicitly enabled, except for the GPT-5.6 Sol Chat Completions
compatibility rule above.

For Azure Responses, use the Azure deployment name and select the API mode:

```bash
AGENT_LLM=azure/gpt-5.6-sol
AGENT_API_MODE=responses
AGENT_RESPONSES_STATE_MODE=stateless
```

`AGENT_RESPONSES_STATE_MODE=provider-default` restores the provider's default
storage behavior. Stateless mode is recommended for isolated benchmark trials
and zero-data-retention-compatible deployments.

## Run

Local smoke:

```bash
uv run car-bench-run scenarios/track_1_agent_under_test/local_smoke.toml --show-logs
```

Docker smoke:

```bash
uv run python generate_compose.py --scenario scenarios/track_1_agent_under_test/local_docker_smoke.toml
docker compose --env-file .env -f scenarios/track_1_agent_under_test/docker-compose.yml up --abort-on-container-exit
```

## Read More

- [Main README](../../README.md): setup, validation modes, and submission shape.
- [Development guide](../../docs/development-guide.md): detailed A2A turn
  contract.
- [Harnessing guide](../../docs/agent-under-test-harnessing.md): what advanced
  internal harnesses may and may not do.
