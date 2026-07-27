# IJCAI-ECAI 2026 Competition
# CAR-bench: Building Reliable LLM Agents Under Real-World Uncertainty

[![Paper](https://img.shields.io/badge/ACL%202026-2026.acl--long.1886-b31b1b.svg)](https://aclanthology.org/2026.acl-long.1886/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![A2A](https://img.shields.io/badge/A2A-Protocol-blue.svg)](https://a2a-protocol.org)
[![Website](https://img.shields.io/badge/Website-CAR--bench-blue)](https://car-bench.github.io/car-bench/)

Dockerized A2A starter kit for the CAR-bench Challenge at IJCAI-ECAI 2026.

[Overview](#overview) | [Setup](#setup) | [Build An Agent](#build-an-agent) | [Validate](#validate-your-agent) | [Submit](#submission-instructions) | [Evaluation](#evaluation-summary) | [Read More](#read-more)

---

## Overview

[CAR-bench](https://github.com/CAR-bench/car-bench) evaluates whether tool-using
LLM agents behave reliably in realistic, uncertain, user-facing settings. The
benchmark is instantiated as an in-car voice assistant domain with ambiguous
requests, mutable vehicle/environment state, domain policies, and unavailable
capabilities.

This repository turns CAR-bench into a competition-ready A2A evaluation harness.
Participants build a dockerized **agent under test**. The evaluator sends the
agent policy context, user messages, tool definitions, and tool results. The
agent returns either user-facing text or tool calls. The evaluator remains the
only component that executes CAR-bench tools and computes scores.

The official competition has two tracks:

| Track | Goal | Starter |
| --- | --- | --- |
| **Track 1: Open Track** | Use any model, provider, framework, or architecture to maximize reliability. The Best Innovation Award focuses on agent harnessing and reliability design. | [`src/track_1_agent_under_test/`](src/track_1_agent_under_test/) |
| **Track 2: Cerebras Fast-Reasoning** | Use direct Cerebras-hosted `gpt-oss` inference and compute-aware harnessing to turn fast inference into better reliability under Track 2 inference-compute constraints. Track 2 registration is closed. | [`src/track_2_agent_under_test_cerebras/`](src/track_2_agent_under_test_cerebras/) and planner variant |

Final ranking is performed by the organizers on a hidden test set. The local
`local_test_set.toml`, `local_docker_test_set.toml`, and `ghcr_test_set.toml`
scenarios are for development validation only.

Docker-based scenarios use the official organizer-published evaluator image,
`ghcr.io/car-bench/car-bench-evaluator:latest`. Participants should build or
publish only their **agent under test** image; they should not self-host,
modify, or submit an evaluator image.

CAR-bench includes:

| Dimension | Details |
| --- | --- |
| Tools | 58 interconnected navigation, vehicle-control, charging, weather, and productivity tools |
| Policies | 19 domain-specific policies agents must follow |
| User model | LLM-simulated multi-turn user |
| Tasks | 254 public CAR-bench tasks across Base, Hallucination, and Disambiguation categories |
| Main reliability metric | `Pass^3`: a task must pass all 3 independent trials |

---

## Architecture

```text
Participant development
  -> choose Track 1 or Track 2 starter
  -> build an A2A agent under test
  -> validate locally and in Docker
  -> publish a GHCR Docker image + config
  -> organizers run hidden evaluation

Runtime exchange

CAR-bench evaluator
  - owns simulated user, tools, environment state, trajectories, and scoring
  - sends A2A messages to the submitted agent
  - executes only the tool calls returned by the agent

        A2A text/data messages
              <---->

Agent under test
  - receives policy/user text, tool definitions, and tool results
  - maintains its own conversation state per context_id
  - returns user-facing text and/or tool-call data
```

The benchmark boundary matters: agents may do internal planning, verification,
reranking, memory, or multi-pass reasoning, but must use only evaluator-provided
inputs and must not inspect hidden evaluator/task state or execute CAR-bench
tools directly.

---

## Setup

### Common Setup

For serious participation, create a fork of this starter repository first, then
clone your fork. That gives you a clean place for your agent code, Dockerfile,
scenario configs, and technical-report notes. Official submission will still be
a public digest-pinned GHCR agent image plus `scenario.toml`, not a pull
request to this repository.

```bash
git clone https://github.com/YOUR_ORG_OR_USERNAME/car-bench-ijcai.git
cd car-bench-ijcai

python3.11 -m venv .venv
source .venv/bin/activate

./scripts/setup_car_bench.sh
cp .env.example .env
```

`./scripts/setup_car_bench.sh` clones the original CAR-bench repository into
`third_party/car-bench/`. That checkout is ignored by git and treated as a local
dependency for evaluator runs.

Set at least the evaluator key in `.env`:

```bash
GEMINI_API_KEY=...
```

### Track 1 Setup

Track 1 can use any model provider. Install the Track 1 template dependencies:

```bash
uv sync --extra track-1-agent --extra car-bench-evaluator
```

Then add whatever provider keys your agent implementation needs to `.env`, for
example:

```bash
AGENT_LLM=anthropic/claude-haiku-4-5-20251001
ANTHROPIC_API_KEY=...
```

The Track 1 starter is documented in
[`src/track_1_agent_under_test/README.md`](src/track_1_agent_under_test/README.md).
Its Docker baseline supports Chat Completions and Responses through
`AGENT_API_MODE`. The organizer Azure GPT-5.6 Sol profile uses Responses and
leaves temperature and reasoning effort unset, applying the provider-default
reasoning level while retaining function tools.

### Track 2 Setup

Track 2 uses direct Cerebras `gpt-oss` inference through the Cerebras Python SDK.
Participants should use Cerebras-hosted `gpt-oss` models. The direct executor
baseline image uses `gpt-oss-120b`, omits the reasoning and temperature fields
so Cerebras applies provider defaults, and sets a 2,048-token completion ceiling
as a rate-limit guardrail. The planner/executor template separately defaults the
private planner to `gpt-oss-120b`, with `TRACK2_PLANNER_REASONING_EFFORT=high`
and executor effort `medium`.

```bash
uv sync --extra track-2-agent --extra car-bench-evaluator
```

Then add the evaluator and Cerebras keys to `.env`:

```bash
GEMINI_API_KEY=...
CEREBRAS_API_KEY=...
```

For the planner/executor template, the default planner is also Cerebras
`gpt-oss-120b`:

```bash
TRACK2_PLANNER_MODEL=gpt-oss-120b
TRACK2_PLANNER_REASONING_EFFORT=high
TRACK2_PLANNER_MAX_COMPLETION_TOKENS=4096
```

Leave `TRACK2_PLANNER_TEMPERATURE` and `TRACK2_TEMPERATURE` unset unless the
provider should receive an explicit temperature value.

Public Cerebras development-tier limits can be strict. Use smoke scenarios
first and keep `TRACK2_MAX_COMPLETION_TOKENS` tight. The reference templates
retry reactively only after a Cerebras 429, using
`x-ratelimit-reset-tokens-minute` when Cerebras provides it and falling back to
`retry-after` otherwise. Provider queue pressure uses jittered local backoff.
Cerebras 429s write JSON reports to
`/tmp/car-bench-rate-limit-reports` by default. Expect organizers to provide
increased Cerebras rate limits compared with a free personal account; access
details will follow soon. Participants may self-host the open-source models
used by Cerebras during ordinary development. Codex Pro is not the submitted-agent
runtime for these templates.

Track 2 uses inference-compute constraints. Participants may use up to 5
sequential LLM calls for each baseline LLM step; parallel calls inside one step
are allowed. Token usage is limited to 500k tokens on average per task,
including input, reasoning, and output tokens. The reference baseline uses
about 54k tokens on average per task. Report token usage through
`Message.metadata.turn_metrics.prompt_tokens`,
`completion_tokens`, and `thinking_tokens`, aggregated across all internal LLM
calls before the final response for that assistant step. Sequential-call
compliance is documented through the technical-report architecture diagram, not
through a new A2A metadata field.

Track 2 details live in the agent READMEs:

| Reference | README |
| --- | --- |
| Direct Cerebras agent | [`src/track_2_agent_under_test_cerebras/README.md`](src/track_2_agent_under_test_cerebras/README.md) |
| Planner/executor agent | [`src/track_2_agent_under_test_cerebras_planner/README.md`](src/track_2_agent_under_test_cerebras_planner/README.md) |

---

## Build An Agent

The most important contract is simple: each turn, the evaluator sends your agent
the information it is allowed to use, and your agent sends back one
benchmark-visible response.

| Turn situation | Evaluator sends | Your agent returns |
| --- | --- | --- |
| First turn of a task | text Part with `System: ... User: ...` plus data Part with `{"tools": [...]}` | text Part for a user response, data Part with `{"tool_calls": [...]}`, or both |
| After your agent called tools | data Part with `{"tool_results": [...]}` | another text response and/or tool-call data |
| After your agent spoke to the user | next simulated user message as a text Part | another text response and/or tool-call data |
| Any turn | same `context_id` for the task | maintain your own per-context conversation state |

The evaluator executes all CAR-bench tools. Your agent should only request tool
calls by returning data like:

```json
{"tool_calls": [{"tool_name": "get_weather", "arguments": {"location_or_poi_id": "loc_123"}}]}
```

For exact A2A shapes, protobuf helper usage, metadata, and code references, read:

- [Inbound messages: what your agent receives](docs/development-guide.md#inbound-messages--what-your-agent-receives)
- [Outbound messages: what your agent should return](docs/development-guide.md#outbound-messages--what-your-agent-should-return)
- [Agent executor contract](docs/development-guide.md#agent-executor-contract)

### Reference Agents

| Agent | Package | Scenario Directory | Best For |
| --- | --- | --- | --- |
| Track 1 template | [`src/track_1_agent_under_test/`](src/track_1_agent_under_test/) | [`scenarios/track_1_agent_under_test/`](scenarios/track_1_agent_under_test/) | Building your own provider/model integration |
| Track 2 Cerebras | [`src/track_2_agent_under_test_cerebras/`](src/track_2_agent_under_test_cerebras/) | [`scenarios/track_2_agent_under_test_cerebras/`](scenarios/track_2_agent_under_test_cerebras/) | Direct Cerebras next-action baseline |
| Track 2 planner/executor | [`src/track_2_agent_under_test_cerebras_planner/`](src/track_2_agent_under_test_cerebras_planner/) | [`scenarios/track_2_agent_under_test_cerebras_planner/`](scenarios/track_2_agent_under_test_cerebras_planner/) | Cerebras `gpt-oss` planner with high reasoning plus Cerebras `gpt-oss` executor with medium reasoning |

---

## Validate Your Agent

Use the same progression for either track: first local smoke tests, then local
Docker builds, then GHCR image validation.

### Scenario Files

Scenario TOML files are the run configs for CAR-bench evaluations. They choose
which evaluator and agent to start or pull, which environment variables to pass,
which task split and task counts to run, and how many trials to execute. Each
agent directory under `scenarios/` has the same six-file matrix:

| Scenario File | Purpose |
| --- | --- |
| `local_smoke.toml` | Local Python, train split, one task from each task type, one trial |
| `local_test_set.toml` | Local Python, public CAR-bench test split, three trials |
| `local_docker_smoke.toml` | Official evaluator image plus local agent build, train smoke |
| `local_docker_test_set.toml` | Official evaluator image plus local agent build, public CAR-bench test split |
| `ghcr_smoke.toml` | Official evaluator image plus published agent image, train smoke |
| `ghcr_test_set.toml` | Official evaluator image plus published agent image, public CAR-bench test split |

### A. Local Smoke And Debug

Fastest way to iterate on code. Agents run as local Python processes.

| Track | Command |
| --- | --- |
| Track 1 | `uv run car-bench-run scenarios/track_1_agent_under_test/local_smoke.toml --show-logs` |
| Track 2 Cerebras | `uv run car-bench-run scenarios/track_2_agent_under_test_cerebras/local_smoke.toml --show-logs` |
| Track 2 planner/executor | `uv run car-bench-run scenarios/track_2_agent_under_test_cerebras_planner/local_smoke.toml --show-logs` |

Use the corresponding `local_test_set.toml` only after the smoke scenario works.
Local test-set runs are development validation, not official final evaluation.

### B. Docker Local Build

Use this before publishing. It verifies that your Dockerfile and runtime
environment work without local Python process assumptions. The evaluator service
is pulled from the official organizer image; only the agent-under-test service
is built from your local Dockerfile.

| Track | Generate Compose | Run |
| --- | --- | --- |
| Track 1 | `uv run python generate_compose.py --scenario scenarios/track_1_agent_under_test/local_docker_smoke.toml` | `docker compose --env-file .env -f scenarios/track_1_agent_under_test/docker-compose.yml up --abort-on-container-exit` |
| Track 2 Cerebras | `uv run python generate_compose.py --scenario scenarios/track_2_agent_under_test_cerebras/local_docker_smoke.toml` | `docker compose --env-file .env -f scenarios/track_2_agent_under_test_cerebras/docker-compose.yml up --abort-on-container-exit` |

For the Track 2 planner/executor agent, use the same commands
with their scenario directories. `generate_compose.py` writes
`docker-compose.yml` and `a2a-scenario.toml` next to the selected Docker
scenario; those generated files are ignored by git.

### C. GHCR Image Validation

Use this to test the same kind of image/config that organizers will run.
Participant agent images must be pullable by the organizers. For the public
competition workflow, publish the GHCR package as **public** and do not bake API
keys or other secrets into the image.

Build and push an `linux/amd64` image:

```bash
docker build --platform linux/amd64 \
  -f src/track_1_agent_under_test/Dockerfile.track-1-agent-under-test \
  -t ghcr.io/yourusername/your-agent:latest .

docker push ghcr.io/yourusername/your-agent:latest
```

After the first push, open the package page and check **Package settings**:

```text
https://github.com/users/yourusername/packages/container/package/your-agent
```

For organization-owned images, use:

```text
https://github.com/orgs/your-org/packages/container/package/your-agent
```

Set **Package visibility** to **Public**, or explicitly grant organizer access
if the image must stay private. The `image = "..."`
value in your `ghcr_*.toml` file must match the image you pushed.

Then update your `ghcr_smoke.toml` or `ghcr_test_set.toml` image reference and
validate it:

```bash
uv run python generate_compose.py --scenario scenarios/track_1_agent_under_test/ghcr_smoke.toml
docker compose --env-file .env -f scenarios/track_1_agent_under_test/docker-compose.yml up --abort-on-container-exit
```

For Track 2, use the matching Track 2 scenario directory and Dockerfile.

If you intentionally want CI publishing, this repo includes a disabled opt-in
template at
[`.github/workflows/publish-ghcr.yml.disabled`](.github/workflows/publish-ghcr.yml.disabled).
It has no push/tag trigger; rename and configure it only if you want to use it.

Results are written under `output/<agent-name>/` with filenames that include
timestamp, scenario, task selection, trial count, and reliable model/reasoning
hints when the scenario exposes them.

### Organizer hidden-dataset runner

Organizers run the final hidden evaluation through the separate resumable
competition command. It validates the submitted Docker scenario, mounts the
hidden CSVs only into the official evaluator, and persists every task-trial
before advancing:

```bash
uv run car-bench-competition start submissions/track_2/team-17/scenario.toml \
  --track track_2 \
  --team-id team-17 \
  --team-name "Team Seventeen" \
  --dataset-dir hidden_dataset \
  --results-root competition_results \
  --env-file .env
```

Interrupted runs resume by run directory, and completed runs can be aggregated
into a CSV/JSON leaderboard and standard SVG plots:

```bash
uv run car-bench-competition resume competition_results/track_2/team-17/<run-id>
uv run car-bench-competition aggregate --results-root competition_results --track track_2
```

See [`docs/organizer-hidden-evaluation.md`](docs/organizer-hidden-evaluation.md)
for folder layout, recovery behavior, privacy boundaries, score accounting, and
paused-run review.

---

## Submission Instructions

The submission Google Form link will be announced by the organizers and added
to the competition website before the official evaluation rounds. The expected
submission shape is:

1. A public GHCR Docker image for your agent under test, pinned by digest.
2. A `scenario.toml` file using the official evaluator image and your
   agent-under-test image/config.
3. Required and optional environment variable or secret names, excluding secret
   values.
4. Track selection: `track_1`, `track_2`, or `both`.
5. A 4-page IJCAI-format technical report that cites CAR-bench. Use the
   [IJCAI author kit](https://www.ijcai.org/authors_kit). Track 2 reports
   should include an architecture diagram so the sequential-call constraint can
   be audited.

All LLM model names, provider routes, deployment names, API bases, service
tiers, and reasoning-effort selectors must be configurable through environment
variables. Organizers may host models differently from your development setup.

Use this copy-pasteable `scenario.toml` as a starting point:

```toml
[evaluator]
image = "ghcr.io/car-bench/car-bench-evaluator:latest"

[evaluator.env]
GEMINI_API_KEY = "${GEMINI_API_KEY:?Set GEMINI_API_KEY}"
LOGURU_LEVEL = "${LOGURU_LEVEL:-INFO}"

[agent_under_test]
image = "ghcr.io/your-org/your-agent@sha256:replace_with_digest"

[agent_under_test.env]
AGENT_LLM = "${AGENT_LLM:?Set AGENT_LLM}"
AGENT_API_BASE = "${AGENT_API_BASE:-}"
AGENT_API_KEY = "${AGENT_API_KEY:?Set AGENT_API_KEY}"
AGENT_TEMPERATURE = "${AGENT_TEMPERATURE:-}"
LOGURU_LEVEL = "${LOGURU_LEVEL:-INFO}"

[config]
num_trials = 3
task_split = "hidden"
tasks_base_num_tasks = -1
tasks_hallucination_num_tasks = -1
tasks_disambiguation_num_tasks = -1
max_steps = 50
```

Field notes:

- `[evaluator]` must use the official organizer-published evaluator image.
  Participants do not submit, modify, or self-host evaluator images for
  official evaluation.
- `[evaluator.env]` may reference `GEMINI_API_KEY`, `GOOGLE_API_KEY`, and
  `LOGURU_LEVEL`; organizers provide evaluator secrets for official runs.
- `[agent_under_test].image` must point to a public digest-pinned GHCR image,
  not a mutable tag.
- `[agent_under_test.env]` lists the env vars organizers must provide. Use
  `${VAR:?message}` for required vars and `${VAR:-}` or `${VAR:-default}` for
  optional vars. Do not include actual API keys, tokens, passwords, or secret
  values.
- `[config]` must use `task_split = "hidden"` for official submission and `-1`
  for each task-count field so the full hidden set is selected.
- Name environment variables however you want, but every model/provider choice
  needed to run the agent must be configurable through env vars.

The organizers will run submitted Docker agents and configs on controlled
evaluation infrastructure. Final ranking is determined from hidden test set
evaluation, not from local public development scenarios.

---

## Evaluation Summary

CAR-bench evaluates different reliability failures across three task types:

| Task Type | Public CAR-bench Tasks | What It Tests |
| --- | ---: | --- |
| Base | 100 | Correct tool use, final state, intermediate state, and policy compliance |
| Hallucination | 98 | Whether the agent acknowledges missing capabilities/data instead of fabricating |
| Disambiguation | 56 | Whether the agent resolves ambiguity through preferences or clarification before acting |

Each task receives fine-grained metric scores such as action correctness,
required information-gathering tools, tool execution validity, policy
compliance, and user end-conversation behavior. A task reward is 1 only when all
required metrics for that task pass.

The competition reports consistency:

| Metric | Meaning |
| --- | --- |
| `Pass^3` | Task passes in all 3 independent trials. This is the main deployment-readiness score. |
| `Pass@3` | Task passes in at least 1 of 3 trials. This measures latent capability. |

For implementation details, see the original CAR-bench reward calculators in
`third_party/car-bench/car_bench/envs/reward_calculators.py` after running
`./scripts/setup_car_bench.sh`.

---

## Project Structure

```text
src/
  agentbeats/                         inherited internal A2A runner helpers
  evaluator/                          CAR-bench evaluator A2A server
  track_1_agent_under_test/           Track 1 minimal template
  track_2_agent_under_test_cerebras/  Track 2 direct Cerebras agent
  track_2_agent_under_test_cerebras_planner/
                                      Track 2 planner/executor agent
  tool_call_types.py                  shared tool-call data models
  turn_metrics.py                     shared metadata keys

scenarios/
  track_1_agent_under_test/
  track_2_agent_under_test_cerebras/
  track_2_agent_under_test_cerebras_planner/

docs/
  development-guide.md                detailed A2A turn contract
  agent-under-test-harnessing.md      allowed harness boundaries
  cerebras-harness-patterns.md        Track 2 model/harness patterns
```

---

## Read More

Use this reading path when building your own agent:

1. **Pick a starter**
   - Track 1: [`src/track_1_agent_under_test/README.md`](src/track_1_agent_under_test/README.md)
   - Track 2 direct Cerebras: [`src/track_2_agent_under_test_cerebras/README.md`](src/track_2_agent_under_test_cerebras/README.md)
   - Track 2 planner/executor: [`src/track_2_agent_under_test_cerebras_planner/README.md`](src/track_2_agent_under_test_cerebras_planner/README.md)
2. **Understand the turn contract**
   - [`docs/development-guide.md`](docs/development-guide.md)
3. **Design a more sophisticated harness**
   - [`docs/agent-under-test-harnessing.md`](docs/agent-under-test-harnessing.md)
   - [`docs/cerebras-harness-patterns.md`](docs/cerebras-harness-patterns.md)
4. **Protocol background**
   - [`docs/a2a-introduction.md`](docs/a2a-introduction.md)

Rules of thumb:

- Preserve the A2A boundary.
- Maintain conversation state per `context_id`.
- Return benchmark-visible text and tool calls in message parts, not hidden metadata.
- Let the evaluator execute CAR-bench tools.
- Do not inspect hidden task/evaluator state or add private vehicle-capability tools.

---

## Citation

If you use CAR-bench in your research, please cite:

```bibtex
@inproceedings{kirmayr-etal-2026-car,
    title = "{CAR}-bench: Evaluating the Consistency and Limit-Awareness of {LLM} Agents under Real-World Uncertainty",
    author = "Kirmayr, Johannes  and
      Stappen, Lukas  and
      Andre, Elisabeth",
    editor = "Liakata, Maria  and
      Moreira, Viviane P.  and
      Zhang, Jiajun  and
      Jurgens, David",
    booktitle = "Proceedings of the 64th Annual Meeting of the {A}ssociation for {C}omputational {L}inguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2026",
    address = "San Diego, California, United States",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2026.acl-long.1886/",
    doi = "10.18653/v1/2026.acl-long.1886",
    pages = "40599--40618",
    ISBN = "979-8-89176-390-6"
}
```

---

## Important Links

- Original CAR-bench: [github.com/CAR-bench/car-bench](https://github.com/CAR-bench/car-bench)
- Competition website: [car-bench.github.io/car-bench](https://car-bench.github.io/car-bench/)
- Paper: [aclanthology.org/2026.acl-long.1886](https://aclanthology.org/2026.acl-long.1886/)
- A2A Protocol: [a2a-protocol.org](https://a2a-protocol.org)
