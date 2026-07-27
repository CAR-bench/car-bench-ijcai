# Organizer Hidden-Dataset Evaluations

This workflow is for competition organizers. It runs one submitted agent image
against the organizer-held hidden CSV files while keeping both detailed results
and recovery state outside the containers.

The hidden dataset is never copied into the evaluator or A2A client image. It is
mounted read-only into the evaluator container for the duration of a run. The
participant container receives only the normal A2A messages produced by the
evaluator. The generated network also requires a per-run organizer token for
evaluator POST requests. That token is given to the trusted client and evaluator
only, so the participant cannot start a forged evaluation request.

## Prerequisites

1. Install the evaluator and development dependencies:

   ```bash
   ./scripts/setup_car_bench.sh
   uv sync --extra car-bench-evaluator --dev
   ```

2. Put the three organizer-only files in one directory:

   ```text
   hidden_dataset/
     hidden_base.csv
     hidden_hallucination.csv
     hidden_disambiguation.csv
   ```

3. Put evaluator and participant API credentials in `.env`. The submitted
   scenario must contain environment-variable references, never credential
   values.

4. Ensure Docker and Docker Compose are available. The runner pulls both image
   references before creating the immutable run manifest. Requested references,
   resolved digests, image IDs, creation times, and available OCI
   version/revision labels are recorded as provenance.

## Start One Team

Use a stable lowercase team ID for paths and an optional display name:

```bash
caffeinate -i uv run car-bench-competition start submissions/track_1/team-17/scenario.toml \
  --track track_1 \
  --team-id team-17 \
  --team-name "Team Seventeen" \
  --dataset-dir hidden_dataset \
  --results-root competition_results \
  --env-file .env
```

```bash
caffeinate -i uv run car-bench-competition start submissions/track_2/team-0/scenario.toml --track track_2 --team-id team-0 --team-name "Organizer Baseline" --dataset-dir hidden_dataset --results-root competition_results --env-file .env
```

Run the same command with different team IDs in separate terminals to evaluate
teams in parallel. Each invocation gets a unique Compose project, network, and
result directory; no host ports are published.

Official mode requires:

- the official evaluator image;
- a digest-pinned participant image;
- `task_split = "hidden"`;
- three trials;
- every Base, Hallucination, and Disambiguation task; and
- `max_steps = 50`.

The participant service uses the same container execution shape as public
Option C validation. Submitted scenarios cannot add builds, mounts, privileged
mode, host networking, evaluator commands, or arbitrary evaluator environment
variables. The evaluator environment accepts only `GEMINI_API_KEY`,
`GOOGLE_API_KEY`, and `LOGURU_LEVEL` references.

### Participant compatibility

Digest-pinned participant images are not rebuilt or modified. The organizer
runner preserves the public Option C contract: the same container entrypoint,
`--host`, `--port`, and `--card-url` arguments, participant environment-variable
expansion, health endpoint, A2A message parts, and tool-call/tool-result shapes.
The new competition run context and persistence paths exist only between the
trusted A2A client and evaluator; they are never sent to the participant.

Participant telemetry remains optional. Agents that predate `turn_metrics` run
and score normally; missing latency, cost, or token telemetry is recorded as
`null`/unknown and never changes the ranking score. Agent scenarios may keep
arbitrary scalar runtime settings such as `MAX_OUTPUT_TOKENS = 2048`. Variables
whose names identify credentials must continue to use `${...}` references.

The intentional compatibility boundary is that an official image must be
self-contained. Participant builds, host mounts, privileged mode, and host
networking are rejected because they would either expose organizer state or
make runs non-reproducible.

Use `--development` only for integration checks with reduced hidden task counts
or trials. Development runs are marked in their manifest and must not be used
for the official leaderboard.

## Recovery and Resume

Each `(category, task ID, trial)` is one durable unit. A completed unit is
written atomically before the evaluator advances. Attempts are separate from
official trials.

The default per-unit timeout is 30 minutes. A timeout, container loss, protocol
failure, or uncaught evaluator error restarts the complete Compose stack from
persisted state. Two retries are allowed after the first attempt. A valid
completed unit with reward zero is recorded normally and is not retried.

If the terminal, host, or Docker daemon stops, resume with:

```bash
uv run car-bench-competition resume \
  competition_results/track_2/team-17/<run-id>
```

Resume verifies the saved scenario, config, dataset fingerprint, and image
provenance before starting Docker. Completed unit files are validated and
skipped.

After three unsuccessful attempts, the run enters `paused` state and exits
nonzero. Review `progress.json`, the current unit's attempt records, and
`private/logs/docker-compose.log`. Once the cause is understood or corrected,
grant a fresh recovery budget explicitly:

```bash
uv run car-bench-competition resume \
  competition_results/track_2/team-17/<run-id> \
  --retry-paused
```

If the dataset or `.env` moved without changing contents, pass
`--dataset-dir` or `--env-file` to `resume`. A changed dataset is rejected even
when it has the same filenames.

## Result Layout

```text
competition_results/<track>/<team-id>/<run-id>/
  manifest.json                 immutable identity and provenance
  progress.json                 current status, unit, attempt, and timestamps
  scenario.toml                 submitted scenario snapshot
  a2a-scenario.toml             generated internal A2A endpoints/run context
  docker-compose.yml            generated isolated runtime
  private/
    units/<category>/<task-id>/trial-<n>.json
    attempts/<category>/<task-id>/trial-<n>/attempt-<n>.json
    results.jsonl
    summary.json
    logs/docker-compose.log
  exports/
    leaderboard-row.json        present only after complete official execution
    summary.json
```

`private/units` is the source of truth. `results.jsonl`, summaries, and exports
are rebuilt deterministically, so they can be regenerated after interruption.
The run directory is host-private by default. The per-run control token exists
only in the organizer process and the trusted evaluator/client container
environments; it is not written to disk or exposed to the participant service.
The evaluator runs with the launching organizer's UID/GID so resumed runs and
their artifacts remain owned and writable by that organizer account.

Private unit records contain task metadata, trajectories, reward components,
and errors. Files under `exports/` contain aggregates and provenance only; they
exclude task IDs, instructions, actions, removed tool details, and trajectories.

## Scores and Accounting

The primary leaderboard value is the unweighted mean of Base,
Hallucination, and Disambiguation `Pass^3`. `Pass^k`, `Pass@k`, per-category
scores, and task-trial reward totals are stored separately. Incomplete runs
have no primary score and cannot emit a leaderboard row.

Raw A2A latency is measured by the evaluator and is the trusted comparison
default. LLM latency, provider quota wait, cost, and token usage come from the
participant-controlled `turn_metrics` metadata. Missing metadata is stored as
missing rather than zero, and every token aggregate includes reporting
coverage. Self-reported usage is diagnostic and never controls rank.

## Build the Leaderboard

After team runs complete:

```bash
uv run car-bench-competition aggregate \
  --results-root competition_results \
  --track track_2
```

The command considers the latest run for each team. It stops if any latest run
is incomplete or development-only, rejects mixed dataset/evaluator/config
fingerprints, assigns equal ranks to exact score ties, and writes:

```text
competition_results/track_2/aggregate/
  leaderboard.json
  leaderboard.csv
  plots/
    macro-pass3.svg
    category-pass3.svg
    consistency-gap.svg
    latency.svg
    tokens.svg
    retries.svg
```

Use `--include-legacy` only when aggregating an older schema-v1 development
result directory that contains no schema-v2 competition rows. Legacy and
official schema-v2 rows are never mixed into one leaderboard.
