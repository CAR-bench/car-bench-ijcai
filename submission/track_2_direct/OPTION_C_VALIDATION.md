# README Option C validation record

Validated on 2026-07-17 PDT (2026-07-18 UTC).

## Release identity

- Source revision:
  `3255c9320bfa36e3fe0c4647ffc401b78e7259a6`
- Publish workflow:
  `https://github.com/puyanguvic/car-bench-ijcai/actions/runs/29625464113`
- Submitted image:
  `ghcr.io/puyanguvic/car-bench-track-2-direct@sha256:173fd630e1691b40af55e731e82023c1df9cfcae52d3b84420c1d307eccea6d1`

An anonymous registry inspection using an empty Docker configuration directory
succeeded. The digest resolves directly to one OCI image manifest (not a
multi-platform index); its image configuration is `linux/amd64`, and its
runtime user is `carbench`.

## Command and outcome

The exact digest above was placed in
`scenarios/track_2_agent_under_test_cerebras/ghcr_smoke.toml`, then validated
with README option C:

```bash
uv run python generate_compose.py \
  --scenario scenarios/track_2_agent_under_test_cerebras/ghcr_smoke.toml
docker compose --env-file .env \
  -f scenarios/track_2_agent_under_test_cerebras/docker-compose.yml \
  up --abort-on-container-exit --exit-code-from a2a-client
```

Observed release checks:

- the public digest was pulled and both Agent Cards returned HTTP 200;
- the evaluator completed all three public-format smoke splits;
- `a2a-client` exited with code 0 and wrote a result artifact;
- the artifact contained final-response `turn_metrics` with `prompt_tokens`,
  `completion_tokens`, and `thinking_tokens` (nine recorded projections); and
- Compose services were removed after validation.

The local artifact was
`20260718-014214__track_2_agent_under_test_cerebras-ghcr_smoke__train-trials1-base1-hall1-dis1.json`
with SHA-256
`55c43af4ec481a39c609554e2ef9fbe4315816d0340bb892f2f6689036115bbf`.

## Scope of this validation

This is a packaging, public-access, platform, A2A-contract, metrics, and result
mount validation. It is not a performance claim. During the run, the personal
Cerebras development account exhausted its daily token quota after two
successful completions. The provider requested a delay of 86,400 seconds;
PACT correctly rejected that delay because it exceeded the configured 600
second cumulative-wait bound and failed closed. Organizers have indicated that
the official Track 2 route will use increased quota.
