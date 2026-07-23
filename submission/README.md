# CAR-bench Track 2 final-submission package

The only final agent in this package is PACT, built from
`src/track_2_agent_under_test_cerebras/`. Its submission template is
[`track_2_direct/scenario.toml.template`](track_2_direct/scenario.toml.template).
No legacy planner/executor submission template is shipped: do not build,
render, validate, or submit that archived source-only variant.

Do not put any API key or secret value in the final TOML. It must contain only
environment-variable interpolation expressions.

## Ready-to-submit artifacts

- [`track_2_direct/scenario.toml`](track_2_direct/scenario.toml): validated
  hidden-set TOML with the final digest.
- [`track_2_direct/FORM_ANSWERS.md`](track_2_direct/FORM_ANSWERS.md): copy-paste
  answers in form order.
- [`track_2_direct/OPTION_C_VALIDATION.md`](track_2_direct/OPTION_C_VALIDATION.md):
  public-image and A2A validation record.
- [`technical_report/PACT_CAR-bench_Track2_Technical_Report.pdf`](technical_report/PACT_CAR-bench_Track2_Technical_Report.pdf):
  four-page IJCAI-format report.

## Release checklist

1. Build the PACT agent for `linux/amd64`.
2. Run the manual GitHub Actions workflow
   [`publish-track2-ghcr.yml`](../.github/workflows/publish-track2-ghcr.yml)
   to push the PACT image to GHCR. It grants `packages: write` only to that job
   and records the immutable digest in the workflow summary.
3. Set the newly created GHCR package to **Public**. Then render the final
   scenario without manually editing it:

   ```bash
   uv run python scripts/prepare_track2_submission.py \
     --variant direct \
     --image-digest "${PACT_DIGEST}" \
     --output dist/track2-direct/scenario.toml
   ```

   Set `PACT_DIGEST` to the actual `sha256:` value recorded by the publish
   workflow. The utility rejects mutable tags, literal secrets, non-hidden task
   splits, and incorrect task counts.
4. Put the same digest and PACT environment block into
   `scenarios/track_2_agent_under_test_cerebras/ghcr_smoke.toml`, and run README
   option C. Retain the output as release evidence.
5. By **July 19, 2026, 23:59 AoE**, submit the agent form: selected Track 2,
   digest-pinned public GHCR image, full hidden-set `scenario.toml`, immutable
   source link, description, inference setup, and checklist confirmations.
6. Submit the four-page IJCAI-format technical report **separately** by
   **July 26, 2026, 23:59 AoE**. An agent submission without this report is not
   eligible for awards.

PACT uses one proposal, at most one verification-guided repair, and a selective
semantic audit for action-bearing plans: at most three sequential completions
per compilation. Exact confirmations and successful actions advance the same
verified suffix without another completion. It reports aggregate, disjoint
prompt, visible-completion, and thinking usage through A2A `turn_metrics`.

The final report sources and build instructions are in
[`technical_report/`](technical_report/).
