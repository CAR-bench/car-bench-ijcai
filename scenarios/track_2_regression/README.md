# Track 2 targeted regression scenarios

These scenarios are development-only. They target task IDs that failed in two
previous public Track 1 full-pass runs, so that controller changes can be
validated before spending an elevated-rate-limit Track 2 evaluation window.

They are not official submission scenarios and must never be submitted with
`task_split = "hidden"` replaced by public task filters.

Run the direct regression after a controller or prompt change:

```bash
uv run car-bench-run \
  scenarios/track_2_regression/direct_persistent_failures.toml --show-logs
```

Use the resulting failure report to decide whether a change should advance to
the 30-task public batch and then the full public three-trial evaluation.
