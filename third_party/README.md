# Third-Party CAR-bench Dependency

This repository wraps the upstream [CAR-bench](https://github.com/CAR-bench/car-bench)
benchmark, but does not vendor the benchmark source into git.

## Setup

Clone the local dependency before installing the evaluator extra or building the
green evaluator image:

```bash
./scripts/setup_car_bench.sh
```

The script clones CAR-bench into `third_party/car-bench/` at the pinned,
release-validated commit `6bc1900750e561e79831dd6a8e757ac37080ba0f`. That
directory is a local ignored dependency and can be deleted/recreated at any
time. To intentionally test another upstream revision, set `CAR_BENCH_REF` to
the desired commit SHA before invoking the script.

The green evaluator imports CAR-bench from this path. Purple reference agents do
not need the CAR-bench checkout at runtime.

## Running the Benchmark

After setup, install the normal extras and run any scenario from `scenarios/`:

```bash
uv sync --extra car-bench-agent --extra car-bench-evaluator
uv run agentbeats-run scenarios/purple_car_bench_agent/local.toml
```
