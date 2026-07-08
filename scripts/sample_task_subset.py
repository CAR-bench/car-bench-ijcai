"""
Sample a reproducible random subset of task IDs per category (base /
hallucination / disambiguation) and print a [config] block ready to paste
into a scenario TOML's task_id_filter fields.

This exists because num_tasks / shuffle in the underlying benchmark only
reorder or take a fixed-order prefix of the task list — they do not draw a
random sample. task_id_filter is the only way to pin an actual random subset,
so this script computes that subset once (with a fixed seed for
reproducibility) and hands you the exact TOML to run.

Usage:
    uv run python scripts/sample_task_subset.py --split test --n 5 --seed 42
"""
import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "third_party" / "car-bench"))
from car_bench.envs import get_env  # noqa: E402

TASK_TYPES = ["base", "hallucination", "disambiguation"]


def sample_ids(task_type: str, task_split: str, n: int, seed: int) -> list[str]:
    env = get_env(
        "car_voice_assistant",
        user_strategy="human",
        user_model=None,
        policy_evaluator_strategy="human",
        task_type=task_type,
        task_split=task_split,
    )
    all_ids = [t.task_id for t in env.tasks]
    rng = random.Random(seed)
    if n >= len(all_ids):
        return sorted(all_ids)
    return sorted(rng.sample(all_ids, n))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--n", type=int, default=5, help="Tasks per category")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"# Sampled with seed={args.seed}, n={args.n} per category, split={args.split}")
    print(f"# Paste into a scenario TOML's [config] block.\n")
    print("[config]")
    print(f'task_split = "{args.split}"')
    for task_type in TASK_TYPES:
        ids = sample_ids(task_type, args.split, args.n, args.seed)
        print(f"tasks_{task_type}_task_id_filter = {ids}")
        # num_tasks is ignored once task_id_filter is set, but keep it
        # consistent for readability / in case task_id_filter is removed.
        print(f"tasks_{task_type}_num_tasks = {len(ids)}")


if __name__ == "__main__":
    main()
