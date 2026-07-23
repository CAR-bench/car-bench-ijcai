#!/usr/bin/env python3
"""Summarize CAR-bench result JSON files and classify failed tasks."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence


SCORE_KEYS = (
    "r_actions",
    "r_actions_final",
    "r_actions_intermediate",
    "r_tool_subset",
    "r_tool_execution",
    "r_policy",
    "r_user_end_conversation",
    "r_outputs",
)

DETAIL_KEYS = (
    "tool_subset_missing_tools",
    "tool_execution_errors",
    "policy_llm_errors",
    "policy_aut_errors",
    "outputs",
)


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _is_failure_score(value: Any) -> bool:
    score = _as_float(value)
    return score is not None and score < 1.0


def _nonempty(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, (dict, list, tuple, set, str)):
        return bool(value)
    return bool(value)


def _compact(value: Any, limit: int = 260) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        text = value
    else:
        try:
            text = json.dumps(value, ensure_ascii=False, sort_keys=True)
        except TypeError:
            text = str(value)
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return f"{text[: limit - 3]}..."


def _fmt_number(value: Any) -> str:
    number = _as_float(value)
    if number is None:
        return "n/a"
    if number.is_integer():
        return str(int(number))
    return f"{number:.3f}".rstrip("0").rstrip(".")


def _action_names(actions: Any) -> list[str]:
    names: list[str] = []
    for action in actions or []:
        if isinstance(action, dict):
            name = action.get("name")
        else:
            name = str(action)
        if name:
            names.append(str(name))
    return names


def _join_names(names: Sequence[str], limit: int = 18) -> str:
    if not names:
        return "(none)"
    visible = list(names[:limit])
    if len(names) > limit:
        visible.append(f"... +{len(names) - limit}")
    return " -> ".join(visible)


def iter_result_entries(data: dict[str, Any], all_results: bool = False) -> Iterable[tuple[int, dict[str, Any]]]:
    results = data.get("results") or []
    if all_results:
        for index, entry in enumerate(results):
            if isinstance(entry, dict):
                yield index, entry
        return

    final_result = data.get("final_result")
    if isinstance(final_result, dict):
        yield max(len(results) - 1, 0), final_result
        return

    if results and isinstance(results[-1], dict):
        yield len(results) - 1, results[-1]


def iter_task_records(entry: dict[str, Any]) -> Iterable[tuple[str, dict[str, Any]]]:
    details = entry.get("detailed_results_by_split") or {}
    if details:
        for split, records in details.items():
            for record in records or []:
                if isinstance(record, dict):
                    yield str(split), record
        return

    rewards = entry.get("task_rewards_by_split") or {}
    for split, split_rewards in rewards.items():
        for task_id, reward in (split_rewards or {}).items():
            yield str(split), {"task_id": task_id, "reward": reward, "reward_info": {"info": {}}}


def classify_failure(info: dict[str, Any], item: dict[str, Any]) -> list[str]:
    categories: list[str] = []

    if _nonempty(info.get("tool_execution_errors")) or _nonempty(item.get("error")):
        categories.append("tool_execution")
    if _nonempty(info.get("tool_subset_missing_tools")) or _is_failure_score(info.get("r_tool_subset")):
        categories.append("missing_tool_subset")
    if (
        _nonempty(info.get("policy_llm_errors"))
        or _nonempty(info.get("policy_aut_errors"))
        or _is_failure_score(info.get("r_policy"))
    ):
        categories.append("policy")
    if any(
        _is_failure_score(info.get(key))
        for key in ("r_actions", "r_actions_final", "r_actions_intermediate")
    ):
        categories.append("action_sequence")
    if _is_failure_score(info.get("r_user_end_conversation")):
        categories.append("user_end_conversation")
    if _is_failure_score(info.get("r_outputs")) or _nonempty(info.get("outputs")):
        categories.append("output")

    return categories or ["unknown"]


def collect_failures(entry: dict[str, Any]) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    for split, item in iter_task_records(entry):
        reward = item.get("reward")
        if reward is None:
            reward = (item.get("reward_info") or {}).get("reward")
        if not _is_failure_score(reward):
            continue

        reward_info = item.get("reward_info") or {}
        info = reward_info.get("info") or {}
        task = item.get("task") or {}
        issues: list[str] = []
        for key in SCORE_KEYS:
            value = info.get(key)
            if _is_failure_score(value):
                issues.append(f"{key}={_fmt_number(value)}")
        for key in DETAIL_KEYS:
            value = info.get(key)
            if _nonempty(value):
                issues.append(f"{key}={_compact(value)}")
        if _nonempty(item.get("error")):
            issues.append(f"error={_compact(item.get('error'))}")

        failures.append(
            {
                "split": split,
                "task_id": item.get("task_id") or task.get("task_id") or "(unknown)",
                "task_type": task.get("task_type") or "(unknown)",
                "reward": reward,
                "instruction": task.get("instruction") or "",
                "expected_actions": _action_names(task.get("actions")),
                "actual_actions": _action_names(reward_info.get("actions") or item.get("actions")),
                "categories": classify_failure(info, item),
                "issues": issues,
            }
        )
    return failures


def _split_counts(entry: dict[str, Any]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for split, _item in iter_task_records(entry):
        counts[split] += 1
    return counts


def _format_counter(counter: Counter[str]) -> str:
    if not counter:
        return "(none)"
    return ", ".join(f"{key}={counter[key]}" for key in sorted(counter))


def render_report(
    path: Path,
    data: dict[str, Any],
    *,
    all_results: bool = False,
    max_failures: int = 20,
) -> str:
    lines: list[str] = [f"== {path} =="]
    metadata = data.get("metadata") or {}
    if metadata:
        scenario = metadata.get("scenario_name") or metadata.get("scenario_path") or "unknown"
        model = metadata.get("model") or "unknown"
        task_selection = metadata.get("task_selection") or "unknown"
        lines.append(f"scenario: {scenario}")
        lines.append(f"model: {model}")
        lines.append(f"task_selection: {task_selection}")

    entries = list(iter_result_entries(data, all_results=all_results))
    if not entries:
        lines.append("no result entries found")
        return "\n".join(lines)

    for result_index, entry in entries:
        split_counts = _split_counts(entry)
        total_tasks = sum(split_counts.values())
        failures = collect_failures(entry)
        failure_split_counts = Counter(failure["split"] for failure in failures)
        category_counts: Counter[str] = Counter()
        for failure in failures:
            category_counts.update(failure["categories"])

        lines.append("")
        lines.append(
            "result {index}: score {score}/{max_score}, pass_rate {pass_rate}%, "
            "successful_llm_time {llm_time}s, time_used {time_used}s, quota_wait {quota_wait}s".format(
                index=result_index,
                score=_fmt_number(entry.get("score")),
                max_score=_fmt_number(entry.get("max_score")),
                pass_rate=_fmt_number(entry.get("pass_rate")),
                llm_time=_fmt_number(entry.get("successful_llm_time_used")),
                time_used=_fmt_number(entry.get("time_used")),
                quota_wait=_fmt_number(entry.get("quota_wait_time")),
            )
        )
        lines.append(f"tasks: total={total_tasks}, by_split={_format_counter(split_counts)}")
        lines.append(
            "failures: total={total}, by_split={by_split}, by_category={by_category}".format(
                total=len(failures),
                by_split=_format_counter(failure_split_counts),
                by_category=_format_counter(category_counts),
            )
        )

        for failure in failures[:max_failures]:
            categories = ",".join(failure["categories"])
            instruction = _compact(failure["instruction"], limit=180)
            lines.append(
                "- [{split}] {task_id} reward={reward} type={task_type} category={categories}".format(
                    split=failure["split"],
                    task_id=failure["task_id"],
                    reward=_fmt_number(failure["reward"]),
                    task_type=failure["task_type"],
                    categories=categories,
                )
            )
            if instruction:
                lines.append(f"  instruction: {instruction}")
            lines.append(f"  expected: {_join_names(failure['expected_actions'])}")
            lines.append(f"  actual: {_join_names(failure['actual_actions'])}")
            if failure["issues"]:
                lines.append(f"  issues: {'; '.join(failure['issues'])}")

        if len(failures) > max_failures:
            lines.append(f"... {len(failures) - max_failures} more failures omitted")

    return "\n".join(lines)


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path, help="CAR-bench result JSON file(s)")
    parser.add_argument(
        "--all-results",
        action="store_true",
        help="summarize every result entry instead of only the final result",
    )
    parser.add_argument(
        "--max-failures",
        type=int,
        default=20,
        help="maximum failed task records to print per result entry",
    )
    parser.add_argument(
        "--fail-on-failure",
        action="store_true",
        help="exit non-zero when any summarized result contains failed tasks",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    any_failures = False

    for index, path in enumerate(args.paths):
        if index:
            print()
        data = json.loads(path.read_text())
        print(
            render_report(
                path,
                data,
                all_results=args.all_results,
                max_failures=max(args.max_failures, 0),
            )
        )
        for _result_index, entry in iter_result_entries(data, all_results=args.all_results):
            if collect_failures(entry):
                any_failures = True

    return 1 if args.fail_on_failure and any_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
