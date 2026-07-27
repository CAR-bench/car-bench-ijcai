"""Deterministic score, latency, token, and leaderboard aggregation."""

from __future__ import annotations

import csv
import json
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from competition.storage import atomic_write_json


CATEGORIES = ("base", "hallucination", "disambiguation")


def _nearest_rank(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[index]


def _stats(values: Iterable[float]) -> dict[str, float | int | None]:
    numbers = [float(value) for value in values]
    if not numbers:
        return {"count": 0, "total": None, "mean": None, "median": None, "p95": None}
    return {
        "count": len(numbers),
        "total": sum(numbers),
        "mean": statistics.fmean(numbers),
        "median": statistics.median(numbers),
        "p95": _nearest_rank(numbers, 0.95),
    }


def _pass_metrics(
    records: list[dict[str, Any]],
    expected_tasks: dict[str, list[str]],
    num_trials: int,
) -> tuple[dict[str, Any], bool]:
    by_category: dict[str, dict[str, dict[int, bool]]] = {
        category: {task_id: {} for task_id in expected_tasks[category]}
        for category in CATEGORIES
    }
    for record in records:
        unit = record["unit"]
        category = unit["category"]
        task_id = unit["task_id"]
        trial = int(unit["trial"])
        if category in by_category and task_id in by_category[category]:
            by_category[category][task_id][trial] = bool(record["passed"])

    category_complete = {
        category: bool(tasks)
        and all(set(trials) == set(range(num_trials)) for trials in tasks.values())
        for category, tasks in by_category.items()
    }
    complete = all(category_complete.values())
    per_category: dict[str, dict[str, float | None]] = {}
    for category, tasks in by_category.items():
        scores: dict[str, float | None] = {}
        denominator = len(tasks)
        for k in range(1, num_trials + 1):
            if not category_complete[category] or denominator == 0:
                scores[f"Pass^{k}"] = None
                scores[f"Pass@{k}"] = None
                continue
            scores[f"Pass^{k}"] = sum(
                all(trials[index] for index in range(k)) for trials in tasks.values()
            ) / denominator
            scores[f"Pass@{k}"] = sum(
                any(trials[index] for index in range(k)) for trials in tasks.values()
            ) / denominator
        per_category[category] = scores

    macro: dict[str, float | None] = {}
    for k in range(1, num_trials + 1):
        for prefix in ("Pass^", "Pass@"):
            key = f"{prefix}{k}"
            values = [per_category[category][key] for category in CATEGORIES]
            non_null_values = [value for value in values if value is not None]
            macro[key] = (
                statistics.fmean(non_null_values)
                if len(non_null_values) == len(values)
                else None
            )
    return {"macro": macro, "by_category": per_category}, complete


def build_summary(
    manifest: dict[str, Any],
    records: list[dict[str, Any]],
    *,
    progress: dict[str, Any] | None = None,
    attempts: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    expected_tasks = {
        category: list(manifest["dataset"]["files"][category]["task_ids"])
        for category in CATEGORIES
    }
    num_trials = int(manifest["config"]["num_trials"])
    expected_units = sum(len(task_ids) for task_ids in expected_tasks.values()) * num_trials
    pass_scores, score_complete = _pass_metrics(records, expected_tasks, num_trials)

    rewards = [float(record["reward"]) for record in records]
    raw_a2a = [
        float(record["timing"]["a2a_raw"]["total_ms"])
        for record in records
        if record.get("timing", {}).get("a2a_raw", {}).get("total_ms") is not None
    ]
    raw_turns = [
        float(value)
        for record in records
        for value in record.get("timing", {}).get("a2a_raw", {}).get("turns_ms", [])
    ]
    llm_latency = [
        float(record["timing"]["llm_self_reported_ms"])
        for record in records
        if record.get("timing", {}).get("llm_self_reported_ms") is not None
    ]
    adjusted_a2a = [
        float(record["timing"]["a2a_quota_adjusted"]["total_ms"])
        for record in records
        if record.get("timing", {}).get("a2a_quota_adjusted", {}).get("total_ms")
        is not None
    ]
    quota_wait = [
        float(record["timing"]["quota_wait_self_reported_ms"])
        for record in records
        if record.get("timing", {}).get("quota_wait_self_reported_ms") is not None
    ]
    llm_calls = [
        float(record["telemetry"]["num_llm_calls"])
        for record in records
        if record.get("telemetry", {}).get("num_llm_calls") is not None
    ]
    costs = [
        float(record["telemetry"]["cost"])
        for record in records
        if record.get("telemetry", {}).get("cost") is not None
    ]

    token_fields = ("prompt_tokens", "completion_tokens", "thinking_tokens", "total_tokens")
    token_values: dict[str, list[float]] = {field: [] for field in token_fields}
    token_reported_units = 0
    token_field_reported_units = {field: 0 for field in token_fields}
    for record in records:
        tokens = record.get("telemetry", {}).get("tokens", {})
        if tokens.get("reported"):
            token_reported_units += 1
        for field in token_fields:
            value = tokens.get(field)
            if value is not None:
                token_field_reported_units[field] += 1
                token_values[field].append(float(value))

    attempt_rows = attempts or []
    completed_unit_keys = {
        (
            record["unit"]["category"],
            record["unit"]["task_id"],
            record["unit"]["trial"],
        )
        for record in records
    }
    incomplete_attempt_counts = Counter(
        (
            row.get("unit", {}).get("category"),
            row.get("unit", {}).get("task_id"),
            row.get("unit", {}).get("trial"),
        )
        for row in attempt_rows
        if (
            row.get("unit", {}).get("category"),
            row.get("unit", {}).get("task_id"),
            row.get("unit", {}).get("trial"),
        )
        not in completed_unit_keys
    )
    retries = sum(
        max(0, int(record.get("attempt_count", 1)) - 1)
        for record in records
    ) + sum(max(0, count - 1) for count in incomplete_attempt_counts.values())
    total_attempt_count = sum(
        int(record.get("attempt_count", 1)) for record in records
    ) + sum(incomplete_attempt_counts.values())
    attempt_outcomes = Counter(
        str(row.get("outcome") or "unknown") for row in attempt_rows
    )
    units_complete = score_complete and len(records) == expected_units
    complete = units_complete and (
        progress is None or progress.get("status") == "completed"
    )
    trial_passes = sum(bool(record["passed"]) for record in records)
    summary: dict[str, Any] = {
        "schema_version": 2,
        "run": {
            "run_id": manifest["run_id"],
            "team_id": manifest["team_id"],
            "team_name": manifest["team_name"],
            "track": manifest["track"],
            "created_at": manifest["created_at"],
            "dataset_fingerprint": manifest["dataset"]["fingerprint"],
            "scenario_sha256": manifest["scenario_sha256"],
            "evaluator_image": manifest["images"]["evaluator"],
            "agent_image": manifest["images"]["agent_under_test"],
        },
        "execution": {
            "status": "completed" if complete else (progress or {}).get("status", "incomplete"),
            "complete": complete,
            "official_eligible": complete and not bool(manifest.get("development")),
            "expected_units": expected_units,
            "completed_units": len(records),
            "incomplete_units": expected_units - len(records),
            "attempt_count": total_attempt_count,
            "retry_count": retries,
            "timeout_count": attempt_outcomes["timeout"],
            "evaluator_error_count": attempt_outcomes["error"],
            "container_exit_count": attempt_outcomes["container_exit"],
            "infrastructure_error_count": (
                attempt_outcomes["timeout"]
                + attempt_outcomes["error"]
                + attempt_outcomes["container_exit"]
            ),
        },
        "scores": {
            "primary_metric": "macro_Pass^3",
            "primary_value": (
                pass_scores["macro"].get("Pass^3") if complete else None
            ),
            "pass": pass_scores,
            "task_trial_reward": {
                "score": sum(rewards),
                "max_score": expected_units,
                "completed_max_score": len(records),
                "completed_passes": trial_passes,
                "completed_pass_rate": (
                    trial_passes / len(records) if records else None
                ),
            },
        },
        "latency": {
            "a2a_raw_per_task_trial_ms": {
                "source": "evaluator_measured",
                "reporting_coverage": len(raw_a2a) / len(records) if records else None,
                **_stats(raw_a2a),
            },
            "a2a_raw_per_turn_ms": {
                "source": "evaluator_measured",
                **_stats(raw_turns),
            },
            "a2a_quota_adjusted_per_task_trial_ms": {
                "source": "derived_using_participant_self_reported_quota",
                "reporting_coverage": (
                    len(adjusted_a2a) / len(records) if records else None
                ),
                **_stats(adjusted_a2a),
            },
            "llm_self_reported_per_task_trial_ms": {
                "source": "participant_self_reported",
                "reporting_coverage": (
                    len(llm_latency) / len(records) if records else None
                ),
                **_stats(llm_latency),
            },
            "quota_wait_self_reported_per_task_trial_ms": {
                "source": "participant_self_reported",
                "reporting_coverage": (
                    len(quota_wait) / len(records) if records else None
                ),
                **_stats(quota_wait),
            },
        },
        "participant_telemetry": {
            "source": "participant_self_reported",
            "reported_units": sum(
                bool(
                    record.get("telemetry", {}).get(
                        "complete",
                        record.get("telemetry", {}).get("reported"),
                    )
                )
                for record in records
            ),
            "completed_units": len(records),
            "llm_calls": {
                "reporting_coverage": len(llm_calls) / len(records) if records else None,
                **_stats(llm_calls),
            },
            "cost": {
                "reporting_coverage": len(costs) / len(records) if records else None,
                **_stats(costs),
            },
        },
        "tokens": {
            "source": "participant_self_reported",
            "reported_units": token_reported_units,
            "completed_units": len(records),
            "coverage": token_reported_units / len(records) if records else None,
            "metrics": {
                field: {
                    "reporting_coverage": (
                        token_field_reported_units[field] / len(records)
                        if records
                        else None
                    ),
                    **_stats(values),
                }
                for field, values in token_values.items()
            },
        },
    }

    average_total = summary["tokens"]["metrics"]["total_tokens"]["mean"]
    coverage = summary["tokens"]["coverage"]
    summary["tokens"]["track_2_500k_status"] = (
        "not_applicable"
        if manifest["track"] != "track_2"
        else "unknown"
        if coverage != 1.0 or average_total is None
        else "within_limit"
        if float(average_total) <= 500_000
        else "over_limit"
    )
    return summary


def _analytics_row(record: dict[str, Any]) -> dict[str, Any]:
    """Drop trajectories and task ground truth from the consolidated JSONL."""

    return {
        key: value
        for key, value in record.items()
        if key not in {"trajectory", "task"}
    }


def write_run_derivatives(store, manifest: dict[str, Any]) -> dict[str, Any]:
    records = list(store.iter_units())
    records.sort(
        key=lambda row: (
            CATEGORIES.index(row["unit"]["category"]),
            int(row["unit"]["trial"]),
            manifest["dataset"]["files"][row["unit"]["category"]]["task_ids"].index(
                row["unit"]["task_id"]
            ),
        )
    )
    jsonl_path = store.private_dir / "results.jsonl"
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = jsonl_path.with_name(f".{jsonl_path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(_analytics_row(record), ensure_ascii=False) + "\n")
        handle.flush()
    temporary.replace(jsonl_path)

    progress = store.progress()
    attempts = list(store.iter_attempts())
    summary = build_summary(
        manifest,
        records,
        progress=progress,
        attempts=attempts,
    )
    atomic_write_json(store.private_dir / "summary.json", summary)
    atomic_write_json(store.exports_dir / "summary.json", summary)

    leaderboard_row = {
        "schema_version": 2,
        "run_id": manifest["run_id"],
        "team_id": manifest["team_id"],
        "team_name": manifest["team_name"],
        "track": manifest["track"],
        "complete": summary["execution"]["complete"],
        "official_eligible": summary["execution"]["official_eligible"],
        "primary_metric": "macro_Pass^3",
        "primary_score": summary["scores"]["primary_value"],
        "pass_scores": summary["scores"]["pass"],
        "task_trial_pass_rate": summary["scores"]["task_trial_reward"]["completed_pass_rate"],
        "latency": summary["latency"],
        "tokens": summary["tokens"],
        "retry_count": summary["execution"]["retry_count"],
        "recovery": {
            key: summary["execution"][key]
            for key in (
                "attempt_count",
                "retry_count",
                "timeout_count",
                "evaluator_error_count",
                "container_exit_count",
                "infrastructure_error_count",
            )
        },
        "provenance": {
            "dataset_fingerprint": manifest["dataset"]["fingerprint"],
            "scenario_sha256": manifest["scenario_sha256"],
            "evaluator_image": manifest["images"]["evaluator"],
            "config_fingerprint": manifest["config_fingerprint"],
        },
    }
    if summary["execution"]["official_eligible"]:
        atomic_write_json(store.exports_dir / "leaderboard-row.json", leaderboard_row)
    else:
        leaderboard_path = store.exports_dir / "leaderboard-row.json"
        if leaderboard_path.exists():
            leaderboard_path.unlink()
    return summary


def rank_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ordered = sorted(rows, key=lambda row: (-float(row["primary_score"]), row["team_id"]))
    ranked = []
    previous_score = None
    previous_rank = 0
    for position, row in enumerate(ordered, start=1):
        score = float(row["primary_score"])
        rank = previous_rank if previous_score == score else position
        ranked.append({"rank": rank, **row})
        previous_score = score
        previous_rank = rank
    return ranked


def write_leaderboard(rows: list[dict[str, Any]], output_dir: Path) -> list[dict[str, Any]]:
    ranked = rank_rows(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output_dir / "leaderboard.json", {"schema_version": 2, "rows": ranked})
    with (output_dir / "leaderboard.csv").open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "rank", "team_id", "team_name", "run_id", "track", "primary_metric",
            "primary_score", "task_trial_pass_rate", "retry_count",
            "timeout_count", "error_count", "a2a_mean_ms", "token_mean",
            "token_coverage",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in ranked:
            writer.writerow(
                {
                    "rank": row["rank"],
                    "team_id": row["team_id"],
                    "team_name": row["team_name"],
                    "run_id": row["run_id"],
                    "track": row["track"],
                    "primary_metric": row["primary_metric"],
                    "primary_score": row["primary_score"],
                    "task_trial_pass_rate": row["task_trial_pass_rate"],
                    "retry_count": row["retry_count"],
                    "timeout_count": row.get("recovery", {}).get("timeout_count", 0),
                    "error_count": row.get("recovery", {}).get(
                        "infrastructure_error_count", 0
                    ),
                    "a2a_mean_ms": row["latency"]["a2a_raw_per_task_trial_ms"]["mean"],
                    "token_mean": row["tokens"]["metrics"]["total_tokens"]["mean"],
                    "token_coverage": row["tokens"]["coverage"],
                }
            )
    return ranked
