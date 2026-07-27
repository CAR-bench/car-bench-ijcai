"""Resume-aware evaluator loop used inside the official evaluator container."""

from __future__ import annotations

import hashlib
import os
from typing import Any, Callable

from a2a.helpers.proto_helpers import new_data_part, new_text_message, new_text_part
from a2a.types import TaskState

from competition.analytics import CATEGORIES, write_run_derivatives
from competition.dataset import install_hidden_task_loader, load_hidden_dataset
from competition.evaluation import normalize_unit_record, timed_single_task_trial, utc_now
from competition.storage import RunStorageError, RunStore


class CompetitionEvaluationPaused(RuntimeError):
    """Raised when a unit exhausted its automatic recovery budget."""


def _trial_seed(
    base_seed: int,
    dataset_fingerprint: str,
    category: str,
    task_id: str,
    trial: int,
) -> int:
    value = f"{base_seed}:{dataset_fingerprint}:{category}:{task_id}:{trial}"
    return int(hashlib.sha256(value.encode("utf-8")).hexdigest()[:8], 16)


def _unit_identity(dataset_fingerprint: str, category: str, task_id: str, trial: int):
    return {
        "dataset_fingerprint": dataset_fingerprint,
        "category": category,
        "task_id": task_id,
        "trial": int(trial),
    }


def _same_unit(left: Any, right: Any) -> bool:
    if not isinstance(left, dict) or not isinstance(right, dict):
        return False
    return all(left.get(key) == right.get(key) for key in ("category", "task_id", "trial"))


def _validate_manifest(req, manifest: dict[str, Any], dataset) -> None:
    context = req.run_context
    assert context is not None
    checks = {
        "run_id": context.run_id,
        "team_id": context.team_id,
        "team_name": context.team_name,
        "track": context.track,
    }
    for field, expected in checks.items():
        if manifest.get(field) != expected:
            raise RunStorageError(
                f"Run manifest mismatch for {field}: {manifest.get(field)!r} != {expected!r}"
            )
    if manifest.get("config") != req.config:
        raise RunStorageError("Run config differs from the immutable manifest")
    manifest_fingerprint = manifest.get("dataset", {}).get("fingerprint")
    if manifest_fingerprint != dataset.fingerprint:
        raise RunStorageError(
            "Hidden dataset fingerprint differs from the immutable run manifest"
        )
    if context.dataset_fingerprint != dataset.fingerprint:
        raise RunStorageError("Evaluation request contains the wrong dataset fingerprint")
    for category in CATEGORIES:
        expected_ids = manifest["dataset"]["files"][category]["task_ids"]
        available_ids = dataset.task_ids(category)
        if expected_ids != available_ids[: len(expected_ids)]:
            raise RunStorageError(f"Hidden task order changed for {category}")


async def run_competition_evaluation(
    *,
    req,
    updater,
    logger,
    agent_factory_builder: Callable[[str], Callable[..., Any]],
    args_builder: Callable[[dict[str, Any], str], Any],
) -> dict[str, Any]:
    """Execute or resume every hidden task-trial and persist it immediately."""

    context = req.run_context
    if context is None:
        raise ValueError("Missing competition run context")
    if req.config.get("task_split") != "hidden":
        raise ValueError("Competition run context is only valid for task_split='hidden'")

    data_dir = os.environ.get("CAR_BENCH_HIDDEN_DATA_DIR", "/hidden-dataset")
    configured_run_dir = os.environ.get("CAR_BENCH_RUN_DIR", context.run_dir)
    if configured_run_dir != context.run_dir:
        raise RunStorageError("Run directory environment does not match the request")

    dataset = load_hidden_dataset(data_dir)
    install_hidden_task_loader(dataset)
    store = RunStore(configured_run_dir)
    manifest = store.manifest()
    _validate_manifest(req, manifest, dataset)
    max_attempts = int(manifest.get("recovery", {}).get("total_attempts", 3))
    if max_attempts < 1:
        raise RunStorageError("Recovery total_attempts must be at least one")

    progress = store.progress()
    write_run_derivatives(store, manifest)
    agent_factory = agent_factory_builder(str(req.agent_under_test))
    expected_units = int(manifest["expected_units"])

    for category in CATEGORIES:
        args = args_builder(req.config, category)
        args.max_steps = int(req.config["max_steps"])
        task_ids = list(manifest["dataset"]["files"][category]["task_ids"])
        for trial in range(int(req.config["num_trials"])):
            for selected_index, task_id in enumerate(task_ids):
                task_index = dataset.index_for(category, task_id)
                existing = store.load_unit(category, task_id, trial)
                if existing is not None:
                    continue

                unit = _unit_identity(dataset.fingerprint, category, task_id, trial)
                progress = store.progress()
                attempts = store.attempt_records(category, task_id, trial)
                budget_start = (
                    int(progress.get("attempt_budget_start", 0))
                    if _same_unit(progress.get("current_unit"), unit)
                    else 0
                )
                attempts_in_budget = max(0, len(attempts) - budget_start)
                if attempts_in_budget >= max_attempts:
                    progress.update(
                        {
                            "status": "paused",
                            "current_unit": unit,
                            "last_error": "automatic recovery budget exhausted",
                            "updated_at": utc_now(),
                        }
                    )
                    store.update_progress(progress)
                    write_run_derivatives(store, manifest)
                    raise CompetitionEvaluationPaused(
                        f"Paused at {category}/{task_id}/trial-{trial}: "
                        "automatic recovery budget exhausted"
                    )

                attempt = store.next_attempt_number(category, task_id, trial)
                started_at = utc_now()
                trial_seed = _trial_seed(
                    int(context.base_seed), dataset.fingerprint, category, task_id, trial
                )
                progress.update(
                    {
                        "status": "running",
                        "current_unit": unit,
                        "current_attempt": attempt,
                        "current_trial_seed": trial_seed,
                        "current_unit_started_at": started_at,
                        "attempt_budget_start": budget_start,
                        "last_error": None,
                        "updated_at": started_at,
                    }
                )
                store.update_progress(progress)
                await updater.update_status(
                    TaskState.TASK_STATE_WORKING,
                    new_text_message(
                        f"Running {category} task {selected_index + 1}/{len(task_ids)}, "
                        f"trial {trial + 1}/{req.config['num_trials']}"
                    ),
                )

                result, unit_started, completed_at, duration_seconds = timed_single_task_trial(
                    args=args,
                    task_index=task_index,
                    task_id=task_id,
                    trial=trial,
                    trial_seed=trial_seed,
                    agent_factory=agent_factory,
                )
                error = (result.info or {}).get("error")
                if error:
                    attempt_record = {
                        "schema_version": 2,
                        "unit": unit,
                        "attempt": attempt,
                        "trial_seed": trial_seed,
                        "started_at": unit_started,
                        "completed_at": completed_at,
                        "duration_seconds": round(duration_seconds, 6),
                        "outcome": "error",
                        "error": str(error),
                        "traceback": (result.info or {}).get("traceback"),
                    }
                    store.write_attempt(attempt_record)
                    progress.update(
                        {
                            "status": "recovering",
                            "current_unit": unit,
                            "current_attempt": attempt,
                            "last_error": str(error),
                            "updated_at": completed_at,
                        }
                    )
                    store.update_progress(progress)
                    logger.error(
                        "Competition task-trial failed and will be recovered",
                        category=category,
                        task_id=task_id,
                        trial=trial,
                        attempt=attempt,
                        error=str(error),
                    )
                    raise RuntimeError(
                        f"Task-trial failed: {category}/{task_id}/trial-{trial}: {error}"
                    )

                unit_record = normalize_unit_record(
                    result=result,
                    dataset_fingerprint=dataset.fingerprint,
                    category=category,
                    task_id=task_id,
                    trial=trial,
                    trial_seed=trial_seed,
                    attempt_count=attempt,
                    started_at=unit_started,
                    completed_at=completed_at,
                    duration_seconds=duration_seconds,
                )
                store.write_unit(unit_record)
                store.write_attempt(
                    {
                        "schema_version": 2,
                        "unit": unit,
                        "attempt": attempt,
                        "trial_seed": trial_seed,
                        "started_at": unit_started,
                        "completed_at": completed_at,
                        "duration_seconds": round(duration_seconds, 6),
                        "outcome": "completed",
                        "reward": unit_record["reward"],
                    }
                )
                completed_units = len(list(store.iter_units()))
                progress.update(
                    {
                        "status": "running",
                        "current_unit": None,
                        "current_attempt": None,
                        "current_trial_seed": None,
                        "current_unit_started_at": None,
                        "attempt_budget_start": 0,
                        "completed_units": completed_units,
                        "expected_units": expected_units,
                        "last_error": None,
                        "updated_at": completed_at,
                    }
                )
                store.update_progress(progress)
                write_run_derivatives(store, manifest)

    progress = store.progress()
    progress.update(
        {
            "status": "completed",
            "current_unit": None,
            "current_attempt": None,
            "current_trial_seed": None,
            "current_unit_started_at": None,
            "completed_units": expected_units,
            "updated_at": utc_now(),
        }
    )
    store.update_progress(progress)
    summary = write_run_derivatives(store, manifest)
    await updater.add_artifact(
        parts=[
            new_text_part(
                "CAR-bench hidden evaluation completed: "
                f"macro Pass^3={summary['scores']['primary_value']:.6f}"
            ),
            new_data_part(summary),
        ],
        name="Result",
    )
    return summary
