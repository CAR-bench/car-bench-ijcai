"""Organizer CLI for starting, resuming, and aggregating hidden evaluations."""

from __future__ import annotations

import argparse
import json
import os
import secrets
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import tomllib

from competition.analytics import write_leaderboard, write_run_derivatives
from competition.compose import generate_competition_compose, generate_internal_scenario
from competition.dataset import HiddenDatasetError, load_hidden_dataset
from competition.evaluation import utc_now
from competition.plots import write_standard_plots
from competition.scenario import (
    DIGEST_IMAGE_RE,
    OFFICIAL_EVALUATOR_REPOSITORY,
    ScenarioValidationError,
    canonical_hash,
    load_and_validate_scenario,
    resolve_docker_image,
    sha256_text,
    validate_team_id,
)
from competition.storage import RunLock, RunStorageError, RunStore, atomic_write_json, load_json


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TIMEOUT_SECONDS = 30 * 60
DEFAULT_TOTAL_ATTEMPTS = 3


def _parse_iso(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _new_run_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{uuid4().hex[:8]}"


def _project_name(track: str, team_id: str, run_id: str) -> str:
    short_track = "t1" if track == "track_1" else "t2"
    prefix = f"carbench-{short_track}-"
    suffix = f"-{run_id[-8:]}"
    team_length = 63 - len(prefix) - len(suffix)
    team_component = team_id[:team_length].rstrip("-") or "team"
    return f"{prefix}{team_component}{suffix}"


def _expected_units(dataset, config: dict[str, Any]) -> int:
    counts = 0
    for category in ("base", "hallucination", "disambiguation"):
        available = len(dataset.tasks_by_category[category])
        configured = int(config.get(f"tasks_{category}_num_tasks", -1))
        counts += available if configured == -1 else min(available, max(0, configured))
    return counts * int(config["num_trials"])


def _selected_dataset_manifest(dataset, config: dict[str, Any]) -> dict[str, Any]:
    manifest = json.loads(json.dumps(dataset.manifest))
    for category in ("base", "hallucination", "disambiguation"):
        configured = int(config.get(f"tasks_{category}_num_tasks", -1))
        file_entry = manifest["files"][category]
        if configured != -1:
            file_entry["task_ids"] = file_entry["task_ids"][: max(0, configured)]
            file_entry["selected_row_count"] = len(file_entry["task_ids"])
        else:
            file_entry["selected_row_count"] = file_entry["row_count"]
    return manifest


def _write_runtime_files(
    *,
    store: RunStore,
    manifest: dict[str, Any],
    scenario: dict[str, Any],
    dataset_dir: Path,
) -> None:
    internal = generate_internal_scenario(manifest=manifest, config=manifest["config"])
    (store.root / "a2a-scenario.toml").write_text(internal, encoding="utf-8")
    compose = generate_competition_compose(
        manifest=manifest,
        scenario=scenario,
        run_dir=store.root,
        dataset_dir=dataset_dir,
        project_root=PROJECT_ROOT,
    )
    (store.root / "docker-compose.yml").write_text(compose, encoding="utf-8")


def _compose_base(manifest: dict[str, Any], store: RunStore, env_file: Path) -> list[str]:
    return [
        "docker", "compose",
        "--project-name", manifest["compose_project"],
        "--env-file", str(env_file),
        "-f", str(store.root / "docker-compose.yml"),
    ]


def _compose_environment(control_token: str) -> dict[str, str]:
    return {**os.environ, "CAR_BENCH_ORGANIZER_TOKEN": control_token}


def _compose_down(
    manifest: dict[str, Any],
    store: RunStore,
    env_file: Path,
    control_token: str,
) -> None:
    try:
        subprocess.run(
            [*_compose_base(manifest, store, env_file), "down", "--remove-orphans"],
            check=False,
            env=_compose_environment(control_token),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=60,
        )
    except subprocess.TimeoutExpired:
        pass


def _tee_process_output(process: subprocess.Popen, log_path: Path) -> threading.Thread:
    def copy_output() -> None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as log:
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="")
                log.write(line)
                log.flush()

    thread = threading.Thread(target=copy_output, daemon=True)
    thread.start()
    return thread


def _attempts_since_budget(store: RunStore, progress: dict[str, Any]) -> int:
    unit = progress.get("current_unit")
    if not isinstance(unit, dict):
        return 0
    records = store.attempt_records(unit["category"], unit["task_id"], int(unit["trial"]))
    return max(0, len(records) - int(progress.get("attempt_budget_start", 0)))


def _record_host_failure(
    store: RunStore,
    *,
    outcome: str,
    error: str,
) -> bool:
    progress = store.progress()
    unit = progress.get("current_unit")
    if not isinstance(unit, dict):
        return False
    attempt = int(progress.get("current_attempt") or store.next_attempt_number(
        unit["category"], unit["task_id"], int(unit["trial"])
    ))
    existing = store.attempt_records(unit["category"], unit["task_id"], int(unit["trial"]))
    if any(int(record.get("attempt", 0)) == attempt for record in existing):
        return False
    completed_at = utc_now()
    started_at = progress.get("current_unit_started_at") or completed_at
    started = _parse_iso(started_at)
    completed = _parse_iso(completed_at)
    duration = max(0.0, (completed - started).total_seconds()) if started and completed else None
    store.write_attempt(
        {
            "schema_version": 2,
            "unit": unit,
            "attempt": attempt,
            "trial_seed": progress.get("current_trial_seed"),
            "started_at": started_at,
            "completed_at": completed_at,
            "duration_seconds": duration,
            "outcome": outcome,
            "error": error,
            "source": "organizer_runner",
        }
    )
    progress.update(
        {
            "status": "recovering",
            "last_error": error,
            "updated_at": completed_at,
        }
    )
    store.update_progress(progress)
    return True


def _run_compose_until_terminal(
    store: RunStore,
    manifest: dict[str, Any],
    env_file: Path,
    control_token: str,
) -> int:
    timeout_seconds = int(manifest["recovery"]["task_timeout_seconds"])
    total_attempts = int(manifest["recovery"]["total_attempts"])
    startup_failures = 0

    def finalize_completed() -> None:
        summary = write_run_derivatives(store, manifest)
        if not summary.get("execution", {}).get("complete"):
            raise RunStorageError(
                "Run status is completed but canonical task-trial units are incomplete"
            )

    while True:
        progress = store.progress()
        if progress.get("status") == "completed":
            finalize_completed()
            print(f"Evaluation already complete: {store.root}")
            return 0
        if progress.get("status") == "paused":
            print(f"Evaluation is paused for review: {store.root}", file=sys.stderr)
            return 2

        command = [
            *_compose_base(manifest, store, env_file),
            "up", "--abort-on-container-exit", "--exit-code-from", "a2a-client",
        ]
        process = subprocess.Popen(
            command,
            env=_compose_environment(control_token),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        output_thread = _tee_process_output(
            process, store.private_dir / "logs" / "docker-compose.log"
        )
        timed_out = False
        try:
            while process.poll() is None:
                time.sleep(2)
                try:
                    progress = store.progress()
                except RunStorageError:
                    continue
                started = _parse_iso(progress.get("current_unit_started_at"))
                if started is None or not isinstance(progress.get("current_unit"), dict):
                    continue
                elapsed = (datetime.now(timezone.utc) - started).total_seconds()
                if elapsed > timeout_seconds:
                    timed_out = True
                    print(
                        f"Task-trial exceeded {timeout_seconds}s; restarting from persisted state.",
                        file=sys.stderr,
                    )
                    _compose_down(manifest, store, env_file, control_token)
                    break
            try:
                return_code = process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.terminate()
                return_code = process.wait(timeout=15)
        except KeyboardInterrupt:
            _compose_down(manifest, store, env_file, control_token)
            progress = store.progress()
            progress.update({"status": "interrupted", "updated_at": utc_now()})
            store.update_progress(progress)
            return 130
        finally:
            _compose_down(manifest, store, env_file, control_token)
            output_thread.join(timeout=5)

        if timed_out:
            _record_host_failure(
                store,
                outcome="timeout",
                error=f"task-trial timeout after {timeout_seconds} seconds",
            )
        else:
            progress = store.progress()
            if progress.get("status") == "completed":
                finalize_completed()
                print(f"Evaluation completed: {store.root}")
                return 0
            if isinstance(progress.get("current_unit"), dict):
                _record_host_failure(
                    store,
                    outcome="container_exit",
                    error=f"Docker Compose exited with code {return_code}",
                )
            else:
                startup_failures += 1
                progress.update(
                    {
                        "status": "recovering",
                        "last_error": f"Docker stack exited before a task started (code {return_code})",
                        "updated_at": utc_now(),
                    }
                )
                store.update_progress(progress)

        progress = store.progress()
        if isinstance(progress.get("current_unit"), dict):
            if _attempts_since_budget(store, progress) >= total_attempts:
                progress.update(
                    {
                        "status": "paused",
                        "last_error": progress.get("last_error") or "recovery budget exhausted",
                        "updated_at": utc_now(),
                    }
                )
                store.update_progress(progress)
                print(f"Evaluation paused after {total_attempts} attempts: {store.root}", file=sys.stderr)
                return 2
        elif startup_failures >= total_attempts:
            progress.update(
                {
                    "status": "paused",
                    "last_error": "Docker stack startup recovery budget exhausted",
                    "updated_at": utc_now(),
                }
            )
            store.update_progress(progress)
            print(f"Evaluation paused after repeated Docker startup failures: {store.root}", file=sys.stderr)
            return 2


def start_run(args: argparse.Namespace) -> int:
    team_id = validate_team_id(args.team_id)
    env_file = args.env_file.expanduser().resolve()
    if not env_file.is_file():
        raise ScenarioValidationError(f"Environment file not found: {env_file}")
    scenario, scenario_text = load_and_validate_scenario(
        args.scenario, development=args.development
    )
    dataset = load_hidden_dataset(args.dataset_dir)
    evaluator_image = resolve_docker_image(scenario["evaluator"]["image"])
    agent_image = resolve_docker_image(scenario["agent_under_test"]["image"])
    run_id = _new_run_id()
    results_root = args.results_root.expanduser().resolve()
    run_dir = results_root / args.track / team_id / run_id
    selected_dataset = _selected_dataset_manifest(dataset, scenario["config"])
    created_at = utc_now()
    manifest = {
        "schema_version": 2,
        "run_id": run_id,
        "team_id": team_id,
        "team_name": args.team_name or team_id,
        "track": args.track,
        "created_at": created_at,
        "development": bool(args.development),
        "base_seed": int(args.base_seed),
        "scenario_sha256": sha256_text(scenario_text),
        "config": scenario["config"],
        "config_fingerprint": canonical_hash(scenario["config"]),
        "dataset": selected_dataset,
        "dataset_source_path": str(dataset.root),
        "images": {
            "evaluator": evaluator_image,
            "agent_under_test": agent_image,
        },
        "agent_metadata": {
            key: scenario["agent_under_test"][key]
            for key in ("name", "result_label", "result_model", "result_reasoning_effort")
            if key in scenario["agent_under_test"]
        },
        "expected_units": _expected_units(dataset, scenario["config"]),
        "recovery": {
            "task_timeout_seconds": int(args.task_timeout_seconds),
            "total_attempts": int(args.retries) + 1,
        },
        "compose_project": _project_name(args.track, team_id, run_id),
        "env_file_path": str(env_file),
        "runtime": {
            "host_uid": os.getuid(),
            "host_gid": os.getgid(),
        },
    }
    store = RunStore(run_dir)
    store.initialize(manifest, scenario_text)
    _write_runtime_files(
        store=store,
        manifest=manifest,
        scenario=scenario,
        dataset_dir=dataset.root,
    )
    print(f"Created competition run: {run_dir}")
    with RunLock(run_dir):
        return _run_compose_until_terminal(
            store,
            manifest,
            env_file,
            secrets.token_urlsafe(32),
        )


def _grant_paused_retry(store: RunStore) -> None:
    progress = store.progress()
    if progress.get("status") != "paused":
        return
    unit = progress.get("current_unit")
    if isinstance(unit, dict):
        attempts = store.attempt_records(unit["category"], unit["task_id"], int(unit["trial"]))
        progress["attempt_budget_start"] = len(attempts)
    else:
        progress["attempt_budget_start"] = 0
    progress.setdefault("operator_events", []).append(
        {"event": "retry_paused", "at": utc_now(), "previous_error": progress.get("last_error")}
    )
    progress.update({"status": "recovering", "last_error": None, "updated_at": utc_now()})
    store.update_progress(progress)


def _validate_image_provenance(
    manifest: dict[str, Any], scenario: dict[str, Any]
) -> None:
    images = manifest.get("images")
    if not isinstance(images, dict):
        raise RunStorageError("Run manifest has no image provenance")
    for role in ("evaluator", "agent_under_test"):
        record = images.get(role)
        configured = scenario.get(role, {}).get("image")
        if not isinstance(record, dict):
            raise RunStorageError(f"Run manifest has invalid {role} image provenance")
        if record.get("requested") != configured:
            raise RunStorageError(f"Saved {role} image differs from the manifest")
        resolved = record.get("resolved")
        if not isinstance(resolved, str) or not DIGEST_IMAGE_RE.fullmatch(resolved):
            raise RunStorageError(f"Saved {role} image is not resolved to a digest")
        if not isinstance(record.get("image_id"), str):
            raise RunStorageError(f"Saved {role} image ID is missing")
    evaluator_resolved = images["evaluator"]["resolved"]
    if not evaluator_resolved.startswith(f"{OFFICIAL_EVALUATOR_REPOSITORY}@sha256:"):
        raise RunStorageError("Saved evaluator digest is not from the official repository")


def resume_run(args: argparse.Namespace) -> int:
    store = RunStore(args.run_dir)
    manifest = store.manifest()
    scenario_path = store.root / "scenario.toml"
    scenario_text = scenario_path.read_text(encoding="utf-8")
    if sha256_text(scenario_text) != manifest["scenario_sha256"]:
        raise RunStorageError("Saved scenario differs from the immutable manifest")
    if canonical_hash(manifest["config"]) != manifest["config_fingerprint"]:
        raise RunStorageError("Saved config fingerprint is invalid")
    scenario, _ = load_and_validate_scenario(
        scenario_path, development=bool(manifest.get("development"))
    )
    _validate_image_provenance(manifest, scenario)
    dataset_dir = (
        args.dataset_dir.expanduser().resolve()
        if args.dataset_dir
        else Path(manifest["dataset_source_path"]).expanduser().resolve()
    )
    dataset = load_hidden_dataset(dataset_dir)
    if dataset.fingerprint != manifest["dataset"]["fingerprint"]:
        raise RunStorageError("Hidden dataset changed; refusing to resume the run")
    env_file = (
        args.env_file.expanduser().resolve()
        if args.env_file
        else Path(manifest["env_file_path"]).expanduser().resolve()
    )
    if not env_file.is_file():
        raise RunStorageError(f"Environment file not found: {env_file}")
    _write_runtime_files(
        store=store,
        manifest=manifest,
        scenario=scenario,
        dataset_dir=dataset_dir,
    )
    with RunLock(store.root):
        progress = store.progress()
        if progress.get("status") == "paused" and not args.retry_paused:
            raise RunStorageError(
                "Run is paused for review; pass --retry-paused after resolving the cause"
            )
        if args.retry_paused:
            _grant_paused_retry(store)
        return _run_compose_until_terminal(
            store,
            manifest,
            env_file,
            secrets.token_urlsafe(32),
        )


def _legacy_row(path: Path, payload: dict[str, Any], track: str) -> dict[str, Any] | None:
    final = payload.get("final_result")
    metadata = payload.get("metadata")
    if not isinstance(final, dict) or not isinstance(metadata, dict):
        return None
    score = final.get("pass_power_k_scores", {}).get("Pass^3")
    if score is None:
        return None
    team_name = str(metadata.get("agent_name") or path.parent.name)
    team_id = "".join(character.lower() if character.isalnum() else "-" for character in team_name).strip("-") or "legacy"
    macro = {
        **final.get("pass_power_k_scores", {}),
        **final.get("pass_at_k_scores", {}),
    }
    by_category = {}
    for category in ("base", "hallucination", "disambiguation"):
        by_category[category] = {
            **final.get("pass_power_k_scores_by_split", {}).get(category, {}),
            **final.get("pass_at_k_scores_by_split", {}).get(category, {}),
        }
    return {
        "schema_version": 1,
        "run_id": path.stem,
        "team_id": team_id,
        "team_name": team_name,
        "track": track,
        "complete": True,
        "primary_metric": "macro_Pass^3",
        "primary_score": score,
        "pass_scores": {
            "macro": macro,
            "by_category": by_category,
        },
        "task_trial_pass_rate": final.get("pass_rate", 0) / 100,
        "latency": {"a2a_raw_per_task_trial_ms": {"mean": None}},
        "tokens": {"coverage": None, "metrics": {"total_tokens": {"mean": None}}},
        "retry_count": 0,
        "recovery": {
            "attempt_count": 0,
            "retry_count": 0,
            "timeout_count": 0,
            "evaluator_error_count": 0,
            "container_exit_count": 0,
            "infrastructure_error_count": 0,
        },
        "provenance": {"legacy": True},
    }


def aggregate_runs(args: argparse.Namespace) -> int:
    root = args.results_root.expanduser().resolve()
    rows: list[dict[str, Any]] = []
    manifests = list(root.glob(f"{args.track}/*/*/manifest.json"))
    latest_manifests: dict[
        str, tuple[str, str, Path, dict[str, Any]]
    ] = {}
    for path in manifests:
        manifest = load_json(path)
        if manifest.get("schema_version") != 2:
            continue
        team_id = str(manifest.get("team_id") or path.parents[1].name)
        run_id = str(manifest.get("run_id") or path.parent.name)
        created_at = str(manifest.get("created_at") or "")
        if (
            manifest.get("track") != args.track
            or team_id != path.parents[1].name
            or run_id != path.parent.name
        ):
            raise RunStorageError(f"Run manifest identity does not match its path: {path}")
        current = latest_manifests.get(team_id)
        if current is None or (created_at, run_id) > (current[0], current[1]):
            latest_manifests[team_id] = (created_at, run_id, path, manifest)

    incomplete: list[str] = []
    for team_id, (_, run_id, manifest_path, manifest) in latest_manifests.items():
        run_dir = manifest_path.parent
        summary_path = run_dir / "exports" / "summary.json"
        row_path = run_dir / "exports" / "leaderboard-row.json"
        if not summary_path.is_file() or not row_path.is_file():
            incomplete.append(f"{team_id}/{run_id}")
            continue
        summary = load_json(summary_path)
        row = load_json(row_path)
        if not summary.get("execution", {}).get("official_eligible"):
            incomplete.append(f"{team_id}/{run_id}")
            continue
        if (
            row.get("run_id") != manifest.get("run_id")
            or row.get("team_id") != manifest.get("team_id")
            or row.get("track") != args.track
            or not row.get("official_eligible")
        ):
            raise RunStorageError(
                f"Leaderboard export does not match its manifest: {row_path}"
            )
        expected_provenance = {
            "dataset_fingerprint": manifest["dataset"]["fingerprint"],
            "scenario_sha256": manifest["scenario_sha256"],
            "evaluator_image": manifest["images"]["evaluator"],
            "config_fingerprint": manifest["config_fingerprint"],
        }
        if row.get("provenance") != expected_provenance:
            raise RunStorageError(
                f"Leaderboard export provenance does not match its manifest: {row_path}"
            )
        row["_path"] = str(row_path)
        rows.append(row)

    if incomplete:
        raise RunStorageError(
            "Latest runs are incomplete or ineligible: " + ", ".join(sorted(incomplete))
        )

    if not latest_manifests:
        for path in root.glob(f"{args.track}/*/*/exports/leaderboard-row.json"):
            row = load_json(path)
            if row.get("official_eligible", row.get("complete")) and row.get("track") == args.track:
                row["_path"] = str(path)
                rows.append(row)

    if not rows and args.include_legacy and not latest_manifests:
        for path in root.rglob("*.json"):
            try:
                payload = load_json(path)
            except RunStorageError:
                continue
            legacy_row = _legacy_row(path, payload, args.track)
            if legacy_row:
                legacy_row["_path"] = str(path)
                rows.append(legacy_row)
    if not rows:
        raise RunStorageError(f"No complete {args.track} leaderboard rows found under {root}")

    latest_by_team: dict[str, dict[str, Any]] = {}
    for row in rows:
        team_id = row["team_id"]
        current_row = latest_by_team.get(team_id)
        if current_row is None or row["run_id"] > current_row["run_id"]:
            latest_by_team[team_id] = row
    rows = list(latest_by_team.values())

    v2_rows = [row for row in rows if row.get("schema_version") == 2]
    if v2_rows:
        compatibility = {
            json.dumps(
                {
                    "dataset": row["provenance"]["dataset_fingerprint"],
                    "evaluator": row["provenance"]["evaluator_image"],
                    "config": row["provenance"]["config_fingerprint"],
                },
                sort_keys=True,
            )
            for row in v2_rows
        }
        if len(compatibility) != 1:
            raise RunStorageError(
                "Complete runs use incompatible dataset, evaluator, or config fingerprints"
            )
        if len(v2_rows) != len(rows):
            raise RunStorageError("Legacy and schema-v2 results cannot be mixed in one leaderboard")

    for row in rows:
        row.pop("_path", None)
    output_dir = root / args.track / "aggregate"
    ranked = write_leaderboard(rows, output_dir)
    write_standard_plots(ranked, output_dir / "plots")
    print(f"Wrote leaderboard and plots to {output_dir}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="car-bench-competition",
        description="Run and aggregate resumable hidden CAR-bench evaluations.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    start = subparsers.add_parser("start", help="Start a new team evaluation")
    start.add_argument("scenario", type=Path)
    start.add_argument("--track", choices=("track_1", "track_2"), required=True)
    start.add_argument("--team-id", required=True)
    start.add_argument("--team-name")
    start.add_argument("--dataset-dir", type=Path, default=Path("hidden_dataset"))
    start.add_argument("--results-root", type=Path, default=Path("competition_results"))
    start.add_argument("--env-file", type=Path, default=Path(".env"))
    start.add_argument("--task-timeout-seconds", type=int, default=DEFAULT_TIMEOUT_SECONDS)
    start.add_argument("--retries", type=int, default=2)
    start.add_argument("--base-seed", type=int, default=10)
    start.add_argument("--development", action="store_true")
    start.set_defaults(func=start_run)

    resume = subparsers.add_parser("resume", help="Resume an existing run")
    resume.add_argument("run_dir", type=Path)
    resume.add_argument("--dataset-dir", type=Path)
    resume.add_argument("--env-file", type=Path)
    resume.add_argument("--retry-paused", action="store_true")
    resume.set_defaults(func=resume_run)

    aggregate = subparsers.add_parser("aggregate", help="Build a track leaderboard")
    aggregate.add_argument("--results-root", type=Path, default=Path("competition_results"))
    aggregate.add_argument("--track", choices=("track_1", "track_2"), required=True)
    aggregate.add_argument("--include-legacy", action="store_true")
    aggregate.set_defaults(func=aggregate_runs)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        if getattr(args, "task_timeout_seconds", 1) < 1:
            raise ScenarioValidationError("task timeout must be at least one second")
        if getattr(args, "retries", 0) < 0:
            raise ScenarioValidationError("retries must not be negative")
        raise SystemExit(args.func(args))
    except (
        HiddenDatasetError,
        ScenarioValidationError,
        RunStorageError,
        subprocess.CalledProcessError,
        OSError,
    ) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
