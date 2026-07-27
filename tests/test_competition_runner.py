import asyncio
import csv
import json
import os
import tempfile
import tomllib
import unittest
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from agentbeats.client_cli import parse_toml as parse_client_toml
from competition.analytics import build_summary, rank_rows, write_run_derivatives
from competition.auth import OrganizerTokenMiddleware
from competition.cli import (
    _grant_paused_retry,
    _legacy_row,
    _project_name,
    _validate_image_provenance,
    aggregate_runs,
)
from competition.compose import generate_competition_compose, generate_internal_scenario
from competition.dataset import (
    HiddenDatasetError,
    install_hidden_task_loader,
    load_hidden_dataset,
)
from competition.evaluation import execute_orchestrator, normalize_unit_record
from competition.evaluator_run import _trial_seed
from competition.scenario import (
    ScenarioValidationError,
    load_and_validate_scenario,
)
from competition.storage import RunLock, RunStorageError, RunStore, atomic_write_json


CSV_FIELDS = [
    "task_id",
    "calendar_id",
    "actions",
    "persona",
    "instruction",
    "context_init_config",
    "task_type",
    "split",
    "disambiguation_element_internal",
    "disambiguation_element_user",
    "disambiguation_element_note",
    "removed_part",
]


def _dataset_rows():
    return {
        "hidden_base.csv": [
            {
                "task_id": "b_1",
                "calendar_id": "cal_1",
                "actions": "[]",
                "persona": "persona",
                "instruction": "instruction",
                "context_init_config": "{}",
                "task_type": "base",
                "split": "",
                "removed_part": "",
            }
        ],
        "hidden_hallucination.csv": [
            {
                "task_id": "h_1",
                "calendar_id": "cal_2",
                "actions": "[]",
                "persona": "persona",
                "instruction": "instruction",
                "context_init_config": "{}",
                "task_type": "hallucination_missing_tool",
                "split": "hidden",
                "removed_part": '["missing_tool"]',
            }
        ],
        "hidden_disambiguation.csv": [
            {
                "task_id": "d_1",
                "calendar_id": "cal_3",
                "actions": "[]",
                "persona": "persona",
                "instruction": "instruction",
                "context_init_config": "{}",
                "task_type": "disambiguation_user",
                "split": "hidden",
                "removed_part": "",
            }
        ],
    }


def _write_dataset(root: Path, mutate=None):
    rows_by_file = _dataset_rows()
    if mutate:
        mutate(rows_by_file)
    for filename, rows in rows_by_file.items():
        with (root / filename).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow(row)


def _scenario_text(extra_agent="", image=None):
    agent_image = image or f"ghcr.io/example/team@sha256:{'a' * 64}"
    return f"""
[evaluator]
image = "ghcr.io/car-bench/car-bench-evaluator:latest"

[evaluator.env]
GEMINI_API_KEY = "${{GEMINI_API_KEY:?required}}"

[agent_under_test]
image = "{agent_image}"
{extra_agent}

[agent_under_test.env]
TEAM_API_KEY = "${{TEAM_API_KEY:?required}}"

[config]
num_trials = 3
task_split = "hidden"
tasks_base_num_tasks = -1
tasks_hallucination_num_tasks = -1
tasks_disambiguation_num_tasks = -1
max_steps = 50
""".strip()


def _manifest(dataset, root: Path):
    return {
        "schema_version": 2,
        "run_id": "20260722T120000Z-12345678",
        "team_id": "team-1",
        "team_name": "Team One",
        "track": "track_2",
        "created_at": "2026-07-22T12:00:00+00:00",
        "development": False,
        "base_seed": 10,
        "scenario_sha256": "scenario-hash",
        "config": {
            "num_trials": 3,
            "task_split": "hidden",
            "tasks_base_num_tasks": -1,
            "tasks_hallucination_num_tasks": -1,
            "tasks_disambiguation_num_tasks": -1,
            "max_steps": 50,
        },
        "config_fingerprint": "config-hash",
        "dataset": dataset.manifest,
        "dataset_source_path": str(dataset.root),
        "images": {
            "evaluator": {
                "requested": "ghcr.io/car-bench/car-bench-evaluator:latest",
                "resolved": f"ghcr.io/car-bench/car-bench-evaluator@sha256:{'b' * 64}",
                "image_id": f"sha256:{'c' * 64}",
            },
            "agent_under_test": {
                "requested": f"ghcr.io/example/team@sha256:{'a' * 64}",
                "resolved": f"ghcr.io/example/team@sha256:{'a' * 64}",
                "image_id": f"sha256:{'d' * 64}",
            },
        },
        "agent_metadata": {},
        "expected_units": 9,
        "recovery": {"task_timeout_seconds": 1800, "total_attempts": 3},
        "compose_project": "carbench-t2-team-1-12345678",
        "env_file_path": str(root / ".env"),
        "runtime": {"host_uid": 1000, "host_gid": 1000},
    }


def _unit(dataset, category, task_id, trial, passed, telemetry=True):
    return {
        "schema_version": 2,
        "unit": {
            "dataset_fingerprint": dataset.fingerprint,
            "category": category,
            "task_id": task_id,
            "trial": trial,
        },
        "status": "completed",
        "attempt_count": 1,
        "trial_seed": trial,
        "started_at": "2026-07-22T12:00:00+00:00",
        "completed_at": "2026-07-22T12:00:01+00:00",
        "duration_seconds": 1,
        "reward": 1.0 if passed else 0.0,
        "passed": passed,
        "reward_info": {"private": "reward detail"},
        "task": {"instruction": "HIDDEN INSTRUCTION"},
        "trajectory": [{"role": "user", "content": "HIDDEN TRAJECTORY"}],
        "error": None,
        "timing": {
            "a2a_raw": {"source": "evaluator_measured", "turns_ms": [100.0], "total_ms": 100.0},
            "a2a_quota_adjusted": {"turns_ms": [90.0], "total_ms": 90.0},
            "llm_self_reported_ms": 80.0 if telemetry else None,
            "quota_wait_self_reported_ms": 10.0 if telemetry else None,
        },
        "telemetry": {
            "source": "participant_self_reported",
            "reported": telemetry,
            "tokens": {
                "reported": telemetry,
                "prompt_tokens": 10 if telemetry else None,
                "completion_tokens": 4 if telemetry else None,
                "thinking_tokens": 1 if telemetry else None,
                "total_tokens": 15 if telemetry else None,
            },
        },
        "evaluator": {},
    }


class CompetitionRunnerTest(unittest.TestCase):
    def test_hidden_loader_accepts_optional_split_and_extra_columns(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _write_dataset(root)
            base_row = _dataset_rows()["hidden_base.csv"][0]
            base_fields = [field for field in CSV_FIELDS if field != "split"] + [
                "organizer_note"
            ]
            with (root / "hidden_base.csv").open(
                "w", encoding="utf-8", newline=""
            ) as handle:
                writer = csv.DictWriter(handle, fieldnames=base_fields)
                writer.writeheader()
                writer.writerow(
                    {
                        **{key: value for key, value in base_row.items() if key != "split"},
                        "organizer_note": "ignored",
                    }
                )
            dataset = load_hidden_dataset(root)
            self.assertEqual(dataset.task_ids("base"), ["b_1"])
            self.assertEqual(dataset.manifest["files"]["base"]["row_count"], 1)
            self.assertEqual(len(dataset.fingerprint), 64)

    def test_hidden_loader_rejects_duplicate_and_wrong_category(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)

            def duplicate(rows):
                rows["hidden_hallucination.csv"][0]["task_id"] = "b_1"

            _write_dataset(root, duplicate)
            with self.assertRaisesRegex(HiddenDatasetError, "duplicate"):
                load_hidden_dataset(root)

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)

            def wrong_type(rows):
                rows["hidden_base.csv"][0]["task_type"] = "disambiguation_user"

            _write_dataset(root, wrong_type)
            with self.assertRaisesRegex(HiddenDatasetError, "does not belong"):
                load_hidden_dataset(root)

    def test_hidden_loader_rejects_missing_file_and_malformed_json(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _write_dataset(root)
            (root / "hidden_base.csv").unlink()
            with self.assertRaisesRegex(HiddenDatasetError, "Missing"):
                load_hidden_dataset(root)

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)

            def malformed(rows):
                rows["hidden_hallucination.csv"][0]["actions"] = "not-json"

            _write_dataset(root, malformed)
            with self.assertRaisesRegex(HiddenDatasetError, "hidden_hallucination.csv:2"):
                load_hidden_dataset(root)

    def test_hidden_loader_hook_never_calls_public_loader_for_hidden(self):
        from competition import dataset as dataset_module

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _write_dataset(root)
            dataset = load_hidden_dataset(root)
            public_tasks = [object()]
            upstream = mock.Mock(return_value=public_tasks)
            loader_module = SimpleNamespace(_load_tasks=upstream)
            with (
                mock.patch.object(dataset_module, "_ORIGINAL_TASK_LOADER", None),
                mock.patch.object(dataset_module, "_INSTALLED_DATASET", None),
            ):
                install_hidden_task_loader(dataset, loader_module=loader_module)
                self.assertEqual(
                    loader_module._load_tasks("base", "hidden"),
                    dataset.tasks_by_category["base"],
                )
                upstream.assert_not_called()
                self.assertIs(loader_module._load_tasks("base", "test"), public_tasks)
                upstream.assert_called_once_with("base", "test")

    def test_official_scenario_validation_and_rejections(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "scenario.toml"
            path.write_text(_scenario_text(), encoding="utf-8")
            data, _ = load_and_validate_scenario(path)
            self.assertEqual(data["config"]["max_steps"], 50)
            self.assertEqual(data["config"]["user_model"], "gemini-3.5-flash")
            self.assertEqual(data["config"]["user_provider"], "gemini")
            self.assertEqual(
                data["config"]["policy_evaluator_model"],
                "gemini-3.5-flash",
            )
            self.assertEqual(
                data["config"]["policy_evaluator_provider"],
                "gemini",
            )

            path.write_text(_scenario_text(image="ghcr.io/example/team:latest"), encoding="utf-8")
            with self.assertRaisesRegex(ScenarioValidationError, "digest-pinned"):
                load_and_validate_scenario(path)

            path.write_text(_scenario_text(extra_agent='volumes = ["/:/host"]'), encoding="utf-8")
            with self.assertRaisesRegex(ScenarioValidationError, "Unsupported"):
                load_and_validate_scenario(path)

            path.write_text(
                _scenario_text().replace(
                    'TEAM_API_KEY = "${TEAM_API_KEY:?required}"',
                    'TEAM_API_KEY = "${TEAM_API_KEY:?required}"\nMAX_OUTPUT_TOKENS = 2048',
                ),
                encoding="utf-8",
            )
            data, _ = load_and_validate_scenario(path)
            self.assertEqual(data["agent_under_test"]["env"]["MAX_OUTPUT_TOKENS"], 2048)

            path.write_text(
                _scenario_text().replace(
                    "max_steps = 50",
                    'max_steps = 50\nuser_model = "gemini-2.5-flash"',
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                ScenarioValidationError,
                "Official hidden config mismatch",
            ):
                load_and_validate_scenario(path)

    def test_compose_mounts_hidden_data_and_results_only_into_evaluator(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data_dir = root / "hidden"
            data_dir.mkdir()
            _write_dataset(data_dir)
            dataset = load_hidden_dataset(data_dir)
            manifest = _manifest(dataset, root)
            scenario = tomllib.loads(_scenario_text())
            run_dir = root / "run"
            run_dir.mkdir()
            compose = generate_competition_compose(
                manifest=manifest,
                scenario=scenario,
                run_dir=run_dir,
                dataset_dir=data_dir,
                project_root=Path.cwd(),
            )
            evaluator_section, rest = compose.split("  agent-under-test:", 1)
            agent_section, client_section = rest.split("  a2a-client:", 1)
            self.assertIn(str(data_dir), evaluator_section)
            self.assertIn(str(run_dir), evaluator_section)
            self.assertNotIn(str(data_dir), agent_section)
            self.assertNotIn(str(run_dir), agent_section)
            self.assertNotIn(str(data_dir), client_section)
            self.assertNotIn("target: /run-state", client_section)
            self.assertIn("CAR_BENCH_ORGANIZER_TOKEN", evaluator_section)
            self.assertNotIn("CAR_BENCH_ORGANIZER_TOKEN", agent_section)
            self.assertIn("CAR_BENCH_ORGANIZER_TOKEN", client_section)
            self.assertNotIn("cap_drop:", agent_section)
            self.assertNotIn("security_opt:", agent_section)

    def test_internal_scenario_carries_validated_run_context(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _write_dataset(root)
            dataset = load_hidden_dataset(root)
            manifest = _manifest(dataset, root)
            text = generate_internal_scenario(manifest=manifest, config=manifest["config"])
            request, endpoint = parse_client_toml(tomllib.loads(text))
            self.assertEqual(endpoint, "http://evaluator:9009")
            self.assertEqual(request.run_context.team_id, "team-1")
            self.assertEqual(request.run_context.dataset_fingerprint, dataset.fingerprint)

    def test_run_store_skips_valid_units_and_rejects_corruption(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data_dir = root / "hidden"
            data_dir.mkdir()
            _write_dataset(data_dir)
            dataset = load_hidden_dataset(data_dir)
            manifest = _manifest(dataset, root)
            store = RunStore(root / "run")
            store.initialize(manifest, _scenario_text())
            record = _unit(dataset, "base", "b_1", 0, True)
            store.write_unit(record)
            self.assertEqual(store.load_unit("base", "b_1", 0), record)
            with self.assertRaises(RunStorageError):
                with RunLock(store.root):
                    with RunLock(store.root):
                        pass

            path = store.unit_path("base", "b_1", 0)
            corrupt = dict(record)
            corrupt["status"] = "working"
            atomic_write_json(path, corrupt)
            with self.assertRaisesRegex(RunStorageError, "Invalid completed"):
                store.load_unit("base", "b_1", 0)

    def test_staging_file_is_ignored_and_paused_retry_gets_fresh_budget(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data_dir = root / "hidden"
            data_dir.mkdir()
            _write_dataset(data_dir)
            dataset = load_hidden_dataset(data_dir)
            manifest = _manifest(dataset, root)
            store = RunStore(root / "run")
            store.initialize(manifest, _scenario_text())
            staging = store.units_dir / "base" / "b_1" / ".trial-0.json.partial"
            staging.parent.mkdir(parents=True)
            staging.write_text('{"truncated":', encoding="utf-8")
            self.assertEqual(list(store.iter_units()), [])

            unit = {
                "dataset_fingerprint": dataset.fingerprint,
                "category": "base",
                "task_id": "b_1",
                "trial": 0,
            }
            for attempt in range(1, 4):
                store.write_attempt(
                    {
                        "schema_version": 2,
                        "unit": unit,
                        "attempt": attempt,
                        "outcome": "timeout",
                    }
                )
            progress = store.progress()
            progress.update({"status": "paused", "current_unit": unit})
            store.update_progress(progress)
            _grant_paused_retry(store)
            updated = store.progress()
            self.assertEqual(updated["status"], "recovering")
            self.assertEqual(updated["attempt_budget_start"], 3)
            self.assertEqual(updated["operator_events"][-1]["event"], "retry_paused")

    def test_trial_seed_is_stable_across_retries(self):
        first = _trial_seed(10, "f" * 64, "base", "b_1", 2)
        second = _trial_seed(10, "f" * 64, "base", "b_1", 2)
        different_trial = _trial_seed(10, "f" * 64, "base", "b_1", 1)
        self.assertEqual(first, second)
        self.assertNotEqual(first, different_trial)

    def test_compose_project_name_retains_unique_run_suffix(self):
        team_id = "a" * 64
        first = _project_name("track_2", team_id, "run-11111111")
        second = _project_name("track_2", team_id, "run-22222222")
        self.assertLessEqual(len(first), 63)
        self.assertTrue(first.endswith("11111111"))
        self.assertTrue(second.endswith("22222222"))
        self.assertNotEqual(first, second)

    def test_resume_image_provenance_rejects_manifest_mismatch(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _write_dataset(root)
            dataset = load_hidden_dataset(root)
            manifest = _manifest(dataset, root)
            scenario = tomllib.loads(_scenario_text())
            _validate_image_provenance(manifest, scenario)
            manifest["images"]["agent_under_test"]["requested"] = (
                f"ghcr.io/example/other@sha256:{'e' * 64}"
            )
            with self.assertRaisesRegex(RunStorageError, "differs"):
                _validate_image_provenance(manifest, scenario)

    def test_complete_summary_uses_macro_pass3_and_nullable_telemetry(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _write_dataset(root)
            dataset = load_hidden_dataset(root)
            manifest = _manifest(dataset, root)
            records = []
            outcomes = {
                "base": (True, True, True),
                "hallucination": (True, False, True),
                "disambiguation": (True, True, True),
            }
            task_ids = {"base": "b_1", "hallucination": "h_1", "disambiguation": "d_1"}
            for category, trials in outcomes.items():
                for trial, passed in enumerate(trials):
                    records.append(
                        _unit(
                            dataset,
                            category,
                            task_ids[category],
                            trial,
                            passed,
                            telemetry=not (category == "hallucination" and trial == 1),
                        )
                    )
            summary = build_summary(manifest, records)
            self.assertTrue(summary["execution"]["complete"])
            self.assertAlmostEqual(summary["scores"]["primary_value"], 2 / 3)
            self.assertEqual(summary["scores"]["pass"]["macro"]["Pass@3"], 1.0)
            self.assertEqual(summary["tokens"]["reported_units"], 8)
            self.assertAlmostEqual(summary["tokens"]["coverage"], 8 / 9)
            self.assertEqual(summary["tokens"]["track_2_500k_status"], "unknown")

    def test_sanitized_derivatives_do_not_contain_hidden_task_content(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data_dir = root / "hidden"
            data_dir.mkdir()
            _write_dataset(data_dir)
            dataset = load_hidden_dataset(data_dir)
            manifest = _manifest(dataset, root)
            store = RunStore(root / "run")
            store.initialize(manifest, _scenario_text())
            task_ids = {"base": "b_1", "hallucination": "h_1", "disambiguation": "d_1"}
            for category, task_id in task_ids.items():
                for trial in range(3):
                    store.write_unit(_unit(dataset, category, task_id, trial, True))
            progress = store.progress()
            progress["status"] = "completed"
            progress["completed_units"] = 9
            store.update_progress(progress)
            write_run_derivatives(store, manifest)
            exports = "\n".join(
                path.read_text(encoding="utf-8") for path in store.exports_dir.glob("*.json")
            )
            self.assertNotIn("HIDDEN INSTRUCTION", exports)
            self.assertNotIn("HIDDEN TRAJECTORY", exports)
            self.assertNotIn("task_id", exports)

    def test_all_units_do_not_publish_until_run_status_is_completed(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _write_dataset(root)
            dataset = load_hidden_dataset(root)
            manifest = _manifest(dataset, root)
            store = RunStore(root / "run")
            store.initialize(manifest, _scenario_text())
            task_ids = {
                "base": "b_1",
                "hallucination": "h_1",
                "disambiguation": "d_1",
            }
            for category, task_id in task_ids.items():
                for trial in range(3):
                    store.write_unit(_unit(dataset, category, task_id, trial, True))
            summary = write_run_derivatives(store, manifest)
            self.assertFalse(summary["execution"]["complete"])
            self.assertIsNone(summary["scores"]["primary_value"])
            self.assertFalse((store.exports_dir / "leaderboard-row.json").exists())

    def test_aggregate_writes_leaderboard_and_standard_plots(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data_dir = root / "hidden"
            data_dir.mkdir()
            _write_dataset(data_dir)
            dataset = load_hidden_dataset(data_dir)
            manifest = _manifest(dataset, root)
            run_dir = (
                root
                / "results"
                / "track_2"
                / manifest["team_id"]
                / manifest["run_id"]
            )
            store = RunStore(run_dir)
            store.initialize(manifest, _scenario_text())
            task_ids = {
                "base": "b_1",
                "hallucination": "h_1",
                "disambiguation": "d_1",
            }
            for category, task_id in task_ids.items():
                for trial in range(3):
                    record = _unit(dataset, category, task_id, trial, True)
                    store.write_unit(record)
                    store.write_attempt(
                        {
                            "schema_version": 2,
                            "unit": record["unit"],
                            "attempt": 1,
                            "outcome": "completed",
                        }
                    )
            progress = store.progress()
            progress.update({"status": "completed", "completed_units": 9})
            store.update_progress(progress)
            write_run_derivatives(store, manifest)

            result = aggregate_runs(
                Namespace(
                    results_root=root / "results",
                    track="track_2",
                    include_legacy=False,
                )
            )
            aggregate_dir = root / "results" / "track_2" / "aggregate"
            self.assertEqual(result, 0)
            self.assertTrue((aggregate_dir / "leaderboard.json").is_file())
            self.assertTrue((aggregate_dir / "leaderboard.csv").is_file())
            self.assertEqual(len(list((aggregate_dir / "plots").glob("*.svg"))), 6)

    def test_max_steps_is_passed_explicitly(self):
        class FakeOrchestrator:
            def execute(self, **kwargs):
                return kwargs

        result = execute_orchestrator(FakeOrchestrator(), "env", 7, 50)
        self.assertEqual(result["env"], "env")
        self.assertEqual(result["task_index"], 7)
        self.assertEqual(result["max_num_steps"], 50)

    def test_telemetry_distinguishes_reported_zero_from_missing(self):
        class Result:
            def __init__(self, metrics):
                self.metrics = metrics

            def model_dump(self, mode):
                return {
                    "reward": 1,
                    "info": {
                        "total_llm_induced_latency_ms": 0,
                        "total_agent_cost": 0,
                    },
                    "traj": [
                        {
                            "role": "assistant",
                            "turn_metrics": self.metrics,
                            "evaluator_metrics": {
                                "a2a_turn_time_ms": 25,
                                "a2a_effective_turn_time_ms": 25,
                            },
                        }
                    ],
                }

        common = {
            "dataset_fingerprint": "f" * 64,
            "category": "base",
            "task_id": "b_1",
            "trial": 0,
            "trial_seed": 1,
            "attempt_count": 1,
            "started_at": "2026-07-22T12:00:00+00:00",
            "completed_at": "2026-07-22T12:00:01+00:00",
            "duration_seconds": 1,
        }
        fields = [
            "prompt_tokens",
            "completion_tokens",
            "thinking_tokens",
            "quota_wait_time_ms",
            "num_llm_calls",
            "avg_llm_call_time_ms",
            "cost",
        ]
        reported = normalize_unit_record(
            result=Result({**{field: 0 for field in fields}, "_reported_fields": fields}),
            **common,
        )
        missing = normalize_unit_record(
            result=Result({"num_llm_calls": 1, "_reported_fields": ["num_llm_calls"]}),
            **common,
        )
        self.assertEqual(reported["telemetry"]["tokens"]["total_tokens"], 0)
        self.assertEqual(reported["telemetry"]["cost"], 0)
        self.assertIsNone(missing["telemetry"]["tokens"]["total_tokens"])
        self.assertIsNone(missing["telemetry"]["cost"])
        self.assertIsNone(missing["timing"]["quota_wait_self_reported_ms"])

    def test_evaluator_post_requires_organizer_token_when_enabled(self):
        middleware = OrganizerTokenMiddleware(app=lambda scope, receive, send: None)

        async def accepted(_request):
            return SimpleNamespace(status_code=200)

        async def exercise(method, token=None):
            headers = {}
            if token is not None:
                headers["X-CAR-BENCH-ORGANIZER-TOKEN"] = token
            request = SimpleNamespace(method=method, headers=headers)
            return await middleware.dispatch(request, accepted)

        with mock.patch.dict(
            os.environ, {"CAR_BENCH_ORGANIZER_TOKEN": "organizer-token"}, clear=False
        ):
            self.assertEqual(asyncio.run(exercise("POST")).status_code, 403)
            self.assertEqual(
                asyncio.run(exercise("POST", "organizer-token")).status_code,
                200,
            )
            self.assertEqual(asyncio.run(exercise("GET")).status_code, 200)

    def test_tied_scores_receive_same_rank(self):
        rows = [
            {"team_id": "b", "primary_score": 0.5},
            {"team_id": "a", "primary_score": 0.5},
            {"team_id": "c", "primary_score": 0.4},
        ]
        ranked = rank_rows(rows)
        self.assertEqual([(row["team_id"], row["rank"]) for row in ranked], [("a", 1), ("b", 1), ("c", 3)])

    def test_schema_v1_development_result_is_readable(self):
        payload = {
            "metadata": {"agent_name": "Legacy Team"},
            "final_result": {
                "pass_power_k_scores": {"Pass^1": 0.7, "Pass^3": 0.5},
                "pass_at_k_scores": {"Pass@1": 0.7, "Pass@3": 0.9},
                "pass_power_k_scores_by_split": {
                    category: {"Pass^3": 0.5}
                    for category in ("base", "hallucination", "disambiguation")
                },
                "pass_at_k_scores_by_split": {
                    category: {"Pass@3": 0.9}
                    for category in ("base", "hallucination", "disambiguation")
                },
                "pass_rate": 60,
            },
        }
        row = _legacy_row(Path("legacy-result.json"), payload, "track_1")
        self.assertIsNotNone(row)
        self.assertEqual(row["schema_version"], 1)
        self.assertEqual(row["primary_score"], 0.5)


if __name__ == "__main__":
    unittest.main()
