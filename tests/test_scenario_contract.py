import stat
import tempfile
import tomllib
import unittest
from contextlib import redirect_stdout
from datetime import datetime
from io import StringIO
from pathlib import Path

from loguru import logger as loguru_logger

from agentbeats.client_cli import (
    build_output_payload,
    parse_toml as parse_client_toml,
    resolve_output_path,
)
from agentbeats.run_scenario import parse_toml as parse_runner_toml
from generate_compose import (
    compose_up_command,
    generate_a2a_scenario,
    generate_docker_compose,
    parse_scenario,
    prepare_results_mount,
)

loguru_logger.disable("agentbeats.run_scenario")

EXPECTED_SCENARIO_FILES = {
    "local_smoke.toml",
    "local_public_batch.toml",
    "local_test_set.toml",
    "local_docker_smoke.toml",
    "local_docker_test_set.toml",
    "ghcr_smoke.toml",
    "ghcr_test_set.toml",
}

SCENARIO_DIRS = [
    Path("scenarios/track_1_agent_under_test"),
    Path("scenarios/track_2_agent_under_test_cerebras"),
    Path("scenarios/track_2_agent_under_test_cerebras_planner"),
]


class ScenarioContractTest(unittest.TestCase):
    def _scenario(self) -> dict:
        return {
            "evaluator": {
                "build": {
                    "context": ".",
                    "dockerfile": "src/evaluator/Dockerfile.evaluator",
                },
                "env": {"GEMINI_API_KEY": "${GEMINI_API_KEY}"},
            },
            "agent_under_test": {
                "build": {
                    "context": ".",
                    "dockerfile": "src/track_1_agent_under_test/Dockerfile.track-1-agent-under-test",
                },
                "env": {"AGENT_LLM": "gemini/gemini-2.5-flash"},
            },
            "config": {"num_trials": 1},
        }

    def test_compose_generation_uses_evaluator_and_agent_under_test(self) -> None:
        compose = generate_docker_compose(
            self._scenario(),
            output_dir=Path("scenarios/track_1_agent_under_test"),
        )

        self.assertIn("  evaluator:", compose)
        self.assertIn("  agent-under-test:", compose)
        self.assertIn("  a2a-client:", compose)
        self.assertIn("agent-network:", compose)
        self.assertIn("context: ../..", compose)
        self.assertIn("../../output:/home/carbench/app/output", compose)
        self.assertIn("docker compose --env-file .env", compose)
        self.assertNotIn("green-agent", compose)

    def test_agent_scenario_directories_use_standard_matrix(self) -> None:
        for scenario_dir in SCENARIO_DIRS:
            with self.subTest(scenario_dir=str(scenario_dir)):
                files = {
                    path.name
                    for path in scenario_dir.glob("*.toml")
                    if path.name != "a2a-scenario.toml"
                }
                self.assertEqual(files, EXPECTED_SCENARIO_FILES)

                for name in (
                    "local_smoke.toml",
                    "local_docker_smoke.toml",
                    "ghcr_smoke.toml",
                ):
                    data = tomllib.loads((scenario_dir / name).read_text())
                    config = data["config"]
                    self.assertEqual(config["num_trials"], 1)
                    self.assertEqual(config["task_split"], "train")
                    self.assertEqual(config["tasks_base_num_tasks"], 1)
                    self.assertEqual(config["tasks_hallucination_num_tasks"], 1)
                    self.assertEqual(config["tasks_disambiguation_num_tasks"], 1)

                data = tomllib.loads(
                    (scenario_dir / "local_public_batch.toml").read_text()
                )
                config = data["config"]
                self.assertEqual(config["num_trials"], 1)
                self.assertEqual(config["task_split"], "test")
                self.assertEqual(config["tasks_base_num_tasks"], 10)
                self.assertEqual(config["tasks_hallucination_num_tasks"], 10)
                self.assertEqual(config["tasks_disambiguation_num_tasks"], 10)

                for name in (
                    "local_test_set.toml",
                    "local_docker_test_set.toml",
                    "ghcr_test_set.toml",
                ):
                    data = tomllib.loads((scenario_dir / name).read_text())
                    config = data["config"]
                    self.assertEqual(config["num_trials"], 3)
                    self.assertEqual(config["task_split"], "test")
                    self.assertEqual(config["tasks_base_num_tasks"], -1)
                    self.assertEqual(config["tasks_hallucination_num_tasks"], -1)
                    self.assertEqual(config["tasks_disambiguation_num_tasks"], -1)

    def test_compose_up_command_uses_root_env_file(self) -> None:
        command = compose_up_command(
            Path("scenarios/track_1_agent_under_test/docker-compose.yml")
        )

        self.assertEqual(
            command,
            "docker compose --env-file .env -f "
            "scenarios/track_1_agent_under_test/docker-compose.yml up "
            "--abort-on-container-exit --exit-code-from a2a-client",
        )

    def test_results_mount_is_writable_by_an_arbitrary_container_uid(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "output"
            agent_dir = prepare_results_mount(root, "example agent")

            self.assertEqual(agent_dir, root / "example-agent")
            for directory in (root, agent_dir):
                mode = directory.stat().st_mode
                self.assertTrue(mode & stat.S_IWOTH)
                self.assertTrue(mode & stat.S_IXOTH)
                self.assertTrue(mode & stat.S_ISVTX)

    def test_generated_a2a_scenario_uses_singular_aut_contract(self) -> None:
        scenario = generate_a2a_scenario(self._scenario())

        self.assertIn("[evaluator]", scenario)
        self.assertIn("[agent_under_test]", scenario)
        self.assertIn("[run]", scenario)
        self.assertIn('endpoint = "http://agent-under-test:9009"', scenario)
        self.assertNotIn("[[participants]]", scenario)

    def test_timestamped_output_path_uses_agent_model_and_effort(self) -> None:
        data = {
            "run": {
                "agent_name": "cerebras-planner",
                "scenario_name": "cerebras-planner/local_docker_smoke",
                "agent_metadata": {
                    "TRACK2_PLANNER_MODEL": "gpt-oss-120b",
                    "TRACK2_PLANNER_REASONING_EFFORT": "high",
                    "TRACK2_EXECUTOR_MODEL": "gpt-oss-120b",
                    "TRACK2_EXECUTOR_REASONING_EFFORT": "medium",
                },
            },
            "evaluator": {"endpoint": "http://127.0.0.1:8081"},
            "agent_under_test": {"endpoint": "http://127.0.0.1:8080"},
        }

        path = resolve_output_path("output", Path("scenario.toml"), data)

        self.assertIsNotNone(path)
        self.assertEqual(path.parts[0:2], ("output", "cerebras-planner"))
        self.assertIn("gpt-oss-120b_to_gpt-oss-120b", path.name)
        self.assertIn("high_to_medium", path.name)
        self.assertIn("split-unspecified", path.name)

    def test_timestamped_output_path_recognizes_pact_metadata(self) -> None:
        scenario = self._scenario()
        scenario["agent_under_test"]["env"] = {
            "PACT_COMPILER_MODEL": "${PACT_COMPILER_MODEL:-gpt-oss-120b}",
            "PACT_COMPILER_REASONING_EFFORT": (
                "${PACT_COMPILER_REASONING_EFFORT:-medium}"
            ),
        }
        generated = generate_a2a_scenario(scenario)
        data = tomllib.loads(generated)

        path = resolve_output_path("output", Path("scenario.toml"), data)

        self.assertIsNotNone(path)
        self.assertIn("gpt-oss-120b", path.name)
        self.assertIn("medium", path.name)

    def test_output_path_omits_unreliable_user_model(self) -> None:
        data = {
            "evaluator": {"endpoint": "http://127.0.0.1:8081"},
            "agent_under_test": {"endpoint": "http://127.0.0.1:8080"},
            "config": {
                "task_split": "test",
                "num_trials": 1,
                "user_model": "gemini/gemini-2.5-flash",
                "tasks_base_num_tasks": 2,
                "tasks_hallucination_num_tasks": 0,
                "tasks_disambiguation_num_tasks": 0,
            },
        }

        path = resolve_output_path(
            "output", Path("scenarios/custom/local_smoke.toml"), data
        )

        self.assertIsNotNone(path)
        self.assertIn("test-trials1-base2-hall0-dis0", path.name)
        self.assertNotIn("gemini", path.name)
        self.assertNotIn("model-unspecified", path.name)
        self.assertNotIn("effort-unspecified", path.name)

    def test_output_payload_promotes_final_summary_and_metadata(self) -> None:
        req, evaluator_url = parse_client_toml(
            {
                "evaluator": {"endpoint": "http://127.0.0.1:8081"},
                "agent_under_test": {"endpoint": "http://127.0.0.1:8080"},
            }
        )
        payload = build_output_payload(
            req=req,
            evaluator_url=evaluator_url,
            scenario_path=Path("scenarios/track_1_agent_under_test/local_smoke.toml"),
            scenario_data={
                "agent_under_test": {
                    "endpoint": "http://127.0.0.1:8080",
                    "cmd": "python server.py --agent-llm gemini/gemini-2.5-flash --reasoning-effort low",
                },
                "config": {"num_trials": 1},
            },
            artifact_records=[
                {
                    "name": "Result",
                    "text_parts": ["CAR-bench Results\nOverall Pass Rate: 50.0%"],
                    "data_parts": [
                        {
                            "summary": {"pass_rate": 50.0},
                            "score": 1,
                            "max_score": 2,
                            "pass_rate": 50.0,
                        }
                    ],
                }
            ],
            started_at=datetime.fromisoformat("2026-05-13T10:00:00+00:00"),
            completed_at=datetime.fromisoformat("2026-05-13T10:01:00+00:00"),
        )

        self.assertEqual(payload["summary"], {"pass_rate": 50.0})
        self.assertEqual(payload["metadata"]["model"], "gemini/gemini-2.5-flash")
        self.assertEqual(payload["metadata"]["reasoning_effort"], "low")
        self.assertEqual(
            payload["metadata"]["task_selection"], "split-unspecified-trials1"
        )
        self.assertEqual(payload["metadata"]["wall_time_seconds"], 60.0)
        self.assertEqual(payload["metadata"]["quota_wait_seconds"], 0.0)

    def test_output_payload_subtracts_quota_wait_from_wall_time(self) -> None:
        req, evaluator_url = parse_client_toml(
            {
                "evaluator": {"endpoint": "http://127.0.0.1:8081"},
                "agent_under_test": {"endpoint": "http://127.0.0.1:8080"},
            }
        )
        payload = build_output_payload(
            req=req,
            evaluator_url=evaluator_url,
            scenario_path=Path(
                "scenarios/track_2_agent_under_test_cerebras/local_smoke.toml"
            ),
            scenario_data={
                "agent_under_test": {
                    "endpoint": "http://127.0.0.1:8080",
                    "cmd": "python server.py --executor-model gpt-oss-120b",
                },
                "config": {"num_trials": 1},
            },
            artifact_records=[
                {
                    "name": "Result",
                    "text_parts": ["CAR-bench Results\nOverall Pass Rate: 100.0%"],
                    "data_parts": [
                        {
                            "summary": {"pass_rate": 100.0},
                            "score": 1,
                            "max_score": 1,
                            "pass_rate": 100.0,
                            "quota_wait_time": 15.0,
                        }
                    ],
                }
            ],
            started_at=datetime.fromisoformat("2026-05-13T10:00:00+00:00"),
            completed_at=datetime.fromisoformat("2026-05-13T10:01:00+00:00"),
        )

        self.assertEqual(payload["metadata"]["raw_wall_time_seconds"], 60.0)
        self.assertEqual(payload["metadata"]["quota_wait_seconds"], 15.0)
        self.assertEqual(payload["metadata"]["wall_time_seconds"], 45.0)

    def test_client_cli_parses_new_shape(self) -> None:
        req, evaluator_url = parse_client_toml(
            {
                "evaluator": {"endpoint": "http://127.0.0.1:8081"},
                "agent_under_test": {"endpoint": "http://127.0.0.1:8080"},
                "config": {"num_trials": 1},
            }
        )

        self.assertEqual(evaluator_url, "http://127.0.0.1:8081")
        self.assertEqual(str(req.agent_under_test), "http://127.0.0.1:8080/")
        self.assertEqual(req.config["num_trials"], 1)

    def test_local_runner_parses_new_shape(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "scenario.toml"
            path.write_text(
                """
[evaluator]
endpoint = "http://127.0.0.1:8081"
cmd = "python src/evaluator/server.py --host 127.0.0.1 --port 8081"

[agent_under_test]
endpoint = "http://127.0.0.1:8080"
cmd = "python src/track_1_agent_under_test/server.py --host 127.0.0.1 --port 8080"

[config]
num_trials = 1
""".strip()
            )

            cfg = parse_runner_toml(str(path))

        self.assertEqual(cfg["evaluator"]["port"], 8081)
        self.assertEqual(cfg["agent_under_test"]["port"], 8080)
        self.assertEqual(cfg["config"]["num_trials"], 1)

    def test_old_scenario_shape_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            parse_client_toml(
                {
                    "green_agent": {"endpoint": "http://127.0.0.1:8081"},
                    "participants": [
                        {"role": "agent", "endpoint": "http://127.0.0.1:8080"}
                    ],
                }
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "scenario.toml"
            path.write_text(
                """
[green_agent]
endpoint = "http://127.0.0.1:8081"

[[participants]]
role = "agent"
endpoint = "http://127.0.0.1:8080"
""".strip()
            )
            with self.assertRaises(SystemExit), redirect_stdout(StringIO()):
                parse_scenario(path)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "scenario.toml"
            path.write_text(
                """
[evaluator]
endpoint = "http://127.0.0.1:8081"

[[participants]]
name = "agent"
endpoint = "http://127.0.0.1:8080"
""".strip()
            )
            with self.assertRaises(SystemExit), redirect_stdout(StringIO()):
                parse_scenario(path)


if __name__ == "__main__":
    unittest.main()
