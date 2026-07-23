import importlib.util
import unittest
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "summarize_car_bench_results.py"
SPEC = importlib.util.spec_from_file_location("summarize_car_bench_results", SCRIPT_PATH)
assert SPEC is not None
summarize_car_bench_results = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(summarize_car_bench_results)


class ResultSummarizerTest(unittest.TestCase):
    def test_render_report_classifies_failed_tasks(self) -> None:
        data = {
            "metadata": {
                "scenario_name": "track_1_agent_under_test/local_public_batch",
                "model": "openrouter/openai/gpt-oss-120b:free",
                "task_selection": "test-trials1-base2-hall0-dis0",
            },
            "results": [
                {
                    "score": 1.0,
                    "max_score": 2.0,
                    "pass_rate": 50.0,
                    "successful_llm_time_used": 12.5,
                    "time_used": 40.0,
                    "quota_wait_time": 3.0,
                    "detailed_results_by_split": {
                        "base": [
                            {
                                "task_id": "base_0",
                                "reward": 1.0,
                                "task": {"task_id": "base_0", "actions": []},
                                "reward_info": {"info": {}, "actions": []},
                            },
                            {
                                "task_id": "base_1",
                                "reward": 0.0,
                                "task": {
                                    "task_id": "base_1",
                                    "task_type": "base",
                                    "instruction": "Defrost the windshield.",
                                    "actions": [{"name": "set_window_defrost"}],
                                },
                                "reward_info": {
                                    "info": {
                                        "r_actions": 0.0,
                                        "r_tool_subset": 0.0,
                                        "tool_subset_missing_tools": ["set_window_defrost"],
                                    },
                                    "actions": [{"name": "respond"}],
                                },
                            },
                        ],
                        "hallucination": [
                            {
                                "task_id": "hallucination_1",
                                "reward": 0.0,
                                "task": {
                                    "task_id": "hallucination_1",
                                    "task_type": "hallucination_missing_tool",
                                    "instruction": "Use an unavailable tool.",
                                    "actions": [{"name": "get_weather"}],
                                },
                                "reward_info": {
                                    "info": {
                                        "r_tool_execution": 0.0,
                                        "tool_execution_errors": ["missing_tool: unavailable"],
                                    },
                                    "actions": [{"name": "unavailable"}],
                                },
                            }
                        ],
                    },
                }
            ],
        }

        report = summarize_car_bench_results.render_report(Path("result.json"), data)

        self.assertIn("score 1/2", report)
        self.assertIn("failures: total=2", report)
        self.assertIn("by_split=base=1, hallucination=1", report)
        self.assertIn("action_sequence=1", report)
        self.assertIn("missing_tool_subset=1", report)
        self.assertIn("tool_execution=1", report)
        self.assertIn("expected: set_window_defrost", report)
        self.assertIn("actual: respond", report)


if __name__ == "__main__":
    unittest.main()
