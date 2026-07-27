import csv
import json
import sys
import unittest
from pathlib import Path


CAR_BENCH_ROOT = Path(__file__).resolve().parents[1] / "third_party" / "car-bench"
sys.path.insert(0, str(CAR_BENCH_ROOT))

from car_bench.envs.car_voice_assistant.wiki import WIKI_LLM_POL_LINES
from car_bench.envs.policy_evaluator import (
    _evaluate_fog_light_batches,
    policy_errors_during_runtime,
)
from car_bench.envs.reward_calculators import enhance_policy_line_with_context


ROUTE_PRESENTATION_POLICY_FRAGMENT = (
    "If you get multiple alternative routes from the tool"
)


def _tool_call(name: str, **arguments) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "arguments": json.dumps(arguments),
        },
    }


def _exterior_light_status(low_beams: bool, high_beams: bool) -> dict:
    return {
        "role": "tool",
        "name": "get_exterior_lights_status",
        "content": json.dumps(
            {
                "status": "SUCCESS",
                "result": {
                    "fog_lights": False,
                    "head_lights_low_beams": low_beams,
                    "head_lights_high_beams": high_beams,
                },
            }
        ),
    }


class HiddenBaseEvaluatorRegressionTest(unittest.TestCase):
    def _fog_policy_errors(self, trajectory: list[dict]) -> list[str]:
        token = policy_errors_during_runtime.set([])
        try:
            _evaluate_fog_light_batches(trajectory)
            return list(policy_errors_during_runtime.get())
        finally:
            policy_errors_during_runtime.reset(token)

    def test_fog_light_companions_in_same_batch_are_accepted(self) -> None:
        trajectory = [
            _exterior_light_status(low_beams=False, high_beams=True),
            {
                "role": "assistant",
                "tool_calls": [
                    _tool_call("set_fog_lights", on=True),
                    _tool_call("set_head_lights_low_beams", on=True),
                    _tool_call("set_head_lights_high_beams", on=False),
                ],
            },
        ]

        self.assertEqual(self._fog_policy_errors(trajectory), [])

    def test_fog_light_companion_deferred_to_later_batch_is_rejected(self) -> None:
        trajectory = [
            _exterior_light_status(low_beams=False, high_beams=True),
            {
                "role": "assistant",
                "tool_calls": [
                    _tool_call("set_head_lights_low_beams", on=True),
                    _tool_call("set_fog_lights", on=True),
                ],
            },
            {
                "role": "assistant",
                "tool_calls": [
                    _tool_call("set_head_lights_high_beams", on=False),
                ],
            },
        ]

        errors = self._fog_policy_errors(trajectory)

        self.assertEqual(len(errors), 1)
        self.assertIn("AUT-POL:013", errors[0])

    def test_route_presentation_policy_is_only_applied_to_route_traces(self) -> None:
        route_policy = next(
            line
            for line in WIKI_LLM_POL_LINES
            if ROUTE_PRESENTATION_POLICY_FRAGMENT in line
        )

        self.assertIsNone(
            enhance_policy_line_with_context(
                route_policy,
                performed_action_names={"get_weather"},
                difference_is_more_than_3_degrees=False,
            )
        )
        enhanced = enhance_policy_line_with_context(
            route_policy,
            performed_action_names={"get_routes_from_start_to_destination"},
            difference_is_more_than_3_degrees=False,
        )
        self.assertIsNotNone(enhanced)
        self.assertIn("Prior explicit user authorization", enhanced)
        self.assertIn("every route-tool result independently", enhanced)

    def test_b2_has_explicit_order_and_policy_safe_ground_truth(self) -> None:
        dataset_path = (
            Path(__file__).resolve().parents[1]
            / "hidden_dataset"
            / "hidden_base.csv"
        )
        with dataset_path.open(newline="") as dataset:
            task = next(
                row
                for row in csv.DictReader(dataset)
                if row["task_id"] == "b_2"
            )

        action_names = [action["name"] for action in json.loads(task["actions"])]

        self.assertIn("first turn on the fog lights", task["instruction"])
        self.assertIn("then clear the fogging front windshield", task["instruction"])
        self.assertLess(
            action_names.index("set_fog_lights"),
            action_names.index("get_climate_settings"),
        )
        self.assertLess(
            action_names.index("get_vehicle_window_positions"),
            action_names.index("set_air_conditioning"),
        )
        self.assertLess(
            action_names.index("set_air_conditioning"),
            action_names.index("set_window_defrost"),
        )

    def test_b2_derived_rows_preserve_ordering_and_h2_removed_tool(self) -> None:
        dataset_dir = Path(__file__).resolve().parents[1] / "hidden_dataset"
        cases = (
            ("hidden_disambiguation.csv", "d_2"),
            ("hidden_hallucination.csv", "h_2"),
        )

        for filename, task_id in cases:
            with self.subTest(task_id=task_id):
                with (dataset_dir / filename).open(newline="") as dataset:
                    task = next(
                        row
                        for row in csv.DictReader(dataset)
                        if row["task_id"] == task_id
                    )

                actions = json.loads(task["actions"])
                action_names = [action["name"] for action in actions]

                self.assertIn("first turn on the fog lights", task["instruction"])
                self.assertLess(
                    action_names.index("set_fog_lights"),
                    action_names.index("get_climate_settings"),
                )
                self.assertLess(
                    action_names.index("set_air_conditioning"),
                    action_names.index("set_window_defrost"),
                )

                if task_id == "h_2":
                    airflow_action = next(
                        action
                        for action in actions
                        if action["name"] == "set_fan_airflow_direction"
                    )
                    self.assertEqual(airflow_action["kwargs"], {})
                    self.assertEqual(
                        json.loads(task["removed_part"]),
                        ["set_fan_airflow_direction"],
                    )


if __name__ == "__main__":
    unittest.main()
