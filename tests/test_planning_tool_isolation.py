import json
import sys
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace


CAR_BENCH_ROOT = Path(__file__).resolve().parents[1] / "third_party" / "car-bench"
sys.path.insert(0, str(CAR_BENCH_ROOT))

from car_bench.envs.base import Env
from car_bench.envs.car_voice_assistant.context.dynamic_context_state import (
    ContextState,
    context_state,
)
from car_bench.envs.car_voice_assistant.tools.cross_domain.planning import PlanningTool
from car_bench.types import Task, TaskType


def _invoke_planning(**kwargs) -> dict:
    return json.loads(PlanningTool.invoke(data={}, **kwargs))


def _create_plan(title: str) -> dict:
    return _invoke_planning(
        command="create",
        plan_id="shared-plan-id",
        title=title,
        steps=[
            {
                "step_description": f"Task-local step for {title}",
                "step_dependent_on": [],
            }
        ],
    )


class PlanningToolIsolationTest(unittest.TestCase):
    def setUp(self) -> None:
        PlanningTool.reset()

    def test_state_persists_within_one_task_context(self) -> None:
        self.assertEqual(_create_plan("Task A")["status"], "SUCCESS")

        result = _invoke_planning(command="get", plan_id="shared-plan-id")

        self.assertEqual(result["status"], "SUCCESS")
        self.assertEqual(result["result"]["plan_details"]["title"], "Task A")

    def test_reset_removes_plans_from_the_previous_task(self) -> None:
        self.assertEqual(_create_plan("Task A")["status"], "SUCCESS")

        PlanningTool.reset()
        result = _invoke_planning(command="get", plan_id="shared-plan-id")

        self.assertEqual(result["status"], "FAILURE")
        self.assertIn("No plan found", result["errors"]["PLANNING_TOOL_ERROR"])

    def test_environment_reset_starts_with_an_empty_plan_store(self) -> None:
        self.assertEqual(_create_plan("Previous task")["status"], "SUCCESS")
        task = Task(
            task_id="planning-reset",
            calendar_id="calendar",
            actions=[],
            persona="persona",
            instruction="instruction",
            context_init_config={},
            task_type=TaskType.BASE,
            removed_part=None,
            disambiguation_element_internal=None,
        )
        env = object.__new__(Env)
        env.tools_map = {"planning_tool": PlanningTool}
        env.tasks = [task]
        env.data_load_func = lambda: {}
        env.user = SimpleNamespace(reset=lambda **_: "initial observation")

        token = context_state.set(ContextState())
        try:
            env.reset(task_index=0)
        finally:
            context_state.reset(token)

        result = _invoke_planning(command="get", plan_id="shared-plan-id")
        self.assertEqual(result["status"], "FAILURE")
        self.assertIn("No plan found", result["errors"]["PLANNING_TOOL_ERROR"])

    def test_parallel_tasks_can_reuse_the_same_plan_id(self) -> None:
        reset_barrier = threading.Barrier(2)
        read_barrier = threading.Barrier(2)

        def run_task(title: str) -> tuple[str, str]:
            PlanningTool.reset()
            reset_barrier.wait()
            create_result = _create_plan(title)
            read_barrier.wait()
            get_result = _invoke_planning(
                command="get", plan_id="shared-plan-id"
            )
            return (
                create_result["status"],
                get_result["result"]["plan_details"]["title"],
            )

        with ThreadPoolExecutor(max_workers=2) as executor:
            results = list(executor.map(run_task, ["Task A", "Task B"]))

        self.assertCountEqual(
            results,
            [
                ("SUCCESS", "Task A"),
                ("SUCCESS", "Task B"),
            ],
        )


if __name__ == "__main__":
    unittest.main()
