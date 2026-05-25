from __future__ import annotations

from src.planner.planner import Planner


def _planner() -> Planner:
    return Planner(policy=lambda messages: {"content": ""}, min_subtasks=1, max_subtasks=1)


def test_planner_repairs_truncated_json_tail() -> None:
    planner = _planner()
    raw = """
{
  "sub_tasks": [
    {
      "task_id": "task_1",
      "task_type": "search",
      "description": "Read the official GitHub productivity study",
      "dependencies": [],
      "context_keys": [],
      "timeout_seconds": 120,
      "priority": 1,
      "expected_type": "factual",
      "search_hints": ["GitHub Copilot", "productivity"]
    },
"""

    dag = planner._parse_plan(raw)
    assert "task_1" in dag

    task_map = planner.get_task_map_from_dag(dag, raw)
    assert task_map["task_1"].description == "Read the official GitHub productivity study"


def test_planner_generate_plan_falls_back_to_single_search_task() -> None:
    truncated = """
{
  "sub_tasks": [
    {
      "task_id": "task_1",
      "task_type": "search",
      "description": "Retrieve empirical GitHub Copilot productivity evidence",
      "dependencies": [],
      "context_keys": [],
      "timeout_seconds": 120,
      "priority": 1,
      "expected_type": "factual",
      "search_hints": ["GitHub Copilot", "https://github.blog/research/copi
"""
    planner = Planner(policy=lambda messages: {"content": truncated}, min_subtasks=1, max_subtasks=1)

    dag = planner.generate_plan(
        "Analyze GitHub Copilot productivity using https://github.blog/research/copilot and compare code quality claims."
    )
    task_map = planner.get_task_map_from_dag(dag, planner._last_raw_json)

    assert "task_1" in dag
    assert task_map["task_1"].description.startswith("Analyze GitHub Copilot productivity")
    assert all("http" not in hint.lower() for hint in task_map["task_1"].search_hints)
