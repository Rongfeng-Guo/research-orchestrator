from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_real_evidence_iteration import build_iteration_plan  # noqa: E402


def test_build_iteration_plan_uses_stable_paths_and_commands() -> None:
    args = SimpleNamespace(
        config="configs/real_evidence_smoke.yaml",
        queries_file="data/queries/real_evidence_seed_20260516.jsonl",
        cache_dir="data/search_cache/real",
        resurfaced_dir="data/search_cache/real_resurfaced",
        analysis_dir="outputs/analysis_real",
        train_dir="outputs/train_real",
        audit_dir="outputs/audit_real",
        artifacts_dir="artifacts",
        session_prefix="real_test",
        search_policy_mode="heuristic",
        include_evaluation=True,
        allow_mock_evidence=False,
        heldout_queries_file=None,
    )

    plan = build_iteration_plan(args, timestamp="20260516_130000")

    assert plan["session_prefix"] == "real_test"
    assert plan["paths"]["cache_file"] == "data\\search_cache\\real\\real_test.jsonl"
    assert plan["paths"]["resurfaced_file"] == "data\\search_cache\\real_resurfaced\\real_test_resurfaced.jsonl"
    assert plan["paths"]["analysis_json"] == "outputs\\analysis_real\\search_cache_analysis_real_test_resurfaced.json"
    assert plan["paths"]["train_summary_file"] == "outputs\\train_real\\real_test_train_summary.json"
    assert plan["paths"]["model_file"] == "artifacts\\search_policy_real_test.json"
    assert plan["paths"]["audit_dir"] == "outputs\\audit_real"
    assert "--allow_mock_evidence" not in plan["commands"]["build_cache"]
    assert plan["commands"]["build_cache"][-2:] == ["--search-policy-mode", "heuristic"]
    assert plan["commands"]["train_policy"][1] == "scripts/train_search_policy_iteration.py"
    assert "--output-summary" in plan["commands"]["train_policy"]
    assert plan["commands"]["audit_paper_readiness"][1] == "scripts/audit_paper_readiness.py"
    assert "--output-json" in plan["commands"]["audit_paper_readiness"]
    assert "--output-md" in plan["commands"]["audit_paper_readiness"]


def test_build_iteration_plan_supports_train_and_heldout_splits() -> None:
    args = SimpleNamespace(
        config="configs/real_evidence_smoke.yaml",
        queries_file="data/queries/real_train.jsonl",
        heldout_queries_file="data/queries/real_heldout.jsonl",
        cache_dir="data/search_cache/real",
        resurfaced_dir="data/search_cache/real_resurfaced",
        analysis_dir="outputs/analysis_real",
        train_dir="outputs/train_real",
        audit_dir="outputs/audit_real",
        artifacts_dir="artifacts",
        session_prefix="real_test",
        search_policy_mode="heuristic",
        include_evaluation=True,
        allow_mock_evidence=True,
    )

    plan = build_iteration_plan(args, timestamp="20260516_130000")

    assert plan["paths"]["cache_file"] == "data\\search_cache\\real\\real_test_train.jsonl"
    assert plan["paths"]["heldout_cache_file"] == "data\\search_cache\\real\\real_test_heldout.jsonl"
    assert plan["paths"]["resurfaced_file"] == "data\\search_cache\\real_resurfaced\\real_test_train_resurfaced.jsonl"
    assert plan["paths"]["heldout_resurfaced_file"] == "data\\search_cache\\real_resurfaced\\real_test_heldout_resurfaced.jsonl"
    assert plan["paths"]["analysis_jsons"][0].startswith(
        "outputs\\analysis_real\\search_cache_analysis_real_test_train"
    )
    assert plan["paths"]["analysis_jsons"][1].startswith(
        "outputs\\analysis_real\\search_cache_analysis_real_test_heldo"
    )
    assert plan["paths"]["analysis_jsons"][0].endswith(".json")
    assert plan["paths"]["analysis_jsons"][1].endswith(".json")
    assert plan["commands"]["build_cache"][1] == "scripts/build_search_cache.py"
    assert plan["commands"]["build_heldout_cache"][1] == "scripts/build_search_cache.py"
    assert "--allow_mock_evidence" in plan["commands"]["build_cache"]
    assert "--allow_mock_evidence" in plan["commands"]["build_heldout_cache"]
    assert plan["commands"]["train_policy"][plan["commands"]["train_policy"].index("--inputs") + 1] == (
        "data\\search_cache\\real_resurfaced\\real_test_train_resurfaced.jsonl"
    )
    audit_command = plan["commands"]["audit_paper_readiness"]
    analysis_start = audit_command.index("--analysis-json") + 1
    train_json_index = audit_command.index("--train-json")
    assert audit_command[analysis_start:train_json_index] == plan["paths"]["analysis_jsons"]
