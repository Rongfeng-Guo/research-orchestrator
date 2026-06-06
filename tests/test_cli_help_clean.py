from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
PYTHON_CI_WORKFLOW = PROJECT_ROOT / ".github" / "workflows" / "python-ci.yml"

CLI_HELP_CASES = [
    ("scripts/run_repl.py", "Research Orchestrator 交互式 REPL"),
    ("scripts/run_single.py", "Research Orchestrator 单条查询运行脚本"),
    ("scripts/run_eval.py", "Research Orchestrator 标准评测脚本"),
    ("scripts/run_benchmark.py", "Research Orchestrator Benchmark"),
    ("scripts/run_ablation.py", "Research Orchestrator 消融实验脚本"),
    ("scripts/run_all_experiments.py", "Research Orchestrator 批量实验脚本"),
    ("scripts/run_evolution.py", "自进化训练脚本"),
    ("scripts/run_judge.py", "MiMo Judge 深度评分"),
    ("scripts/run_policy_head2head.py", "Run off/heuristic/learned head-to-head comparison"),
    ("scripts/run_real_evidence_iteration.py", "Run a gated real-evidence cache"),
    ("scripts/analyze_search_cache.py", "Analyze search-cache JSONL artifacts"),
    ("scripts/audit_paper_readiness.py", "Audit whether experiment outputs are paper-ready."),
    ("scripts/build_search_cache.py", "Build offline search-cache dataset from Research Orchestrator runs"),
    ("scripts/mine_natural_browser_queries.py", "Mine natural browser-positive queries from search-cache JSONL"),
    ("scripts/resurface_search_cache_citations.py", "Re-surface inline citations in existing search-cache JSONL files."),
    ("scripts/split_query_bank.py", "Create deterministic train/heldout query files from a JSONL query bank."),
    ("scripts/split_search_cache_dataset.py", "Create train / held-out splits from search-cache JSONL"),
    ("scripts/train_evidence_policy.py", "Train the learned evidence policy from exported transition datasets."),
    ("scripts/train_search_policy_iteration.py", "Train search policy only when cache quality passes readiness checks"),
    ("scripts/audit_research_readiness.py", "Audit repository-level research readiness signals"),
    ("scripts/index_research_outputs.py", "Index research-facing output artifacts"),
    ("scripts/render_research_brief.py", "Render a concise research brief from indexed outputs"),
    ("scripts/render_research_dashboard.py", "Render a research dashboard from indexed outputs"),
    ("scripts/refresh_research_outputs.py", "Refresh repository-level research reporting artifacts"),
    (
        "scripts/run_policy_head2head_cv.py",
        "Run query-level cross-fold policy preparation and optional head-to-head",
    ),
]


def _uses_argparse_argument_parser(path: Path) -> bool:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr == "ArgumentParser":
            return True
        if isinstance(func, ast.Name) and func.id == "ArgumentParser":
            return True
    return False


@pytest.mark.parametrize(
    ("script_path", "expected_text"),
    CLI_HELP_CASES,
)
def test_cli_help_is_clean(script_path: str, expected_text: str) -> None:
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"

    result = subprocess.run(
        [sys.executable, script_path, "--help"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        check=False,
    )

    assert result.returncode == 0
    assert expected_text in result.stdout
    assert "sentence-transformers not installed" not in result.stdout
    assert "sentence-transformers not installed" not in result.stderr


def test_all_argparse_scripts_have_help_smoke_coverage() -> None:
    covered_scripts = {script_path for script_path, _ in CLI_HELP_CASES}
    argparse_scripts = {
        str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")
        for path in SCRIPTS_DIR.glob("*.py")
        if path.name != "__init__.py" and _uses_argparse_argument_parser(path)
    }

    assert argparse_scripts == covered_scripts


def test_python_ci_smokes_every_covered_cli_help() -> None:
    workflow_text = PYTHON_CI_WORKFLOW.read_text(encoding="utf-8")

    for script_path, _ in CLI_HELP_CASES:
        assert f"python {script_path} --help" in workflow_text
