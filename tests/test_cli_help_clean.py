from __future__ import annotations

import subprocess
import sys
import os
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.mark.parametrize(
    ("script_path", "expected_text"),
    [
        ("scripts/run_repl.py", "Research Orchestrator 交互式 REPL"),
        ("scripts/run_single.py", "Research Orchestrator 单条查询运行脚本"),
        ("scripts/run_eval.py", "Research Orchestrator 标准评测脚本"),
        ("scripts/run_benchmark.py", "Research Orchestrator Benchmark"),
        ("scripts/run_ablation.py", "Research Orchestrator 消融实验脚本"),
    ],
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
