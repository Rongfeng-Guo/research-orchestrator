from __future__ import annotations

import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_run_repl_help_is_clean() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/run_repl.py", "--help"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "Research Orchestrator 交互式 REPL" in result.stdout
    assert "sentence-transformers not installed" not in result.stdout
    assert "sentence-transformers not installed" not in result.stderr
