from __future__ import annotations

import importlib
import tomllib
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_pyproject_identity_matches_repository() -> None:
    pyproject = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    project = pyproject["project"]

    assert project["name"] == "research-orchestrator"
    assert project["version"] == "0.1.1"
    assert project["urls"]["Repository"] == "https://github.com/Rongfeng-Guo/research-orchestrator"
    assert project["authors"][0]["name"] == "Rongfeng Guo"


def test_console_script_targets_are_importable() -> None:
    pyproject = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    scripts = pyproject["project"]["scripts"]

    for target in scripts.values():
        module_name, func_name = target.split(":")
        module = importlib.import_module(module_name)
        assert hasattr(module, func_name)
