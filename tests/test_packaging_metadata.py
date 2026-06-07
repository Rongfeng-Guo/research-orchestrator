from __future__ import annotations

import importlib
import re
import tomllib
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

EXPECTED_CONSOLE_SCRIPTS = {
    "run-research": "scripts.run_single:main",
    "run-eval": "scripts.run_eval:main",
    "run-ablation": "scripts.run_ablation:main",
    "run-benchmark": "scripts.run_benchmark:main",
    "run-judge": "scripts.run_judge:main",
    "run-evolution": "scripts.run_evolution:main",
}


def _dependency_name(requirement: str) -> str:
    match = re.match(r"[A-Za-z0-9_.-]+", requirement.strip())
    assert match is not None, f"Could not parse dependency name from {requirement!r}"
    return match.group(0).lower().replace("_", "-")


def _active_requirements_txt_names() -> set[str]:
    names: set[str] = set()
    for line in (PROJECT_ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines():
        requirement = line.split("#", 1)[0].strip()
        if not requirement or requirement.startswith("-"):
            continue
        names.add(_dependency_name(requirement))
    return names


def test_pyproject_identity_matches_repository() -> None:
    pyproject = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    project = pyproject["project"]

    assert project["name"] == "research-orchestrator"
    assert project["version"] == "0.1.1"
    assert project["urls"]["Repository"] == "https://github.com/Rongfeng-Guo/research-orchestrator"
    assert project["authors"][0]["name"] == "Rongfeng Guo"


def test_mit_license_file_is_present() -> None:
    license_text = (PROJECT_ROOT / "LICENSE").read_text(encoding="utf-8")

    assert "MIT License" in license_text
    assert "Copyright (c) 2026 Rongfeng Guo" in license_text


def test_pyproject_runtime_dependencies_cover_active_requirements_txt() -> None:
    pyproject = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    pyproject_names = {_dependency_name(item) for item in pyproject["project"]["dependencies"]}

    assert _active_requirements_txt_names() <= pyproject_names


def test_console_script_registry_is_stable() -> None:
    pyproject = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert pyproject["project"]["scripts"] == EXPECTED_CONSOLE_SCRIPTS


def test_console_script_targets_are_importable() -> None:
    pyproject = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    scripts = pyproject["project"]["scripts"]

    for target in scripts.values():
        module_name, func_name = target.split(":")
        module = importlib.import_module(module_name)
        assert hasattr(module, func_name)
