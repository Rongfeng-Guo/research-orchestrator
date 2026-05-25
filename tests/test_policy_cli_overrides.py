from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_eval import apply_policy_overrides as apply_eval_overrides
from scripts.run_single import apply_policy_overrides as apply_single_overrides
from scripts.build_search_cache import (  # noqa: E402
    apply_policy_overrides as apply_build_cache_overrides,
    build_cache_records,
    validate_real_evidence_preflight,
)
from src.orchestrator.schemas import ResearchReport  # noqa: E402


def _base_config() -> dict:
    return {
        "evidence_policy": {
            "mode": "heuristic",
            "model_path": "artifacts/evidence_policy.json",
        },
        "search_policy": {
            "enabled": False,
            "heuristic_fallback": True,
            "model_path": "artifacts/search_policy.json",
        },
    }


def test_run_single_overrides_to_learned_modes() -> None:
    config = _base_config()
    args = SimpleNamespace(
        evidence_policy_mode="learned",
        evidence_policy_model_path="artifacts/custom_evidence.json",
        search_policy_mode="learned",
        search_policy_model_path="artifacts/custom_search.json",
    )

    updated = apply_single_overrides(config, args)

    assert updated["evidence_policy"]["mode"] == "learned"
    assert updated["evidence_policy"]["model_path"] == "artifacts/custom_evidence.json"
    assert updated["search_policy"]["enabled"] is True
    assert updated["search_policy"]["heuristic_fallback"] is False
    assert updated["search_policy"]["model_path"] == "artifacts/custom_search.json"


def test_run_eval_overrides_disable_search_policy() -> None:
    config = _base_config()
    args = SimpleNamespace(
        evidence_policy_mode=None,
        evidence_policy_model_path=None,
        search_policy_mode="off",
        search_policy_model_path=None,
    )

    updated = apply_eval_overrides(config, args)

    assert updated["evidence_policy"]["mode"] == "heuristic"
    assert updated["search_policy"]["enabled"] is False


def test_build_search_cache_overrides_enable_heuristic_search_policy() -> None:
    config = _base_config()
    args = SimpleNamespace(
        evidence_policy_mode=None,
        evidence_policy_model_path=None,
        search_policy_mode="heuristic",
        search_policy_model_path="artifacts/search_policy_heuristic_probe.json",
    )

    updated = apply_build_cache_overrides(config, args)

    assert updated["search_policy"]["mode"] == "heuristic"
    assert updated["search_policy"]["enabled"] is True
    assert updated["search_policy"]["heuristic_fallback"] is True
    assert updated["search_policy"]["model_path"] == "artifacts/search_policy_heuristic_probe.json"


def test_build_search_cache_preflight_blocks_mock_evidence(monkeypatch) -> None:
    monkeypatch.delenv("SERPAPI_KEY", raising=False)
    monkeypatch.delenv("SEARCH_BACKEND", raising=False)
    config = {
        "tools": {
            "web_search": {
                "mock_mode": True,
            }
        }
    }

    failures = validate_real_evidence_preflight(config)

    assert "tools.web_search.mock_mode=true" in failures
    assert "missing_SERPAPI_KEY_for_SEARCH_BACKEND=serpapi" in failures


def test_build_search_cache_preflight_allows_real_search_key(monkeypatch) -> None:
    monkeypatch.setenv("SEARCH_BACKEND", "serpapi")
    monkeypatch.setenv("SERPAPI_KEY", "test-key")
    config = {
        "tools": {
            "web_search": {
                "mock_mode": False,
            }
        }
    }

    assert validate_real_evidence_preflight(config) == []
    assert validate_real_evidence_preflight(config, allow_mock_evidence=True) == []


def test_build_search_cache_preflight_blocks_missing_model_backend(monkeypatch) -> None:
    monkeypatch.setenv("SEARCH_BACKEND", "serpapi")
    monkeypatch.setenv("SERPAPI_KEY", "test-key")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    config = {
        "model": {
            "backend": "openai",
            "backend_mapping": {
                "solver": "openai",
            },
        },
        "tools": {
            "web_search": {
                "mock_mode": False,
            }
        },
    }

    failures = validate_real_evidence_preflight(config)

    assert failures == [
        "missing_OPENAI_API_KEY_or_OPENAI_BASE_URL_for_model_backend=openai"
    ]


def test_build_cache_records_marks_empty_failed_report_as_failed() -> None:
    class _Orchestrator:
        _results = []

    def _init_modules(_config, session_id=""):
        return {"orchestrator": _Orchestrator()}

    async def _run_research(_query, _config, _modules, return_report=False):
        report = ResearchReport(
            query=_query,
            content="Research failed due to persistent errors or global timeout.",
            confidence=0.0,
        )
        return "formatted failure", report

    records, manifest = build_cache_records(
        query_items=[{"id": "q1", "query": "test query"}],
        config={},
        config_path="config.yaml",
        source_label="unit",
        initialize_modules_fn=_init_modules,
        run_research_fn=_run_research,
    )

    assert records[0]["status"] == "failed"
    assert records[0]["error"] == "orchestrator_failed_or_empty_report"
    assert manifest["num_success"] == 0
    assert manifest["num_failed"] == 1


def test_build_cache_records_marks_all_failed_task_results_as_failed() -> None:
    class _Orchestrator:
        _results = [
            SimpleNamespace(
                task_id="task_1",
                status="failed",
                output="tool error",
                trajectory=[],
                action_log=[],
                token_usage=0,
                confidence=0.0,
                metadata={},
            )
        ]

    def _init_modules(_config, session_id=""):
        return {"orchestrator": _Orchestrator()}

    async def _run_research(_query, _config, _modules, return_report=False):
        report = ResearchReport(
            query=_query,
            content="A synthesized fallback summary was still produced.",
            confidence=0.0,
        )
        return "formatted fallback", report

    records, manifest = build_cache_records(
        query_items=[{"id": "q1", "query": "test query"}],
        config={},
        config_path="config.yaml",
        source_label="unit",
        initialize_modules_fn=_init_modules,
        run_research_fn=_run_research,
    )

    assert records[0]["status"] == "failed"
    assert records[0]["error"] == "orchestrator_failed_or_empty_report"
    assert manifest["num_success"] == 0
    assert manifest["num_failed"] == 1
