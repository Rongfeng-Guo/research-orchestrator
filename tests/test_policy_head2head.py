from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_policy_head2head import _apply_mode_overrides, _evaluate_report, _render_markdown


def _base_config() -> dict:
    return {
        "evidence_policy": {
            "mode": "heuristic",
            "model_path": "artifacts/evidence_policy.json",
        },
        "search_policy": {
            "enabled": True,
            "heuristic_fallback": True,
            "model_path": "artifacts/search_policy.json",
        },
    }


def test_apply_mode_overrides_sets_expected_flags() -> None:
    args = SimpleNamespace(
        evidence_policy_mode=None,
        evidence_policy_model_path=None,
        search_policy_model_path="artifacts/new_search_policy.json",
    )

    off_cfg = _apply_mode_overrides(_base_config(), "off", args)
    heur_cfg = _apply_mode_overrides(_base_config(), "heuristic", args)
    learned_cfg = _apply_mode_overrides(_base_config(), "learned", args)

    assert off_cfg["search_policy"]["enabled"] is False
    assert off_cfg["search_policy"]["mode"] == "off"
    assert heur_cfg["search_policy"]["enabled"] is True
    assert heur_cfg["search_policy"]["mode"] == "heuristic"
    assert heur_cfg["search_policy"]["heuristic_fallback"] is True
    assert learned_cfg["search_policy"]["enabled"] is True
    assert learned_cfg["search_policy"]["mode"] == "learned"
    assert learned_cfg["search_policy"]["heuristic_fallback"] is False
    assert learned_cfg["search_policy"]["model_path"] == "artifacts/new_search_policy.json"


def test_render_markdown_contains_conclusion_hints() -> None:
    payload = {
        "created_at": "2026-05-14T00:00:00",
        "config_path": "configs/aliyun_smoke.yaml",
        "query_count": 3,
        "runs": [
            {"mode": "off", "summary": {"num_success": 3, "num_total": 3, "avg_composite_score": 0.40, "avg_factual_accuracy": 0.42, "avg_citation_coverage": 0.35, "avg_search_policy_score": 0.41, "avg_estimated_token_cost": 100.0, "avg_tool_calls": 1.0}},
            {"mode": "heuristic", "summary": {"num_success": 3, "num_total": 3, "avg_composite_score": 0.50, "avg_factual_accuracy": 0.52, "avg_citation_coverage": 0.45, "avg_search_policy_score": 0.51, "avg_estimated_token_cost": 95.0, "avg_tool_calls": 1.2, "avg_policy_advice_count": 1.0, "avg_guardrail_trigger_count": 0.0, "avg_policy_enforce_stop_count": 0.1}},
            {"mode": "learned", "summary": {"num_success": 3, "num_total": 3, "avg_composite_score": 0.58, "avg_factual_accuracy": 0.60, "avg_citation_coverage": 0.50, "avg_search_policy_score": 0.59, "avg_estimated_token_cost": 90.0, "avg_tool_calls": 1.1, "avg_policy_advice_count": 1.8, "avg_guardrail_trigger_count": 0.7, "avg_policy_enforce_stop_count": 0.4}},
        ],
    }

    md = _render_markdown(payload)
    assert "Best quality mode" in md
    assert "`learned`" in md
    assert "Learned vs Heuristic" in md
    assert "citation_delta" in md
    assert "Policy Instrumentation" in md
    assert "avg_guardrail_triggers" in md


def test_evaluate_report_uses_question_id_and_metric_keys(monkeypatch) -> None:
    calls: list[str | None] = []

    class DummyBench:
        def evaluate_report(self, report, question_id=None, report_metadata=None):  # noqa: ANN001
            calls.append(question_id)
            return {
                "metrics": {
                    "factual_accuracy": 0.61,
                    "citation_coverage": 0.42,
                    "logical_consistency": 0.73,
                },
                "search_policy_metrics": {
                    "search_policy_score": 0.58,
                    "budget_efficiency": 0.67,
                },
                "composite_score": 0.55,
            }

    class DummyReport:
        metadata = {"search_cost": {"tool_calls": 2}}

    monkeypatch.setattr("scripts.run_policy_head2head.ResearchBench", DummyBench)

    result = _evaluate_report(DummyReport(), "tech_001")

    assert calls == ["tech_001"]
    assert result["composite_score"] == 0.55
    assert result["logical_consistency"] == 0.73
    assert result["search_policy_score"] == 0.58


def test_evaluate_report_falls_back_for_custom_query_items(monkeypatch) -> None:
    class DummyBench:
        questions: list[dict[str, str]] = []

        def evaluate_report(self, report, question_id=None, report_metadata=None):  # noqa: ANN001
            raise ValueError(f"未找到题目 ID: {question_id}")

        def _extract_report_payload(self, report, report_metadata=None):  # noqa: ANN001
            metadata = dict(getattr(report, "metadata", {}) or {})
            if isinstance(report_metadata, dict):
                metadata.update(report_metadata)
            return getattr(report, "content", ""), metadata, getattr(report, "sources", [])

    class DummyReport:
        content = "NIST AI Risk Management Framework [1]. UK AI Safety Institute [2]. FTC voice cloning [3]."
        metadata = {"search_cost": {"tool_calls": 2}}
        sources = []

    monkeypatch.setattr("scripts.run_policy_head2head.ResearchBench", DummyBench)

    result = _evaluate_report(
        DummyReport(),
        {
            "id": "custom_001",
            "query": "Compare NIST, UK AISI, and FTC voice cloning approaches.",
            "expected_topics": ["NIST", "AI Safety Institute", "FTC voice cloning"],
            "domain": "法律治理",
        },
    )

    assert result["composite_score"] >= 0.0
    assert result["logical_consistency"] >= 0.0
    assert result["citation_coverage"] >= 0.0
