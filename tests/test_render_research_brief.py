from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.render_research_brief import build_brief, write_markdown  # noqa: E402
from scripts.index_research_outputs import build_index  # noqa: E402


def test_build_brief_extracts_blockers_and_recommendation(sample_research_outputs: Path) -> None:
    index_payload = build_index(sample_research_outputs)

    brief = build_brief(index_payload)

    assert brief["recommended_policy_artifact"] == "search_policy_20260515_natural_train_v1.json"
    assert brief["blocked_by_preflight"] is True
    assert brief["missing_backends"] == ["openai"]
    assert brief["head2head_query_count"] == 5
    assert brief["cross_fold_label"] == "policy_head2head_cv_fast_probe8_live"
    assert brief["cross_fold_count"] == 4
    assert brief["cross_fold_query_count"] == 8
    assert brief["cross_fold_quality_delta"] == 0.055
    assert brief["cross_fold_macro_quality_delta"] == 0.05
    assert brief["cross_fold_domains"] == ["finance", "medicine", "technology"]


def test_build_brief_uses_index_summary_when_artifact_files_are_missing() -> None:
    index_payload = {
        "artifacts": [
            {
                "artifact_type": "policy_head2head",
                "label": "policy_head2head_natural_train_v1_balanced",
                "path_json": "outputs/missing/head2head.json",
                "query_count": 5,
                "sampling_strategy": "balanced_domains",
                "quality_delta": 0.0,
                "macro_quality_delta": 0.0,
                "blocked_by_preflight": True,
                "missing_backends": ["openai"],
                "domains": ["technology", "medicine", "finance"],
            },
            {
                "artifact_type": "research_audit",
                "label": "Research Readiness Audit",
                "path_json": "outputs/missing/research_readiness_audit.json",
            },
        ]
    }

    brief = build_brief(index_payload)

    assert brief["recommended_policy_artifact"] is None
    assert brief["blocked_by_preflight"] is True
    assert brief["missing_backends"] == ["openai"]
    assert brief["head2head_query_count"] == 5
    assert brief["cross_fold_label"] is None


def test_write_markdown_includes_cross_fold_summary(sample_research_outputs: Path, tmp_path: Path) -> None:
    brief = build_brief(build_index(sample_research_outputs))
    output_path = tmp_path / "research_brief.md"

    write_markdown(brief, output_path)
    markdown = output_path.read_text(encoding="utf-8")

    assert "Latest cross-fold run `policy_head2head_cv_fast_probe8_live`" in markdown
    assert "quality_delta=+0.0550" in markdown
    assert "macro_quality_delta=+0.0500" in markdown
    assert "Cross-fold evidence" in markdown
