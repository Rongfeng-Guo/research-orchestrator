from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.render_research_brief import build_brief  # noqa: E402
from scripts.index_research_outputs import build_index  # noqa: E402


def test_build_brief_extracts_blockers_and_recommendation(sample_research_outputs: Path) -> None:
    index_payload = build_index(sample_research_outputs)

    brief = build_brief(index_payload)

    assert brief["recommended_policy_artifact"] == "search_policy_20260515_natural_train_v1.json"
    assert brief["blocked_by_preflight"] is True
    assert brief["missing_backends"] == ["openai"]
    assert brief["head2head_query_count"] == 5


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
