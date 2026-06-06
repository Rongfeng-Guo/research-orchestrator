from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.index_research_outputs import build_index, write_markdown  # noqa: E402


def test_build_index_finds_research_artifacts(sample_research_outputs: Path) -> None:
    payload = build_index(sample_research_outputs)

    assert payload["artifact_count"] >= 2
    artifact_types = {item["artifact_type"] for item in payload["artifacts"]}
    assert "research_audit" in artifact_types
    assert "policy_head2head" in artifact_types
    assert "policy_head2head_cv" in artifact_types

    head2head_entries = [item for item in payload["artifacts"] if item["artifact_type"] == "policy_head2head"]
    assert any(item.get("blocked_by_preflight") for item in head2head_entries)

    cv_entries = [item for item in payload["artifacts"] if item["artifact_type"] == "policy_head2head_cv"]
    assert cv_entries[0]["fold_count"] == 4
    assert cv_entries[0]["quality_delta"] == 0.055
    assert cv_entries[0]["macro_quality_delta"] == 0.05
    assert cv_entries[0]["domains"] == ["finance", "medicine", "technology"]


def test_write_markdown_includes_cross_fold_highlights(sample_research_outputs: Path, tmp_path: Path) -> None:
    payload = build_index(sample_research_outputs)
    output_path = tmp_path / "research_output_index.md"

    write_markdown(payload, output_path)
    markdown = output_path.read_text(encoding="utf-8")

    assert "policy_head2head_cv" in markdown
    assert "folds=4" in markdown
    assert "quality_delta=+0.0550" in markdown
