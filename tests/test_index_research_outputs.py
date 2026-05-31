from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.index_research_outputs import build_index  # noqa: E402


def test_build_index_finds_research_artifacts() -> None:
    payload = build_index(PROJECT_ROOT / "outputs")

    assert payload["artifact_count"] >= 2
    artifact_types = {item["artifact_type"] for item in payload["artifacts"]}
    assert "research_audit" in artifact_types
    assert "policy_head2head" in artifact_types

    head2head_entries = [item for item in payload["artifacts"] if item["artifact_type"] == "policy_head2head"]
    assert any(item.get("blocked_by_preflight") for item in head2head_entries)
