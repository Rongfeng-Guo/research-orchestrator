from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.refresh_research_outputs import refresh_research_outputs  # noqa: E402


def test_refresh_research_outputs_rebuilds_full_reporting_chain(tmp_path: Path) -> None:
    outputs_dir = tmp_path / "outputs"

    manifest = refresh_research_outputs(PROJECT_ROOT, outputs_dir, source_outputs_dir=PROJECT_ROOT / "outputs")

    assert manifest["headline"]["recommended_policy_artifact"] == "search_policy_20260515_natural_train_v1.json"
    assert manifest["headline"]["head2head_blocked_by_preflight"] is True
    assert manifest["headline"]["missing_backends"] == ["openai"]
    assert len(manifest["generated"]) == 4
    assert manifest["source_outputs_dir"].endswith("research-orchestrator\\outputs")

    generated_paths = [
        outputs_dir / "research_audit" / "research_readiness_audit.json",
        outputs_dir / "research_index" / "research_output_index.json",
        outputs_dir / "research_brief" / "research_brief.json",
        outputs_dir / "research_dashboard" / "research_dashboard.json",
        outputs_dir / "research_dashboard" / "research_dashboard.html",
        outputs_dir / "research_refresh_manifest.json",
    ]
    for path in generated_paths:
        assert path.exists(), f"missing generated path: {path}"
