from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.resurface_search_cache_citations import analysis_stem_for_output, resurface_records  # noqa: E402


def _record() -> dict:
    return {
        "status": "success",
        "cache_id": "cache_001",
        "query": "测试 query",
        "query_id": "tech_001",
        "confidence": 0.6,
        "report_content": (
            "## 执行摘要\n"
            "这是第一段较长正文，没有显式引用，但尾部已经有参考文献和 URL。\n\n"
            "## 分析\n"
            "这是第二段较长正文，同样没有显式引用。\n\n"
            "**参考文献**\n"
            "1. 示例来源 A. https://example.com/source-a\n"
        ),
        "sources": [
            {
                "url": "https://example.com/source-a",
                "title": "source a",
                "snippet": "snippet",
            }
        ],
        "report_metadata": {
            "policy_trace": [
                {"action_type": "assistant_response", "tool_calls_count": 1},
                {
                    "action_type": "tool_call",
                    "tool_name": "browser",
                    "result_count": 1,
                    "top_urls": ["https://example.com/source-a"],
                },
                {"action_type": "stop", "stop_reason": "done"},
            ],
        },
        "evaluation": {
            "metrics": {
                "factual_accuracy": 0.4,
                "logical_consistency": 0.8,
                "citation_coverage": 0.0,
                "bias": 0.9,
                "comprehensiveness": 0.5,
            },
            "search_policy_metrics": {
                "citation_grounding": 1.0,
            },
            "composite_score": 0.5,
        },
        "reward_breakdown": {"reward": 0.0},
        "reward": 0.0,
    }


def test_resurface_records_updates_content_metrics_and_reward() -> None:
    updated_records, summary = resurface_records([_record()])
    updated = updated_records[0]

    assert "[1]" in updated["report_content"]
    assert updated["evaluation"]["metrics"]["citation_coverage"] > 0.0
    assert updated["evaluation"]["search_policy_metrics"]["citation_grounding"] == 1.0
    assert updated["citation_resurface"]["citation_delta"] > 0.0
    assert updated["citation_resurface"]["content_changed"] is True
    assert "reward_breakdown" in updated
    assert summary["num_changed"] == 1
    assert summary["num_citation_improved"] == 1


def test_resurface_cli_writes_jsonl_and_analysis(tmp_path: Path) -> None:
    input_path = tmp_path / "input.jsonl"
    output_path = tmp_path / "output.jsonl"
    analysis_dir = tmp_path / "analysis"
    input_path.write_text(json.dumps(_record(), ensure_ascii=False) + "\n", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "resurface_search_cache_citations.py"),
            "--inputs",
            str(input_path),
            "--output-file",
            str(output_path),
            "--analysis-output-dir",
            str(analysis_dir),
        ],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    updated = json.loads(output_path.read_text(encoding="utf-8").splitlines()[0])
    analysis_path = analysis_dir / "search_cache_analysis_output.md"

    assert "[resurface] JSONL written" in result.stdout
    assert "[1]" in updated["report_content"]
    assert updated["citation_resurface"]["citation_delta"] > 0.0
    assert analysis_path.exists()


def test_analysis_stem_for_output_truncates_long_names() -> None:
    long_name = "x" * 160

    stem = analysis_stem_for_output(
        Path(f"{long_name}.jsonl"),
        output_file_requested=True,
        timestamp="20260520_120000",
    )

    assert stem.startswith("search_cache_analysis_")
    assert len(stem) <= 48
