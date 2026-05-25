from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.metrics.rule_based import RuleBasedMetrics  # noqa: E402
from src.agents.summarizer import SummarizerAgent  # noqa: E402
from src.core.runner import _format_report  # noqa: E402
from src.orchestrator.schemas import AgentResult, AgentStatus  # noqa: E402


def _successful_result(task_id: str, url: str) -> AgentResult:
    return AgentResult(
        task_id=task_id,
        status=AgentStatus.SUCCESS,
        output="sub-task output",
        trajectory=[
            {
                "role": "tool",
                "result": {
                    "results": [
                        {
                            "url": url,
                            "title": f"source for {task_id}",
                            "snippet": "snippet",
                        }
                    ]
                },
            }
        ],
        confidence=0.8,
    )


def _source(url: str, title: str = "source") -> dict[str, str]:
    return {"url": url, "title": title, "snippet": "snippet"}


def test_parse_report_surfaces_explicit_citations_into_report_content() -> None:
    agent = SummarizerAgent(name="summarizer", policy=None)
    content = (
        "## 执行摘要\n"
        "这是第一段完整的分析内容，用于验证在正文中自动补显式引用标记，而不是只在末尾列参考来源。\n"
        "## 关键发现\n"
        "这是第二段分析内容，同样没有任何现成引用，需要由后处理写入 citation marker。\n"
        "Overall Confidence: 0.80"
    )

    report = agent._parse_report(
        "测试 query",
        content,
        [_successful_result("task_1", "https://example.com/source-a")],
    )

    assert "[1]" in report.content
    assert "## 参考来源" in report.content
    assert "https://example.com/source-a" in report.content
    assert RuleBasedMetrics.citation_coverage(report.content) > 0.2


def test_inject_inline_citations_spreads_markers_across_long_reports() -> None:
    agent = SummarizerAgent(name="summarizer", policy=None)
    body = "\n".join(
        f"这是第 {idx} 段较长的事实性分析内容，用于验证长报告不会只在开头象征性补少量引用。"
        for idx in range(1, 13)
    )

    updated = agent._surface_explicit_citations(
        body,
        [
            _source("https://example.com/a", "A"),
            _source("https://example.org/b", "B"),
            _source("https://example.net/c", "C"),
        ],
    )

    assert updated.count("[1]") >= 1
    assert updated.count("[2]") >= 1
    assert updated.count("[3]") >= 1
    assert sum(updated.count(f"[{idx}]") for idx in range(1, 4)) >= 6
    assert "## 参考来源" in updated


def test_format_report_does_not_duplicate_reference_section() -> None:
    agent = SummarizerAgent(name="summarizer", policy=None)
    report = agent._parse_report(
        "测试 query",
        "这是唯一一段足够长的正文内容，用于检查最终格式化时不会重复附加第二份参考来源列表。Overall Confidence: 0.70",
        [_successful_result("task_1", "https://example.com/source-a")],
    )

    formatted = _format_report(report, elapsed=1.23)

    assert formatted.count("## 参考来源") == 1


def test_parse_report_injects_body_citations_even_with_existing_bold_reference_block() -> None:
    agent = SummarizerAgent(name="summarizer", policy=None)
    content = (
        "# 报告标题\n\n"
        "## 执行摘要\n"
        "这是第一段较长的正文内容，用于模拟真实 cache 中已经拿到来源、但最终报告只在尾部写了参考文献而没有正文显式 citation 的情况。\n\n"
        "## 分析\n"
        "这是第二段较长的正文内容，继续描述结论和影响，但仍然没有任何正文引用标记。\n\n"
        "**参考文献**\n"
        "1. 示例来源 A. https://example.com/source-a\n"
        "2. 示例来源 B. https://example.com/source-b\n\n"
        "Overall Confidence: 0.81"
    )

    report = agent._parse_report(
        "测试 query",
        content,
        [_successful_result("task_1", "https://example.com/source-a")],
    )

    body, refs = agent._split_reference_section(report.content)

    assert "[1]" in body
    assert "**参考文献**" in refs
    assert "## 参考来源" not in report.content
    assert report.content.count("https://example.com/source-a") == 1
    assert RuleBasedMetrics.citation_coverage(report.content) > 0.2


def test_format_report_recognizes_bold_reference_section() -> None:
    agent = SummarizerAgent(name="summarizer", policy=None)
    report = agent._parse_report(
        "测试 query",
        (
            "这是正文内容，包含足够长的分析文字，用于检查格式化器不会在已有加粗参考文献区之后重复追加参考来源。\n\n"
            "**参考文献**\n"
            "1. 示例来源 A. https://example.com/source-a\n"
            "Overall Confidence: 0.70"
        ),
        [_successful_result("task_1", "https://example.com/source-a")],
    )

    formatted = _format_report(report, elapsed=1.23)

    assert "**参考文献**" in formatted
    assert "## 参考来源" not in formatted
