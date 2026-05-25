from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.summarizer import SummarizerAgent  # noqa: E402
from src.orchestrator.schemas import AgentResult, AgentStatus  # noqa: E402


def test_parse_report_aggregates_policy_stats_from_subtasks() -> None:
    agent = SummarizerAgent(name="summarizer", policy=None)
    results = [
        AgentResult(
            task_id="task_1",
            status=AgentStatus.SUCCESS,
            output="Result 1",
            confidence=0.8,
            metadata={
                "search_cost": {
                    "tool_calls": 2,
                    "search_calls": 1,
                    "browser_calls": 1,
                    "estimated_token_cost": 100,
                    "estimated_tool_cost": 40,
                    "assistant_turns": 2,
                },
                "route_stats": {
                    "tool_counts": {"web_search": 1, "browser": 1},
                    "queries_used": ["query a"],
                    "search_backends": ["bing_html"],
                    "domains_seen": ["example.com"],
                    "top_urls": ["https://example.com/a"],
                },
                "policy_stats": {
                    "policy_advice_count": 2,
                    "policy_continue_count": 1,
                    "policy_enforce_stop_count": 1,
                    "policy_stage_counts": {"assistant_no_tool": 1, "post_tool": 1},
                    "recommended_action_counts": {"search": 1, "stop": 1},
                    "guardrail_trigger_count": 1,
                    "guardrail_reasons": {"no_search_yet_force_search": 1},
                    "stop_signal_reasons": {"learned_policy_stop": 1},
                    "tool_call_truncation_count": 0,
                },
            },
            action_log=[{"action_type": "policy_advice", "guardrail_reason": "no_search_yet_force_search"}],
        ),
        AgentResult(
            task_id="task_2",
            status=AgentStatus.SUCCESS,
            output="Result 2",
            confidence=0.7,
            metadata={
                "search_cost": {
                    "tool_calls": 1,
                    "search_calls": 1,
                    "browser_calls": 0,
                    "estimated_token_cost": 50,
                    "estimated_tool_cost": 20,
                    "assistant_turns": 1,
                },
                "route_stats": {
                    "tool_counts": {"web_search": 1},
                    "queries_used": ["query b"],
                    "search_backends": ["bing_html"],
                    "domains_seen": ["example.org"],
                    "top_urls": ["https://example.org/b"],
                },
                "policy_stats": {
                    "policy_advice_count": 1,
                    "policy_continue_count": 0,
                    "policy_enforce_stop_count": 0,
                    "policy_stage_counts": {"post_tool": 1},
                    "recommended_action_counts": {"stop": 1},
                    "guardrail_trigger_count": 1,
                    "guardrail_reasons": {"repetitive_search_force_stop": 1},
                    "stop_signal_reasons": {"search_limit": 1},
                    "tool_call_truncation_count": 1,
                },
            },
            action_log=[{"action_type": "policy_advice", "guardrail_reason": "repetitive_search_force_stop"}],
        ),
    ]

    report = agent._parse_report("test query", "Body\nOverall Confidence: 0.90", results)
    policy_stats = report.metadata["policy_stats"]

    assert policy_stats["policy_advice_count"] == 3
    assert policy_stats["policy_continue_count"] == 1
    assert policy_stats["policy_enforce_stop_count"] == 1
    assert policy_stats["guardrail_trigger_count"] == 2
    assert policy_stats["tool_call_truncation_count"] == 1
    assert policy_stats["policy_stage_counts"] == {"assistant_no_tool": 1, "post_tool": 2}
    assert policy_stats["recommended_action_counts"] == {"search": 1, "stop": 2}
    assert policy_stats["guardrail_reasons"] == {
        "no_search_yet_force_search": 1,
        "repetitive_search_force_stop": 1,
    }
    assert policy_stats["stop_signal_reasons"] == {
        "learned_policy_stop": 1,
        "search_limit": 1,
    }
