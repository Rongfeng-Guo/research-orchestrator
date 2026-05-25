from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.metrics.search_policy import SearchPolicyMetrics  # noqa: E402
from src.agents.researcher import ResearcherAgent  # noqa: E402
from src.core.runner import create_search_policy, resolve_search_policy_mode  # noqa: E402
from src.orchestrator.schemas import AgentStatus, SubTask, TaskType  # noqa: E402
from src.search_policy import LearnedSearchPolicy, SearchPolicyDatasetBuilder, SearchPolicyTrainer  # noqa: E402


def _reward_breakdown() -> dict[str, float]:
    return {
        "process_score": 0.72,
        "efficiency_reward": 0.81,
        "grounding_reward": 0.76,
        "policy_score_reward": 0.69,
    }


def _record_with_browser_route() -> dict:
    return {
        "status": "success",
        "cache_id": "cache_browser_route",
        "query": "比较 GPT-4o 与 Claude 3.5 的网页评测差异",
        "reward": 0.58,
        "reward_breakdown": _reward_breakdown(),
        "report_metadata": {
            "policy_trace": [
                {
                    "task_id": "task_1",
                    "turn": 0,
                    "action_type": "assistant_response",
                    "tool_calls_count": 1,
                },
                {
                    "task_id": "task_1",
                    "turn": 0,
                    "action_type": "tool_call",
                    "tool_name": "web_search",
                    "query_text": "GPT-4o Claude 3.5 benchmark",
                    "backend": "mock",
                    "latency_ms": 120,
                    "result_count": 5,
                    "top_urls": [
                        "https://example.com/a",
                        "https://example.org/b",
                    ],
                    "estimated_token_cost": 90,
                },
                {
                    "task_id": "task_1",
                    "turn": 1,
                    "action_type": "assistant_response",
                    "tool_calls_count": 1,
                },
                {
                    "task_id": "task_1",
                    "turn": 1,
                    "action_type": "tool_call",
                    "tool_name": "browser",
                    "query_text": "",
                    "backend": "browser",
                    "latency_ms": 210,
                    "result_count": 1,
                    "top_urls": ["https://example.com/a"],
                    "estimated_token_cost": 60,
                },
                {
                    "task_id": "task_1",
                    "turn": 1,
                    "action_type": "stop_signal",
                    "stop_reason": "search_limit",
                },
                {
                    "task_id": "task_1",
                    "turn": 2,
                    "action_type": "stop",
                    "stop_reason": "model_finished_no_tool_calls",
                },
            ]
        },
    }


def _manual_row(label_action: str, reward: float, **features: float) -> dict:
    template = {name: 0.0 for name in SearchPolicyDatasetBuilder.FEATURE_NAMES}
    template.update(features)
    return {
        "label_action": label_action,
        "reward": reward,
        "features": template,
        "feature_vector": [
            float(template[name]) for name in SearchPolicyDatasetBuilder.FEATURE_NAMES
        ],
    }


def _training_rows() -> list[dict]:
    return [
        _manual_row(
            "search",
            0.40,
            tool_calls_so_far=0.0,
            search_calls_so_far=0.0,
            browser_calls_so_far=0.0,
            assistant_turns_so_far=1.0,
        ),
        _manual_row(
            "search",
            0.55,
            tool_calls_so_far=1.0,
            search_calls_so_far=1.0,
            browser_calls_so_far=0.0,
            unique_query_count=1.0,
            unique_domain_count=1.0,
            budget_ratio=0.10,
            recent_query_overlap=0.15,
        ),
        _manual_row(
            "browser",
            0.70,
            tool_calls_so_far=1.0,
            search_calls_so_far=1.0,
            browser_calls_so_far=0.0,
            successful_tool_call_ratio=1.0,
            avg_result_count=4.0,
            last_result_count=4.0,
            unique_query_count=1.0,
            unique_domain_count=1.0,
            last_action_was_search=1.0,
            browser_available=1.0,
        ),
        _manual_row(
            "browser",
            0.82,
            tool_calls_so_far=1.0,
            search_calls_so_far=1.0,
            browser_calls_so_far=0.0,
            successful_tool_call_ratio=1.0,
            avg_result_count=6.0,
            last_result_count=6.0,
            unique_query_count=1.0,
            unique_domain_count=2.0,
            last_action_was_search=1.0,
            browser_available=1.0,
        ),
        _manual_row(
            "stop",
            0.88,
            tool_calls_so_far=2.0,
            search_calls_so_far=2.0,
            browser_calls_so_far=1.0,
            successful_tool_call_ratio=1.0,
            unique_query_count=2.0,
            unique_domain_count=3.0,
            budget_ratio=0.92,
            has_stop_signal=1.0,
        ),
        _manual_row(
            "stop",
            0.78,
            tool_calls_so_far=2.0,
            search_calls_so_far=2.0,
            browser_calls_so_far=0.0,
            empty_result_ratio=0.50,
            unique_query_count=2.0,
            unique_domain_count=1.0,
            budget_ratio=0.86,
            recent_query_overlap=0.84,
        ),
    ]


def test_search_policy_dataset_builder_extracts_step_rows() -> None:
    builder = SearchPolicyDatasetBuilder()

    rows = builder.build_rows_from_records([_record_with_browser_route()])

    assert [row["label_action"] for row in rows] == ["search", "browser", "stop"]
    assert rows[0]["features"]["tool_calls_so_far"] == 0.0
    assert rows[1]["features"]["search_calls_so_far"] == 1.0
    assert rows[1]["features"]["browser_available"] == 1.0
    assert rows[1]["features"]["last_action_was_search"] == 1.0
    assert rows[2]["features"]["tool_calls_so_far"] == 2.0
    assert rows[2]["features"]["has_stop_signal"] == 1.0

    summary = builder.summarize(rows)
    assert summary["num_examples"] == 3
    assert summary["label_distribution"]["search"] == 1
    assert summary["label_distribution"]["browser"] == 1
    assert summary["label_distribution"]["stop"] == 1


def test_search_policy_metrics_track_citation_source_quality() -> None:
    report = (
        "第一段说明模型能力差异。 [1]\n\n"
        "第二段补充评测方法和限制。 [2]\n\n"
        "第三段给出结论。\n\n"
        "## 参考来源\n"
        "1. [A](https://example.com/a)\n"
        "2. [B](https://example.org/b)\n"
    )
    metadata = {
        "policy_trace": [
            {
                "action_type": "tool_call",
                "tool_name": "web_search",
                "backend": "serpapi",
                "result_count": 2,
                "top_urls": ["https://example.com/a", "https://example.org/b"],
            }
        ]
    }
    sources = [
        {"url": "https://example.com/a", "title": "A"},
        {"url": "https://example.org/b", "title": "B"},
    ]

    metrics = SearchPolicyMetrics.breakdown(report, metadata, sources)

    assert metrics["inline_citation_count"] == 2.0
    assert metrics["unique_inline_citation_count"] == 2.0
    assert metrics["reference_source_count"] == 2.0
    assert metrics["cited_source_count"] == 2.0
    assert metrics["source_utilization"] == 1.0
    assert metrics["citation_density"] > 0.0
    assert metrics["citation_quality_score"] > 0.5


def test_search_policy_trainer_fit_save_load_roundtrip(tmp_path: Path) -> None:
    rows = _training_rows()
    model_path = tmp_path / "search_policy.json"

    trainer = SearchPolicyTrainer(model_path=model_path)
    stats = trainer.fit(
        rows,
        epochs=320,
        learning_rate=0.18,
        l2=1e-4,
        validation_ratio=0.0,
    )

    assert stats["train_examples"] == len(rows)
    assert stats["train_accuracy"] >= 0.83
    assert model_path.exists()

    predictions = trainer.predict_rows(rows)
    assert predictions.count("browser") >= 2
    assert predictions.count("stop") >= 2

    loaded = SearchPolicyTrainer.load(model_path)
    assert loaded.predict_rows(rows) == predictions


def test_learned_search_policy_runtime_recommends_browser_after_search(tmp_path: Path) -> None:
    model_path = tmp_path / "search_policy.json"
    trainer = SearchPolicyTrainer(model_path=model_path)
    trainer.fit(
        _training_rows(),
        epochs=320,
        learning_rate=0.18,
        l2=1e-4,
        validation_ratio=0.0,
    )

    policy = LearnedSearchPolicy(
        model_path=model_path,
        continue_threshold=0.55,
        stop_threshold=0.70,
        heuristic_fallback=False,
    )
    state = policy.new_state()
    policy.observe_action(state, {
        "action_type": "assistant_response",
        "tool_calls_count": 1,
    })
    policy.observe_action(state, {
        "action_type": "tool_call",
        "tool_name": "web_search",
        "query_text": "GPT-4o Claude 3.5 benchmark",
        "backend": "mock",
        "latency_ms": 120,
        "result_count": 5,
        "top_urls": ["https://example.com/a"],
        "estimated_token_cost": 90,
    })

    recommendation = policy.recommend(state)
    assert recommendation["label"] == "browser"
    assert recommendation["can_continue"] is True
    assert recommendation["browser_url"] == "https://example.com/a"


def test_learned_search_policy_runtime_prefers_non_pdf_browser_url(tmp_path: Path) -> None:
    model_path = tmp_path / "search_policy.json"
    trainer = SearchPolicyTrainer(model_path=model_path)
    trainer.fit(
        _training_rows(),
        epochs=320,
        learning_rate=0.18,
        l2=1e-4,
        validation_ratio=0.0,
    )

    policy = LearnedSearchPolicy(
        model_path=model_path,
        continue_threshold=0.55,
        stop_threshold=0.70,
        heuristic_fallback=False,
    )
    state = policy.new_state()
    policy.observe_action(state, {
        "action_type": "assistant_response",
        "tool_calls_count": 1,
    })
    policy.observe_action(state, {
        "action_type": "tool_call",
        "tool_name": "web_search",
        "query_text": "AI regulation official sources",
        "backend": "bing_html",
        "latency_ms": 120,
        "result_count": 3,
        "top_urls": [
            "https://example.com/policy.pdf",
            "https://official.example.org/policy",
            "https://another.example.net/report",
        ],
        "estimated_token_cost": 90,
    })

    recommendation = policy.recommend(state)
    assert recommendation["label"] == "browser"
    assert recommendation["can_continue"] is True
    assert recommendation["browser_url"] == "https://official.example.org/policy"


def test_researcher_agent_guardrail_forces_search_before_zero_tool_stop() -> None:
    agent = ResearcherAgent(
        name="researcher_guardrail_search",
        policy=_GuidanceAwarePolicy(),
        tools=[_FakeWebSearchTool()],
    )
    task = SubTask(
        task_id="task_guardrail_search",
        task_type=TaskType.SEARCH,
        description="比较欧盟 AI 法案与中国生成式 AI 暂行办法的透明度要求",
    )

    adjusted = agent._apply_search_policy_guardrails(
        {
            "label": "stop",
            "source": "learned",
            "confidence": 0.91,
            "browser_url": "",
            "features": {
                "tool_calls_so_far": 0.0,
                "search_calls_so_far": 0.0,
                "browser_calls_so_far": 0.0,
            },
            "can_continue": False,
            "enforce_stop": True,
            "guidance": "stop now",
        },
        task=task,
        context={"query": task.description},
        stage="assistant_no_tool",
        preferred_search_tool="web_search",
    )

    assert adjusted["label"] == "search"
    assert adjusted["can_continue"] is True
    assert adjusted["enforce_stop"] is False
    assert adjusted["guardrail_reason"] == "no_search_yet_force_search"
    assert "web_search" in adjusted["guidance"]


def test_researcher_agent_guardrail_forces_browser_before_stop_for_seeded_queries() -> None:
    agent = ResearcherAgent(
        name="researcher_guardrail_browser",
        policy=_GuidanceAwarePolicy(),
        tools=[_FakeWebSearchTool()],
    )
    task = SubTask(
        task_id="task_guardrail_browser",
        task_type=TaskType.SEARCH,
        description=(
            "Compare the EU AI Act and China's interim measures. "
            "Prefer these primary sources first: https://eur-lex.europa.eu/eli/reg/2024/1689/oj/eng "
            "and https://www.cac.gov.cn/2023-07/13/c_1690898327029107.htm ."
        ),
    )

    adjusted = agent._apply_search_policy_guardrails(
        {
            "label": "stop",
            "source": "learned",
            "confidence": 0.88,
            "browser_url": "https://eur-lex.europa.eu/eli/reg/2024/1689/oj/eng",
            "features": {
                "tool_calls_so_far": 1.0,
                "search_calls_so_far": 1.0,
                "browser_calls_so_far": 0.0,
                "browser_available": 1.0,
                "last_result_count": 5.0,
            },
            "can_continue": False,
            "enforce_stop": True,
            "guidance": "stop now",
        },
        task=task,
        context={"query": task.description},
        stage="post_tool",
        preferred_search_tool="web_search",
    )

    assert adjusted["label"] == "browser"
    assert adjusted["can_continue"] is True
    assert adjusted["enforce_stop"] is False
    assert adjusted["guardrail_reason"] == "seeded_query_force_browser_before_stop"
    assert "browser" in adjusted["guidance"]
    assert "eur-lex.europa.eu" in adjusted["guidance"]


def test_researcher_agent_guardrail_forces_stop_on_repetitive_continue() -> None:
    agent = ResearcherAgent(
        name="researcher_guardrail_stop",
        policy=_GuidanceAwarePolicy(),
        tools=[_FakeWebSearchTool()],
    )
    task = SubTask(
        task_id="task_guardrail_stop",
        task_type=TaskType.SEARCH,
        description="比较清洁氢融资和规模化瓶颈",
    )

    adjusted = agent._apply_search_policy_guardrails(
        {
            "label": "search",
            "source": "learned",
            "confidence": 0.79,
            "browser_url": "",
            "features": {
                "tool_calls_so_far": 4.0,
                "search_calls_so_far": 3.0,
                "browser_calls_so_far": 1.0,
                "recent_query_overlap": 0.92,
                "last_result_count": 4.0,
                "budget_ratio": 1.2,
            },
            "can_continue": True,
            "enforce_stop": False,
            "guidance": "search again",
        },
        task=task,
        context={"query": task.description},
        stage="post_tool",
        preferred_search_tool="web_search",
    )

    assert adjusted["label"] == "stop"
    assert adjusted["can_continue"] is False
    assert adjusted["enforce_stop"] is True
    assert adjusted["guardrail_reason"] == "repetitive_search_force_stop"
    assert "Stop now" in adjusted["guidance"]


class _FakeWebSearchTool:
    name = "web_search"

    @staticmethod
    def get_openai_tool_schema() -> dict:
        return {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "Search the web",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                    },
                    "required": ["query"],
                },
            },
        }

    async def execute(self, query: str) -> dict:
        return {
            "backend": "mock",
            "results": [
                {
                    "title": "Benchmark article",
                    "url": "https://example.com/a",
                    "snippet": f"Search result for {query}",
                }
            ],
            "total": 1,
        }


class _TrackingWebSearchTool(_FakeWebSearchTool):
    def __init__(self) -> None:
        self.queries: list[str] = []

    async def execute(self, query: str) -> dict:
        self.queries.append(query)
        return await super().execute(query=query)


class _GuidanceAwarePolicy:
    def __init__(self) -> None:
        self.calls = 0
        self.tools = []

    def set_tools(self, tools: list[dict]) -> None:
        self.tools = tools

    def __call__(self, messages: list[dict]) -> dict:
        self.calls += 1
        last_user = next(
            (str(item.get("content", "") or "") for item in reversed(messages) if item.get("role") == "user"),
            "",
        )
        last_tool = next(
            (str(item.get("content", "") or "") for item in reversed(messages) if item.get("role") == "tool"),
            "",
        )
        if self.calls == 1:
            return {
                "content": "我先直接总结，不调用工具。",
                "tool_calls": [],
            }
        if last_tool:
            return {
                "content": "最终总结\nConfidence: 0.82",
                "tool_calls": [],
            }
        if "[SEARCH POLICY]" in last_user:
            return {
                "content": "执行搜索",
                "tool_calls": [
                    {
                        "id": "call_web_1",
                        "type": "function",
                        "function": {
                            "name": "web_search",
                            "arguments": json.dumps(
                                {"query": "GPT-4o Claude 3.5 benchmark"},
                                ensure_ascii=False,
                            ),
                        },
                    }
                ],
            }
        return {"content": "最终总结\nConfidence: 0.82", "tool_calls": []}


class _TooManyToolCallsPolicy:
    def set_tools(self, tools: list[dict]) -> None:
        self.tools = tools

    def __call__(self, messages: list[dict]) -> dict:
        last_tool = next(
            (str(item.get("content", "") or "") for item in reversed(messages) if item.get("role") == "tool"),
            "",
        )
        if last_tool:
            return {"content": "最终总结\nConfidence: 0.75", "tool_calls": []}
        return {
            "content": "",
            "tool_calls": [
                {
                    "id": f"call_{idx}",
                    "type": "function",
                    "function": {
                        "name": "web_search",
                        "arguments": json.dumps({"query": f"query {idx}"}, ensure_ascii=False),
                    },
                }
                for idx in range(3)
            ],
        }


class _SingleSearchPolicy:
    def set_tools(self, tools: list[dict]) -> None:
        self.tools = tools

    def __call__(self, messages: list[dict]) -> dict:
        last_tool = next(
            (str(item.get("content", "") or "") for item in reversed(messages) if item.get("role") == "tool"),
            "",
        )
        if last_tool:
            return {"content": "最终总结\nConfidence: 0.71", "tool_calls": []}
        return {
            "content": "执行搜索",
            "tool_calls": [
                {
                    "id": "call_seeded_1",
                    "type": "function",
                    "function": {
                        "name": "web_search",
                        "arguments": json.dumps(
                            {"query": "EU AI Act high-risk systems transparency accountable parties provisions"},
                            ensure_ascii=False,
                        ),
                    },
                }
            ],
        }


def test_researcher_agent_truncates_tool_calls_to_configured_budget() -> None:
    agent = ResearcherAgent(
        name="researcher_budget_test",
        policy=_TooManyToolCallsPolicy(),
        tools=[_FakeWebSearchTool()],
        max_tool_calls_per_turn=1,
        max_tool_calls_per_task=1,
    )
    task = SubTask(
        task_id="task_budget",
        task_type=TaskType.SEARCH,
        description="测试工具预算约束",
    )

    result = asyncio.run(agent.run(task, {"query": task.description}))

    tool_calls = [item for item in result.action_log if item.get("action_type") == "tool_call"]
    assert result.status == AgentStatus.SUCCESS
    assert len(tool_calls) == 1
    assert any(item.get("action_type") == "tool_calls_truncated" for item in result.action_log)


def test_researcher_agent_uses_search_policy_guidance_when_model_skips_tool_calls(tmp_path: Path) -> None:
    model_path = tmp_path / "search_policy.json"
    trainer = SearchPolicyTrainer(model_path=model_path)
    trainer.fit(
        _training_rows(),
        epochs=320,
        learning_rate=0.18,
        l2=1e-4,
        validation_ratio=0.0,
    )

    runtime_policy = LearnedSearchPolicy(
        model_path=model_path,
        continue_threshold=0.55,
        stop_threshold=0.70,
        heuristic_fallback=False,
    )
    llm_policy = _GuidanceAwarePolicy()
    agent = ResearcherAgent(
        name="researcher_test",
        policy=llm_policy,
        tools=[_FakeWebSearchTool()],
        search_policy=runtime_policy,
    )
    task = SubTask(
        task_id="task_search_policy",
        task_type=TaskType.SEARCH,
        description="比较 GPT-4o 与 Claude 3.5 的网页评测差异",
    )

    result = asyncio.run(agent.run(task, {"query": task.description}))

    assert result.status == AgentStatus.SUCCESS
    assert llm_policy.calls >= 3
    assert any(
        item.get("action_type") == "policy_advice" and item.get("can_continue")
        for item in result.action_log
    )
    assert any(
        item.get("action_type") == "tool_call" and item.get("tool_name") == "web_search"
        for item in result.action_log
    )
    policy_stats = result.metadata.get("policy_stats", {})
    assert policy_stats.get("policy_advice_count", 0) >= 1
    assert policy_stats.get("policy_continue_count", 0) >= 1
    assert policy_stats.get("guardrail_trigger_count", 0) >= 1
    assert policy_stats.get("guardrail_reasons", {}).get("no_search_yet_force_search", 0) >= 1


def test_researcher_agent_injects_primary_source_urls_into_web_search_query() -> None:
    tool = _TrackingWebSearchTool()
    agent = ResearcherAgent(
        name="researcher_seed_hint_test",
        policy=_SingleSearchPolicy(),
        tools=[tool],
        max_tool_calls_per_turn=1,
        max_tool_calls_per_task=1,
    )
    task = SubTask(
        task_id="task_seed_urls",
        task_type=TaskType.SEARCH,
        description=(
            "Compare the EU AI Act and China's interim measures. "
            "Prefer these primary sources first: https://eur-lex.europa.eu/eli/reg/2024/1689/oj/eng "
            "and https://www.cac.gov.cn/2023-07/13/c_1690898327029107.htm ."
        ),
    )

    result = asyncio.run(agent.run(task, {"query": task.description}))

    assert result.status == AgentStatus.SUCCESS
    assert len(tool.queries) == 1
    assert "https://eur-lex.europa.eu/eli/reg/2024/1689/oj/eng" in tool.queries[0]
    assert "https://www.cac.gov.cn/2023-07/13/c_1690898327029107.htm" in tool.queries[0]
    tool_actions = [item for item in result.action_log if item.get("action_type") == "tool_call"]
    assert tool_actions[0]["original_query_text"] == (
        "EU AI Act high-risk systems transparency accountable parties provisions"
    )


def test_search_policy_mode_resolution_prefers_explicit_or_legacy_heuristic() -> None:
    assert resolve_search_policy_mode({"search_policy": {"mode": "off", "enabled": True}}) == "off"
    assert resolve_search_policy_mode({"search_policy": {"enabled": True, "heuristic_fallback": True, "model_path": "artifacts/search_policy.json"}}) == "heuristic"
    assert resolve_search_policy_mode({"search_policy": {"enabled": True, "heuristic_fallback": False}}) == "learned"


def test_create_search_policy_honors_heuristic_mode_even_when_model_exists(tmp_path: Path) -> None:
    model_path = tmp_path / "search_policy.json"
    trainer = SearchPolicyTrainer(model_path=model_path)
    trainer.fit(
        _training_rows(),
        epochs=320,
        learning_rate=0.18,
        l2=1e-4,
        validation_ratio=0.0,
    )

    heuristic_policy = create_search_policy({
        "search_policy": {
            "mode": "heuristic",
            "enabled": True,
            "model_path": str(model_path),
            "heuristic_fallback": True,
        }
    })
    learned_policy = create_search_policy({
        "search_policy": {
            "mode": "learned",
            "enabled": True,
            "model_path": str(model_path),
            "heuristic_fallback": False,
        }
    })

    assert heuristic_policy is not None
    assert heuristic_policy.is_trained is False
    assert learned_policy is not None
    assert learned_policy.is_trained is True
