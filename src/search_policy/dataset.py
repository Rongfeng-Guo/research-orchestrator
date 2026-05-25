from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


__all__ = ["SearchPolicyDatasetBuilder"]


class SearchPolicyDatasetBuilder:
    """Build step-level route/stop samples from search-cache JSONL records."""

    FEATURE_NAMES = [
        "tool_calls_so_far",
        "search_calls_so_far",
        "browser_calls_so_far",
        "assistant_turns_so_far",
        "successful_tool_call_ratio",
        "empty_result_ratio",
        "avg_result_count",
        "last_result_count",
        "avg_latency_ms",
        "last_latency_ms",
        "unique_query_count",
        "unique_domain_count",
        "unique_backend_count",
        "query_repeat_ratio",
        "domain_diversity",
        "budget_ratio",
        "recent_query_overlap",
        "last_action_was_search",
        "last_action_was_browser",
        "browser_available",
        "has_stop_signal",
    ]

    ACTION_SPACE = [
        "search",
        "browser",
        "stop",
    ]

    def load_jsonl(self, path: str | Path) -> list[dict[str, Any]]:
        path = Path(path)
        rows: list[dict[str, Any]] = []
        if not path.exists():
            return rows
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                raw = line.strip()
                if not raw:
                    continue
                try:
                    rows.append(json.loads(raw))
                except json.JSONDecodeError:
                    continue
        return rows

    def build_rows_from_paths(self, paths: list[str | Path]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for path in paths:
            rows.extend(self.build_rows_from_records(self.load_jsonl(path), source_path=str(path)))
        return rows

    def build_rows_from_records(
        self,
        records: list[dict[str, Any]],
        *,
        source_path: str = "",
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for record_idx, record in enumerate(records):
            rows.extend(
                self.build_rows_from_record(
                    record,
                    record_idx=record_idx,
                    source_path=source_path,
                )
            )
        return rows

    def build_rows_from_record(
        self,
        record: dict[str, Any],
        *,
        record_idx: int = 0,
        source_path: str = "",
    ) -> list[dict[str, Any]]:
        if not isinstance(record, dict):
            return []
        if str(record.get("status", "") or "success") != "success":
            return []

        query = str(record.get("query", "") or "")
        reward = self._safe_float(record.get("reward", 0.0))
        reward_breakdown = record.get("reward_breakdown", {})
        if not isinstance(reward_breakdown, dict):
            reward_breakdown = {}

        policy_trace = self._get_policy_trace(record)
        if not policy_trace:
            return []

        rows: list[dict[str, Any]] = []
        state = self.new_state()

        for action_idx, action in enumerate(policy_trace):
            if not isinstance(action, dict):
                continue

            action_type = str(action.get("action_type", "") or "")
            if action_type == "assistant_response":
                self.observe_action(state, action)
                continue
            if action_type == "stop_signal":
                self.observe_action(state, action)
                continue

            label = self._action_to_label(action)
            if label is None:
                continue

            if label == "stop" and state["tool_calls_so_far"] <= 0:
                stop_reason = str(action.get("stop_reason", "") or "")
                if stop_reason in {
                    "non_searchable_direct_analysis",
                    "direct_analysis_error",
                    "policy_runtime_error",
                    "tool_failure_explanation",
                    "tool_error",
                }:
                    continue

            features = self.features_from_state(state)
            row_reward = self._row_reward(
                label=label,
                reward=reward,
                reward_breakdown=reward_breakdown,
            )
            rows.append(
                {
                    "sample_id": f"{record_idx:05d}:{action_idx:05d}",
                    "query": query,
                    "label_action": label,
                    "reward": row_reward,
                    "record_reward": reward,
                    "features": features,
                    "feature_vector": [float(features[name]) for name in self.FEATURE_NAMES],
                    "action_type": action_type,
                    "tool_name": str(action.get("tool_name", "") or ""),
                    "stop_reason": str(action.get("stop_reason", "") or ""),
                    "source_path": source_path,
                    "cache_id": str(record.get("cache_id", "") or ""),
                    "source_label": str(record.get("source_label", "") or ""),
                }
            )

            self.observe_action(state, action)

        return rows

    def summarize(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        if not rows:
            return {
                "num_examples": 0,
                "avg_reward": 0.0,
                "label_distribution": {},
            }

        label_distribution: dict[str, int] = {}
        rewards: list[float] = []
        for row in rows:
            label = str(row.get("label_action", "") or "")
            label_distribution[label] = label_distribution.get(label, 0) + 1
            rewards.append(self._safe_float(row.get("reward", 0.0)))

        return {
            "num_examples": len(rows),
            "avg_reward": round(sum(rewards) / len(rewards), 4),
            "label_distribution": label_distribution,
        }

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _clamp(value: float, low: float, high: float) -> float:
        return max(low, min(high, value))

    @staticmethod
    def _get_policy_trace(record: dict[str, Any]) -> list[dict[str, Any]]:
        structured = record.get("structured_actions", [])
        if isinstance(structured, list) and structured:
            return [item for item in structured if isinstance(item, dict)]

        metadata = record.get("report_metadata", {})
        if isinstance(metadata, dict):
            policy_trace = metadata.get("policy_trace", [])
            if isinstance(policy_trace, list):
                return [item for item in policy_trace if isinstance(item, dict)]

        return []

    @staticmethod
    def _action_to_label(action: dict[str, Any]) -> str | None:
        action_type = str(action.get("action_type", "") or "")
        if action_type == "stop":
            return "stop"
        if action_type != "tool_call":
            return None

        tool_name = str(action.get("tool_name", "") or "")
        if tool_name in {"web_search", "arxiv_reader"}:
            return "search"
        if tool_name == "browser":
            return "browser"
        return None

    @classmethod
    def new_state(cls) -> dict[str, Any]:
        return cls._initial_state()

    @classmethod
    def features_from_state(cls, state: dict[str, Any]) -> dict[str, float]:
        return cls._state_to_features(state)

    @classmethod
    def observe_action(cls, state: dict[str, Any], action: dict[str, Any]) -> None:
        if not isinstance(action, dict):
            return

        action_type = str(action.get("action_type", "") or "")
        if action_type == "assistant_response":
            state["assistant_turns_so_far"] = int(state.get("assistant_turns_so_far", 0) or 0) + 1
            return
        if action_type == "stop_signal":
            state["has_stop_signal"] = 1.0
            return
        if action_type != "tool_call":
            return

        label = cls._action_to_label(action)
        if label is None:
            return
        cls._update_state_from_tool_action(state, action, label)

    @classmethod
    def _initial_state(cls) -> dict[str, Any]:
        return {
            "assistant_turns_so_far": 0,
            "tool_calls_so_far": 0,
            "search_calls_so_far": 0,
            "browser_calls_so_far": 0,
            "successful_tool_calls_so_far": 0,
            "empty_result_calls_so_far": 0,
            "cumulative_result_count": 0.0,
            "cumulative_latency_ms": 0.0,
            "cumulative_token_cost": 0.0,
            "query_history": [],
            "query_set": set(),
            "domain_set": set(),
            "backend_set": set(),
            "last_result_count": 0.0,
            "last_latency_ms": 0.0,
            "last_action": "",
            "has_stop_signal": 0.0,
            "last_tool_name": "",
            "last_query_text": "",
            "last_top_urls": [],
        }

    @classmethod
    def _state_to_features(cls, state: dict[str, Any]) -> dict[str, float]:
        tool_calls = max(0, int(state.get("tool_calls_so_far", 0) or 0))
        search_calls = max(0, int(state.get("search_calls_so_far", 0) or 0))
        browser_calls = max(0, int(state.get("browser_calls_so_far", 0) or 0))
        successful = max(0, int(state.get("successful_tool_calls_so_far", 0) or 0))
        empty_calls = max(0, int(state.get("empty_result_calls_so_far", 0) or 0))
        cumulative_results = max(0.0, cls._safe_float(state.get("cumulative_result_count", 0.0)))
        cumulative_latency = max(0.0, cls._safe_float(state.get("cumulative_latency_ms", 0.0)))
        cumulative_tokens = max(0.0, cls._safe_float(state.get("cumulative_token_cost", 0.0)))

        avg_result_count = cumulative_results / max(tool_calls, 1)
        avg_latency_ms = cumulative_latency / max(tool_calls, 1)
        unique_query_count = len(state.get("query_set", set()))
        unique_domain_count = len(state.get("domain_set", set()))
        unique_backend_count = len(state.get("backend_set", set()))
        query_repeat_ratio = 0.0
        if search_calls > 0:
            query_repeat_ratio = max(0.0, 1.0 - unique_query_count / max(search_calls, 1))

        last_action = str(state.get("last_action", "") or "")
        features = {
            "tool_calls_so_far": float(tool_calls),
            "search_calls_so_far": float(search_calls),
            "browser_calls_so_far": float(browser_calls),
            "assistant_turns_so_far": float(max(0, int(state.get("assistant_turns_so_far", 0) or 0))),
            "successful_tool_call_ratio": successful / max(tool_calls, 1),
            "empty_result_ratio": empty_calls / max(tool_calls, 1),
            "avg_result_count": float(avg_result_count),
            "last_result_count": max(0.0, cls._safe_float(state.get("last_result_count", 0.0))),
            "avg_latency_ms": float(avg_latency_ms),
            "last_latency_ms": max(0.0, cls._safe_float(state.get("last_latency_ms", 0.0))),
            "unique_query_count": float(unique_query_count),
            "unique_domain_count": float(unique_domain_count),
            "unique_backend_count": float(unique_backend_count),
            "query_repeat_ratio": cls._clamp(query_repeat_ratio, 0.0, 1.0),
            "domain_diversity": cls._clamp(unique_domain_count / max(tool_calls, 1), 0.0, 1.0),
            "budget_ratio": cls._clamp(cumulative_tokens / 2500.0, 0.0, 3.0),
            "recent_query_overlap": cls._recent_query_overlap(state.get("query_history", [])),
            "last_action_was_search": 1.0 if last_action == "search" else 0.0,
            "last_action_was_browser": 1.0 if last_action == "browser" else 0.0,
            "browser_available": 1.0 if search_calls > 0 and browser_calls == 0 else 0.0,
            "has_stop_signal": cls._clamp(cls._safe_float(state.get("has_stop_signal", 0.0)), 0.0, 1.0),
        }
        return features

    @classmethod
    def _update_state_from_tool_action(
        cls,
        state: dict[str, Any],
        action: dict[str, Any],
        label: str,
    ) -> None:
        state["tool_calls_so_far"] = int(state.get("tool_calls_so_far", 0) or 0) + 1
        if label == "search":
            state["search_calls_so_far"] = int(state.get("search_calls_so_far", 0) or 0) + 1
        elif label == "browser":
            state["browser_calls_so_far"] = int(state.get("browser_calls_so_far", 0) or 0) + 1

        result_count = max(0.0, cls._safe_float(action.get("result_count", 0.0)))
        latency_ms = max(0.0, cls._safe_float(action.get("latency_ms", 0.0)))
        estimated_token_cost = max(0.0, cls._safe_float(action.get("estimated_token_cost", 0.0)))
        error = str(action.get("error", "") or "")

        if not error and result_count > 0:
            state["successful_tool_calls_so_far"] = int(state.get("successful_tool_calls_so_far", 0) or 0) + 1
        else:
            state["empty_result_calls_so_far"] = int(state.get("empty_result_calls_so_far", 0) or 0) + 1

        state["cumulative_result_count"] = cls._safe_float(state.get("cumulative_result_count", 0.0)) + result_count
        state["cumulative_latency_ms"] = cls._safe_float(state.get("cumulative_latency_ms", 0.0)) + latency_ms
        state["cumulative_token_cost"] = cls._safe_float(state.get("cumulative_token_cost", 0.0)) + estimated_token_cost
        state["last_result_count"] = result_count
        state["last_latency_ms"] = latency_ms
        state["last_action"] = label
        state["last_tool_name"] = str(action.get("tool_name", "") or "")
        state["last_query_text"] = str(action.get("query_text", "") or "")

        query_text = str(action.get("query_text", "") or "").strip().lower()
        if query_text:
            query_history = state.setdefault("query_history", [])
            query_history.append(query_text)
            query_set = state.setdefault("query_set", set())
            query_set.add(query_text)

        backend = str(action.get("backend", "") or "").strip().lower()
        if backend:
            backend_set = state.setdefault("backend_set", set())
            backend_set.add(backend)

        for url in action.get("top_urls", []) if isinstance(action.get("top_urls"), list) else []:
            domain = cls._extract_domain(str(url or ""))
            if domain:
                domain_set = state.setdefault("domain_set", set())
                domain_set.add(domain)
        state["last_top_urls"] = [
            str(url or "")
            for url in action.get("top_urls", [])
            if isinstance(url, str) and str(url or "").strip()
        ]

    @classmethod
    def _row_reward(
        cls,
        *,
        label: str,
        reward: float,
        reward_breakdown: dict[str, Any],
    ) -> float:
        if not reward_breakdown:
            return cls._clamp(reward, -1.0, 1.0)

        process_score = cls._safe_float(reward_breakdown.get("process_score", 0.0))
        efficiency = cls._safe_float(reward_breakdown.get("efficiency_reward", 0.0))
        grounding = cls._safe_float(reward_breakdown.get("grounding_reward", 0.0))
        policy_score = cls._safe_float(reward_breakdown.get("policy_score_reward", 0.0))

        if label == "stop":
            local = 0.55 * efficiency + 0.45 * process_score
        elif label == "browser":
            local = 0.60 * grounding + 0.40 * process_score
        else:
            local = 0.60 * policy_score + 0.40 * process_score

        local_reward = cls._clamp(local * 2.0 - 1.0, -1.0, 1.0)
        return round(cls._clamp(0.65 * reward + 0.35 * local_reward, -1.0, 1.0), 4)

    @staticmethod
    def _extract_domain(url: str) -> str:
        if not url:
            return ""
        cleaned = url.strip().strip(".,;:)]}>\"'")
        try:
            return urlparse(cleaned).netloc.lower()
        except ValueError:
            return ""

    @classmethod
    def _recent_query_overlap(cls, query_history: list[str]) -> float:
        if not isinstance(query_history, list) or len(query_history) < 2:
            return 0.0
        left = cls._tokenize(query_history[-2])
        right = cls._tokenize(query_history[-1])
        if not left or not right:
            return 0.0
        return cls._clamp(len(left & right) / len(left | right), 0.0, 1.0)

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        if not isinstance(text, str):
            return set()
        lowered = text.strip().lower()
        if not lowered:
            return set()
        tokens = set(re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]", lowered))
        if tokens:
            return tokens
        return {lowered}
