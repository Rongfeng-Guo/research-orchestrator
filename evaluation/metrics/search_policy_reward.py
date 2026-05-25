#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluation/metrics/search_policy_reward.py
================================================================================
Reward shaping for offline agentic-search training.

This module converts:
  - final-answer quality signals
  - search-policy metrics
  - process reward traces

into a scalar reward in [-1, 1] plus an interpretable breakdown. It is designed
for cached offline datasets, where we want a stable reward target without
requiring online Judge calls for every experiment.
================================================================================
"""

from __future__ import annotations

from typing import Any

from .search_policy import SearchPolicyMetrics


class SearchPolicyReward:
    """Offline reward shaping for search-policy learning."""

    DEFAULT_WEIGHTS = {
        "quality": 0.55,
        "process": 0.45,
    }

    @staticmethod
    def _clamp01(value: Any, default: float = 0.0) -> float:
        try:
            return max(0.0, min(1.0, float(value)))
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _mean(values: list[float], default: float = 0.0) -> float:
        cleaned = [float(v) for v in values if isinstance(v, (int, float))]
        if not cleaned:
            return default
        return sum(cleaned) / len(cleaned)

    @staticmethod
    def _trace_mean_to_unit_interval(process_reward_trace: Any) -> float:
        if not isinstance(process_reward_trace, list) or not process_reward_trace:
            return 0.0
        cleaned: list[float] = []
        for value in process_reward_trace:
            if not isinstance(value, (int, float)):
                continue
            clipped = max(-1.0, min(1.0, float(value)))
            cleaned.append((clipped + 1.0) / 2.0)
        return SearchPolicyReward._mean(cleaned, default=0.0)

    @classmethod
    def breakdown(
        cls,
        *,
        report_text: str,
        report_metadata: dict[str, Any] | None = None,
        report_sources: list[dict[str, Any]] | None = None,
        evaluation_result: dict[str, Any] | None = None,
        process_reward_trace: list[float] | None = None,
        confidence: float | None = None,
        weights: dict[str, float] | None = None,
    ) -> dict[str, float]:
        policy_breakdown = SearchPolicyMetrics.breakdown(
            report_text,
            report_metadata,
            report_sources,
        )

        evaluation_metrics = {}
        if isinstance(evaluation_result, dict):
            maybe_metrics = evaluation_result.get("metrics", {})
            if isinstance(maybe_metrics, dict):
                evaluation_metrics = maybe_metrics

        factual_accuracy = cls._clamp01(
            evaluation_metrics.get("factual_accuracy", confidence if confidence is not None else 0.0)
        )
        citation_quality = cls._clamp01(
            evaluation_metrics.get("citation_coverage", policy_breakdown.get("citation_grounding", 0.0))
        )
        comprehensiveness = cls._clamp01(
            evaluation_metrics.get("comprehensiveness", confidence if confidence is not None else 0.0)
        )
        transition_quality = cls._clamp01(
            evaluation_metrics.get("evidence_transition_quality", 0.0)
        )

        confidence_proxy = cls._clamp01(confidence, default=0.0)
        process_trace_unit = cls._trace_mean_to_unit_interval(process_reward_trace)

        quality_score = cls._mean(
            [
                factual_accuracy,
                citation_quality,
                comprehensiveness,
                max(transition_quality, confidence_proxy * 0.5),
            ],
            default=confidence_proxy,
        )

        policy_score = cls._clamp01(policy_breakdown.get("search_policy_score", 0.0))
        efficiency_score = cls._mean(
            [
                cls._clamp01(policy_breakdown.get("budget_efficiency", 0.0)),
                cls._clamp01(policy_breakdown.get("stop_efficiency", 0.0)),
                cls._clamp01(policy_breakdown.get("successful_tool_call_ratio", 0.0)),
            ],
            default=0.0,
        )
        grounding_score = cls._clamp01(policy_breakdown.get("citation_grounding", 0.0))
        diversity_score = cls._clamp01(policy_breakdown.get("source_diversity", 0.0))

        process_score = cls._mean(
            [
                policy_score,
                efficiency_score,
                grounding_score,
                diversity_score,
                process_trace_unit,
            ],
            default=policy_score,
        )

        reward_weights = dict(cls.DEFAULT_WEIGHTS)
        if isinstance(weights, dict):
            reward_weights.update(weights)
        quality_weight = max(0.0, float(reward_weights.get("quality", 0.55)))
        process_weight = max(0.0, float(reward_weights.get("process", 0.45)))
        weight_sum = quality_weight + process_weight
        if weight_sum <= 0.0:
            quality_weight = 0.55
            process_weight = 0.45
            weight_sum = 1.0

        reward_raw = (
            quality_weight * quality_score + process_weight * process_score
        ) / weight_sum
        reward = max(-1.0, min(1.0, reward_raw * 2.0 - 1.0))

        return {
            "factual_accuracy_reward": factual_accuracy,
            "citation_quality_reward": citation_quality,
            "comprehensiveness_reward": comprehensiveness,
            "transition_quality_reward": transition_quality,
            "confidence_proxy_reward": confidence_proxy,
            "policy_score_reward": policy_score,
            "efficiency_reward": efficiency_score,
            "grounding_reward": grounding_score,
            "diversity_reward": diversity_score,
            "process_trace_mean_reward": process_trace_unit,
            "quality_score": quality_score,
            "process_score": process_score,
            "reward_raw": reward_raw,
            "reward": reward,
        }
