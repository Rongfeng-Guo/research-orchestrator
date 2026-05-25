"""
M6 自进化引擎 - Evidence Transition Dataset

将 evidence transition trace 展平为可监督学习的数据样本，支持：
1. JSONL 导出，便于快速实验和调试
2. 特征向量化，供 learned policy 训练
3. 简单统计摘要，便于分析数据质量
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["EvidenceTransitionDatasetBuilder"]


class EvidenceTransitionDatasetBuilder:
    """将 evidence transition trace 转成监督学习样本。"""

    FEATURE_NAMES = [
        "uncertainty",
        "coverage_ratio",
        "evidence_strength",
        "open_conflicts",
        "claim_count",
        "support_edge_count",
        "contradiction_edge_count",
        "avg_source_trust",
        "missing_term_count",
        "candidate_action_margin",
        "candidate_top_score",
        "candidate_second_score",
        "candidate_action_entropy",
        "evidence_graph_quality",
        "claim_support_coverage",
        "consistency_score",
    ]

    ACTION_SPACE = [
        "search",
        "retrieve_more",
        "verify",
        "expand_subquestion",
        "stop",
    ]

    def build_from_collected_batch(self, batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """把 collector 的 batch 输出展开为逐步样本。"""
        rows: list[dict[str, Any]] = []
        for item_idx, item in enumerate(batch):
            if not isinstance(item, dict):
                continue
            rows.extend(self.build_from_collected_item(item, item_idx=item_idx))
        return rows

    def build_from_collected_item(
        self,
        item: dict[str, Any],
        item_idx: int = 0,
    ) -> list[dict[str, Any]]:
        """把单条 collector 输出展开为逐步样本。"""
        query = str(item.get("query", ""))
        transitions = item.get("evidence_transition_trace", []) or item.get("evidence_transitions", [])
        rewards = item.get("process_reward_trace", [])
        if not isinstance(transitions, list):
            transitions = []
        if not isinstance(rewards, list):
            rewards = []

        rows: list[dict[str, Any]] = []
        for step_idx, transition in enumerate(transitions):
            if not isinstance(transition, dict):
                continue

            before = transition.get("before", {}) if isinstance(transition.get("before"), dict) else {}
            after = transition.get("after", {}) if isinstance(transition.get("after"), dict) else {}
            features = self._snapshot_to_features(before)
            feature_vector = [float(features[name]) for name in self.FEATURE_NAMES]
            reward = self._transition_reward(transition, rewards, step_idx)
            label_action = str(
                transition.get("label_action")
                or before.get("recommended_action")
                or "search"
            )

            rows.append(
                {
                    "sample_id": f"{item_idx:04d}:{step_idx:04d}",
                    "query": query,
                    "step": int(transition.get("step", step_idx + 1) or step_idx + 1),
                    "task_id": str(transition.get("task_id", "")),
                    "task_type": str(transition.get("task_type", "")),
                    "status": str(transition.get("status", "")),
                    "label_action": label_action,
                    "executed_action": str(transition.get("task_type", "")),
                    "reward": float(reward),
                    "features": features,
                    "feature_vector": feature_vector,
                    "before": before,
                    "after": after,
                    "delta": transition.get("delta", {}) if isinstance(transition.get("delta"), dict) else {},
                    "candidate_actions": before.get("candidate_actions", []),
                    "report_metadata": item.get("report_metadata", {}),
                    "final_evidence_snapshot": item.get("final_evidence_snapshot", {}),
                }
            )

        return rows

    def export_jsonl(self, rows: list[dict[str, Any]], path: str | Path) -> str:
        """导出为 JSONL 文件。"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        return str(path)

    def load_jsonl(self, path: str | Path) -> list[dict[str, Any]]:
        """从 JSONL 加载样本。"""
        path = Path(path)
        rows: list[dict[str, Any]] = []
        if not path.exists():
            return rows
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return rows

    def summarize(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        """给导出的样本做轻量统计。"""
        if not rows:
            return {
                "num_examples": 0,
                "avg_reward": 0.0,
                "label_distribution": {},
                "status_distribution": {},
            }

        reward_values = [float(row.get("reward", 0.0) or 0.0) for row in rows]
        label_distribution: dict[str, int] = {}
        status_distribution: dict[str, int] = {}
        for row in rows:
            label = str(row.get("label_action", "") or "")
            status = str(row.get("status", "") or "")
            label_distribution[label] = label_distribution.get(label, 0) + 1
            status_distribution[status] = status_distribution.get(status, 0) + 1

        return {
            "num_examples": len(rows),
            "avg_reward": round(sum(reward_values) / len(reward_values), 4),
            "label_distribution": label_distribution,
            "status_distribution": status_distribution,
        }

    @staticmethod
    def _snapshot_to_features(snapshot: dict[str, Any]) -> dict[str, float]:
        """把 snapshot 压成固定维度的数值特征。"""
        candidate_actions = snapshot.get("candidate_actions", []) if isinstance(snapshot, dict) else []
        candidate_scores: list[float] = []
        if isinstance(candidate_actions, list):
            for item in candidate_actions:
                if isinstance(item, dict):
                    candidate_scores.append(float(item.get("score", 0.0) or 0.0))
        candidate_scores.sort(reverse=True)

        if len(candidate_scores) >= 2:
            top_score = candidate_scores[0]
            second_score = candidate_scores[1]
        elif candidate_scores:
            top_score = candidate_scores[0]
            second_score = 0.0
        else:
            top_score = 0.0
            second_score = 0.0

        probs = np.array(candidate_scores, dtype=np.float64)
        if probs.size > 0:
            probs = np.maximum(probs, 1e-8)
            probs = probs / probs.sum()
            entropy = float(-(probs * np.log(probs)).sum() / np.log(len(probs) + 1e-8))
        else:
            entropy = 0.0

        features = {
            "uncertainty": float(snapshot.get("uncertainty", 1.0) or 1.0),
            "coverage_ratio": float(snapshot.get("coverage_ratio", 0.0) or 0.0),
            "evidence_strength": float(snapshot.get("evidence_strength", 0.0) or 0.0),
            "open_conflicts": float(snapshot.get("open_conflicts", 0.0) or 0.0),
            "claim_count": float(snapshot.get("claim_count", 0.0) or 0.0),
            "support_edge_count": float(snapshot.get("support_edge_count", 0.0) or 0.0),
            "contradiction_edge_count": float(snapshot.get("contradiction_edge_count", 0.0) or 0.0),
            "avg_source_trust": float(snapshot.get("avg_source_trust", snapshot.get("avg_source_trust", 0.0)) or 0.0),
            "missing_term_count": float(len(snapshot.get("missing_terms", []) or [])),
            "candidate_action_margin": max(0.0, float(top_score - second_score)),
            "candidate_top_score": float(top_score),
            "candidate_second_score": float(second_score),
            "candidate_action_entropy": float(entropy),
            "evidence_graph_quality": float(snapshot.get("evidence_graph_quality", 0.0) or 0.0),
            "claim_support_coverage": float(snapshot.get("claim_support_coverage", 0.0) or 0.0),
            "consistency_score": float(snapshot.get("consistency_score", 0.0) or 0.0),
        }
        return features

    @staticmethod
    def _transition_reward(
        transition: dict[str, Any],
        rewards: list[float],
        idx: int,
    ) -> float:
        """优先用 process_reward_trace，否则回退到 delta 估算。"""
        if idx < len(rewards):
            try:
                return float(rewards[idx])
            except (TypeError, ValueError):
                pass

        delta = transition.get("delta", {}) if isinstance(transition.get("delta"), dict) else {}
        uncertainty = abs(float(delta.get("uncertainty", 0.0) or 0.0))
        coverage = float(delta.get("coverage_ratio", 0.0) or 0.0)
        conflicts = -float(delta.get("open_conflicts", 0.0) or 0.0)
        strength = float(delta.get("evidence_strength", 0.0) or 0.0)
        reward = 0.35 * (-uncertainty) + 0.35 * coverage + 0.20 * conflicts + 0.10 * strength
        return max(-1.0, min(1.0, round(reward, 4)))
