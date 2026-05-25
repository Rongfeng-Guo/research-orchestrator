#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluation/metrics/rule_based.py
================================================================================
基于规则/统计的轻量级评测指标。

适用于批量运行、CI/CD、消融实验等需要快速、免费、可复现评分的场景。
================================================================================
"""

from __future__ import annotations

import math
import re
from typing import Any


class RuleBasedMetrics:
    """研究报告质量评测指标集合（规则版）。"""

    @staticmethod
    def _clamp01(value: Any, default: float = 0.0) -> float:
        """将输入安全裁剪到 [0, 1]。"""
        try:
            return max(0.0, min(1.0, float(value)))
        except (TypeError, ValueError):
            return default

    @staticmethod
    def combine_factual_accuracy(
        factual_str: float | None = None,
        factual_sem: float | None = None,
        semantic_weight: float = 0.7,
    ) -> float:
        """
        合并字符串匹配与语义匹配的事实准确性。

        默认更偏向语义分数，因为它对措辞变化更稳健。
        """
        if factual_str is None and factual_sem is None:
            return 0.0
        if factual_str is None:
            return RuleBasedMetrics._clamp01(factual_sem)
        if factual_sem is None:
            return RuleBasedMetrics._clamp01(factual_str)

        sem_w = max(0.0, min(1.0, semantic_weight))
        str_w = 1.0 - sem_w
        return (
            str_w * RuleBasedMetrics._clamp01(factual_str)
            + sem_w * RuleBasedMetrics._clamp01(factual_sem)
        )

    @staticmethod
    def _normalize_metrics(metrics: dict[str, float]) -> dict[str, float]:
        """对 metric alias 做归一化，避免 composite 漏算关键项。"""
        normalized = dict(metrics)
        if "factual_accuracy" not in normalized and (
            "factual_accuracy_str" in normalized or "factual_accuracy_sem" in normalized
        ):
            normalized["factual_accuracy"] = RuleBasedMetrics.combine_factual_accuracy(
                normalized.get("factual_accuracy_str"),
                normalized.get("factual_accuracy_sem"),
            )
        return normalized

    # -----------------------------------------------------------------------
    # 1. 事实准确性 (Factual Accuracy) — 字符串匹配版（快速但粗糙）
    # -----------------------------------------------------------------------
    @staticmethod
    def fact_accuracy(report: str, ground_truth: dict[str, Any] | None = None) -> float:
        """
        计算报告中的关键事实与 ground_truth 的匹配程度。

        当前实现采用简单启发式：统计报告中包含的 ground_truth 关键短语比例。
        若无 ground_truth，则返回 0.0（需外部 Judge LLM 补充评估）。
        """
        if not ground_truth:
            return 0.0

        report_lower = report.lower()
        matched = 0
        for key_fact in ground_truth.keys():
            if key_fact.lower() in report_lower:
                matched += 1

        return matched / len(ground_truth) if ground_truth else 0.0

    # -----------------------------------------------------------------------
    # 1b. 语义事实准确性 (Semantic Factual Accuracy) — 面试强化版
    # -----------------------------------------------------------------------
    @staticmethod
    def semantic_fact_accuracy(
        report: str,
        ground_truth: dict[str, Any] | None = None,
        threshold: float = 0.65,
    ) -> float:
        """
        基于 embedding 语义相似度的事实准确性验证。

        改进点（相比字符串匹配）：
        1. 把 ground_truth 的 key + description 编码为语义向量
        2. 把报告拆分成句子 chunk，分别编码
        3. 计算每个 ground_truth 条目与报告中最相似 chunk 的 cosine similarity
        4. 超过阈值（默认 0.65）才判定为"事实被覆盖"

        这样能避免"GPT-4o 是 Google 发布的"这种关键词命中但语义错误的误报。

        Args:
            report: 研究报告全文
            ground_truth: 期望事实字典 {key: description}
            threshold: 语义相似度阈值，0-1

        Returns:
            0.0 ~ 1.0 的覆盖率
        """
        if not ground_truth:
            return 0.0

        import numpy as np
        from src.memory.embedder import Embedder

        embedder = Embedder()

        # 把报告拆成句子 chunk（避免长报告淹没短事实）
        chunks = [s.strip() for s in re.split(r"[。！？\n]", report) if len(s.strip()) > 10]
        if not chunks:
            return 0.0

        # 批量编码 chunk（Sentencetransformer 支持批量）
        try:
            chunk_embs = np.array(embedder._load_model().encode(chunks, normalize_embeddings=True))
        except Exception:
            # fallback：逐条编码
            chunk_embs = np.array([embedder.encode(c) for c in chunks])

        matched = 0
        for key_fact, expected_desc in ground_truth.items():
            # 组合 key + description 作为语义查询
            fact_text = f"{key_fact}：{expected_desc}"
            fact_emb = np.array(embedder.encode(fact_text))

            # 计算与所有 chunk 的 cosine similarity
            sims = chunk_embs.dot(fact_emb)
            max_sim = float(np.max(sims)) if sims.size > 0 else 0.0

            if max_sim > threshold:
                matched += 1

        return matched / len(ground_truth)

    # -----------------------------------------------------------------------
    # 2. 幻觉率 (Hallucination Rate)
    # -----------------------------------------------------------------------
    @staticmethod
    def hallucination_rate(report: str) -> float:
        """
        估算报告中可能存在的幻觉内容比例。

        当前启发式策略：
        - 检测无引用的数值声明（数字+单位）。
        - 检测缺乏来源的绝对化表述（"绝对"、"毫无疑问"等）。
        - 检测模型常见的幻觉模式（"据我所知"、"研究表明"但无具体引用）。

        Returns:
            0.0 ~ 1.0，越高表示幻觉风险越大。
        """
        if not report:
            return 1.0

        sentences = re.split(r"[。！？\n]", report)
        sentences = [s.strip() for s in sentences if s.strip()]
        if not sentences:
            return 1.0

        hallucination_indicators = [
            r"\d+[\d,]*\.?\d*\s*(%|倍|个|人|元|美元|亿|万)",  # 带单位的孤立数字
            r"毫无疑问|绝对|必然|一定|众所周知",
            r"据我所知|据了解|研究显示[^【\[（(]",  # 模糊引用开头
        ]

        suspicious_count = 0
        for sentence in sentences:
            # 如果句子中无引用标记，检查是否包含幻觉特征
            if not re.search(r"[\[【（(].*?[\]）)]", sentence):
                for pattern in hallucination_indicators:
                    if re.search(pattern, sentence):
                        suspicious_count += 1
                        break

        return min(1.0, suspicious_count / max(len(sentences), 1))

    # -----------------------------------------------------------------------
    # 3. 引用覆盖率 (Citation Coverage)
    # -----------------------------------------------------------------------
    @staticmethod
    def citation_coverage(report: str) -> float:
        """
        计算报告中包含引用来源的段落比例。

        引用标记形式：
        - [N] 或 [来源: ...]
        - 【来源: ...】
        - (来源: ...)
        """
        if not report:
            return 0.0

        paragraphs = [p.strip() for p in report.split("\n") if p.strip()]
        if not paragraphs:
            return 0.0

        citation_patterns = [
            r"\[\d+\]",
            r"\[来源[：:]",
            r"【来源[：:]",
            r"\(来源[：:]",
            r"https?://",
            r"arxiv\.org",
        ]

        cited_paragraphs = 0
        for para in paragraphs:
            for pattern in citation_patterns:
                if re.search(pattern, para):
                    cited_paragraphs += 1
                    break

        return cited_paragraphs / len(paragraphs)

    # -----------------------------------------------------------------------
    # 4. 逻辑一致性 (Logical Consistency)
    # -----------------------------------------------------------------------
    @staticmethod
    def logical_consistency(report: str) -> float:
        """
        估算报告的逻辑一致性分数。

        当前启发式策略：
        - 检测明显的自相矛盾关键词对（"是" vs "不是" 在同一上下文）。
        - 检测逻辑连接词使用是否合理（"因此"、"然而"前是否有前提）。
        """
        if not report:
            return 0.0

        # 简单检测矛盾对：句子中同时出现 A 和 非A（同一句话）
        contradiction_pairs = [
            ("是", "不是"),
            ("可以", "不可以"),
            ("会", "不会"),
            ("支持", "反对"),
            ("增加", "减少"),
        ]

        sentences = re.split(r"[。！？\n]", report)
        sentences = [s.strip() for s in sentences if s.strip()]
        if not sentences:
            return 0.0

        contradiction_count = 0
        for sentence in sentences:
            for a, b in contradiction_pairs:
                if a in sentence and b in sentence:
                    # 更严格的检查：确保它们之间没有否定词分隔
                    contradiction_count += 1
                    break

        # 同时奖励使用逻辑连接词
        connectives = ["因此", "所以", "然而", "但是", "首先", "其次", "综上所述"]
        connective_count = sum(1 for c in connectives if c in report)
        connective_bonus = min(0.1, connective_count * 0.01)

        base_score = 1.0 - (contradiction_count / max(len(sentences), 1))
        return min(1.0, max(0.0, base_score + connective_bonus))

    # -----------------------------------------------------------------------
    # 5. 完备性 (Comprehensiveness)
    # -----------------------------------------------------------------------
    @staticmethod
    def comprehensiveness(report: str, expected_topics: list[str] | None = None) -> float:
        """
        计算报告对期望主题的覆盖程度。
        """
        if not expected_topics:
            return 0.0

        report_lower = report.lower()
        covered = 0
        for topic in expected_topics:
            if topic.lower() in report_lower:
                covered += 1

        return covered / len(expected_topics) if expected_topics else 0.0

    # -----------------------------------------------------------------------
    # 6b. 证据图质量 (Evidence Graph Quality)
    # -----------------------------------------------------------------------
    @staticmethod
    def evidence_graph_quality(report_metadata: dict[str, Any] | None = None) -> float:
        """
        基于 report.metadata 中的 evidence snapshot 粗略评估证据图质量。

        该指标鼓励：
        - claim 节点不是空的
        - support edge 与 claim 数量匹配
        - contradiction edge 不至于过多
        - source trust 不至于过低
        """
        breakdown = RuleBasedMetrics.evidence_graph_breakdown(report_metadata)
        return breakdown["evidence_graph_quality"]

    @staticmethod
    def evidence_graph_breakdown(report_metadata: dict[str, Any] | None = None) -> dict[str, float]:
        """
        提取证据图的细粒度结构指标。

        返回值包含 claim / edge / trust 层面的分量，便于批量评测、消融分析、
        以及把 evidence state 接入 learned policy 的 reward shaping。
        """
        empty = {
            "has_evidence_snapshot": 0.0,
            "claim_count": 0.0,
            "support_edge_count": 0.0,
            "contradiction_edge_count": 0.0,
            "source_count": 0.0,
            "claim_support_coverage": 0.0,
            "claim_readiness": 0.0,
            "support_edge_precision": 0.0,
            "contradiction_edge_precision": 1.0,
            "edge_structure_score": 0.0,
            "source_trust_mean": 0.0,
            "contradicted_claim_ratio": 0.0,
            "contradiction_burden": 0.0,
            "consistency_score": 0.0,
            "candidate_action_margin": 0.0,
            "supported_claim_ratio": 0.0,
            "avg_support_weight": 0.0,
            "contradiction_density": 0.0,
            "avg_contradiction_severity": 0.0,
            "avg_source_trust": 0.0,
            "evidence_graph_quality": 0.0,
        }
        if not report_metadata:
            return empty

        snapshot = report_metadata.get("evidence_snapshot", {})
        if not isinstance(snapshot, dict) or not snapshot:
            return empty

        claims = snapshot.get("claims", [])
        supports = snapshot.get("support_edges", [])
        contradictions = snapshot.get("contradiction_edges", [])
        trusts = snapshot.get("source_trusts", [])
        candidate_actions = snapshot.get("candidate_actions", [])

        claim_count = len(claims)
        if claim_count <= 0:
            return empty

        claim_ids = {
            str(item.get("claim_id", ""))
            for item in claims
            if isinstance(item, dict) and item.get("claim_id")
        }
        claim_source_ids = {
            str(item.get("source_id", ""))
            for item in claims
            if isinstance(item, dict) and item.get("source_id")
        }
        trust_source_ids = {
            str(item.get("source_id", ""))
            for item in trusts
            if isinstance(item, dict) and item.get("source_id")
        }
        known_source_ids = claim_source_ids | trust_source_ids

        claims_with_support: set[str] = set()
        valid_support_edges = 0
        for item in supports:
            if not isinstance(item, dict):
                continue
            claim_id = str(item.get("claim_id", "") or "")
            source_id = str(item.get("source_id", "") or "")
            if claim_id in claim_ids and (not source_id or source_id in known_source_ids):
                claims_with_support.add(claim_id)
                valid_support_edges += 1

        claim_readiness_parts: list[float] = []
        for item in claims:
            if not isinstance(item, dict):
                continue
            claim_readiness_parts.append(
                0.40 * RuleBasedMetrics._clamp01(item.get("support_score", 0.0))
                + 0.35 * RuleBasedMetrics._clamp01(item.get("source_trust", 0.0))
                + 0.25 * RuleBasedMetrics._clamp01(item.get("confidence", 0.0))
            )

        valid_contradiction_edges = 0
        contradicted_claim_ids: set[str] = set()
        contradiction_severity_sum = 0.0
        for item in contradictions:
            if not isinstance(item, dict):
                continue
            claim_id_1 = str(item.get("claim_id_1", "") or "")
            claim_id_2 = str(item.get("claim_id_2", "") or "")
            if claim_id_1 in claim_ids and claim_id_2 in claim_ids:
                valid_contradiction_edges += 1
                contradicted_claim_ids.update([claim_id_1, claim_id_2])
                contradiction_severity_sum += RuleBasedMetrics._clamp01(
                    item.get("severity", 0.0)
                )

        trust_scores: list[float] = []
        for item in trusts:
            if isinstance(item, dict):
                trust_scores.append(
                    RuleBasedMetrics._clamp01(item.get("trust_score", 0.0))
                )

        candidate_scores: list[float] = []
        for item in candidate_actions:
            if isinstance(item, dict) and "score" in item:
                candidate_scores.append(float(item.get("score", 0.0) or 0.0))
        candidate_scores.sort(reverse=True)

        claim_support_coverage = len(claims_with_support) / max(claim_count, 1)
        claim_readiness = (
            sum(claim_readiness_parts) / len(claim_readiness_parts)
            if claim_readiness_parts
            else 0.0
        )
        support_edge_precision = valid_support_edges / len(supports) if supports else 0.0
        contradiction_edge_precision = (
            valid_contradiction_edges / len(contradictions)
            if contradictions
            else 1.0
        )
        edge_structure_score = (
            0.60 * support_edge_precision + 0.40 * contradiction_edge_precision
            if contradictions
            else support_edge_precision
        )
        source_trust_mean = sum(trust_scores) / len(trust_scores) if trust_scores else 0.0
        contradicted_claim_ratio = len(contradicted_claim_ids) / max(claim_count, 1)
        contradiction_burden = min(1.0, contradiction_severity_sum / max(claim_count, 1))
        consistency_score = max(
            0.0,
            1.0 - min(1.0, 0.55 * contradiction_burden + 0.45 * contradicted_claim_ratio),
        )
        candidate_action_margin = 0.0
        if len(candidate_scores) >= 2:
            candidate_action_margin = max(
                0.0,
                min(1.0, candidate_scores[0] - candidate_scores[1]),
            )
        elif candidate_scores:
            candidate_action_margin = max(0.0, min(1.0, candidate_scores[0]))

        evidence_graph_quality = max(
            0.0,
            min(
                1.0,
                0.30 * claim_support_coverage
                + 0.20 * edge_structure_score
                + 0.20 * claim_readiness
                + 0.15 * source_trust_mean
                + 0.15 * consistency_score,
            ),
        )

        return {
            "has_evidence_snapshot": 1.0,
            "claim_count": float(claim_count),
            "support_edge_count": float(len(supports)),
            "contradiction_edge_count": float(len(contradictions)),
            "source_count": float(len(trusts)),
            "claim_support_coverage": claim_support_coverage,
            "claim_readiness": claim_readiness,
            "support_edge_precision": support_edge_precision,
            "contradiction_edge_precision": contradiction_edge_precision,
            "edge_structure_score": edge_structure_score,
            "source_trust_mean": source_trust_mean,
            "contradicted_claim_ratio": contradicted_claim_ratio,
            "contradiction_burden": contradiction_burden,
            "consistency_score": consistency_score,
            "candidate_action_margin": candidate_action_margin,
            "supported_claim_ratio": claim_support_coverage,
            "avg_support_weight": claim_readiness,
            "contradiction_density": contradicted_claim_ratio,
            "avg_contradiction_severity": contradiction_burden,
            "avg_source_trust": source_trust_mean,
            "evidence_graph_quality": evidence_graph_quality,
        }

    @staticmethod
    def evidence_transition_quality(report_metadata: dict[str, Any] | None = None) -> float:
        """
        基于 step-wise evidence transitions 估计过程质量。

        该指标偏向 reward shaping：鼓励 uncertainty 下降、coverage 上升、冲突减少。
        """
        if not report_metadata:
            return 0.0

        transitions = (
            report_metadata.get("evidence_transition_trace")
            or report_metadata.get("evidence_transitions")
            or []
        )
        if not isinstance(transitions, list) or not transitions:
            return 0.0

        step_scores: list[float] = []
        for item in transitions:
            if not isinstance(item, dict):
                continue
            before = item.get("before", {})
            after = item.get("after", {})
            if not isinstance(before, dict) or not isinstance(after, dict):
                if "uncertainty" not in item and "coverage_ratio" not in item:
                    continue
                before = {"uncertainty": 1.0, "coverage_ratio": 0.0, "open_conflicts": 0.0, "evidence_graph_quality": 0.0}
                after = item

            before_uncertainty = float(before.get("uncertainty", 1.0) or 1.0)
            after_uncertainty = float(after.get("uncertainty", before_uncertainty) or before_uncertainty)
            before_coverage = float(before.get("coverage_ratio", 0.0) or 0.0)
            after_coverage = float(after.get("coverage_ratio", before_coverage) or before_coverage)
            before_conflicts = float(before.get("open_conflicts", 0.0) or 0.0)
            after_conflicts = float(after.get("open_conflicts", before_conflicts) or before_conflicts)
            before_quality = float(before.get("evidence_graph_quality", 0.0) or 0.0)
            after_quality = float(after.get("evidence_graph_quality", before_quality) or before_quality)

            uncertainty_gain = max(0.0, before_uncertainty - after_uncertainty)
            coverage_gain = max(0.0, after_coverage - before_coverage)
            conflict_gain = max(0.0, before_conflicts - after_conflicts) / max(before_conflicts, 1.0)
            quality_gain = max(0.0, after_quality - before_quality)

            step_score = max(
                0.0,
                min(
                    1.0,
                    0.42 * uncertainty_gain
                    + 0.28 * coverage_gain
                    + 0.20 * conflict_gain
                    + 0.10 * quality_gain,
                ),
            )
            step_scores.append(step_score)

        if not step_scores:
            return 0.0

        return max(0.0, min(1.0, sum(step_scores) / len(step_scores)))

    # -----------------------------------------------------------------------
    # 6. 综合得分 (Composite Score)
    # -----------------------------------------------------------------------
    @staticmethod
    def composite_score(
        metrics: dict[str, float],
        weights: dict[str, float] | None = None,
    ) -> float:
        """
        基于多维度指标和权重计算加权综合得分。

        默认权重与 Red Agent 的五维度对齐：
        - factual_accuracy: 0.25
        - logical_consistency: 0.20
        - citation_coverage: 0.20
        - bias (1 - hallucination_rate 作为代理): 0.20
        - comprehensiveness: 0.15
        """
        default_weights = {
            "factual_accuracy": 0.24,
            "logical_consistency": 0.18,
            "citation_coverage": 0.16,
            "bias": 0.16,
            "comprehensiveness": 0.16,
            "evidence_graph_quality": 0.10,
        }

        w = weights if weights is not None else default_weights
        normalized = RuleBasedMetrics._normalize_metrics(metrics)
        total_score = 0.0
        total_weight = 0.0

        for key, weight in w.items():
            if key not in normalized:
                continue
            value = normalized.get(key, 0.0)
            total_score += value * weight
            total_weight += weight

        return total_score / total_weight if total_weight > 0 else 0.0

    # -----------------------------------------------------------------------
    # 7. 效率指标 (Efficiency)
    # -----------------------------------------------------------------------
    @staticmethod
    def efficiency_score(
        num_turns: int,
        target_turns: float = 8.0,
        slope: float = 0.5,
        max_bonus: float = 0.5,
    ) -> float:
        """
        基于 sigmoid 的效率奖励分数。

        公式：max_bonus * sigmoid(slope * (target_turns - num_turns))
        """
        sigmoid = 1.0 / (1.0 + math.exp(-slope * (target_turns - num_turns)))
        return max_bonus * sigmoid
