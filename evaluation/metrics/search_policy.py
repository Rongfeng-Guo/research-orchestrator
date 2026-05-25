#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluation/metrics/search_policy.py
================================================================================
Search-policy oriented metrics for offline analysis and reward shaping.

These metrics operate on structured traces stored in report.metadata:
  - policy_trace
  - search_cost
  - route_stats

The goal is not to replace final-answer quality metrics, but to quantify whether
the agent searched efficiently, diversified sources, and grounded its citations.
================================================================================
"""

from __future__ import annotations

import re
from typing import Any
from urllib.parse import urlparse


class SearchPolicyMetrics:
    """Search-policy metrics for agentic search trajectories."""

    @staticmethod
    def _clamp01(value: Any, default: float = 0.0) -> float:
        try:
            return max(0.0, min(1.0, float(value)))
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _normalize_url(url: str) -> str:
        if not isinstance(url, str) or not url.strip():
            return ""
        cleaned = url.strip().strip(".,;:)]}>\"'")
        try:
            parsed = urlparse(cleaned)
        except ValueError:
            return cleaned.lower()

        scheme = parsed.scheme.lower()
        netloc = parsed.netloc.lower()
        path = parsed.path.rstrip("/")
        if not netloc:
            return cleaned.lower()
        return f"{scheme}://{netloc}{path}"

    @staticmethod
    def _extract_domain(url: str) -> str:
        normalized = SearchPolicyMetrics._normalize_url(url)
        if not normalized:
            return ""
        try:
            return urlparse(normalized).netloc.lower()
        except ValueError:
            return ""

    @staticmethod
    def _extract_cited_urls(
        report_text: str,
        report_sources: list[dict[str, Any]] | None = None,
    ) -> list[str]:
        urls: list[str] = []
        seen: set[str] = set()

        if isinstance(report_text, str) and report_text:
            for match in re.findall(r"https?://[^\s\])>\"']+", report_text):
                normalized = SearchPolicyMetrics._normalize_url(match)
                if normalized and normalized not in seen:
                    seen.add(normalized)
                    urls.append(normalized)

        if isinstance(report_sources, list):
            for item in report_sources:
                if not isinstance(item, dict):
                    continue
                normalized = SearchPolicyMetrics._normalize_url(str(item.get("url", "") or ""))
                if normalized and normalized not in seen:
                    seen.add(normalized)
                    urls.append(normalized)

        return urls

    @staticmethod
    def _split_reference_section(report_text: str) -> tuple[str, str]:
        pattern = re.compile(
            r"(?mi)^\s{0,3}"
            r"(?:#{1,6}\s*)?"
            r"(?:\*\*|__)?\s*"
            r"(参考来源|参考文献|References|Sources)"
            r"\s*(?:\*\*|__)?\s*[:：]?\s*$"
        )
        match = pattern.search(report_text or "")
        if not match:
            return report_text or "", ""
        return (report_text or "")[:match.start()].rstrip(), (report_text or "")[match.start():].strip()

    @staticmethod
    def _extract_inline_citation_numbers(report_text: str) -> list[int]:
        body, _ = SearchPolicyMetrics._split_reference_section(report_text)
        numbers: list[int] = []
        for raw in re.findall(r"\[(\d+)\]", body or ""):
            try:
                numbers.append(int(raw))
            except ValueError:
                continue
        return numbers

    @staticmethod
    def _count_citable_paragraphs(report_text: str) -> int:
        body, _ = SearchPolicyMetrics._split_reference_section(report_text)
        count = 0
        in_fence = False
        for raw_line in (body or "").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("```"):
                in_fence = not in_fence
                continue
            if in_fence:
                continue
            if line.startswith("#") or line.startswith("|") or line.startswith("---"):
                continue
            if re.match(r"^(整体置信度|Overall Confidence|置信度|总体信心评分)[:：]", line, flags=re.I):
                continue
            if len(line) < 16:
                continue
            count += 1
        return count

    @staticmethod
    def _citation_source_quality(
        report_text: str,
        report_sources: list[dict[str, Any]] | None,
        source_diversity: float,
        citation_grounding: float,
    ) -> dict[str, float]:
        source_urls = SearchPolicyMetrics._extract_cited_urls("", report_sources)
        inline_numbers = SearchPolicyMetrics._extract_inline_citation_numbers(report_text)
        unique_inline_numbers = sorted(set(inline_numbers))
        body, _ = SearchPolicyMetrics._split_reference_section(report_text)
        body_urls = SearchPolicyMetrics._extract_cited_urls(body, None)

        cited_source_indexes: set[int] = set()
        for number in unique_inline_numbers:
            if source_urls and 1 <= number <= len(source_urls):
                cited_source_indexes.add(number - 1)

        if source_urls:
            indexed_sources = {url: idx for idx, url in enumerate(source_urls)}
            source_domains = {
                SearchPolicyMetrics._extract_domain(url): idx
                for idx, url in enumerate(source_urls)
                if SearchPolicyMetrics._extract_domain(url)
            }
            for url in body_urls:
                if url in indexed_sources:
                    cited_source_indexes.add(indexed_sources[url])
                    continue
                domain = SearchPolicyMetrics._extract_domain(url)
                if domain and domain in source_domains:
                    cited_source_indexes.add(source_domains[domain])

        citable_paragraphs = SearchPolicyMetrics._count_citable_paragraphs(report_text)
        inline_count = len(inline_numbers) + len(body_urls)
        cited_source_count = len(cited_source_indexes) if source_urls else len(set(body_urls))
        reference_source_count = len(source_urls)
        citation_density = (
            min(1.0, inline_count / citable_paragraphs)
            if citable_paragraphs > 0
            else 0.0
        )
        source_utilization = (
            min(1.0, cited_source_count / reference_source_count)
            if reference_source_count > 0
            else 0.0
        )
        citation_quality_score = SearchPolicyMetrics._clamp01(
            0.35 * citation_grounding
            + 0.25 * citation_density
            + 0.25 * source_utilization
            + 0.15 * source_diversity
        )

        return {
            "inline_citation_count": float(inline_count),
            "unique_inline_citation_count": float(len(unique_inline_numbers) + len(set(body_urls))),
            "reference_source_count": float(reference_source_count),
            "cited_source_count": float(cited_source_count),
            "citable_paragraph_count": float(citable_paragraphs),
            "source_utilization": source_utilization,
            "citation_density": citation_density,
            "citation_quality_score": citation_quality_score,
        }

    @staticmethod
    def breakdown(
        report_text: str,
        report_metadata: dict[str, Any] | None = None,
        report_sources: list[dict[str, Any]] | None = None,
    ) -> dict[str, float]:
        empty = {
            "has_policy_trace": 0.0,
            "action_count": 0.0,
            "tool_call_count": 0.0,
            "search_call_count": 0.0,
            "browser_call_count": 0.0,
            "unique_query_count": 0.0,
            "unique_domain_count": 0.0,
            "unique_backend_count": 0.0,
            "successful_tool_call_ratio": 0.0,
            "query_diversity": 0.0,
            "source_diversity": 0.0,
            "citation_grounding": 0.0,
            "inline_citation_count": 0.0,
            "unique_inline_citation_count": 0.0,
            "reference_source_count": 0.0,
            "cited_source_count": 0.0,
            "citable_paragraph_count": 0.0,
            "source_utilization": 0.0,
            "citation_density": 0.0,
            "citation_quality_score": 0.0,
            "stop_efficiency": 0.0,
            "budget_efficiency": 0.0,
            "search_policy_score": 0.0,
        }
        if not report_metadata or not isinstance(report_metadata, dict):
            return empty

        policy_trace = report_metadata.get("policy_trace", [])
        search_cost = report_metadata.get("search_cost", {})
        route_stats = report_metadata.get("route_stats", {})
        if not isinstance(policy_trace, list) or not policy_trace:
            return empty

        tool_calls = [item for item in policy_trace if isinstance(item, dict) and item.get("action_type") == "tool_call"]
        action_count = len(policy_trace)
        tool_call_count = len(tool_calls)
        search_call_count = 0
        browser_call_count = 0
        successful_tool_calls = 0

        query_texts: list[str] = []
        unique_queries: set[str] = set()
        traced_urls: set[str] = set()
        traced_domains: set[str] = set()
        traced_backends: set[str] = set()

        for action in tool_calls:
            tool_name = str(action.get("tool_name", "") or "")
            if tool_name == "web_search":
                search_call_count += 1
            elif tool_name == "browser":
                browser_call_count += 1

            error = str(action.get("error", "") or "")
            result_count = action.get("result_count", 0)
            try:
                result_count_num = int(result_count or 0)
            except (TypeError, ValueError):
                result_count_num = 0
            if not error and result_count_num > 0:
                successful_tool_calls += 1

            query_text = str(action.get("query_text", "") or "").strip().lower()
            if query_text:
                query_texts.append(query_text)
                unique_queries.add(query_text)

            backend = str(action.get("backend", "") or "").strip().lower()
            if backend:
                traced_backends.add(backend)

            top_urls = action.get("top_urls", [])
            if isinstance(top_urls, list):
                for raw_url in top_urls:
                    normalized = SearchPolicyMetrics._normalize_url(str(raw_url or ""))
                    if normalized:
                        traced_urls.add(normalized)
                        domain = SearchPolicyMetrics._extract_domain(normalized)
                        if domain:
                            traced_domains.add(domain)

        if isinstance(route_stats, dict):
            for domain in route_stats.get("domains_seen", []) if isinstance(route_stats.get("domains_seen"), list) else []:
                if isinstance(domain, str) and domain.strip():
                    traced_domains.add(domain.strip().lower())
            for backend in route_stats.get("search_backends", []) if isinstance(route_stats.get("search_backends"), list) else []:
                if isinstance(backend, str) and backend.strip():
                    traced_backends.add(backend.strip().lower())
            for raw_url in route_stats.get("top_urls", []) if isinstance(route_stats.get("top_urls"), list) else []:
                normalized = SearchPolicyMetrics._normalize_url(str(raw_url or ""))
                if normalized:
                    traced_urls.add(normalized)
                    domain = SearchPolicyMetrics._extract_domain(normalized)
                    if domain:
                        traced_domains.add(domain)
            for query_text in route_stats.get("queries_used", []) if isinstance(route_stats.get("queries_used"), list) else []:
                if isinstance(query_text, str) and query_text.strip():
                    normalized_query = query_text.strip().lower()
                    query_texts.append(normalized_query)
                    unique_queries.add(normalized_query)

        cited_urls = SearchPolicyMetrics._extract_cited_urls(report_text, report_sources)
        citation_grounded = 0
        for cited_url in cited_urls:
            cited_domain = SearchPolicyMetrics._extract_domain(cited_url)
            if cited_url in traced_urls or (cited_domain and cited_domain in traced_domains):
                citation_grounded += 1
        citation_grounding = (
            citation_grounded / len(cited_urls) if cited_urls else 0.0
        )

        successful_tool_call_ratio = (
            successful_tool_calls / tool_call_count if tool_call_count > 0 else 0.0
        )
        query_diversity = (
            len(unique_queries) / len(query_texts) if query_texts else 0.0
        )
        source_diversity = (
            min(1.0, len(traced_domains) / max(tool_call_count, 1))
            if tool_call_count > 0
            else 0.0
        )
        citation_quality = SearchPolicyMetrics._citation_source_quality(
            report_text,
            report_sources,
            source_diversity,
            citation_grounding,
        )

        top_level_stop_reason = str(report_metadata.get("stop_reason", "") or "").strip()
        stop_reasons = {}
        if isinstance(route_stats, dict) and isinstance(route_stats.get("stop_reasons"), dict):
            stop_reasons = route_stats["stop_reasons"]
        dominant_stop_reason = top_level_stop_reason
        if not dominant_stop_reason and stop_reasons:
            dominant_stop_reason = max(
                stop_reasons.items(),
                key=lambda item: int(item[1]) if isinstance(item[1], (int, float)) else 0,
            )[0]

        stop_base = 0.0
        if tool_call_count > 0:
            target_calls = 3.0
            stop_base = max(0.0, 1.0 - abs(tool_call_count - target_calls) / target_calls)
        if dominant_stop_reason in {"model_finished_no_tool_calls", "assistant_final_answer", "non_searchable_direct_analysis"}:
            stop_multiplier = 1.0
        elif dominant_stop_reason in {"search_limit", "empty_results"}:
            stop_multiplier = 0.75
        elif dominant_stop_reason in {"tool_error", "policy_runtime_error", "max_turns_reached"}:
            stop_multiplier = 0.35
        else:
            stop_multiplier = 0.8 if dominant_stop_reason else 0.7
        stop_efficiency = SearchPolicyMetrics._clamp01(stop_base * stop_multiplier)

        estimated_token_cost = 0.0
        if isinstance(search_cost, dict):
            try:
                estimated_token_cost = float(search_cost.get("estimated_token_cost", 0.0) or 0.0)
            except (TypeError, ValueError):
                estimated_token_cost = 0.0
        over_budget_calls = max(0.0, tool_call_count - 3.0) * 0.12
        over_budget_tokens = max(0.0, estimated_token_cost - 2500.0) / 5000.0
        budget_efficiency = SearchPolicyMetrics._clamp01(1.0 - over_budget_calls - over_budget_tokens)
        if tool_call_count <= 0:
            budget_efficiency = 0.0

        search_policy_score = SearchPolicyMetrics._clamp01(
            0.20 * successful_tool_call_ratio
            + 0.12 * query_diversity
            + 0.13 * source_diversity
            + 0.18 * citation_grounding
            + 0.12 * citation_quality["citation_quality_score"]
            + 0.10 * stop_efficiency
            + 0.15 * budget_efficiency
        )

        return {
            "has_policy_trace": 1.0,
            "action_count": float(action_count),
            "tool_call_count": float(tool_call_count),
            "search_call_count": float(search_call_count),
            "browser_call_count": float(browser_call_count),
            "unique_query_count": float(len(unique_queries)),
            "unique_domain_count": float(len(traced_domains)),
            "unique_backend_count": float(len(traced_backends)),
            "successful_tool_call_ratio": successful_tool_call_ratio,
            "query_diversity": query_diversity,
            "source_diversity": source_diversity,
            "citation_grounding": citation_grounding,
            **citation_quality,
            "stop_efficiency": stop_efficiency,
            "budget_efficiency": budget_efficiency,
            "search_policy_score": search_policy_score,
        }
