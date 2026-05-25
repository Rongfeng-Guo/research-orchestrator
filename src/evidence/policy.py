from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
import math
import re
import time
from typing import Any, Sequence


_STOPWORDS = {
    "请",
    "分析",
    "比较",
    "对比",
    "评估",
    "探讨",
    "综述",
    "研究",
    "当前",
    "最新",
    "近年",
    "近年来",
    "关于",
    "对于",
    "如何",
    "为什么",
    "怎样",
    "什么",
    "是不是",
    "是否",
    "以及",
    "并且",
    "还有",
    "是否会",
    "的",
    "在",
    "与",
    "和",
    "及",
    "并",
}


def _truncate(text: str, limit: int = 160) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."


def _get_value(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _normalize(obj: Any) -> dict[str, Any]:
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return dict(obj)
    if is_dataclass(obj):
        return asdict(obj)
    return {
        "entry_id": _get_value(obj, "entry_id", ""),
        "claim": _get_value(obj, "claim", ""),
        "source": _get_value(obj, "source", ""),
        "confidence": _get_value(obj, "confidence", 0.0),
        "agent_id": _get_value(obj, "agent_id", ""),
        "timestamp": _get_value(obj, "timestamp", 0.0),
        "evidence_type": _get_value(obj, "evidence_type", "inference"),
        "topic": _get_value(obj, "topic", ""),
        "similarity": _get_value(obj, "similarity", 0.0),
        "status": _get_value(obj, "status", "open"),
        "resolution": _get_value(obj, "resolution", None),
        "conflict_id": _get_value(obj, "conflict_id", ""),
        "entry_id_1": _get_value(obj, "entry_id_1", ""),
        "entry_id_2": _get_value(obj, "entry_id_2", ""),
        "claim_1": _get_value(obj, "claim_1", ""),
        "claim_2": _get_value(obj, "claim_2", ""),
    }


@dataclass
class EvidenceNode:
    entry_id: str
    claim: str
    source: str
    confidence: float
    evidence_type: str
    topic: str
    similarity: float = 0.0
    timestamp: float = 0.0


@dataclass
class EvidenceConflict:
    conflict_id: str
    entry_id_1: str
    entry_id_2: str
    claim_1: str
    claim_2: str
    similarity: float
    status: str
    resolution: str | None = None


@dataclass
class CandidateAction:
    name: str
    score: float
    rationale: str


@dataclass
class SourceTrust:
    source_id: str
    trust_score: float
    trust_label: str
    evidence_type: str
    rationale: str


@dataclass
class ClaimNode:
    claim_id: str
    claim: str
    source_id: str
    source_trust: float
    support_score: float
    confidence: float
    evidence_type: str
    topic: str


@dataclass
class SupportEdge:
    support_id: str
    source_id: str
    claim_id: str
    weight: float
    rationale: str


@dataclass
class ContradictionEdge:
    contradiction_id: str
    claim_id_1: str
    claim_id_2: str
    severity: float
    rationale: str


@dataclass
class EvidenceSnapshot:
    query: str
    nodes: list[EvidenceNode] = field(default_factory=list)
    conflicts: list[EvidenceConflict] = field(default_factory=list)
    claims: list[ClaimNode] = field(default_factory=list)
    support_edges: list[SupportEdge] = field(default_factory=list)
    contradiction_edges: list[ContradictionEdge] = field(default_factory=list)
    source_trusts: list[SourceTrust] = field(default_factory=list)
    candidate_actions: list[CandidateAction] = field(default_factory=list)
    focus_terms: list[str] = field(default_factory=list)
    covered_terms: list[str] = field(default_factory=list)
    missing_terms: list[str] = field(default_factory=list)
    coverage_ratio: float = 0.0
    evidence_strength: float = 0.0
    conflict_ratio: float = 0.0
    uncertainty: float = 1.0
    recommended_action: str = "search"
    rationale: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_markdown(self, max_nodes: int = 5, max_conflicts: int = 3) -> str:
        lines = [
            "## Evidence Snapshot",
            f"- Recommended action: `{self.recommended_action}`",
            f"- Uncertainty: {self.uncertainty:.2f}",
            f"- Evidence strength: {self.evidence_strength:.2f}",
            f"- Coverage ratio: {self.coverage_ratio:.2f}",
            f"- Open conflict ratio: {self.conflict_ratio:.2f}",
        ]

        if self.focus_terms:
            lines.append(f"- Focus terms: {', '.join(self.focus_terms[:10])}")
        if self.missing_terms:
            lines.append(f"- Missing terms: {', '.join(self.missing_terms[:8])}")
        if self.rationale:
            lines.append(f"- Rationale: {self.rationale}")

        if self.candidate_actions:
            lines.append("- Candidate actions:")
            for action in self.candidate_actions[:4]:
                lines.append(
                    f"  - `{action.name}` score={action.score:.2f}: {action.rationale}"
                )

        if self.claims:
            lines.append("- Claims:")
            for claim in self.claims[:max_nodes]:
                lines.append(
                    f"  - [{claim.claim_id}] { _truncate(claim.claim, 140) } "
                    f"(support={claim.support_score:.2f}, trust={claim.source_trust:.2f})"
                )

        if self.support_edges:
            lines.append("- Support edges:")
            for edge in self.support_edges[:max_nodes]:
                lines.append(
                    f"  - {edge.source_id} -> {edge.claim_id} "
                    f"(weight={edge.weight:.2f})"
                )

        if self.source_trusts:
            lines.append("- Source trust:")
            for trust in self.source_trusts[:max_nodes]:
                lines.append(
                    f"  - [{trust.source_id}] {trust.trust_label} "
                    f"({trust.trust_score:.2f})"
                )

        if not self.nodes:
            lines.append("- No evidence nodes available yet.")

        if self.nodes:
            lines.append("- Key evidence:")
            for node in self.nodes[:max_nodes]:
                claim = _truncate(node.claim, 160)
                lines.append(
                    f"  - [{node.entry_id}] {claim} "
                    f"(conf={node.confidence:.2f}, sim={node.similarity:.2f}, src={_truncate(node.source, 48)})"
                )

        if self.conflicts:
            lines.append("- Open conflicts:")
            for conflict in self.conflicts[:max_conflicts]:
                left = _truncate(conflict.claim_1, 80)
                right = _truncate(conflict.claim_2, 80)
                lines.append(
                    f"  - [{conflict.conflict_id}] {left} <-> {right} "
                    f"(sim={conflict.similarity:.2f})"
                )

        return "\n".join(lines)


class EvidenceAwareResearchPolicy:
    """Deterministic evidence-state policy used to steer query refinement."""

    def __init__(
        self,
        stop_threshold: float = 0.32,
        search_threshold: float = 0.22,
        expand_threshold: float = 0.45,
    ) -> None:
        self.stop_threshold = stop_threshold
        self.search_threshold = search_threshold
        self.expand_threshold = expand_threshold

    def analyze(
        self,
        query: str,
        nodes: Sequence[Any] | None = None,
        conflicts: Sequence[Any] | None = None,
    ) -> EvidenceSnapshot:
        norm_nodes = [self._to_node(node) for node in (nodes or [])]
        norm_conflicts = [self._to_conflict(conflict) for conflict in (conflicts or [])]

        focus_terms = self._extract_focus_terms(query)
        covered_terms = self._covered_terms(focus_terms, norm_nodes)
        missing_terms = [term for term in focus_terms if term not in covered_terms]
        coverage_ratio = len(covered_terms) / max(len(focus_terms), 1) if focus_terms else 0.0

        evidence_strength = self._evidence_strength(norm_nodes)
        open_conflicts = [c for c in norm_conflicts if c.status == "open"]
        conflict_ratio = len(open_conflicts) / max(len(norm_nodes), 1) if norm_nodes else 0.0
        uncertainty = self._uncertainty(evidence_strength, coverage_ratio, conflict_ratio)

        candidate_actions = self._candidate_actions(
            evidence_strength=evidence_strength,
            coverage_ratio=coverage_ratio,
            uncertainty=uncertainty,
            open_conflicts=open_conflicts,
            nodes=norm_nodes,
            focus_terms=focus_terms,
            missing_terms=missing_terms,
        )
        candidate_actions.sort(key=lambda a: a.score, reverse=True)
        recommended_action = candidate_actions[0].name if candidate_actions else "search"
        rationale = candidate_actions[0].rationale if candidate_actions else ""
        claims, support_edges, contradiction_edges, source_trusts = self._build_graph(norm_nodes, open_conflicts)

        return EvidenceSnapshot(
            query=query,
            nodes=norm_nodes,
            conflicts=open_conflicts,
            claims=claims,
            support_edges=support_edges,
            contradiction_edges=contradiction_edges,
            source_trusts=source_trusts,
            candidate_actions=candidate_actions,
            focus_terms=focus_terms,
            covered_terms=covered_terms,
            missing_terms=missing_terms,
            coverage_ratio=coverage_ratio,
            evidence_strength=evidence_strength,
            conflict_ratio=conflict_ratio,
            uncertainty=uncertainty,
            recommended_action=recommended_action,
            rationale=rationale,
        )

    def _to_node(self, obj: Any) -> EvidenceNode:
        data = _normalize(obj)
        return EvidenceNode(
            entry_id=str(data.get("entry_id", "")),
            claim=str(data.get("claim", "")),
            source=str(data.get("source", "")),
            confidence=float(data.get("confidence", 0.0) or 0.0),
            evidence_type=str(data.get("evidence_type", "inference")),
            topic=str(data.get("topic", "")),
            similarity=float(data.get("similarity", 0.0) or 0.0),
            timestamp=float(data.get("timestamp", 0.0) or 0.0),
        )

    def _to_conflict(self, obj: Any) -> EvidenceConflict:
        data = _normalize(obj)
        return EvidenceConflict(
            conflict_id=str(data.get("conflict_id", "")),
            entry_id_1=str(data.get("entry_id_1", "")),
            entry_id_2=str(data.get("entry_id_2", "")),
            claim_1=str(data.get("claim_1", "")),
            claim_2=str(data.get("claim_2", "")),
            similarity=float(data.get("similarity", 0.0) or 0.0),
            status=str(data.get("status", "open")),
            resolution=data.get("resolution"),
        )

    def _build_graph(
        self,
        nodes: Sequence[EvidenceNode],
        conflicts: Sequence[EvidenceConflict],
    ) -> tuple[list[ClaimNode], list[SupportEdge], list[ContradictionEdge], list[SourceTrust]]:
        claims: list[ClaimNode] = []
        support_edges: list[SupportEdge] = []
        contradiction_edges: list[ContradictionEdge] = []
        source_trust_map: dict[str, list[float]] = {}
        source_meta: dict[str, tuple[str, str]] = {}

        for node in nodes:
            trust_score = self._source_trust(node)
            support_score = self._support_score(node, trust_score)
            source_id = node.source or node.entry_id
            claims.append(
                ClaimNode(
                    claim_id=node.entry_id,
                    claim=node.claim,
                    source_id=source_id,
                    source_trust=trust_score,
                    support_score=support_score,
                    confidence=node.confidence,
                    evidence_type=node.evidence_type,
                    topic=node.topic,
                )
            )
            support_edges.append(
                SupportEdge(
                    support_id=f"support:{node.entry_id}",
                    source_id=source_id,
                    claim_id=node.entry_id,
                    weight=support_score,
                    rationale=f"Derived from {node.evidence_type} evidence with confidence {node.confidence:.2f}.",
                )
            )
            source_trust_map.setdefault(source_id, []).append(trust_score)
            source_meta[source_id] = (node.evidence_type, node.claim)

        for conflict in conflicts:
            severity = max(0.0, min(1.0, conflict.similarity))
            contradiction_edges.append(
                ContradictionEdge(
                    contradiction_id=conflict.conflict_id,
                    claim_id_1=conflict.entry_id_1,
                    claim_id_2=conflict.entry_id_2,
                    severity=severity,
                    rationale=f"Open conflict with similarity {conflict.similarity:.2f}.",
                )
            )

        source_trusts: list[SourceTrust] = []
        for source_id, scores in source_trust_map.items():
            trust_score = sum(scores) / max(len(scores), 1)
            evidence_type = source_meta.get(source_id, ("inference", ""))[0]
            source_trusts.append(
                SourceTrust(
                    source_id=source_id,
                    trust_score=trust_score,
                    trust_label=self._trust_label(trust_score),
                    evidence_type=evidence_type,
                    rationale=f"Aggregated from {len(scores)} supporting claim(s).",
                )
            )

        source_trusts.sort(key=lambda item: item.trust_score, reverse=True)
        return claims, support_edges, contradiction_edges, source_trusts

    def _source_trust(self, node: EvidenceNode) -> float:
        evidence_weights = {"primary": 1.0, "secondary": 0.8, "inference": 0.6}
        confidence = max(0.0, min(1.0, node.confidence))
        evidence_weight = evidence_weights.get(node.evidence_type, 0.6)
        similarity = max(0.0, min(1.0, node.similarity or 0.0))
        return max(0.0, min(1.0, 0.55 * confidence + 0.30 * evidence_weight + 0.15 * similarity))

    def _support_score(self, node: EvidenceNode, source_trust: float) -> float:
        similarity = max(0.0, min(1.0, node.similarity or 0.0))
        if similarity <= 0.0:
            similarity = 0.55
        confidence = max(0.0, min(1.0, node.confidence))
        return max(0.0, min(1.0, 0.45 * confidence + 0.35 * source_trust + 0.20 * similarity))

    def _trust_label(self, score: float) -> str:
        if score >= 0.85:
            return "high"
        if score >= 0.65:
            return "medium"
        return "low"

    def _extract_focus_terms(self, query: str) -> list[str]:
        if not query:
            return []

        candidates: list[str] = []
        segments = re.split(r"[，,。！？；;\n\t]+", query)
        for segment in segments:
            segment = segment.strip()
            if not segment:
                continue

            sub_segments = re.split(r"(?:和|与|及|并且|并|以及)", segment)
            for sub in sub_segments:
                sub = re.sub(r"^[\s:：\-—]+|[\s:：\-—]+$", "", sub.strip())
                if not sub:
                    continue
                if len(sub) > 8 and "的" in sub:
                    pieces = [p.strip() for p in sub.split("的") if p.strip()]
                else:
                    pieces = [sub]

                for piece in pieces:
                    piece = re.sub(r"^(请|分析|比较|对比|评估|探讨|综述|研究|当前|最新|近年|近年来|关于|对于)+", "", piece)
                    piece = piece.strip()
                    if not piece:
                        continue
                    if piece in _STOPWORDS:
                        continue
                    if len(piece) < 2 and not re.search(r"[A-Za-z0-9]", piece):
                        continue
                    if re.fullmatch(r"[的和与及并且以及]+", piece):
                        continue
                    if re.search(r"[A-Za-z0-9]", piece):
                        matches = re.findall(r"[A-Za-z0-9][A-Za-z0-9\.\-\+]*", piece)
                        candidates.extend(matches or [piece])
                    else:
                        candidates.append(piece)

        # 去重并保持顺序
        seen: set[str] = set()
        result: list[str] = []
        for term in candidates:
            cleaned = term.strip()
            if not cleaned or cleaned in _STOPWORDS or cleaned in seen:
                continue
            seen.add(cleaned)
            result.append(cleaned)
        return result

    def _covered_terms(self, focus_terms: Sequence[str], nodes: Sequence[EvidenceNode]) -> list[str]:
        if not focus_terms or not nodes:
            return []

        combined_texts = [
            " ".join(
                filter(
                    None,
                    [
                        node.claim,
                        node.source,
                        node.topic,
                        node.entry_id,
                    ],
                )
            )
            for node in nodes
        ]

        covered: list[str] = []
        for term in focus_terms:
            if any(term in text for text in combined_texts):
                covered.append(term)
        return covered

    def _evidence_strength(self, nodes: Sequence[EvidenceNode]) -> float:
        if not nodes:
            return 0.0

        now = int(time.time())
        scores: list[float] = []
        for node in nodes:
            confidence = max(0.0, min(1.0, node.confidence))
            similarity = max(0.0, min(1.0, node.similarity or 0.0))
            if similarity <= 0.0:
                similarity = 0.55
            days_old = 0.0
            if node.timestamp:
                days_old = max((now - node.timestamp) / 86400.0, 0.0)
            recency = math.exp(-days_old / 30.0)
            scores.append(confidence * 0.55 + similarity * 0.30 + recency * 0.15)

        return max(0.0, min(1.0, sum(scores) / len(scores)))

    def _uncertainty(self, evidence_strength: float, coverage_ratio: float, conflict_ratio: float) -> float:
        score = 0.55 * evidence_strength + 0.30 * coverage_ratio + 0.15 * (1.0 - min(conflict_ratio, 1.0))
        return max(0.0, min(1.0, 1.0 - score))

    def _candidate_actions(
        self,
        *,
        evidence_strength: float,
        coverage_ratio: float,
        uncertainty: float,
        open_conflicts: Sequence[EvidenceConflict],
        nodes: Sequence[EvidenceNode],
        focus_terms: Sequence[str],
        missing_terms: Sequence[str],
    ) -> list[CandidateAction]:
        open_conflict_count = len(open_conflicts)
        focus_term_count = max(len(focus_terms), 1)
        missing_ratio = 1.0 - coverage_ratio
        conflict_pressure = min(1.0, open_conflict_count / max(len(nodes), 1)) if nodes else 1.0
        missing_term_pressure = len(missing_terms) / focus_term_count

        if not nodes:
            return [
                CandidateAction(
                    name="search",
                    score=1.0,
                    rationale="No evidence nodes are available yet, so the next action should acquire evidence.",
                ),
                CandidateAction(
                    name="expand_subquestion",
                    score=0.35,
                    rationale="With no evidence, the system can also expand the plan to clarify missing subtopics.",
                ),
                CandidateAction(
                    name="retrieve_more",
                    score=0.25,
                    rationale="A retrieval step is possible, but there is not enough evidence to refine it yet.",
                ),
                CandidateAction(
                    name="verify",
                    score=0.0,
                    rationale="Verification is not useful before any evidence has been collected.",
                ),
                CandidateAction(
                    name="stop",
                    score=0.0,
                    rationale="Synthesis would be premature without evidence.",
                ),
            ]

        search_score = max(
            0.0,
            min(
                1.0,
                0.72 * (1.0 - evidence_strength)
                + 0.18 * missing_ratio
                + 0.10 * conflict_pressure,
            ),
        )
        retrieve_more_score = max(
            0.0,
            min(
                1.0,
                0.55 * uncertainty
                + 0.20 * missing_ratio
                + 0.25 * (1.0 - evidence_strength),
            ),
        )
        verify_score = max(
            0.0,
            min(
                1.0,
                0.80 * conflict_pressure
                + 0.20 * (1.0 - evidence_strength),
            ),
        )
        if open_conflict_count > 0:
            verify_score = min(1.0, verify_score + 0.25 + 0.15 * conflict_pressure)
        expand_score = max(
            0.0,
            min(
                1.0,
                0.55 * missing_ratio
                + 0.25 * missing_term_pressure
                + 0.20 * (1.0 - evidence_strength),
            ),
        )
        stop_score = max(
            0.0,
            min(
                1.0,
                0.80 * (1.0 - uncertainty)
                + 0.20 * (1.0 - conflict_pressure),
            ),
        )
        if open_conflict_count > 0:
            stop_score *= max(0.0, 1.0 - 0.85 * conflict_pressure)

        gap_terms = ", ".join(missing_terms[:4]) if missing_terms else "core subtopics"
        return [
            CandidateAction(
                name="search",
                score=search_score,
                rationale=(
                    f"Evidence strength is {evidence_strength:.2f}; additional retrieval can reduce uncertainty faster."
                ),
            ),
            CandidateAction(
                name="retrieve_more",
                score=retrieve_more_score,
                rationale=(
                    f"Uncertainty is {uncertainty:.2f} and coverage is {coverage_ratio:.2f}; targeted retrieval still has value."
                ),
            ),
            CandidateAction(
                name="verify",
                score=verify_score,
                rationale=(
                    f"{open_conflict_count} open conflict(s) remain, so verification can resolve contradictory claims."
                ),
            ),
            CandidateAction(
                name="expand_subquestion",
                score=expand_score,
                rationale=(
                    f"Coverage is incomplete ({coverage_ratio:.2f}); expand the plan around missing terms: {gap_terms}."
                ),
            ),
            CandidateAction(
                name="stop",
                score=stop_score,
                rationale=(
                    f"Low uncertainty ({uncertainty:.2f}) with no strong conflict pressure suggests synthesis is now plausible."
                ),
            ),
        ]
