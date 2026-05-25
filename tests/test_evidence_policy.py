from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evidence.policy import (  # noqa: E402
    EvidenceAwareResearchPolicy,
    EvidenceConflict,
    EvidenceNode,
)
from src.core.runner import create_evidence_policy  # noqa: E402
from src.evidence.learned_policy import LearnedEvidenceAwareResearchPolicy  # noqa: E402
from src.evolution.evidence_dataset import EvidenceTransitionDatasetBuilder  # noqa: E402
from evaluation.metrics.rule_based import RuleBasedMetrics  # noqa: E402


def test_no_evidence_recommends_search() -> None:
    policy = EvidenceAwareResearchPolicy()

    snapshot = policy.analyze("分析 GPT-4o 与 Claude 3.5 的差异", nodes=[], conflicts=[])

    assert snapshot.recommended_action == "search"
    assert snapshot.candidate_actions[0].name == "search"
    assert snapshot.claims == []
    assert snapshot.support_edges == []
    assert snapshot.contradiction_edges == []
    assert snapshot.uncertainty >= 0.80
    assert "No evidence nodes" in snapshot.to_markdown()


def test_open_conflict_recommends_verify() -> None:
    policy = EvidenceAwareResearchPolicy()
    nodes = [
        EvidenceNode(
            entry_id="a",
            claim="GPT-4o 在中文推理上表现更强。",
            source="task:a",
            confidence=0.92,
            evidence_type="primary",
            topic="llm",
            similarity=0.91,
            timestamp=1_700_000_000,
        ),
        EvidenceNode(
            entry_id="b",
            claim="Claude 3.5 在中文推理上表现更强。",
            source="task:b",
            confidence=0.90,
            evidence_type="primary",
            topic="llm",
            similarity=0.89,
            timestamp=1_700_000_100,
        ),
    ]
    conflicts = [
        EvidenceConflict(
            conflict_id="c1",
            entry_id_1="a",
            entry_id_2="b",
            claim_1=nodes[0].claim,
            claim_2=nodes[1].claim,
            similarity=0.88,
            status="open",
        )
    ]

    snapshot = policy.analyze("分析 GPT-4o 与 Claude 3.5 的差异", nodes=nodes, conflicts=conflicts)

    assert snapshot.recommended_action == "verify"
    assert snapshot.candidate_actions[0].name == "verify"
    assert snapshot.conflicts
    assert len(snapshot.claims) == 2
    assert len(snapshot.support_edges) == 2
    assert len(snapshot.contradiction_edges) == 1
    assert len(snapshot.source_trusts) == 2
    assert snapshot.uncertainty < 0.7
    assert RuleBasedMetrics.evidence_graph_quality({"evidence_snapshot": snapshot.to_dict()}) > 0.5


def test_strong_coverage_without_conflicts_recommends_stop() -> None:
    policy = EvidenceAwareResearchPolicy()
    nodes = [
        EvidenceNode(
            entry_id="a",
            claim="GPT-4o 在中文推理上表现更强。",
            source="task:a",
            confidence=0.95,
            evidence_type="primary",
            topic="llm",
            similarity=0.95,
            timestamp=1_700_000_000,
        ),
        EvidenceNode(
            entry_id="b",
            claim="Claude 3.5 在长上下文任务上有优势。",
            source="task:b",
            confidence=0.94,
            evidence_type="primary",
            topic="llm",
            similarity=0.94,
            timestamp=1_700_000_100,
        ),
        EvidenceNode(
            entry_id="c",
            claim="两者差异主要来自上下文窗口和对齐策略。",
            source="task:c",
            confidence=0.91,
            evidence_type="secondary",
            topic="llm",
            similarity=0.92,
            timestamp=1_700_000_200,
        ),
    ]

    snapshot = policy.analyze("分析 GPT-4o 与 Claude 3.5 在中文推理和长上下文上的差异", nodes=nodes, conflicts=[])

    assert snapshot.coverage_ratio > 0.5
    assert snapshot.recommended_action == "stop"
    assert snapshot.candidate_actions[0].name == "stop"
    assert len(snapshot.claims) == 3
    assert len(snapshot.support_edges) == 3
    assert len(snapshot.source_trusts) == 3
    assert snapshot.uncertainty <= 0.32


def _conflict_state() -> tuple[str, list[EvidenceNode], list[EvidenceConflict]]:
    query = "分析 GPT-4o 与 Claude 3.5 的差异"
    nodes = [
        EvidenceNode(
            entry_id="a",
            claim="GPT-4o 在中文推理上表现更强。",
            source="task:a",
            confidence=0.92,
            evidence_type="primary",
            topic="llm",
            similarity=0.91,
            timestamp=1_700_000_000,
        ),
        EvidenceNode(
            entry_id="b",
            claim="Claude 3.5 在中文推理上表现更强。",
            source="task:b",
            confidence=0.90,
            evidence_type="primary",
            topic="llm",
            similarity=0.89,
            timestamp=1_700_000_100,
        ),
    ]
    conflicts = [
        EvidenceConflict(
            conflict_id="c1",
            entry_id_1="a",
            entry_id_2="b",
            claim_1=nodes[0].claim,
            claim_2=nodes[1].claim,
            similarity=0.88,
            status="open",
        )
    ]
    return query, nodes, conflicts


def _stop_state() -> tuple[str, list[EvidenceNode], list[EvidenceConflict]]:
    query = "分析 GPT-4o 与 Claude 3.5 在中文推理和长上下文上的差异"
    nodes = [
        EvidenceNode(
            entry_id="a",
            claim="GPT-4o 在中文推理上表现更强。",
            source="task:a",
            confidence=0.95,
            evidence_type="primary",
            topic="llm",
            similarity=0.95,
            timestamp=1_700_000_000,
        ),
        EvidenceNode(
            entry_id="b",
            claim="Claude 3.5 在长上下文任务上有优势。",
            source="task:b",
            confidence=0.94,
            evidence_type="primary",
            topic="llm",
            similarity=0.94,
            timestamp=1_700_000_100,
        ),
        EvidenceNode(
            entry_id="c",
            claim="两者差异主要来自上下文窗口和对齐策略。",
            source="task:c",
            confidence=0.91,
            evidence_type="secondary",
            topic="llm",
            similarity=0.92,
            timestamp=1_700_000_200,
        ),
    ]
    return query, nodes, []


def _row_from_snapshot(snapshot, label_action: str, reward: float) -> dict:
    features = LearnedEvidenceAwareResearchPolicy.snapshot_features(snapshot)
    return {
        "label_action": label_action,
        "reward": reward,
        "features": features,
        "feature_vector": [
            float(features[name]) for name in EvidenceTransitionDatasetBuilder.FEATURE_NAMES
        ],
    }


def _manual_row(label_action: str, reward: float, **features: float) -> dict:
    template = {name: 0.0 for name in EvidenceTransitionDatasetBuilder.FEATURE_NAMES}
    template.update(features)
    return {
        "label_action": label_action,
        "reward": reward,
        "features": template,
        "feature_vector": [
            float(template[name]) for name in EvidenceTransitionDatasetBuilder.FEATURE_NAMES
        ],
    }


def _training_rows() -> list[dict]:
    heuristic = EvidenceAwareResearchPolicy()
    search_snapshot = heuristic.analyze("比较大模型 agent 研究进展", nodes=[], conflicts=[])
    verify_query, verify_nodes, verify_conflicts = _conflict_state()
    verify_snapshot = heuristic.analyze(verify_query, nodes=verify_nodes, conflicts=verify_conflicts)
    stop_query, stop_nodes, stop_conflicts = _stop_state()
    stop_snapshot = heuristic.analyze(stop_query, nodes=stop_nodes, conflicts=stop_conflicts)
    return [
        _row_from_snapshot(search_snapshot, "search", 0.35),
        _row_from_snapshot(search_snapshot, "search", 0.50),
        _row_from_snapshot(verify_snapshot, "verify", 0.80),
        _row_from_snapshot(verify_snapshot, "verify", 0.60),
        _row_from_snapshot(stop_snapshot, "stop", 0.92),
        _row_from_snapshot(stop_snapshot, "stop", 0.75),
        _manual_row(
            "retrieve_more",
            0.42,
            uncertainty=0.58,
            coverage_ratio=0.34,
            evidence_strength=0.41,
            open_conflicts=0.0,
            claim_count=2.0,
            support_edge_count=2.0,
            contradiction_edge_count=0.0,
            avg_source_trust=0.62,
            missing_term_count=1.0,
            candidate_action_margin=0.12,
            candidate_top_score=0.56,
            candidate_second_score=0.44,
            candidate_action_entropy=0.94,
            evidence_graph_quality=0.52,
            claim_support_coverage=0.66,
            consistency_score=0.73,
        ),
        _manual_row(
            "expand_subquestion",
            0.47,
            uncertainty=0.62,
            coverage_ratio=0.21,
            evidence_strength=0.43,
            open_conflicts=0.0,
            claim_count=1.0,
            support_edge_count=1.0,
            contradiction_edge_count=0.0,
            avg_source_trust=0.58,
            missing_term_count=4.0,
            candidate_action_margin=0.08,
            candidate_top_score=0.51,
            candidate_second_score=0.43,
            candidate_action_entropy=0.98,
            evidence_graph_quality=0.40,
            claim_support_coverage=0.33,
            consistency_score=0.67,
        ),
    ]


def test_learned_policy_falls_back_to_heuristic_when_untrained() -> None:
    policy = LearnedEvidenceAwareResearchPolicy()
    query, nodes, conflicts = _stop_state()

    snapshot = policy.analyze(query, nodes=nodes, conflicts=conflicts)

    assert snapshot.recommended_action == "stop"
    assert snapshot.candidate_actions[0].name == "stop"


def test_learned_policy_fit_and_save_load_roundtrip(tmp_path: Path) -> None:
    rows = _training_rows()
    model_path = tmp_path / "evidence_policy.json"

    policy = LearnedEvidenceAwareResearchPolicy(model_path=model_path, blend_weight=0.85)
    stats = policy.fit(rows, epochs=320, learning_rate=0.16, l2=1e-4)

    assert stats["num_examples"] == len(rows)
    assert stats["train_accuracy"] >= 0.8
    assert model_path.exists()
    assert policy.predict_rows(rows).count("stop") >= 2

    query, nodes, conflicts = _conflict_state()
    snapshot = policy.analyze(query, nodes=nodes, conflicts=conflicts)
    assert snapshot.recommended_action == "verify"
    assert snapshot.candidate_actions[0].name == "verify"

    loaded = LearnedEvidenceAwareResearchPolicy.load(model_path, blend_weight=0.85)
    loaded_snapshot = loaded.analyze(query, nodes=nodes, conflicts=conflicts)
    assert loaded_snapshot.recommended_action == snapshot.recommended_action
    assert loaded_snapshot.candidate_actions[0].score == snapshot.candidate_actions[0].score


def test_create_evidence_policy_loads_learned_model(tmp_path: Path) -> None:
    model_path = tmp_path / "evidence_policy.json"
    trained = LearnedEvidenceAwareResearchPolicy(model_path=model_path, blend_weight=0.85)
    trained.fit(_training_rows(), epochs=320, learning_rate=0.16, l2=1e-4)

    policy = create_evidence_policy(
        {
            "evidence_policy": {
                "mode": "learned",
                "model_path": str(model_path),
                "blend_weight": 0.85,
            }
        }
    )

    assert isinstance(policy, LearnedEvidenceAwareResearchPolicy)
    assert policy.is_trained

    query, nodes, conflicts = _conflict_state()
    snapshot = policy.analyze(query, nodes=nodes, conflicts=conflicts)
    assert snapshot.recommended_action == "verify"


def test_create_evidence_policy_falls_back_when_model_missing(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing_evidence_policy.json"

    policy = create_evidence_policy(
        {
            "evidence_policy": {
                "mode": "learned",
                "model_path": str(missing_path),
                "blend_weight": 0.85,
            }
        }
    )

    assert type(policy) is EvidenceAwareResearchPolicy
