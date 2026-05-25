from __future__ import annotations

from collections import Counter
from dataclasses import replace
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from src.evolution.evidence_dataset import EvidenceTransitionDatasetBuilder

from .policy import CandidateAction, EvidenceAwareResearchPolicy, EvidenceSnapshot

__all__ = ["LearnedEvidenceAwareResearchPolicy"]


class LinearActionRanker:
    """Tiny multinomial linear model for evidence-action ranking."""

    def __init__(
        self,
        feature_names: Sequence[str],
        action_names: Sequence[str],
        *,
        means: Sequence[float] | None = None,
        scales: Sequence[float] | None = None,
        weights: Sequence[Sequence[float]] | None = None,
        bias: Sequence[float] | None = None,
        training_stats: dict[str, Any] | None = None,
    ) -> None:
        self.feature_names = list(feature_names)
        self.action_names = list(action_names)
        self.means = np.asarray(means if means is not None else np.zeros(len(self.feature_names)), dtype=np.float64)
        self.scales = np.asarray(
            scales if scales is not None else np.ones(len(self.feature_names)),
            dtype=np.float64,
        )
        self.weights = np.asarray(
            weights if weights is not None else np.zeros((len(self.action_names), len(self.feature_names))),
            dtype=np.float64,
        )
        self.bias = np.asarray(
            bias if bias is not None else np.zeros(len(self.action_names)),
            dtype=np.float64,
        )
        self.training_stats = dict(training_stats or {})
        self.is_fitted = bool(training_stats)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LinearActionRanker":
        return cls(
            feature_names=data.get("feature_names", []),
            action_names=data.get("action_names", []),
            means=data.get("means", []),
            scales=data.get("scales", []),
            weights=data.get("weights", []),
            bias=data.get("bias", []),
            training_stats=data.get("training_stats", {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": 1,
            "feature_names": list(self.feature_names),
            "action_names": list(self.action_names),
            "means": self.means.tolist(),
            "scales": self.scales.tolist(),
            "weights": self.weights.tolist(),
            "bias": self.bias.tolist(),
            "training_stats": dict(self.training_stats),
        }

    def save(self, path: str | Path) -> str:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
        return str(path)

    @classmethod
    def load(cls, path: str | Path) -> "LinearActionRanker":
        with Path(path).open("r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    def fit(
        self,
        rows: Sequence[dict[str, Any]],
        *,
        epochs: int = 240,
        learning_rate: float = 0.12,
        l2: float = 1e-4,
        seed: int = 7,
    ) -> dict[str, Any]:
        if not rows:
            raise ValueError("No rows available for training.")

        feature_vectors: list[np.ndarray] = []
        labels: list[int] = []
        sample_weights: list[float] = []
        label_counter: Counter[str] = Counter()

        action_index = {name: idx for idx, name in enumerate(self.action_names)}
        for row in rows:
            label = str(row.get("label_action", "") or "")
            if label not in action_index:
                continue
            vector = self._row_to_vector(row)
            feature_vectors.append(vector)
            labels.append(action_index[label])
            reward = float(row.get("reward", 0.0) or 0.0)
            sample_weights.append(float(np.clip(1.0 + reward, 0.25, 2.0)))
            label_counter[label] += 1

        if not feature_vectors:
            raise ValueError("No rows contain labels from the supported action space.")

        X = np.vstack(feature_vectors).astype(np.float64)
        y = np.asarray(labels, dtype=np.int64)
        w = np.asarray(sample_weights, dtype=np.float64)

        self.means = X.mean(axis=0)
        self.scales = X.std(axis=0)
        self.scales[self.scales < 1e-6] = 1.0
        Xn = (X - self.means) / self.scales

        rng = np.random.default_rng(seed)
        self.weights = rng.normal(scale=0.01, size=(len(self.action_names), Xn.shape[1]))
        self.bias = np.zeros(len(self.action_names), dtype=np.float64)

        total_weight = float(w.sum())
        y_one_hot = np.eye(len(self.action_names), dtype=np.float64)[y]
        last_loss = float("inf")

        for epoch in range(epochs):
            logits = Xn @ self.weights.T + self.bias
            probs = self._softmax(logits)
            weighted_error = (probs - y_one_hot) * w[:, None]

            grad_w = weighted_error.T @ Xn / total_weight + l2 * self.weights
            grad_b = weighted_error.sum(axis=0) / total_weight

            self.weights -= learning_rate * grad_w
            self.bias -= learning_rate * grad_b

            loss = self._loss(Xn, y, w, l2)
            if abs(last_loss - loss) < 1e-8:
                break
            last_loss = loss

        train_predictions = self.predict(X)
        accuracy = float(np.mean(train_predictions == y))
        weighted_accuracy = float(np.average(train_predictions == y, weights=w))

        self.training_stats = {
            "num_examples": int(len(X)),
            "num_features": int(X.shape[1]),
            "num_actions": int(len(self.action_names)),
            "epochs": int(epoch + 1),
            "learning_rate": float(learning_rate),
            "l2": float(l2),
            "train_accuracy": round(accuracy, 4),
            "weighted_accuracy": round(weighted_accuracy, 4),
            "label_distribution": dict(label_counter),
            "reward_mean": round(float(np.mean([float(row.get("reward", 0.0) or 0.0) for row in rows])), 4),
        }
        self.is_fitted = True
        return dict(self.training_stats)

    def predict(self, X: Sequence[float] | Sequence[Sequence[float]]) -> np.ndarray:
        probs = self.predict_proba(X)
        return np.asarray([int(np.argmax(row)) for row in probs], dtype=np.int64)

    def predict_proba(self, X: Sequence[float] | Sequence[Sequence[float]]) -> np.ndarray:
        array = np.asarray(X, dtype=np.float64)
        if array.ndim == 1:
            array = array.reshape(1, -1)
        if array.shape[1] != len(self.feature_names):
            raise ValueError(
                f"Expected {len(self.feature_names)} features, received {array.shape[1]}."
            )
        normalized = (array - self.means) / self.scales
        logits = normalized @ self.weights.T + self.bias
        return self._softmax(logits)

    def predict_label(self, X: Sequence[float] | Sequence[Sequence[float]]) -> str | list[str]:
        probs = self.predict_proba(X)
        labels = [self.action_names[int(np.argmax(row))] for row in probs]
        return labels[0] if len(labels) == 1 else labels

    def logits(self, X: Sequence[float] | Sequence[Sequence[float]]) -> np.ndarray:
        array = np.asarray(X, dtype=np.float64)
        if array.ndim == 1:
            array = array.reshape(1, -1)
        normalized = (array - self.means) / self.scales
        return normalized @ self.weights.T + self.bias

    def _loss(self, Xn: np.ndarray, y: np.ndarray, sample_weight: np.ndarray, l2: float) -> float:
        logits = Xn @ self.weights.T + self.bias
        probs = self._softmax(logits)
        idx = np.arange(len(y))
        log_prob = -np.log(np.clip(probs[idx, y], 1e-12, 1.0))
        weighted = float(np.average(log_prob, weights=sample_weight))
        reg = 0.5 * l2 * float(np.sum(self.weights**2))
        return weighted + reg

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        exp = np.exp(shifted)
        denom = np.clip(exp.sum(axis=1, keepdims=True), 1e-12, None)
        return exp / denom

    def _row_to_vector(self, row: dict[str, Any]) -> np.ndarray:
        if isinstance(row.get("feature_vector"), list) and row["feature_vector"]:
            vector = np.asarray(row["feature_vector"], dtype=np.float64)
            if vector.shape[0] == len(self.feature_names):
                return vector
        features = row.get("features", {})
        if not isinstance(features, dict):
            features = {}
        return np.asarray([float(features.get(name, 0.0) or 0.0) for name in self.feature_names], dtype=np.float64)


class LearnedEvidenceAwareResearchPolicy(EvidenceAwareResearchPolicy):
    """Drop-in evidence policy that reranks actions with a learned linear model."""

    ACTION_SPACE = EvidenceTransitionDatasetBuilder.ACTION_SPACE
    FEATURE_NAMES = EvidenceTransitionDatasetBuilder.FEATURE_NAMES

    def __init__(
        self,
        stop_threshold: float = 0.32,
        search_threshold: float = 0.22,
        expand_threshold: float = 0.45,
        *,
        model_path: str | Path | None = None,
        blend_weight: float = 0.7,
        fallback_policy: EvidenceAwareResearchPolicy | None = None,
    ) -> None:
        super().__init__(
            stop_threshold=stop_threshold,
            search_threshold=search_threshold,
            expand_threshold=expand_threshold,
        )
        self.model_path = Path(model_path) if model_path else None
        self.blend_weight = float(np.clip(blend_weight, 0.0, 1.0))
        self.fallback_policy = fallback_policy or EvidenceAwareResearchPolicy(
            stop_threshold=stop_threshold,
            search_threshold=search_threshold,
            expand_threshold=expand_threshold,
        )
        self.model: LinearActionRanker | None = None
        if self.model_path and self.model_path.exists():
            self.model = LinearActionRanker.load(self.model_path)

    @property
    def is_trained(self) -> bool:
        return bool(self.model and self.model.is_fitted)

    def fit(
        self,
        rows: Sequence[dict[str, Any]],
        *,
        epochs: int = 240,
        learning_rate: float = 0.12,
        l2: float = 1e-4,
        seed: int = 7,
        save_path: str | Path | None = None,
    ) -> dict[str, Any]:
        self.model = LinearActionRanker(self.FEATURE_NAMES, self.ACTION_SPACE)
        stats = self.model.fit(
            rows,
            epochs=epochs,
            learning_rate=learning_rate,
            l2=l2,
            seed=seed,
        )
        target_path = Path(save_path) if save_path else self.model_path
        if target_path is not None:
            self.save(target_path)
        return stats

    def save(self, path: str | Path | None = None) -> str:
        if not self.model or not self.model.is_fitted:
            raise ValueError("Cannot save an untrained learned evidence policy.")
        target = Path(path) if path else self.model_path
        if target is None:
            raise ValueError("A save path is required when no model_path was configured.")
        self.model_path = target
        return self.model.save(target)

    @classmethod
    def load(cls, path: str | Path, **kwargs: Any) -> "LearnedEvidenceAwareResearchPolicy":
        policy = cls(model_path=path, **kwargs)
        policy.model = LinearActionRanker.load(path)
        return policy

    def predict_label(self, query: str, nodes: Sequence[Any] | None = None, conflicts: Sequence[Any] | None = None) -> str:
        snapshot = self.analyze(query, nodes=nodes, conflicts=conflicts)
        return snapshot.recommended_action

    def predict_rows(self, rows: Sequence[dict[str, Any]]) -> list[str]:
        if not self.is_trained:
            return []
        vectors = [self._row_to_vector(row) for row in rows]
        return [self.ACTION_SPACE[idx] for idx in self.model.predict(vectors)]

    def _row_to_vector(self, row: dict[str, Any]) -> np.ndarray:
        if not self.model:
            raise ValueError("Model has not been initialized.")
        return self.model._row_to_vector(row)

    @classmethod
    def snapshot_summary(cls, snapshot: EvidenceSnapshot) -> dict[str, Any]:
        trust_scores = [float(item.trust_score) for item in snapshot.source_trusts]
        claim_ids = {item.claim_id for item in snapshot.claims}
        supported_claims = {item.claim_id for item in snapshot.support_edges if item.claim_id in claim_ids}
        contradicted_claims: set[str] = set()
        contradiction_severity = 0.0
        for edge in snapshot.contradiction_edges:
            contradicted_claims.update([edge.claim_id_1, edge.claim_id_2])
            contradiction_severity += max(0.0, min(1.0, float(edge.severity)))

        claim_count = len(snapshot.claims)
        claim_support_coverage = len(supported_claims) / max(claim_count, 1)
        contradicted_claim_ratio = len(contradicted_claims & claim_ids) / max(claim_count, 1)
        contradiction_burden = min(1.0, contradiction_severity / max(claim_count, 1))
        consistency_score = max(
            0.0,
            1.0 - min(1.0, 0.55 * contradiction_burden + 0.45 * contradicted_claim_ratio),
        )

        summary = {
            "recommended_action": snapshot.recommended_action,
            "uncertainty": round(float(snapshot.uncertainty), 4),
            "coverage_ratio": round(float(snapshot.coverage_ratio), 4),
            "evidence_strength": round(float(snapshot.evidence_strength), 4),
            "claim_count": claim_count,
            "support_edge_count": len(snapshot.support_edges),
            "contradiction_edge_count": len(snapshot.contradiction_edges),
            "open_conflicts": len(snapshot.conflicts),
            "avg_source_trust": round(sum(trust_scores) / len(trust_scores), 4) if trust_scores else 0.0,
            "missing_terms": list(snapshot.missing_terms[:6]),
            "candidate_actions": [
                {"name": item.name, "score": round(float(item.score), 4)}
                for item in snapshot.candidate_actions[:3]
            ],
            "claim_support_coverage": round(claim_support_coverage, 4),
            "consistency_score": round(consistency_score, 4),
        }
        summary["evidence_graph_quality"] = TrajectoryQualityProxy.quality(summary)
        return summary

    @classmethod
    def snapshot_features(cls, snapshot: EvidenceSnapshot) -> dict[str, float]:
        return EvidenceTransitionDatasetBuilder._snapshot_to_features(cls.snapshot_summary(snapshot))

    @classmethod
    def snapshot_feature_vector(cls, snapshot: EvidenceSnapshot) -> list[float]:
        features = cls.snapshot_features(snapshot)
        return [float(features[name]) for name in cls.FEATURE_NAMES]

    def analyze(
        self,
        query: str,
        nodes: Sequence[Any] | None = None,
        conflicts: Sequence[Any] | None = None,
    ) -> EvidenceSnapshot:
        snapshot = self.fallback_policy.analyze(query, nodes=nodes, conflicts=conflicts)
        if not self.is_trained:
            return snapshot

        feature_vector = np.asarray(self.snapshot_feature_vector(snapshot), dtype=np.float64)
        learned_probs = self.model.predict_proba(feature_vector)[0]

        heuristic_map = {action.name: action for action in snapshot.candidate_actions}
        blended_actions: list[CandidateAction] = []
        for idx, action_name in enumerate(self.ACTION_SPACE):
            learned_score = float(learned_probs[idx])
            heuristic_action = heuristic_map.get(action_name)
            heuristic_score = float(heuristic_action.score if heuristic_action else 0.0)
            heuristic_rationale = heuristic_action.rationale if heuristic_action else "No heuristic rationale available."
            final_score = (
                self.blend_weight * learned_score
                + (1.0 - self.blend_weight) * heuristic_score
            )
            blended_actions.append(
                CandidateAction(
                    name=action_name,
                    score=round(final_score, 4),
                    rationale=(
                        f"Learned policy p={learned_score:.2f}, heuristic={heuristic_score:.2f}. "
                        f"{heuristic_rationale}"
                    ),
                )
            )

        blended_actions.sort(key=lambda item: item.score, reverse=True)
        top_action = blended_actions[0]
        return replace(
            snapshot,
            candidate_actions=blended_actions,
            recommended_action=top_action.name,
            rationale=top_action.rationale,
        )

    def export_model(self) -> dict[str, Any]:
        if not self.model:
            raise ValueError("Model has not been trained yet.")
        return self.model.to_dict()


class TrajectoryQualityProxy:
    """Shared proxy to keep online features aligned with exported transition rows."""

    @staticmethod
    def quality(state: dict[str, Any]) -> float:
        claim_count = max(float(state.get("claim_count", 0.0) or 0.0), 1.0)
        support_ratio = min(
            1.0,
            float(state.get("support_edge_count", 0.0) or 0.0) / claim_count,
        )
        contradiction_ratio = min(
            1.0,
            float(state.get("open_conflicts", 0.0) or 0.0) / claim_count,
        )
        trust = max(0.0, min(1.0, float(state.get("avg_source_trust", 0.0) or 0.0)))
        coverage = max(0.0, min(1.0, float(state.get("coverage_ratio", 0.0) or 0.0)))
        strength = max(0.0, min(1.0, float(state.get("evidence_strength", 0.0) or 0.0)))
        claim_support_coverage = max(
            0.0,
            min(1.0, float(state.get("claim_support_coverage", 0.0) or 0.0)),
        )
        consistency_score = max(
            0.0,
            min(1.0, float(state.get("consistency_score", 0.0) or 0.0)),
        )
        return round(
            max(
                0.0,
                min(
                    1.0,
                    0.20 * support_ratio
                    + 0.18 * trust
                    + 0.18 * strength
                    + 0.14 * coverage
                    + 0.15 * claim_support_coverage
                    + 0.15 * consistency_score
                    - 0.10 * contradiction_ratio,
                ),
            ),
            4,
        )
