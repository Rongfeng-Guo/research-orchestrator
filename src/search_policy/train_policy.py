#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.search_policy.dataset import SearchPolicyDatasetBuilder  # noqa: E402


__all__ = [
    "LinearSearchPolicyModel",
    "SearchPolicyTrainer",
]


class LinearSearchPolicyModel:
    """Tiny multinomial linear classifier for route/stop decisions."""

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
        self.means = np.asarray(
            means if means is not None else np.zeros(len(self.feature_names)),
            dtype=np.float64,
        )
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
    def from_dict(cls, data: dict[str, Any]) -> "LinearSearchPolicyModel":
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
    def load(cls, path: str | Path) -> "LinearSearchPolicyModel":
        with Path(path).open("r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    def fit(
        self,
        rows: Sequence[dict[str, Any]],
        *,
        epochs: int = 280,
        learning_rate: float = 0.14,
        l2: float = 1e-4,
        seed: int = 7,
    ) -> dict[str, Any]:
        if not rows:
            raise ValueError("No rows available for training.")

        action_index = {name: idx for idx, name in enumerate(self.action_names)}
        feature_vectors: list[np.ndarray] = []
        labels: list[int] = []
        sample_weights: list[float] = []
        label_counter: Counter[str] = Counter()

        for row in rows:
            label = str(row.get("label_action", "") or "")
            if label not in action_index:
                continue
            feature_vectors.append(self._row_to_vector(row))
            labels.append(action_index[label])
            reward = float(np.clip(float(row.get("reward", 0.0) or 0.0), -1.0, 1.0))
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
            "weighted_train_accuracy": round(weighted_accuracy, 4),
            "label_distribution": dict(label_counter),
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
            raise ValueError(f"Expected {len(self.feature_names)} features, received {array.shape[1]}.")
        normalized = (array - self.means) / self.scales
        logits = normalized @ self.weights.T + self.bias
        return self._softmax(logits)

    def predict_label(self, X: Sequence[float] | Sequence[Sequence[float]]) -> str | list[str]:
        probs = self.predict_proba(X)
        labels = [self.action_names[int(np.argmax(row))] for row in probs]
        return labels[0] if len(labels) == 1 else labels

    def _row_to_vector(self, row: dict[str, Any]) -> np.ndarray:
        vector = row.get("feature_vector", [])
        if isinstance(vector, list) and len(vector) == len(self.feature_names):
            return np.asarray(vector, dtype=np.float64)
        features = row.get("features", {})
        if not isinstance(features, dict):
            features = {}
        return np.asarray([float(features.get(name, 0.0) or 0.0) for name in self.feature_names], dtype=np.float64)

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


class SearchPolicyTrainer:
    """Train a lightweight offline search policy from cached trajectories."""

    def __init__(
        self,
        *,
        model_path: str | Path | None = None,
        feature_names: Sequence[str] | None = None,
        action_names: Sequence[str] | None = None,
    ) -> None:
        self.feature_names = list(feature_names or SearchPolicyDatasetBuilder.FEATURE_NAMES)
        self.action_names = list(action_names or SearchPolicyDatasetBuilder.ACTION_SPACE)
        self.model_path = Path(model_path) if model_path else None
        self.model: LinearSearchPolicyModel | None = None
        if self.model_path and self.model_path.exists():
            self.model = LinearSearchPolicyModel.load(self.model_path)

    @property
    def is_trained(self) -> bool:
        return bool(self.model and self.model.is_fitted)

    def fit(
        self,
        rows: Sequence[dict[str, Any]],
        *,
        epochs: int = 280,
        learning_rate: float = 0.14,
        l2: float = 1e-4,
        validation_ratio: float = 0.2,
        seed: int = 7,
        save_path: str | Path | None = None,
    ) -> dict[str, Any]:
        train_rows, eval_rows = self._split_rows(rows, validation_ratio=validation_ratio, seed=seed)
        self.model = LinearSearchPolicyModel(self.feature_names, self.action_names)
        stats = self.model.fit(
            train_rows,
            epochs=epochs,
            learning_rate=learning_rate,
            l2=l2,
            seed=seed,
        )

        train_metrics = self.evaluate(train_rows)
        eval_metrics = self.evaluate(eval_rows) if eval_rows else {}
        heuristic_train = self.evaluate_heuristic(train_rows)
        heuristic_eval = self.evaluate_heuristic(eval_rows) if eval_rows else {}

        stats.update(
            {
                "train_accuracy": train_metrics.get("accuracy", stats.get("train_accuracy", 0.0)),
                "weighted_train_accuracy": train_metrics.get(
                    "weighted_accuracy",
                    stats.get("weighted_train_accuracy", 0.0),
                ),
                "train_examples": len(train_rows),
                "eval_examples": len(eval_rows),
                "eval_accuracy": eval_metrics.get("accuracy", 0.0),
                "weighted_eval_accuracy": eval_metrics.get("weighted_accuracy", 0.0),
                "heuristic_train_accuracy": heuristic_train.get("accuracy", 0.0),
                "heuristic_weighted_train_accuracy": heuristic_train.get("weighted_accuracy", 0.0),
                "heuristic_eval_accuracy": heuristic_eval.get("accuracy", 0.0),
                "heuristic_weighted_eval_accuracy": heuristic_eval.get("weighted_accuracy", 0.0),
            }
        )
        self.model.training_stats = dict(stats)

        target_path = Path(save_path) if save_path else self.model_path
        if target_path is not None:
            self.save(target_path)
        return dict(stats)

    def evaluate(self, rows: Sequence[dict[str, Any]]) -> dict[str, float]:
        if not rows or not self.is_trained:
            return {"accuracy": 0.0, "weighted_accuracy": 0.0}
        predictions = self.predict_rows(rows)
        labels = [str(row.get("label_action", "") or "") for row in rows]
        weights = np.asarray([self._sample_weight(row) for row in rows], dtype=np.float64)
        correct = np.asarray([pred == label for pred, label in zip(predictions, labels)], dtype=np.float64)
        return {
            "accuracy": round(float(correct.mean()), 4),
            "weighted_accuracy": round(float(np.average(correct, weights=weights)), 4),
        }

    def evaluate_heuristic(self, rows: Sequence[dict[str, Any]]) -> dict[str, float]:
        if not rows:
            return {"accuracy": 0.0, "weighted_accuracy": 0.0}
        predictions = [self.heuristic_label(row) for row in rows]
        labels = [str(row.get("label_action", "") or "") for row in rows]
        weights = np.asarray([self._sample_weight(row) for row in rows], dtype=np.float64)
        correct = np.asarray([pred == label for pred, label in zip(predictions, labels)], dtype=np.float64)
        return {
            "accuracy": round(float(correct.mean()), 4),
            "weighted_accuracy": round(float(np.average(correct, weights=weights)), 4),
        }

    def predict_rows(self, rows: Sequence[dict[str, Any]]) -> list[str]:
        if not self.is_trained:
            return []
        vectors = [self._row_to_vector(row) for row in rows]
        indices = self.model.predict(vectors)
        return [self.action_names[int(idx)] for idx in indices]

    def save(self, path: str | Path | None = None) -> str:
        if not self.model or not self.model.is_fitted:
            raise ValueError("Cannot save an untrained search policy.")
        target = Path(path) if path else self.model_path
        if target is None:
            raise ValueError("A save path is required when no model_path was configured.")
        self.model_path = target
        return self.model.save(target)

    @classmethod
    def load(cls, path: str | Path) -> "SearchPolicyTrainer":
        trainer = cls(model_path=path)
        trainer.model = LinearSearchPolicyModel.load(path)
        return trainer

    def _row_to_vector(self, row: dict[str, Any]) -> np.ndarray:
        if not self.model:
            raise ValueError("Model has not been initialized.")
        return self.model._row_to_vector(row)

    @staticmethod
    def _sample_weight(row: dict[str, Any]) -> float:
        reward = float(np.clip(float(row.get("reward", 0.0) or 0.0), -1.0, 1.0))
        return float(np.clip(1.0 + reward, 0.25, 2.0))

    @staticmethod
    def _split_rows(
        rows: Sequence[dict[str, Any]],
        *,
        validation_ratio: float = 0.2,
        seed: int = 7,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        rows_list = [row for row in rows if isinstance(row, dict)]
        if len(rows_list) < 5 or validation_ratio <= 0.0:
            return rows_list, []

        validation_ratio = float(np.clip(validation_ratio, 0.0, 0.5))
        rng = np.random.default_rng(seed)
        indices = np.arange(len(rows_list))
        rng.shuffle(indices)
        split = max(1, int(round(len(rows_list) * validation_ratio)))
        eval_idx = set(indices[:split].tolist())

        train_rows: list[dict[str, Any]] = []
        eval_rows: list[dict[str, Any]] = []
        for idx, row in enumerate(rows_list):
            if idx in eval_idx:
                eval_rows.append(row)
            else:
                train_rows.append(row)
        if not train_rows:
            return rows_list, []
        return train_rows, eval_rows

    @staticmethod
    def heuristic_label(row: dict[str, Any]) -> str:
        features = row.get("features", {})
        if not isinstance(features, dict):
            features = {}

        tool_calls = float(features.get("tool_calls_so_far", 0.0) or 0.0)
        search_calls = float(features.get("search_calls_so_far", 0.0) or 0.0)
        browser_calls = float(features.get("browser_calls_so_far", 0.0) or 0.0)
        last_result_count = float(features.get("last_result_count", 0.0) or 0.0)
        budget_ratio = float(features.get("budget_ratio", 0.0) or 0.0)
        recent_query_overlap = float(features.get("recent_query_overlap", 0.0) or 0.0)
        has_stop_signal = float(features.get("has_stop_signal", 0.0) or 0.0)
        browser_available = float(features.get("browser_available", 0.0) or 0.0)
        last_action_was_search = float(features.get("last_action_was_search", 0.0) or 0.0)
        empty_result_ratio = float(features.get("empty_result_ratio", 0.0) or 0.0)

        if tool_calls <= 0.0:
            return "search"
        if has_stop_signal >= 0.5 or search_calls >= 2.0 or budget_ratio >= 0.85:
            return "stop"
        if last_action_was_search >= 0.5 and browser_available >= 0.5 and last_result_count > 0:
            return "browser"
        if empty_result_ratio >= 0.5 and search_calls >= 1.0:
            return "stop"
        if recent_query_overlap >= 0.8 and browser_calls >= 1.0:
            return "stop"
        return "search"


def _expand_inputs(inputs: list[str]) -> list[Path]:
    dataset_paths: list[Path] = []
    seen: set[Path] = set()

    for raw in inputs:
        path = Path(raw)
        candidates: list[Path]
        if path.is_dir():
            candidates = sorted(path.rglob("*.jsonl"))
        elif path.is_file():
            candidates = [path]
        else:
            candidates = sorted(PROJECT_ROOT.glob(raw))

        for candidate in candidates:
            resolved = candidate.resolve()
            if resolved in seen or not candidate.is_file():
                continue
            dataset_paths.append(candidate)
            seen.add(resolved)

    return dataset_paths


def _print_distribution(title: str, values: list[str]) -> None:
    counts = Counter(values)
    print(title)
    for label in SearchPolicyDatasetBuilder.ACTION_SPACE:
        if counts[label]:
            print(f"  {label}: {counts[label]}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a lightweight offline search policy from search-cache JSONL files.",
    )
    parser.add_argument(
        "--input",
        nargs="+",
        default=["data/search_cache"],
        help="JSONL file(s), directory, or glob pattern(s). Default: data/search_cache",
    )
    parser.add_argument(
        "--output-model",
        type=str,
        default="artifacts/search_policy.json",
        help="Path to save the trained search policy JSON.",
    )
    parser.add_argument("--epochs", type=int, default=280, help="Number of training epochs.")
    parser.add_argument("--learning-rate", type=float, default=0.14, help="Learning rate.")
    parser.add_argument("--l2", type=float, default=1e-4, help="L2 regularization strength.")
    parser.add_argument(
        "--validation-ratio",
        type=float,
        default=0.2,
        help="Fraction of rows reserved for evaluation. Use 0 to train on all rows.",
    )
    args = parser.parse_args()

    dataset_paths = _expand_inputs(args.input)
    if not dataset_paths:
        raise SystemExit("No JSONL files found for the provided input paths.")

    builder = SearchPolicyDatasetBuilder()
    records: list[dict[str, Any]] = []
    for path in dataset_paths:
        records.extend(builder.load_jsonl(path))

    rows = builder.build_rows_from_records(records)
    if not rows:
        raise SystemExit("No usable search-policy rows were found in the selected cache files.")

    trainer = SearchPolicyTrainer(model_path=args.output_model)
    stats = trainer.fit(
        rows,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        l2=args.l2,
        validation_ratio=args.validation_ratio,
    )

    print(f"Loaded {len(records)} records from {len(dataset_paths)} file(s).")
    for path in dataset_paths:
        print(f"  - {path}")
    print(f"Built {len(rows)} search-policy rows.")
    _print_distribution("Label distribution:", [str(row.get("label_action", "") or "") for row in rows])
    print(f"Saved model to {args.output_model}")
    print(f"Train accuracy: {stats.get('train_accuracy', 0.0):.4f}")
    print(f"Weighted train accuracy: {stats.get('weighted_train_accuracy', 0.0):.4f}")
    if stats.get("eval_examples", 0):
        print(f"Eval accuracy: {stats.get('eval_accuracy', 0.0):.4f}")
        print(f"Weighted eval accuracy: {stats.get('weighted_eval_accuracy', 0.0):.4f}")
    print(f"Heuristic train accuracy: {stats.get('heuristic_train_accuracy', 0.0):.4f}")
    if stats.get("eval_examples", 0):
        print(f"Heuristic eval accuracy: {stats.get('heuristic_eval_accuracy', 0.0):.4f}")


if __name__ == "__main__":
    main()
