from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .dataset import SearchPolicyDatasetBuilder
from .train_policy import LinearSearchPolicyModel, SearchPolicyTrainer

__all__ = ["LearnedSearchPolicy"]


class LearnedSearchPolicy:
    """Runtime wrapper for the offline search/browser/stop baseline."""

    ACTION_SPACE = SearchPolicyDatasetBuilder.ACTION_SPACE
    FEATURE_NAMES = SearchPolicyDatasetBuilder.FEATURE_NAMES

    def __init__(
        self,
        *,
        model_path: str | Path | None = None,
        continue_threshold: float = 0.58,
        stop_threshold: float = 0.66,
        heuristic_fallback: bool = True,
    ) -> None:
        self.model_path = Path(model_path) if model_path else None
        self.continue_threshold = float(np.clip(continue_threshold, 0.0, 1.0))
        self.stop_threshold = float(np.clip(stop_threshold, 0.0, 1.0))
        self.heuristic_fallback = bool(heuristic_fallback)
        self.model: LinearSearchPolicyModel | None = None
        if self.model_path and self.model_path.exists():
            self.model = LinearSearchPolicyModel.load(self.model_path)

    @property
    def is_trained(self) -> bool:
        return bool(self.model and self.model.is_fitted)

    @property
    def is_available(self) -> bool:
        return self.is_trained or self.heuristic_fallback

    def new_state(self) -> dict[str, Any]:
        return SearchPolicyDatasetBuilder.new_state()

    def observe_action(self, state: dict[str, Any], action: dict[str, Any]) -> None:
        SearchPolicyDatasetBuilder.observe_action(state, action)

    def feature_vector_from_state(self, state: dict[str, Any]) -> list[float]:
        features = SearchPolicyDatasetBuilder.features_from_state(state)
        return [float(features[name]) for name in self.FEATURE_NAMES]

    def features_from_state(self, state: dict[str, Any]) -> dict[str, float]:
        return SearchPolicyDatasetBuilder.features_from_state(state)

    def predict_proba(self, state: dict[str, Any]) -> tuple[np.ndarray, str]:
        if self.is_trained and self.model is not None:
            vector = np.asarray(self.feature_vector_from_state(state), dtype=np.float64)
            probs = self.model.predict_proba(vector)[0]
            return probs, "learned"

        if not self.heuristic_fallback:
            return np.zeros(len(self.ACTION_SPACE), dtype=np.float64), "disabled"

        features = self.features_from_state(state)
        label = SearchPolicyTrainer.heuristic_label({"features": features})
        probs = np.zeros(len(self.ACTION_SPACE), dtype=np.float64)
        probs[self.ACTION_SPACE.index(label)] = 1.0
        return probs, "heuristic"

    def recommend(
        self,
        state: dict[str, Any],
        *,
        preferred_search_tool: str = "web_search",
    ) -> dict[str, Any]:
        features = self.features_from_state(state)
        probs, source = self.predict_proba(state)
        best_idx = int(np.argmax(probs)) if len(probs) else 0
        label = self.ACTION_SPACE[best_idx]
        confidence = float(probs[best_idx]) if len(probs) else 0.0
        browser_url = self._browser_url_from_state(state)

        can_continue = (
            label in {"search", "browser"}
            and confidence >= self.continue_threshold
            and (label != "browser" or bool(browser_url))
        )
        enforce_stop = label == "stop" and confidence >= self.stop_threshold

        return {
            "label": label,
            "confidence": round(confidence, 4),
            "source": source,
            "probabilities": {
                action_name: round(float(probs[idx]), 4)
                for idx, action_name in enumerate(self.ACTION_SPACE)
            },
            "features": features,
            "feature_vector": [float(features[name]) for name in self.FEATURE_NAMES],
            "preferred_search_tool": preferred_search_tool,
            "browser_url": browser_url,
            "can_continue": can_continue,
            "enforce_stop": enforce_stop,
            "guidance": self._build_guidance(
                label=label,
                confidence=confidence,
                source=source,
                preferred_search_tool=preferred_search_tool,
                browser_url=browser_url,
                features=features,
            ),
        }

    def _build_guidance(
        self,
        *,
        label: str,
        confidence: float,
        source: str,
        preferred_search_tool: str,
        browser_url: str,
        features: dict[str, float],
    ) -> str:
        prefix = f"[SEARCH POLICY] ({source}, p={confidence:.2f})"
        if label == "stop":
            return (
                f"{prefix} Stop now and write the final answer. "
                f"Current budget_ratio={features.get('budget_ratio', 0.0):.2f}, "
                f"tool_calls_so_far={features.get('tool_calls_so_far', 0.0):.0f}, "
                f"has_stop_signal={features.get('has_stop_signal', 0.0):.0f}. "
                "Do not call more tools unless the current evidence is empty."
            )
        if label == "browser" and browser_url:
            return (
                f"{prefix} Open the strongest retrieved page with the 'browser' tool now. "
                f"Use this URL: {browser_url}"
            )
        return (
            f"{prefix} Run one more focused search with the '{preferred_search_tool}' tool before answering. "
            f"recent_query_overlap={features.get('recent_query_overlap', 0.0):.2f}, "
            f"unique_query_count={features.get('unique_query_count', 0.0):.0f}."
        )

    @staticmethod
    def _browser_url_from_state(state: dict[str, Any]) -> str:
        urls = state.get("last_top_urls", [])
        if not isinstance(urls, list):
            return ""
        fallback_pdf = ""
        for url in urls:
            if not isinstance(url, str):
                continue
            normalized = url.strip()
            if not normalized:
                continue
            if normalized.lower().endswith(".pdf"):
                if not fallback_pdf:
                    fallback_pdf = normalized
                continue
            return normalized
        return fallback_pdf
