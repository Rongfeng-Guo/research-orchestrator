from __future__ import annotations

from typing import TYPE_CHECKING

from .dataset import SearchPolicyDatasetBuilder

if TYPE_CHECKING:
    from .runtime_policy import LearnedSearchPolicy
    from .train_policy import LinearSearchPolicyModel, SearchPolicyTrainer

__all__ = [
    "LearnedSearchPolicy",
    "LinearSearchPolicyModel",
    "SearchPolicyDatasetBuilder",
    "SearchPolicyTrainer",
]


def __getattr__(name: str):
    if name == "LearnedSearchPolicy":
        from .runtime_policy import LearnedSearchPolicy

        return LearnedSearchPolicy
    if name in {"LinearSearchPolicyModel", "SearchPolicyTrainer"}:
        from .train_policy import LinearSearchPolicyModel, SearchPolicyTrainer

        exports = {
            "LinearSearchPolicyModel": LinearSearchPolicyModel,
            "SearchPolicyTrainer": SearchPolicyTrainer,
        }
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
