#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evidence import LearnedEvidenceAwareResearchPolicy  # noqa: E402
from src.evolution.evidence_dataset import EvidenceTransitionDatasetBuilder  # noqa: E402


def _expand_inputs(inputs: list[str]) -> list[Path]:
    dataset_paths: list[Path] = []
    seen: set[Path] = set()

    for raw in inputs:
        path = Path(raw)
        candidates: list[Path]
        if path.is_dir():
            candidates = sorted(path.rglob("evidence_transition_dataset.jsonl"))
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


def _load_rows(dataset_paths: list[Path]) -> list[dict]:
    builder = EvidenceTransitionDatasetBuilder()
    rows: list[dict] = []
    for path in dataset_paths:
        rows.extend(builder.load_jsonl(path))
    return rows


def _print_distribution(title: str, values: list[str]) -> None:
    counts = Counter(values)
    print(title)
    for label in LearnedEvidenceAwareResearchPolicy.ACTION_SPACE:
        if counts[label]:
            print(f"  {label}: {counts[label]}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the learned evidence policy from exported transition datasets.",
    )
    parser.add_argument(
        "--input",
        nargs="+",
        default=["evolution_output"],
        help="JSONL file(s), round directory, or glob pattern(s). Default: evolution_output",
    )
    parser.add_argument(
        "--output-model",
        type=str,
        default="artifacts/evidence_policy.json",
        help="Path to save the trained policy JSON.",
    )
    parser.add_argument("--epochs", type=int, default=320, help="Number of training epochs.")
    parser.add_argument("--learning-rate", type=float, default=0.16, help="Learning rate.")
    parser.add_argument("--l2", type=float, default=1e-4, help="L2 regularization strength.")
    parser.add_argument(
        "--blend-weight",
        type=float,
        default=0.85,
        help="Blend weight used when the model reranks heuristic actions at inference time.",
    )
    args = parser.parse_args()

    dataset_paths = _expand_inputs(args.input)
    if not dataset_paths:
        raise SystemExit("No evidence_transition_dataset.jsonl files found for the provided input paths.")

    rows = _load_rows(dataset_paths)
    if not rows:
        raise SystemExit("The selected datasets are empty.")

    policy = LearnedEvidenceAwareResearchPolicy(
        model_path=args.output_model,
        blend_weight=args.blend_weight,
    )
    stats = policy.fit(
        rows,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        l2=args.l2,
    )
    predictions = policy.predict_rows(rows)

    print(f"Loaded {len(rows)} rows from {len(dataset_paths)} dataset file(s).")
    for path in dataset_paths:
        print(f"  - {path}")
    print(f"Saved model to {args.output_model}")
    print(f"Train accuracy: {stats['train_accuracy']:.4f}")
    print(f"Weighted accuracy: {stats['weighted_accuracy']:.4f}")
    _print_distribution("Label distribution:", [str(row.get("label_action", "") or "") for row in rows])
    _print_distribution("Prediction distribution:", predictions)


if __name__ == "__main__":
    main()
