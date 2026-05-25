from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_eval import (  # noqa: E402
    _build_hotpotqa_closed_book_prompt,
    _build_hotpotqa_prompt,
    _create_hotpotqa_context_policy,
    _extract_hotpotqa_prediction,
    _extract_hotpotqa_supporting_facts,
    _format_hotpotqa_indexed_context,
    _postprocess_hotpotqa_prediction,
    _augment_hotpotqa_supporting_facts,
    _run_hotpotqa_context_agent,
)
from evaluation.benchmarks.hotpotqa import HotpotQABenchmark  # noqa: E402


def test_hotpotqa_prompt_includes_context_and_answer_contract() -> None:
    prompt = _build_hotpotqa_prompt(
        "Were Scott Derrickson and Ed Wood of the same nationality?",
        "Ed Wood was American.\nScott Derrickson is American.",
    )

    assert "Final answer:" in prompt
    assert "Use the exact wording" in prompt
    assert "Question: Were Scott Derrickson" in prompt
    assert "Ed Wood was American" in prompt


def test_hotpotqa_closed_book_prompt_has_no_context_section() -> None:
    prompt = _build_hotpotqa_closed_book_prompt("Who wrote Hamlet?")

    assert "Final answer:" in prompt
    assert "Question: Who wrote Hamlet?" in prompt
    assert "Context:" not in prompt


def test_format_hotpotqa_indexed_context_includes_sentence_ids() -> None:
    context = _format_hotpotqa_indexed_context([["Title A", ["First.", "Second."]]])

    assert "## Title A" in context
    assert "[0] First." in context
    assert "[1] Second." in context


def test_extract_hotpotqa_supporting_facts_from_notes() -> None:
    facts = _extract_hotpotqa_supporting_facts("Supporting facts: Scott Derrickson#0; Ed Wood#0.")

    assert facts == [["Scott Derrickson", 0], ["Ed Wood", 0]]


def test_augment_hotpotqa_supporting_facts_from_report_and_context() -> None:
    facts = _augment_hotpotqa_supporting_facts(
        [],
        report_text="Evidence notes mention Scott Derrickson and that Scott Derrickson is an American director.",
        prediction="American",
        raw_context=[["Scott Derrickson", ["Scott Derrickson is an American director."]]],
    )

    assert facts == [["Scott Derrickson", 0]]


def test_extract_hotpotqa_prediction_prefers_final_answer_line() -> None:
    report = """
# Report

Final answer: yes

Both people are described as American in the context.
"""

    assert _extract_hotpotqa_prediction(report) == "yes"


def test_postprocess_hotpotqa_prediction_shortens_common_answer_shapes() -> None:
    assert (
        _postprocess_hotpotqa_prediction(
            "Yes, the Galata Tower and Suleymaniye Mosque are both in Istanbul.",
            "Are the Galata Tower and Suleymaniye Mosque located in the same city?",
        )
        == "yes"
    )
    assert _postprocess_hotpotqa_prediction("three times", "How many times did the flight circle the earth?") == "three"
    assert _postprocess_hotpotqa_prediction("1,840", "How many students were enrolled?") == "1,840 students"
    assert _postprocess_hotpotqa_prediction("2", "How many Grammy awards were won?") == "two Grammy awards"
    assert _postprocess_hotpotqa_prediction("two", "How many Grammy awards were won?") == "two Grammy awards"
    assert _postprocess_hotpotqa_prediction("10", "How many episodes were in the season?") == "ten episodes"
    assert _postprocess_hotpotqa_prediction("over 1 million acres", "How many acres are in the park?") == "over 1 million"
    assert _postprocess_hotpotqa_prediction("Only Vladimir is from Russia.", "Are both people from Russia?") == "no"
    assert _postprocess_hotpotqa_prediction("Allied World War I fighter aircraft", "Where was the weapon found?") == "World War I fighter aircraft"
    assert _postprocess_hotpotqa_prediction("Crystal Palace Football Club Player of the Year", "What award does Crystal Palace present?") == "Player of the Year"
    assert (
        _postprocess_hotpotqa_prediction(
            "Brittany, Cornwall, Ireland, Isle of Man, Scotland, Wales",
            "In which six Western European territories have Celtic languages survived?",
        )
        == "Brittany, Cornwall, Ireland, Isle of Man, Scotland and Wales"
    )
    assert (
        _postprocess_hotpotqa_prediction(
            "Liu Yifei, Liu Ye, Yu Shaoqun, and Leon Lai",
            "Crystal Liu stars in Night Peacock with which three other actresses?",
        )
        == "Liu Ye, Yu Shaoqun and Leon Lai"
    )
    assert _postprocess_hotpotqa_prediction("October 25, 1931", "When was the designer born?") == "born October 25, 1931"
    assert _postprocess_hotpotqa_prediction("historic business district", "What type of district is it?") == "business district"
    assert _postprocess_hotpotqa_prediction("Tugurt", "What language is closely related?") == "The Tugurt language"
    assert _postprocess_hotpotqa_prediction("grass family", "Sporobolus and Zea are in the same what?") == "family"
    assert _postprocess_hotpotqa_prediction("filibuster and scathing rhetoric", "was a master of what?") == "filibuster"
    assert (
        _postprocess_hotpotqa_prediction(
            "Babiana has more species than Ceratophyllum.",
            "Which genus has more species Babiana or Ceratophyllum?",
        )
        == "Babiana"
    )


def test_create_hotpotqa_context_policy_uses_solver_backend_and_caps_tokens(monkeypatch) -> None:
    captured = {}

    class _Router:
        @staticmethod
        def create_backend(backend_name=None, **kwargs):
            captured["backend_name"] = backend_name
            captured["kwargs"] = kwargs
            return object()

    monkeypatch.setattr("scripts.run_eval.ModelRouter", _Router)

    policy = _create_hotpotqa_context_policy(
        {
            "model": {
                "backend": "openai",
                "backend_mapping": {"solver": "custom"},
                "backend_sampling": {
                    "openai": {"temperature": 0.7, "max_tokens": 1200},
                    "modules": {"solver": {"temperature": 0.1, "max_tokens": 900}},
                },
            }
        }
    )

    assert policy is not None
    assert captured["backend_name"] == "custom"
    assert captured["kwargs"]["temperature"] == 0.0
    assert captured["kwargs"]["max_tokens"] == 300


def test_create_hotpotqa_context_policy_allows_lower_cap(monkeypatch) -> None:
    captured = {}

    class _Router:
        @staticmethod
        def create_backend(backend_name=None, **kwargs):
            captured["kwargs"] = kwargs
            return object()

    monkeypatch.setattr("scripts.run_eval.ModelRouter", _Router)

    _create_hotpotqa_context_policy(
        {"model": {"backend": "openai", "backend_sampling": {"openai": {"max_tokens": 1200}}}},
        max_tokens=120,
    )

    assert captured["kwargs"]["max_tokens"] == 120


def test_hotpotqa_shuffle_seed_is_reproducible() -> None:
    bench = HotpotQABenchmark(use_mock=True)

    first = bench.get_samples(n=3, shuffle=True, seed=7)
    second = bench.get_samples(n=3, shuffle=True, seed=7)
    other = bench.get_samples(n=3, shuffle=True, seed=8)

    assert [item["query"] for item in first] == [item["query"] for item in second]
    assert [item["query"] for item in first] != [item["query"] for item in other]


def test_hotpotqa_supporting_and_joint_metrics() -> None:
    bench = HotpotQABenchmark(use_mock=True)
    metrics = bench.evaluate(
        [
            {
                "prediction": "yes",
                "gold": "yes",
                "supporting_facts": [["A", 0], ["B", 1]],
                "gold_supporting_facts": [["A", 0], ["B", 1]],
            },
            {
                "prediction": "wrong",
                "gold": "gold",
                "supporting_facts": [["A", 0]],
                "gold_supporting_facts": [["A", 0], ["B", 1]],
            },
        ],
        metrics=["em", "f1", "pass@1"],
    )

    assert metrics["sp_em"] == 0.5
    assert round(metrics["sp_f1"], 4) == 0.8333
    assert metrics["joint_em"] == 0.5


def test_run_hotpotqa_context_agent_extracts_answer_from_notes(monkeypatch) -> None:
    class _Policy:
        def __init__(self):
            self.calls = 0

        def __call__(self, _messages):
            self.calls += 1
            if self.calls == 1:
                return {"content": "Fact 1: Tivoli Gardens opened on 15 August 1843."}
            if self.calls == 2:
                return {"content": "Final answer: 15 August 1843"}
            return {"content": "Final answer: 15 August 1843"}

    policy = _Policy()
    monkeypatch.setattr("scripts.run_eval._create_hotpotqa_context_policy", lambda _config, max_tokens=300: policy)

    report = _run_hotpotqa_context_agent(
        "When did the park at which Tivolis Koncertsal is located open?",
        "Tivoli Gardens opened on 15 August 1843.",
        {},
    )

    assert report.startswith("Final answer: 15 August 1843")
    assert "Evidence notes:" in report
    assert "Draft answer:" in report
