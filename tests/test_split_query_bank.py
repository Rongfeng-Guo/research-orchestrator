from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.split_query_bank import load_query_bank, render_markdown, split_queries  # noqa: E402


def _query(idx: int, domain: str) -> dict:
    return {
        "id": f"{domain}_{idx:03d}",
        "domain": domain,
        "query": f"{domain} query {idx}",
    }


def test_split_queries_creates_domain_balanced_heldout() -> None:
    queries = [
        *[_query(idx, "tech") for idx in range(1, 9)],
        *[_query(idx, "finance") for idx in range(1, 9)],
        *[_query(idx, "medical") for idx in range(1, 9)],
        *[_query(idx, "energy") for idx in range(1, 9)],
    ]

    split = split_queries(queries, heldout_ratio=0.25, min_heldout_size=8)

    summary = split["summary"]
    assert summary["train"]["num_queries"] == 24
    assert summary["heldout"]["num_queries"] == 8
    assert set(summary["heldout"]["domain_distribution"].values()) == {2}
    assert set(summary["train"]["query_ids"]).isdisjoint(summary["heldout"]["query_ids"])


def test_load_query_bank_accepts_jsonl_and_preserves_metadata(tmp_path: Path) -> None:
    path = tmp_path / "queries.jsonl"
    path.write_text(
        json.dumps({"id": "q1", "query": "hello", "domain": "tech", "expected_topics": ["a"]}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    queries = load_query_bank(path)

    assert queries[0]["id"] == "q1"
    assert queries[0]["expected_topics"] == ["a"]


def test_render_markdown_contains_domain_table() -> None:
    split = split_queries([_query(idx, "tech") for idx in range(1, 12)], min_heldout_size=3)

    md = render_markdown(split["summary"])

    assert "# Query Bank Split" in md
    assert "## Domain Distribution" in md


def test_split_query_bank_cli_writes_outputs(tmp_path: Path) -> None:
    input_path = tmp_path / "queries.jsonl"
    train_output = tmp_path / "queries" / "train.jsonl"
    heldout_output = tmp_path / "queries" / "heldout.jsonl"
    summary_json = tmp_path / "summary" / "split.json"
    summary_md = tmp_path / "summary" / "split.md"
    rows = [_query(idx, "tech") for idx in range(1, 17)]
    input_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "split_query_bank.py"),
            "--queries-file",
            str(input_path),
            "--train-output-file",
            str(train_output),
            "--heldout-output-file",
            str(heldout_output),
            "--summary-json",
            str(summary_json),
            "--summary-md",
            str(summary_md),
            "--prefix",
            "real_test",
            "--min-heldout-size",
            "4",
        ],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(result.stdout)
    assert payload["artifacts"]["train_queries"] == str(train_output)
    assert payload["artifacts"]["heldout_queries"] == str(heldout_output)
    assert payload["artifacts"]["summary_json"] == str(summary_json)
    assert payload["artifacts"]["summary_md"] == str(summary_md)
    assert train_output.exists()
    assert heldout_output.exists()
    assert summary_json.exists()
    assert summary_md.exists()
