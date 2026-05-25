from __future__ import annotations

import asyncio

from src.tools.web_search import WebSearchTool


def test_parse_duckduckgo_html_extracts_real_target_urls() -> None:
    html = """
    <div class="result results_links">
      <h2>
        <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Farticle%3Fa%3D1">Example &amp; Report</a>
      </h2>
      <a class="result__snippet">A useful <b>summary</b> with evidence.</a>
    </div>
    <div class="result results_links">
      <h2>
        <a class="result__a" href="https://news.example.org/story">Second Result</a>
      </h2>
      <div class="result__snippet">Another source snippet.</div>
    </div>
    """

    results = WebSearchTool._parse_duckduckgo_html(html, top_n=5)

    assert results == [
        {
            "title": "Example & Report",
            "url": "https://example.com/article?a=1",
            "snippet": "A useful summary with evidence.",
        },
        {
            "title": "Second Result",
            "url": "https://news.example.org/story",
            "snippet": "Another source snippet.",
        },
    ]


def test_parse_bing_html_extracts_result_blocks() -> None:
    html = """
    <li class="b_algo">
      <h2><a href="https://docs.example.com/page">Docs &amp; Guide</a></h2>
      <div class="b_caption"><p>Official <strong>documentation</strong> snippet.</p></div>
    </li>
    <li class="b_algo">
      <h2><a href="https://news.example.org/story">News Story</a></h2>
      <div class="b_caption"><p>Reported source snippet.</p></div>
    </li>
    """

    results = WebSearchTool._parse_bing_html(html, top_n=5)

    assert results == [
        {
            "title": "Docs & Guide",
            "url": "https://docs.example.com/page",
            "snippet": "Official documentation snippet.",
        },
        {
            "title": "News Story",
            "url": "https://news.example.org/story",
            "snippet": "Reported source snippet.",
        },
    ]


def test_extract_seed_urls_from_query_strips_punctuation_and_deduplicates() -> None:
    query = (
        "Compare https://example.com/policy, https://example.com/policy), "
        "and https://official.test/page?x=1."
    )

    urls = WebSearchTool._extract_seed_urls_from_query(query)

    assert urls == [
        "https://example.com/policy",
        "https://official.test/page?x=1",
    ]


def test_query_for_backend_removes_seed_directive_and_urls() -> None:
    query = (
        "Compare the EU AI Act and China's AI rules. "
        "Prefer these primary sources first: https://eur-lex.europa.eu/eli/reg/2024/1689/oj/eng ; "
        "https://www.cac.gov.cn/2023-07/13/c_1690898327029107.htm . "
        "Explain transparency obligations."
    )

    cleaned = WebSearchTool._query_for_backend(query)

    assert "https://" not in cleaned
    assert "Prefer these primary sources first" not in cleaned
    assert cleaned == "the EU AI Act and China's AI rules. transparency obligations"


def test_merge_seeded_results_prefers_seeded_urls_and_reranks() -> None:
    seeded = [
        {
            "title": "Official Policy",
            "url": "https://official.test/policy",
            "snippet": "Primary source.",
            "source": "query_seed",
            "backend": "bing_html",
        }
    ]
    existing = [
        {
            "title": "Mirror Policy",
            "url": "https://official.test/policy",
            "snippet": "Duplicate source.",
            "rank": 1,
            "domain": "official.test",
            "snippet_length": 17,
            "source": "bing_html",
            "backend": "bing_html",
        },
        {
            "title": "Context Article",
            "url": "https://news.test/story",
            "snippet": "Secondary coverage.",
            "rank": 2,
            "domain": "news.test",
            "snippet_length": 19,
            "source": "bing_html",
            "backend": "bing_html",
        },
    ]

    merged = WebSearchTool._merge_seeded_results(seeded, existing, top_n=3)

    assert [item["url"] for item in merged] == [
        "https://official.test/policy",
        "https://news.test/story",
    ]
    assert [item["rank"] for item in merged] == [1, 2]
    assert merged[0]["source"] == "query_seed"


def test_apply_seeded_results_adds_seed_metadata() -> None:
    tool = WebSearchTool(backend="bing_html")

    async def fake_seeded_results(query: str, backend: str, top_n: int) -> list[dict[str, str]]:
        assert backend == "bing_html"
        return [
            {
                "title": "Official Source",
                "url": "https://official.test/source",
                "snippet": "Direct primary source.",
                "rank": 0,
                "domain": "official.test",
                "snippet_length": 22,
                "backend": backend,
                "source": "query_seed",
            }
        ]

    tool._seeded_results_from_query = fake_seeded_results  # type: ignore[method-assign]

    response = {
        "query": "Inspect https://official.test/source and summarize the rule.",
        "results": [
            {
                "title": "Secondary Coverage",
                "url": "https://news.test/coverage",
                "snippet": "Reported discussion.",
                "rank": 1,
                "domain": "news.test",
                "snippet_length": 20,
                "backend": "bing_html",
                "source": "bing_html",
            }
        ],
        "total": 1,
        "source": "bing_html",
        "backend": "bing_html",
        "cost": {"request_count": 1},
    }

    updated = asyncio.run(
        tool._apply_seeded_results(
            query=response["query"],
            response=response,
            top_n=3,
        )
    )

    assert [item["url"] for item in updated["results"]] == [
        "https://official.test/source",
        "https://news.test/coverage",
    ]
    assert updated["seeded_urls"] == ["https://official.test/source"]
    assert updated["cost"]["seeded_results"] == 1
