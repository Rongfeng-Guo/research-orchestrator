from __future__ import annotations

from src.tools.arxiv_reader import ArxivReaderTool


def test_openalex_headers_use_public_project_identity() -> None:
    tool = ArxivReaderTool(backend="openalex")
    tool.openalex_email = "maintainer@example.com"

    headers = tool._openalex_headers()

    assert headers == {
        "User-Agent": "research-orchestrator",
        "Accept-Encoding": "gzip, deflate",
        "mailto": "maintainer@example.com",
    }
