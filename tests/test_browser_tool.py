from __future__ import annotations

import asyncio
import gzip
import zlib

from src.tools.browser import BrowserTool


class _PdfFetchBrowser(BrowserTool):
    async def _fetch(self, url: str) -> dict[str, object]:
        return {
            "url": url,
            "content_type": "application/pdf",
            "bytes": b"%PDF-1.7",
            "text": "ignored",
        }

    def _extract_pdf_text(self, raw_bytes: bytes, *, source_url: str) -> str:
        assert raw_bytes.startswith(b"%PDF")
        assert source_url.endswith(".pdf")
        return "[PDF: mock.pdf]\n\nImportant policy text."


class _HtmlFetchBrowser(BrowserTool):
    async def _fetch(self, url: str) -> dict[str, object]:
        return {
            "url": url,
            "content_type": "text/html; charset=utf-8",
            "bytes": b"<html></html>",
            "text": "<html><body><main><p>Primary content paragraph.</p></main></body></html>",
        }


def test_browser_tool_prefers_pdf_extraction_for_pdf_urls() -> None:
    tool = _PdfFetchBrowser()

    content = asyncio.run(tool.execute("https://example.com/policy.pdf"))

    assert "Important policy text." in content


def test_browser_tool_extracts_html_text_for_web_pages() -> None:
    tool = _HtmlFetchBrowser()

    content = asyncio.run(tool.execute("https://example.com/article"))

    assert "Primary content paragraph." in content


def test_browser_tool_detects_pdf_by_url_or_content_type() -> None:
    assert BrowserTool._looks_like_pdf("https://example.com/doc.pdf", "text/html") is True
    assert BrowserTool._looks_like_pdf("https://example.com/doc", "application/pdf") is True
    assert BrowserTool._looks_like_pdf("https://example.com/doc", "text/html") is False


def test_browser_tool_decodes_compressed_payloads() -> None:
    plain = b"Primary evidence text."

    import brotli

    assert BrowserTool._decode_content_bytes(gzip.compress(plain), content_encoding="gzip") == plain
    assert BrowserTool._decode_content_bytes(zlib.compress(plain), content_encoding="deflate") == plain
    assert BrowserTool._decode_content_bytes(brotli.compress(plain), content_encoding="br") == plain
