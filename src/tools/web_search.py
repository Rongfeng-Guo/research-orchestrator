"""
网页搜索工具 — 支持多后端：SerpAPI / Bing / 博查AI / 秘塔AI

设计理由：
  通过 .env 中的 SEARCH_BACKEND 切换后端，零源码修改。

后端对比：
  - serpapi: 每月 100 次免费，结果最全（Google 数据），国内可访问
  - bing:    微软搜索 API，国内稳定，需 Azure 订阅 Key
  - bocha:   博查AI搜索，国内索引最全，面向 AI Agent 优化
  - metaso:  秘塔AI搜索，中文语义强，有 research 多轮模式
"""
from __future__ import annotations

import asyncio
import re
import json
import os
import random
import time
from abc import ABC, abstractmethod
from html import unescape
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

import aiohttp

from ..utils.env_config import get_env

__all__ = ["WebSearchTool", "MockWebSearchTool", "BaseWebSearchTool"]

_QUERY_URL_PATTERN = re.compile(r"https?://[^\s<>\"]+")


class BaseWebSearchTool(ABC):
    """网页搜索工具抽象基类。"""

    name: str = "web_search"
    description: str = (
        "Search the web for information. "
        "Supports SerpAPI / Bing / 博查AI(bocha) / 秘塔AI(metaso) / DuckDuckGo / Bing HTML backends. "
        "Input: {'query': str, 'top_n': int(optional, default=5)}. "
        "Output: list of {'title': str, 'url': str, 'snippet': str}."
    )

    @abstractmethod
    async def execute(self, query: str, top_n: int = 5) -> dict[str, Any]:
        """执行搜索并返回结果。"""
        pass

    @staticmethod
    def _extract_domain(url: str) -> str:
        if not url:
            return ""
        try:
            return urlparse(url).netloc.lower()
        except Exception:
            return ""

    def _normalize_results(self, results: list[dict[str, Any]], backend: str) -> list[dict[str, Any]]:
        normalized: list[dict[str, Any]] = []
        for idx, item in enumerate(results, start=1):
            title = str(item.get("title", "") or "")
            url = str(item.get("url", "") or "")
            snippet = str(item.get("snippet", "") or "")
            source = str(item.get("source", backend) or backend)
            normalized.append(
                {
                    "title": title,
                    "url": url,
                    "snippet": snippet,
                    "rank": idx,
                    "domain": self._extract_domain(url),
                    "snippet_length": len(snippet),
                    "backend": backend,
                    "source": source,
                }
            )
        return normalized

    @staticmethod
    def _classify_error(error_msg: str) -> str:
        text = (error_msg or "").lower()
        if any(k in text for k in ("http", "status", "错误", "error")):
            if any(k in text for k in ("401", "403", "unauthorized", "forbidden", "api key", "key")):
                return "auth_error"
            if any(k in text for k in ("429", "quota", "limit", "额度")):
                return "rate_limit"
            if any(k in text for k in ("500", "502", "503", "504")):
                return "backend_server_error"
        if any(k in text for k in ("timeout", "timed out", "网络", "connection", "connect", "dns")):
            return "network_error"
        return "unknown_error"

    def _build_response(
        self,
        query: str,
        backend: str,
        results: list[dict[str, Any]],
        top_n: int,
        started_at: float,
        error: str | None = None,
    ) -> dict[str, Any]:
        normalized_results = self._normalize_results(results[:top_n], backend)
        latency_ms = int((time.perf_counter() - started_at) * 1000)
        resp = {
            "query": query,
            "results": normalized_results,
            "total": len(normalized_results),
            "source": backend,
            "backend": backend,
            "cost": {
                "request_count": 1,
                "requested_top_n": top_n,
                "returned_results": len(normalized_results),
                "latency_ms": latency_ms,
            },
        }
        if error:
            resp["error"] = error
            resp["error_type"] = self._classify_error(error)
        return resp

    def get_openai_tool_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "搜索关键词"},
                        "top_n": {
                            "type": "integer",
                            "description": "返回结果数量",
                            "default": 5,
                        },
                    },
                    "required": ["query"],
                },
            },
        }


class MockWebSearchTool(BaseWebSearchTool):
    """Mock 搜索工具：用于无网络环境的测试和演示。"""

    def __init__(self, delay_ms: tuple[int, int] = (50, 200)) -> None:
        self.delay_ms = delay_ms

    async def execute(self, query: str, top_n: int = 5) -> dict[str, Any]:
        started_at = time.perf_counter()
        await asyncio.sleep(random.randint(*self.delay_ms) / 1000.0)

        query_lower = query.lower()
        mock_db: dict[str, list[dict]] = {
            "transformer": [
                {
                    "title": "Attention Is All You Need",
                    "url": "https://arxiv.org/abs/1706.03762",
                    "snippet": "We propose a new simple network architecture, the Transformer, based solely on attention mechanisms.",
                },
                {
                    "title": "BERT: Pre-training of Deep Bidirectional Transformers",
                    "url": "https://arxiv.org/abs/1810.04805",
                    "snippet": "BERT obtains new state-of-the-art results on eleven natural language processing tasks.",
                },
            ],
            "llm": [
                {
                    "title": "Large Language Models: A Survey",
                    "url": "https://arxiv.org/abs/2303.18223",
                    "snippet": "This survey reviews the recent advances in large language models, including pre-training, adaptation, and applications.",
                },
            ],
            "python": [
                {
                    "title": "Python Documentation",
                    "url": "https://docs.python.org/3/",
                    "snippet": "Official Python programming language documentation.",
                },
            ],
        }

        results: list[dict] = []
        for keyword, entries in mock_db.items():
            if keyword in query_lower:
                results.extend(entries)

        seen = set()
        unique = []
        for r in results:
            key = r["url"]
            if key not in seen:
                seen.add(key)
                unique.append(r)
        results = unique[:top_n]

        if not results:
            results = [
                {
                    "title": f"Mock result for '{query}'",
                    "url": "https://example.com/mock",
                    "snippet": "This is a mock search result for testing purposes.",
                }
            ]

        return self._build_response(
            query=query,
            backend="mock",
            results=results,
            top_n=top_n,
            started_at=started_at,
        )


class WebSearchTool(BaseWebSearchTool):
    """真实网页搜索工具：支持多个真实搜索后端。

    配置优先从 .env / .env.local 读取：
      - SEARCH_BACKEND: 后端选择，可选 "serpapi" | "bing" | "bocha" | "metaso" | "duckduckgo" | "bing_html"
      - SERPAPI_KEY / SERPAPI_ENDPOINT: SerpAPI 配置
      - BING_SEARCH_KEY / BING_SEARCH_ENDPOINT: Bing API 配置
    """

    _session: aiohttp.ClientSession | None = None

    def __init__(self, backend: str | None = None, api_key: str | None = None, api_endpoint: str | None = None) -> None:
        self.backend = (backend or get_env("SEARCH_BACKEND", "serpapi")).lower().strip()

        # SerpAPI 配置
        self.serpapi_key = api_key or get_env("SERPAPI_KEY")
        self.serpapi_endpoint = api_endpoint or get_env("SERPAPI_ENDPOINT", "https://serpapi.com/search")

        # Bing API 配置
        self.bing_key = api_key or get_env("BING_SEARCH_KEY")
        self.bing_endpoint = api_endpoint or get_env("BING_SEARCH_ENDPOINT", "https://api.bing.microsoft.com/v7.0/search")

        # 博查AI 配置
        self.bocha_key = api_key or get_env("BOCHA_API_KEY")
        self.bocha_endpoint = api_endpoint or get_env("BOCHA_API_ENDPOINT", "https://api.bochaai.com/v1/web-search")

        # 秘塔AI 配置
        self.metaso_key = api_key or get_env("METASO_API_KEY")
        self.metaso_endpoint = api_endpoint or get_env("METASO_API_ENDPOINT", "https://metaso.cn/api/open/search/v2")

        # DuckDuckGo HTML 搜索无需 API Key，用于小规模 real-evidence fallback。
        self.duckduckgo_endpoint = api_endpoint or get_env("DUCKDUCKGO_SEARCH_ENDPOINT", "https://html.duckduckgo.com/html/")

        # Bing HTML 搜索无需 API Key，用于公开网页 fallback；正式大规模实验仍建议 API 后端。
        self.bing_html_endpoint = api_endpoint or get_env("BING_HTML_SEARCH_ENDPOINT", "https://www.bing.com/search")

    def _get_session(self) -> aiohttp.ClientSession:
        """获取复用的 ClientSession，避免每次搜索新建连接。"""
        if WebSearchTool._session is None or WebSearchTool._session.closed:
            WebSearchTool._session = aiohttp.ClientSession(
                headers={"Accept-Encoding": "gzip, deflate"}
            )
        return WebSearchTool._session

    @classmethod
    async def close_session(cls) -> None:
        """关闭类级别的共享 session。应在程序退出前调用。"""
        if cls._session is not None and not cls._session.closed:
            await cls._session.close()
            cls._session = None

    def __del__(self):
        """析构时尝试关闭 session（同步环境回退）。"""
        if WebSearchTool._session is not None and not WebSearchTool._session.closed:
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self.close_session())
            except RuntimeError:
                # 无运行中的事件循环，忽略
                pass

    async def execute(self, query: str, top_n: int = 5) -> dict[str, Any]:
        backend_query = self._query_for_backend(query)
        if self.backend == "bing":
            response = await self._bing_execute(backend_query, top_n)
        elif self.backend == "bocha":
            response = await self._bocha_execute(backend_query, top_n)
        elif self.backend == "metaso":
            response = await self._metaso_execute(backend_query, top_n)
        elif self.backend in {"duckduckgo", "ddg"}:
            response = await self._duckduckgo_execute(backend_query, top_n)
        elif self.backend in {"bing_html", "binghtml"}:
            response = await self._bing_html_execute(backend_query, top_n)
        else:
            response = await self._serpapi_execute(backend_query, top_n)
        response["query"] = query
        if backend_query != query:
            response["search_query"] = backend_query
        return await self._apply_seeded_results(query, response, top_n)

    @staticmethod
    def _strip_html(value: str) -> str:
        text = re.sub(r"<[^>]+>", " ", value or "")
        text = unescape(text)
        return re.sub(r"\s+", " ", text).strip()

    @staticmethod
    def _clean_seed_url(url: str) -> str:
        return (url or "").rstrip(".,;:!?)]}\"'")

    @classmethod
    def _extract_seed_urls_from_query(cls, query: str) -> list[str]:
        seen: set[str] = set()
        urls: list[str] = []
        for match in _QUERY_URL_PATTERN.findall(query or ""):
            url = cls._clean_seed_url(match)
            if not url or url in seen:
                continue
            seen.add(url)
            urls.append(url)
        return urls

    @classmethod
    def _query_for_backend(cls, query: str) -> str:
        if not _QUERY_URL_PATTERN.search(query or ""):
            return query

        cleaned = _QUERY_URL_PATTERN.sub(" ", query or "")
        cleanup_patterns = [
            r"\bprefer\s+(?:these|the following)\s+[^.;:]{0,120}?sources?\s+first:?",
            r"\bprefer\s+[^.;:]{0,120}?first:?",
            r"\busing\s+(?:this|these)\s+[^.;:]{0,120}?first:?",
            r"\buse\s+(?:this|these)\s+[^.;:]{0,120}?first:?",
            r"\bconsult\s+(?:this|these)\s+[^.;:]{0,120}?first:?",
        ]
        for pattern in cleanup_patterns:
            cleaned = re.sub(pattern, " ", cleaned, flags=re.IGNORECASE)

        cleaned = re.sub(
            r"(^|[.?!]\s+)(summarize|compare|analyze|evaluate|assess|explain|review|investigate)\s+",
            r"\1",
            cleaned,
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(
            r"\band\s+compare\s+with\s+other\s+reporting\b",
            " ",
            cleaned,
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(r"\s*[;:]\s*", ". ", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned)
        cleaned = re.sub(r"(?:\s*\.\s*){2,}", ". ", cleaned)
        cleaned = cleaned.strip(" .,;")
        return cleaned or query

    @classmethod
    def _extract_title_and_snippet_from_html(cls, html_text: str) -> tuple[str, str]:
        title_match = re.search(
            r"<title[^>]*>(.*?)</title>",
            html_text or "",
            flags=re.IGNORECASE | re.DOTALL,
        )
        title = cls._strip_html(title_match.group(1)) if title_match else ""

        meta_match = re.search(
            r'<meta[^>]+name=["\']description["\'][^>]+content=["\'](.*?)["\']',
            html_text or "",
            flags=re.IGNORECASE | re.DOTALL,
        )
        if not meta_match:
            meta_match = re.search(
                r'<meta[^>]+property=["\']og:description["\'][^>]+content=["\'](.*?)["\']',
                html_text or "",
                flags=re.IGNORECASE | re.DOTALL,
            )
        snippet = cls._strip_html(meta_match.group(1)) if meta_match else ""

        if not snippet:
            paragraph_match = re.search(
                r"<p[^>]*>(.*?)</p>",
                html_text or "",
                flags=re.IGNORECASE | re.DOTALL,
            )
            if paragraph_match:
                snippet = cls._strip_html(paragraph_match.group(1))

        if not snippet:
            body_text = cls._strip_html(html_text or "")
            snippet = body_text[:320].strip()

        return title[:200], snippet[:500]

    async def _fetch_seed_result(self, url: str, backend: str) -> dict[str, Any] | None:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0 Safari/537.36"
            )
        }
        try:
            session = self._get_session()
            async with session.get(
                url,
                headers=headers,
                allow_redirects=True,
                timeout=aiohttp.ClientTimeout(total=15),
            ) as resp:
                if resp.status != 200:
                    return None
                content_type = (resp.headers.get("Content-Type") or "").lower()
                resolved_url = str(resp.url)
                if "html" not in content_type and "xml" not in content_type:
                    path = urlparse(resolved_url).path.rstrip("/").split("/")[-1]
                    title = path or self._extract_domain(resolved_url) or "Seeded source"
                    snippet = f"Direct source seeded from query ({content_type or 'non-html content'})."
                    return {
                        "title": title[:200],
                        "url": resolved_url,
                        "snippet": snippet[:500],
                        "rank": 0,
                        "domain": self._extract_domain(resolved_url),
                        "snippet_length": len(snippet[:500]),
                        "backend": backend,
                        "source": "query_seed",
                    }
                text = await resp.text(errors="ignore")
        except Exception:
            return None

        title, snippet = self._extract_title_and_snippet_from_html(text)
        if not title and not snippet:
            return None
        return {
            "title": title or self._extract_domain(resolved_url),
            "url": resolved_url,
            "snippet": snippet,
            "rank": 0,
            "domain": self._extract_domain(resolved_url),
            "snippet_length": len(snippet),
            "backend": backend,
            "source": "query_seed",
        }

    async def _seeded_results_from_query(self, query: str, backend: str, top_n: int) -> list[dict[str, Any]]:
        seed_urls = self._extract_seed_urls_from_query(query)[:top_n]
        if not seed_urls:
            return []

        fetched = await asyncio.gather(
            *(self._fetch_seed_result(url, backend) for url in seed_urls),
            return_exceptions=True,
        )
        results: list[dict[str, Any]] = []
        for item in fetched:
            if isinstance(item, Exception) or item is None:
                continue
            results.append(item)
        return results

    @staticmethod
    def _merge_seeded_results(
        seeded_results: list[dict[str, Any]],
        existing_results: list[dict[str, Any]],
        top_n: int,
    ) -> list[dict[str, Any]]:
        merged: list[dict[str, Any]] = []
        seen_urls: set[str] = set()

        for item in list(seeded_results) + list(existing_results):
            url = str(item.get("url", "") or "")
            key = url.lower()
            if key and key in seen_urls:
                continue
            if key:
                seen_urls.add(key)
            merged.append(dict(item))
            if len(merged) >= top_n:
                break

        for idx, item in enumerate(merged, start=1):
            item["rank"] = idx
            item["domain"] = item.get("domain") or BaseWebSearchTool._extract_domain(str(item.get("url", "") or ""))
            item["snippet_length"] = len(str(item.get("snippet", "") or ""))
        return merged

    async def _apply_seeded_results(
        self,
        query: str,
        response: dict[str, Any],
        top_n: int,
    ) -> dict[str, Any]:
        seed_urls = self._extract_seed_urls_from_query(query)
        if not seed_urls:
            return response

        backend = str(response.get("backend", self.backend) or self.backend)
        seeded_results = await self._seeded_results_from_query(query, backend, top_n)
        if not seeded_results:
            return response

        merged_results = self._merge_seeded_results(
            seeded_results=seeded_results,
            existing_results=list(response.get("results", [])),
            top_n=top_n,
        )
        updated_response = dict(response)
        updated_response["results"] = merged_results
        updated_response["total"] = len(merged_results)
        updated_response["seeded_urls"] = seed_urls

        cost = dict(updated_response.get("cost", {}))
        cost["seed_url_count"] = len(seed_urls)
        cost["seeded_results"] = len(seeded_results)
        updated_response["cost"] = cost
        return updated_response

    @staticmethod
    def _normalize_duckduckgo_url(raw_href: str) -> str:
        href = unescape(raw_href or "").strip()
        if href.startswith("//"):
            href = "https:" + href
        if href.startswith("/"):
            href = "https://duckduckgo.com" + href
        parsed = urlparse(href)
        if "duckduckgo.com" in parsed.netloc and parsed.path.startswith("/l/"):
            target = parse_qs(parsed.query).get("uddg", [""])[0]
            if target:
                return unquote(target)
        return href

    @classmethod
    def _parse_duckduckgo_html(cls, html_text: str, top_n: int) -> list[dict[str, str]]:
        anchors = list(
            re.finditer(
                r'<a[^>]+class="[^"]*\bresult__a\b[^"]*"[^>]+href="([^"]+)"[^>]*>(.*?)</a>',
                html_text or "",
                flags=re.IGNORECASE | re.DOTALL,
            )
        )
        results: list[dict[str, str]] = []
        seen_urls: set[str] = set()
        for idx, match in enumerate(anchors):
            url = cls._normalize_duckduckgo_url(match.group(1))
            title = cls._strip_html(match.group(2))
            block_end = anchors[idx + 1].start() if idx + 1 < len(anchors) else len(html_text)
            block = html_text[match.end():block_end]
            snippet_match = re.search(
                r'<a[^>]+class="[^"]*\bresult__snippet\b[^"]*"[^>]*>(.*?)</a>',
                block,
                flags=re.IGNORECASE | re.DOTALL,
            )
            if not snippet_match:
                snippet_match = re.search(
                    r'<div[^>]+class="[^"]*\bresult__snippet\b[^"]*"[^>]*>(.*?)</div>',
                    block,
                    flags=re.IGNORECASE | re.DOTALL,
                )
            snippet = cls._strip_html(snippet_match.group(1)) if snippet_match else ""
            if not title or not url or url in seen_urls:
                continue
            seen_urls.add(url)
            results.append({"title": title, "url": url, "snippet": snippet})
            if len(results) >= top_n:
                break
        return results

    async def _duckduckgo_execute(self, query: str, top_n: int) -> dict[str, Any]:
        started_at = time.perf_counter()
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0 Safari/537.36"
            )
        }
        params = {"q": query}
        try:
            session = self._get_session()
            async with session.get(
                self.duckduckgo_endpoint,
                params=params,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=20),
            ) as resp:
                text = await resp.text()
                if resp.status != 200:
                    return self._build_response(
                        query=query,
                        backend="duckduckgo",
                        results=[],
                        top_n=top_n,
                        started_at=started_at,
                        error=f"DuckDuckGo HTML 搜索错误: HTTP {resp.status}",
                    )
        except Exception as e:
            return self._build_response(
                query=query,
                backend="duckduckgo",
                results=[],
                top_n=top_n,
                started_at=started_at,
                error=f"DuckDuckGo HTML 搜索网络错误: {e}",
            )

        results = self._parse_duckduckgo_html(text, top_n)
        if not results:
            return self._build_response(
                query=query,
                backend="duckduckgo",
                results=[],
                top_n=top_n,
                started_at=started_at,
                error="DuckDuckGo HTML 搜索未解析到结果",
            )
        return self._build_response(
            query=query,
            backend="duckduckgo",
            results=results,
            top_n=top_n,
            started_at=started_at,
        )

    @classmethod
    def _parse_bing_html(cls, html_text: str, top_n: int) -> list[dict[str, str]]:
        blocks = re.findall(
            r'<li[^>]+class="[^"]*\bb_algo\b[^"]*"[^>]*>(.*?)</li>',
            html_text or "",
            flags=re.IGNORECASE | re.DOTALL,
        )
        results: list[dict[str, str]] = []
        seen_urls: set[str] = set()
        for block in blocks:
            link_match = re.search(
                r"<h2[^>]*>.*?<a[^>]+href=\"([^\"]+)\"[^>]*>(.*?)</a>.*?</h2>",
                block,
                flags=re.IGNORECASE | re.DOTALL,
            )
            if not link_match:
                continue
            url = unescape(link_match.group(1)).strip()
            title = cls._strip_html(link_match.group(2))
            snippet_match = re.search(
                r"<p[^>]*>(.*?)</p>",
                block,
                flags=re.IGNORECASE | re.DOTALL,
            )
            snippet = cls._strip_html(snippet_match.group(1)) if snippet_match else ""
            if not title or not url or url in seen_urls:
                continue
            seen_urls.add(url)
            results.append({"title": title, "url": url, "snippet": snippet})
            if len(results) >= top_n:
                break
        return results

    async def _bing_html_execute(self, query: str, top_n: int) -> dict[str, Any]:
        started_at = time.perf_counter()
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0 Safari/537.36"
            ),
            "Accept-Encoding": "gzip, deflate",
            "Accept-Language": "en-US,en;q=0.9",
        }
        params = {"q": query, "count": str(top_n), "setlang": "en-US"}
        try:
            session = self._get_session()
            async with session.get(
                self.bing_html_endpoint,
                params=params,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=20),
            ) as resp:
                text = await resp.text(errors="ignore")
                if resp.status != 200:
                    return self._build_response(
                        query=query,
                        backend="bing_html",
                        results=[],
                        top_n=top_n,
                        started_at=started_at,
                        error=f"Bing HTML 搜索错误: HTTP {resp.status}",
                    )
        except Exception as e:
            return self._build_response(
                query=query,
                backend="bing_html",
                results=[],
                top_n=top_n,
                started_at=started_at,
                error=f"Bing HTML 搜索网络错误: {e}",
            )

        results = self._parse_bing_html(text, top_n)
        if not results:
            return self._build_response(
                query=query,
                backend="bing_html",
                results=[],
                top_n=top_n,
                started_at=started_at,
                error="Bing HTML 搜索未解析到结果",
            )
        return self._build_response(
            query=query,
            backend="bing_html",
            results=results,
            top_n=top_n,
            started_at=started_at,
        )

    async def _serpapi_execute(self, query: str, top_n: int) -> dict[str, Any]:
        started_at = time.perf_counter()
        if not self.serpapi_key:
            raise RuntimeError(
                "WebSearchTool (serpapi 后端) 需要 API Key。\n"
                "请在 .env 或 .env.local 中设置 SERPAPI_KEY，\n"
                "或构造函数传入: WebSearchTool(api_key='your_key')\n"
                "如需 Mock 模式，请显式使用 MockWebSearchTool()"
            )

        params = {
            "q": query,
            "num": top_n,
            "api_key": self.serpapi_key,
            "engine": "google",
            "gl": "us",
            "hl": "en",
        }

        try:
            session = self._get_session()
            async with session.get(
                self.serpapi_endpoint,
                params=params,
                timeout=aiohttp.ClientTimeout(total=20),
            ) as resp:
                    data = await resp.json()
                    if resp.status != 200:
                        error_msg = data.get("error", f"HTTP {resp.status}")
                        return {
                            **self._build_response(
                                query=query,
                                backend="serpapi",
                                results=[],
                                top_n=top_n,
                                started_at=started_at,
                            ),
                            "error": f"SerpAPI 错误: {error_msg}",
                            "error_type": self._classify_error(f"SerpAPI 错误: {error_msg}"),
                        }
        except Exception as e:
            return self._build_response(
                query=query,
                backend="serpapi",
                results=[],
                top_n=top_n,
                started_at=started_at,
                error=f"SerpAPI 网络错误: {e}",
            )

        # 解析 SerpAPI 响应
        organic = data.get("organic_results", [])
        results = []
        for item in organic[:top_n]:
            results.append({
                "title": item.get("title", ""),
                "url": item.get("link", ""),
                "snippet": item.get("snippet", ""),
            })

        return self._build_response(
            query=query,
            backend="serpapi",
            results=results,
            top_n=top_n,
            started_at=started_at,
        )

    async def _bing_execute(self, query: str, top_n: int) -> dict[str, Any]:
        started_at = time.perf_counter()
        if not self.bing_key:
            raise RuntimeError(
                "WebSearchTool (bing 后端) 需要 API Key。\n"
                "请在 .env 或 .env.local 中设置 BING_SEARCH_KEY，\n"
                "或在 Azure Portal 创建 Bing Search v7 资源获取 Key。\n"
                "如需 Mock 模式，请显式使用 MockWebSearchTool()"
            )

        headers = {"Ocp-Apim-Subscription-Key": self.bing_key}
        params = {"q": query, "count": top_n, "mkt": "en-US"}

        try:
            session = self._get_session()
            async with session.get(
                self.bing_endpoint,
                params=params,
                timeout=aiohttp.ClientTimeout(total=20),
            ) as resp:
                    data = await resp.json()
                    if resp.status != 200:
                        error_msg = data.get("message", f"HTTP {resp.status}")
                        return {
                            **self._build_response(
                                query=query,
                                backend="bing",
                                results=[],
                                top_n=top_n,
                                started_at=started_at,
                            ),
                            "error": f"Bing API 错误: {error_msg}",
                            "error_type": self._classify_error(f"Bing API 错误: {error_msg}"),
                        }
        except Exception as e:
            return self._build_response(
                query=query,
                backend="bing",
                results=[],
                top_n=top_n,
                started_at=started_at,
                error=f"Bing API 网络错误: {e}",
            )

        # 解析 Bing 响应
        web_pages = data.get("webPages", {}).get("value", [])
        results = []
        for item in web_pages[:top_n]:
            results.append({
                "title": item.get("name", ""),
                "url": item.get("url", ""),
                "snippet": item.get("snippet", ""),
            })

        return self._build_response(
            query=query,
            backend="bing",
            results=results,
            top_n=top_n,
            started_at=started_at,
        )

    async def _bocha_execute(self, query: str, top_n: int) -> dict[str, Any]:
        """博查AI搜索后端。

        文档: https://open.bochaai.com
        特点: 国内网页索引最全，面向 AI Agent 和 RAG 优化，返回结构化摘要。
        """
        started_at = time.perf_counter()
        if not self.bocha_key:
            raise RuntimeError(
                "WebSearchTool (bocha 后端) 需要 API Key。\n"
                "请在 .env 或 .env.local 中设置 BOCHA_API_KEY，\n"
                "或访问 https://open.bochaai.com 注册获取。\n"
                "如需 Mock 模式，请显式使用 MockWebSearchTool()"
            )

        payload = {
            "query": query,
            "summary": True,
            "freshness": "noLimit",
            "count": top_n,
        }
        headers = {
            "Authorization": f"Bearer {self.bocha_key}",
            "Content-Type": "application/json",
        }

        try:
            session = self._get_session()
            async with session.post(
                self.bocha_endpoint,
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=20),
            ) as resp:
                data = await resp.json()
                if resp.status != 200:
                    error_msg = data.get("message", f"HTTP {resp.status}")
                    return {
                        **self._build_response(
                            query=query,
                            backend="bocha",
                            results=[],
                            top_n=top_n,
                            started_at=started_at,
                        ),
                        "error": f"博查AI 错误: {error_msg}",
                        "error_type": self._classify_error(f"博查AI 错误: {error_msg}"),
                    }
        except Exception as e:
            return self._build_response(
                query=query,
                backend="bocha",
                results=[],
                top_n=top_n,
                started_at=started_at,
                error=f"博查AI 网络错误: {e}",
            )

        # 解析博查响应 — 兼容 web-search 和 ai-search 两种端点返回结构
        results: list[dict] = []

        # 结构 A: /v1/web-search → data.webPages.value[]
        web_pages = data.get("data", {}).get("webPages", {}).get("value", [])
        for item in web_pages[:top_n]:
            results.append({
                "title": item.get("name", ""),
                "url": item.get("url", ""),
                "snippet": item.get("snippet", ""),
            })

        # 结构 B: /v1/ai-search → data.messages[] content 里含引用
        if not results:
            messages = data.get("data", {}).get("messages", [])
            for msg in messages[:top_n]:
                content = msg.get("content", "")
                if content:
                    results.append({
                        "title": msg.get("role", "引用")[:30],
                        "url": "",
                        "snippet": content[:500],
                    })

        # 去重：同一篇文章的不同 URL（移动端/PC端/转发）会被当作多条结果
        results = self._deduplicate_results(results)

        return self._build_response(
            query=query,
            backend="bocha",
            results=results,
            top_n=top_n,
            started_at=started_at,
        )

    def _deduplicate_results(self, results: list[dict]) -> list[dict]:
        """对搜索结果去重：基于规范化 URL 和清洗后的标题。"""
        from urllib.parse import urlparse
        import re

        seen_keys: set[str] = set()
        unique: list[dict] = []

        for r in results:
            raw_url = r.get("url", "")
            raw_title = r.get("title", "").strip()

            # --- URL 规范化 ---
            try:
                parsed = urlparse(raw_url)
                netloc = parsed.netloc.lower()
                path = parsed.path.lower().rstrip("/")

                # 去掉移动端前缀
                for prefix in ("m.", "wap.", "mobile.", "app."):
                    if netloc.startswith(prefix):
                        netloc = netloc[len(prefix):]
                        break
                # 去掉 www 前缀
                if netloc.startswith("www."):
                    netloc = netloc[4:]

                # 对常见新闻/博客站，只保留域名+路径（去掉查询参数）
                normalized_url = f"{netloc}{path}"
            except Exception:
                normalized_url = raw_url.lower().strip()

            # --- 标题清洗 ---
            # 去掉常见来源后缀，如 " - 虎嗅网"、"_CSDN博客"、"| 人人都是产品经理"
            cleaned_title = re.sub(
                r"[_\-\s|]*(CSDN博客|虎嗅网|人人都是产品经理|36氪|知乎|搜狐|新浪|网易|腾讯|今日头条|飞书云文档|简书|豆瓣|百度文库|原创力文档|道客巴巴|豆丁网|MBA智库文档|外唐智库|未来智库|中研网|中商产业研究院|三个皮匠报告|book118\.com|doc88\.com|docin\.com|mbalib\.com|askci\.com|chinairn\.com|vzkoo\.com|waitang\.com|sgpjbg\.com|toutiao\.com|sohu\.com|sina\.com|163\.com|qq\.com|ifeng\.com|huxiu\.com|36kr\.com|woshipm\.com|csdn\.net|zhihu\.com|juejin\.cn|segmentfault\.com|cnblogs\.com|简书|知乎专栏|百家号|大鱼号|企鹅号|新浪看点|一点资讯|趣头条|东方财富|雪球|同花顺|财联社|华尔街见闻|界面新闻|澎湃|新京报|南方周末|财新|第一财经|经济观察网|21世纪经济报道|新浪财经|腾讯财经|网易财经|凤凰财经|和讯网|中金在线|东方财富网|中国证券报|上海证券报|证券时报|证券日报|每日经济新闻|第一财经日报|经济参考报|人民日报|新华社|央视新闻|中央广播电视总台|中国日报|环球时报|参考消息|瞭望|半月谈|求是|学习强国|新华网|人民网|中国网|国际在线|中国新闻网|环球网等?)",
                "",
                raw_title,
                flags=re.IGNORECASE,
            ).strip()
            # 再去掉末尾的 " - "、" | "、"_"
            cleaned_title = re.sub(r"[_\-\s|]+$", "", cleaned_title).strip()

            # --- 去重键：优先用 URL，URL 为空时用清洗后的标题 ---
            key = normalized_url if normalized_url else cleaned_title.lower()
            if not key:
                unique.append(r)
                continue

            if key in seen_keys:
                continue
            seen_keys.add(key)
            unique.append(r)

        return unique

    async def _metaso_execute(self, query: str, top_n: int) -> dict[str, Any]:
        """秘塔AI搜索后端。

        文档: https://metaso.cn/open
        特点: 中文语义搜索强，支持 detail / concise / research 模式。
        """
        started_at = time.perf_counter()
        if not self.metaso_key:
            raise RuntimeError(
                "WebSearchTool (metaso 后端) 需要 API Key。\n"
                "请在 .env 或 .env.local 中设置 METASO_API_KEY，\n"
                "或访问 https://metaso.cn/open 注册获取。\n"
                "如需 Mock 模式，请显式使用 MockWebSearchTool()"
            )

        payload = {
            "question": query,
            "lang": "zh",
            "stream": False,
        }
        headers = {
            "Authorization": f"Bearer {self.metaso_key}",
            "Content-Type": "application/json",
        }

        try:
            session = self._get_session()
            async with session.post(
                self.metaso_endpoint,
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                data = await resp.json()
                if resp.status != 200 or data.get("errCode"):
                    error_msg = data.get("errMsg", f"HTTP {resp.status}")
                    return {
                        **self._build_response(
                            query=query,
                            backend="metaso",
                            results=[],
                            top_n=top_n,
                            started_at=started_at,
                        ),
                        "error": f"秘塔AI 错误: {error_msg}",
                        "error_type": self._classify_error(f"秘塔AI 错误: {error_msg}"),
                    }
        except Exception as e:
            return self._build_response(
                query=query,
                backend="metaso",
                results=[],
                top_n=top_n,
                started_at=started_at,
                error=f"秘塔AI 网络错误: {e}",
            )

        # 解析秘塔响应
        results: list[dict] = []
        result_data = data.get("data", {})

        # 1. 优先把 text 字段（秘塔 AI 整理的完整答案）作为高价值结果
        text = result_data.get("text", "")
        if text:
            results.append({
                "title": "秘塔AI搜索总结",
                "url": "",
                "snippet": text[:1500],  # 给足上下文，让 LLM 能直接总结
            })

        # 2. 附加参考文献列表（用于溯源）
        refs = result_data.get("references", [])
        for item in refs[:top_n]:
            snippet_parts = []
            if item.get("title"):
                snippet_parts.append(item["title"])
            if item.get("article_type"):
                snippet_parts.append(f"类型: {item['article_type']}")
            if item.get("date"):
                snippet_parts.append(f"日期: {item['date']}")
            results.append({
                "title": item.get("title", ""),
                "url": item.get("link", ""),
                "snippet": " | ".join(snippet_parts)[:500],
            })

        return self._build_response(
            query=query,
            backend="metaso",
            results=results,
            top_n=top_n,
            started_at=started_at,
        )
