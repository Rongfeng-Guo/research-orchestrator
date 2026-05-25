"""
浏览器工具 (BrowserTool) — 网页全文阅读器

设计理由：
  web_search 只能返回搜索摘要（通常 100-200 字），而深度研究需要阅读网页原文。
  BrowserTool 负责：打开 URL → 提取正文 → 清理广告/导航栏 → 返回结构化文本。

与 web_search 的关系：
  web_search: "找到可能有信息的链接"
  browser: "读这个链接里的具体内容"
  两者是上下游，不是重复。

实现要点：
  - 使用 aiohttp 异步抓取，避免阻塞事件循环
  - 用 BeautifulSoup 提取正文（去除 script/style/nav 等噪声标签）
  - 自动截断超长页面（保留前 N 个段落，防止 token 爆炸）
  - 支持重试和错误降级
"""
from __future__ import annotations

import asyncio
import gzip
import io
import re
import zlib
from abc import ABC, abstractmethod
from typing import Any
from urllib.parse import urlparse

import aiohttp


__all__ = ["BrowserTool", "MockBrowserTool"]

# 默认保留的正文字数上限（防止超长网页占满上下文）
_DEFAULT_MAX_CHARS = 8000

# HTML 中通常包含正文的标签
_CONTENT_TAGS = ["article", "main", "section", "div"]
# 噪声标签（直接移除）
_NOISE_TAGS = ["script", "style", "nav", "header", "footer", "aside", "noscript", "iframe", "svg"]


class BaseBrowserTool(ABC):
    """浏览器工具基类。"""

    name: str = "browser"
    description: str = (
        "Open a URL and extract the main article text. "
        "Use this after web_search when you need to read the full content of a webpage. "
        "Input: {'url': str, 'max_chars': int(optional, default=8000)}. "
        "Output: extracted text content."
    )

    @abstractmethod
    async def execute(self, url: str, max_chars: int = _DEFAULT_MAX_CHARS) -> str:
        """打开 URL，提取正文。

        Args:
            url: 要访问的网页地址。
            max_chars: 返回的最大字符数，超出则截断。

        Returns:
            提取后的正文文本。
        """
        ...

    def get_openai_tool_schema(self) -> dict:
        """返回 OpenAI Function Calling 格式的 schema。"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "The URL of the webpage to open and read",
                        },
                        "max_chars": {
                            "type": "integer",
                            "description": "Maximum characters to return (default: 8000)",
                            "default": 8000,
                        },
                    },
                    "required": ["url"],
                },
            },
        }


class BrowserTool(BaseBrowserTool):
    """真实浏览器工具：异步 HTTP 抓取 + 正文提取。

    配置优先从 .env / .env.local 读取，构造函数参数仅作为覆盖。
    支持的环境变量：
      - BROWSER_TIMEOUT: HTTP 请求超时秒数（默认 15）
      - BROWSER_USER_AGENT: 自定义 User-Agent
    """

    def __init__(self, timeout: int | None = None, user_agent: str | None = None) -> None:
        from ..utils.env_config import get_env, get_env_int

        self.timeout = timeout or get_env_int("BROWSER_TIMEOUT", 15)
        self.user_agent = user_agent or get_env(
            "BROWSER_USER_AGENT",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        )
        self._request_headers = {
            "User-Agent": self.user_agent,
            # 避免部分站点返回 aiohttp 容易解码失败的 brotli 内容。
            "Accept-Encoding": "gzip, deflate",
        }

    async def execute(self, url: str, max_chars: int = _DEFAULT_MAX_CHARS) -> str:
        if not url.startswith(("http://", "https://")):
            return f"[Browser Error] Invalid URL: {url}. URL must start with http:// or https://"

        try:
            response = await self._fetch(url)
            resolved_url = str(response.get("url", url) or url)
            content_type = str(response.get("content_type", "") or "")
            raw_bytes = response.get("bytes", b"")

            if self._looks_like_pdf(resolved_url, content_type):
                text = self._extract_pdf_text(raw_bytes, source_url=resolved_url)
            else:
                html = str(response.get("text", "") or "")
                text = self._extract_text(html)
                text = self._clean_text(text)

            if len(text) > max_chars:
                text = text[:max_chars] + f"\n\n[CONTENT_TRUNCATED: {len(text)} chars total, showing first {max_chars}]"

            return text if text else "[Browser Warning] No meaningful content extracted from the page."

        except aiohttp.ClientError as e:
            return f"[Browser Error] Network error: {type(e).__name__}: {e}"
        except Exception as e:
            return f"[Browser Error] Unexpected: {type(e).__name__}: {e}"

    async def _fetch(self, url: str) -> dict[str, Any]:
        """异步获取网页内容。"""
        try:
            return await self._fetch_once(url, auto_decompress=True)
        except aiohttp.ClientPayloadError:
            # 某些站点会在压缩内容解码阶段失败，降级为读取原始字节再手动解压。
            return await self._fetch_once(url, auto_decompress=False)

    async def _fetch_once(self, url: str, *, auto_decompress: bool) -> dict[str, Any]:
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.timeout),
            headers=self._request_headers,
            auto_decompress=auto_decompress,
        ) as session:
            async with session.get(url, allow_redirects=True) as resp:
                resp.raise_for_status()
                raw_bytes = await resp.read()
                content_type = resp.headers.get("Content-Type", "")
                content_encoding = resp.headers.get("Content-Encoding", "")
                if not auto_decompress:
                    raw_bytes = self._decode_content_bytes(raw_bytes, content_encoding=content_encoding)
                return {
                    "url": str(resp.url),
                    "content_type": content_type,
                    "content_encoding": content_encoding,
                    "bytes": raw_bytes,
                    "text": self._decode_response_bytes(raw_bytes, charset=resp.charset),
                }

    @staticmethod
    def _decode_response_bytes(raw_bytes: bytes, charset: str | None = None) -> str:
        encoding = charset or "utf-8"
        try:
            return raw_bytes.decode(encoding, errors="replace")
        except LookupError:
            return raw_bytes.decode("utf-8", errors="replace")

    @staticmethod
    def _decode_content_bytes(raw_bytes: bytes, *, content_encoding: str | None = None) -> bytes:
        normalized_encoding = (content_encoding or "").strip().lower()
        if not normalized_encoding:
            return raw_bytes

        if normalized_encoding == "gzip":
            return gzip.decompress(raw_bytes)

        if normalized_encoding == "deflate":
            try:
                return zlib.decompress(raw_bytes)
            except zlib.error:
                return zlib.decompress(raw_bytes, -zlib.MAX_WBITS)

        if normalized_encoding == "br":
            try:
                import brotli
            except ImportError as exc:
                raise ValueError("brotli content encoding is not supported in this environment") from exc
            return brotli.decompress(raw_bytes)

        return raw_bytes

    @staticmethod
    def _looks_like_pdf(url: str, content_type: str) -> bool:
        normalized_url = (url or "").lower()
        normalized_type = (content_type or "").lower()
        return normalized_url.endswith(".pdf") or "application/pdf" in normalized_type

    def _extract_pdf_text(self, raw_bytes: bytes, *, source_url: str) -> str:
        source_name = urlparse(source_url).path.rstrip("/").split("/")[-1] or "document.pdf"

        try:
            import pdfplumber

            texts = []
            with pdfplumber.open(io.BytesIO(raw_bytes)) as pdf:
                for i, page in enumerate(pdf.pages, 1):
                    text = page.extract_text()
                    if text:
                        texts.append(f"--- Page {i} ---\n{text}")
                page_count = len(pdf.pages)
            full = "\n\n".join(texts)
            if full.strip():
                return f"[PDF: {source_name}]\n[Pages: {page_count}]\n\n{full}"
        except ImportError:
            pass
        except Exception as e:
            return f"[Browser Error] Failed to extract PDF text from {source_name}: {type(e).__name__}: {e}"

        try:
            from PyPDF2 import PdfReader

            reader = PdfReader(io.BytesIO(raw_bytes))
            texts = []
            for i, page in enumerate(reader.pages, 1):
                text = page.extract_text()
                if text:
                    texts.append(f"--- Page {i} ---\n{text}")
            full = "\n\n".join(texts)
            if full.strip():
                return f"[PDF: {source_name}]\n[Pages: {len(reader.pages)}]\n\n{full}"
            return f"[Browser Warning] PDF opened but no extractable text was found: {source_name}"
        except ImportError:
            return (
                f"[Browser Error] Cannot read PDF content for {source_name}. "
                "Please install pdfplumber or PyPDF2."
            )
        except Exception as e:
            return f"[Browser Error] Failed to extract PDF text from {source_name}: {type(e).__name__}: {e}"

    def _extract_text(self, html: str) -> str:
        """从 HTML 中提取正文。"""
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            # 降级：简单正则提取
            return self._fallback_extract(html)

        soup = BeautifulSoup(html, "html.parser")

        # 移除噪声标签
        for tag_name in _NOISE_TAGS:
            for tag in soup.find_all(tag_name):
                tag.decompose()

        # 策略 1：找 article 或 main 标签（语义化 HTML 常用）
        for tag_name in ["article", "main"]:
            tag = soup.find(tag_name)
            if tag:
                return tag.get_text(separator="\n", strip=True)

        # 策略 2：找最长的 div（启发式：正文通常在最长的 div 中）
        best_div = None
        best_len = 0
        for div in soup.find_all("div"):
            text_len = len(div.get_text(strip=True))
            if text_len > best_len:
                best_len = text_len
                best_div = div

        if best_div and best_len > 200:
            return best_div.get_text(separator="\n", strip=True)

        # 策略 3：降级到整个 body
        body = soup.find("body")
        if body:
            return body.get_text(separator="\n", strip=True)

        return soup.get_text(separator="\n", strip=True)

    def _fallback_extract(self, html: str) -> str:
        """无 BeautifulSoup 时的降级提取。"""
        # 移除 script/style 内容
        html = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE)
        html = re.sub(r"<style[^>]*>.*?</style>", "", html, flags=re.DOTALL | re.IGNORECASE)
        # 移除所有标签，保留文本
        text = re.sub(r"<[^>]+>", "\n", html)
        return self._clean_text(text)

    @staticmethod
    def _clean_text(text: str) -> str:
        """清理提取后的文本。"""
        # 合并多余换行
        text = re.sub(r"\n\s*\n+", "\n\n", text)
        # 去除每行首尾空白
        lines = [line.strip() for line in text.splitlines()]
        # 过滤空行和过短行（通常是导航项）
        lines = [line for line in lines if len(line) > 3]
        return "\n".join(lines)


class MockBrowserTool(BaseBrowserTool):
    """Mock 浏览器工具：用于无网络环境调试。"""

    _MOCK_PAGES: dict[str, str] = {
        "https://example.com/ai-report-2024": """
2024 年全球人工智能发展报告

摘要
2024 年，全球 AI 产业进入爆发期。大语言模型参数量突破万亿级别，
多模态能力显著增强。中美两国在 AI 基础研究和应用落地上持续领跑。

一、市场规模
据 IDC 统计，2024 年全球 AI 市场规模达到 5,540 亿美元，同比增长 38.2%。
其中，生成式 AI 占比 28%，约 1,551 亿美元。

二、技术进展
1. 大语言模型：GPT-4o、Claude 3.5、Gemini 1.5 Pro 等模型在多模态推理上取得突破
2. 代码生成：GitHub Copilot 月活开发者超过 500 万
3. 科学发现：AlphaFold 3 预测几乎所有生物分子结构

三、主要玩家
- OpenAI：估值 1,570 亿美元，年化收入 34 亿美元
- Anthropic：估值 400 亿美元，Claude 系列增长迅速
- Google DeepMind：Gemini 整合进全线产品
- 百度：文心一言用户数突破 3 亿

四、政策与监管
欧盟《人工智能法案》于 2024 年 8 月正式生效，成为全球首部全面监管 AI 的法律。

来源：IDC、OpenAI Blog、Anthropic 官方公告
        """.strip(),
        "https://example.com/quantum-computing": """
量子计算最新进展（2024）

IBM 于 2024 年 12 月发布了 Condor 量子处理器，拥有 1,121 个量子比特，
是目前 publicly available 的最大量子处理器。

Google Quantum AI 团队实现了 surface code 纠错的关键里程碑，
逻辑错误率首次低于物理错误率。

中国科学技术大学潘建伟团队实现了 255 光子量子计算优越性。

来源：IBM Research Blog、Nature、中国科学技术大学官网
        """.strip(),
    }

    async def execute(self, url: str, max_chars: int = _DEFAULT_MAX_CHARS) -> str:
        await asyncio.sleep(0.1)  # 模拟网络延迟

        # 精确匹配
        if url in self._MOCK_PAGES:
            content = self._MOCK_PAGES[url]
            if len(content) > max_chars:
                content = content[:max_chars] + "\n\n[CONTENT_TRUNCATED]"
            return content

        # 模糊匹配：根据 URL 关键词返回通用 mock
        if "wikipedia" in url.lower():
            return f"[Mock Browser] Wikipedia page for {url}\n\nThis is a mock Wikipedia article. In production, BrowserTool would fetch the real Wikipedia content."

        return f"[Mock Browser] Fetched {url}\n\nThis is a generic mock page. Content would be extracted from the real webpage in production mode."


def get_browser_tool(mock_mode: bool = False, **kwargs) -> BaseBrowserTool:
    """工厂函数：根据配置返回 BrowserTool 或 MockBrowserTool。"""
    if mock_mode:
        return MockBrowserTool()
    return BrowserTool(**kwargs)
