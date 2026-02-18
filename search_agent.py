#!/usr/bin/env python3
"""
LocalWebSearch-CDP: API-key-free Local WebSearch
- Multi-LLM backend support (SGLang, Ollama, LM Studio, OpenAI-compatible)
- CDP-based parallel web search (Naver, Google, Brave)
- Smart search with query generation and result summarization
"""

import asyncio
import html
import httpx
import json
import re
import sys
import time
from datetime import datetime
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlparse

# CDP search module
from cdp_search import search_with_cdp

# LLM Adapter layer
from llm_adapters import (
    get_adapter,
    detect_available_backends,
    BaseLLMAdapter,
    LLMResponse,
)

# Configuration - LLM Backend
LLM_BACKEND = "sglang"  # Default: sglang, ollama, lmstudio, openai
LLM_URL = None          # None = use default URL for backend
LLM_MODEL = None        # None = use default model for backend
LLM_ADAPTER: BaseLLMAdapter | None = None  # Will be initialized at startup

# Legacy config (for backwards compatibility)
AGENTCPM_URL = "http://localhost:30001/v1/chat/completions"
USE_CDP_SEARCH = True  # Whether to use CDP search

# Timeout and parallel processing settings
SEARCH_TIMEOUT = 90.0   # Search timeout (seconds)
FETCH_TIMEOUT = 3.0     # URL fetch timeout (seconds) - including Jina
FETCH_LOCAL_TIMEOUT = 2.0   # Local fetch timeout (seconds)
FETCH_GLOBAL_BUDGET = 10.0  # Total fetch budget (seconds) - prioritize fast URLs
MODEL_TIMEOUT = 150.0   # Model call timeout (seconds)
MAX_PARALLEL = 50       # Maximum parallel executions
MAX_SEARCH_QUERIES = 3  # Search tool query limit (limited to 3 due to CDP sequential search)
MAX_OFFICIAL_FETCH = 5  # Auto fetch limit for official URLs
MAX_SOURCES = 20        # Number of source URLs to include in final answer

# Search depth settings
DEPTH_CONFIG = {
    "simple": {"fetch_enabled": False, "max_fetch": 0, "description": "snippets only"},
    "medium": {"fetch_enabled": True, "max_fetch": 5, "description": "fetch top 5 URLs"},
    "deep": {"fetch_enabled": True, "max_fetch": 50, "description": "fetch all URLs"},
}
CURRENT_DEPTH = "medium"  # Default

MAX_OUTPUT_CONTENT = 5000            # Maximum length for model
MIN_LOCAL_CONTENT_LEN = 400          # Minimum local extraction length

# Cache disabled - fresh search each time

OFFICIAL_DOMAINS = {
    "openai.com",
    "help.openai.com",
    "docs.openai.com",
    "github.com",
}

LOG_DIR = Path(__file__).resolve().parent / "outputs" / "search_agent_logs"

# Jina removed - using simple_clean

# Foreign word -> English translation dictionary (rule-based)
FOREIGN_WORD_MAP = {
    # Apple products
    "맥": "Mac", "맥북": "MacBook", "맥프로": "Mac Pro", "맥스튜디오": "Mac Studio",
    "아이폰": "iPhone", "아이패드": "iPad", "아이맥": "iMac",
    "에어팟": "AirPods", "애플워치": "Apple Watch", "애플": "Apple",
    "울트라": "Ultra", "프로": "Pro", "맥스": "Max", "에어": "Air",
    # Numbers (M series chips)
    "엠": "M", "칩": "chip",
    # Other companies/products
    "테슬라": "Tesla", "엔비디아": "NVIDIA", "구글": "Google",
    "마이크로소프트": "Microsoft", "메타": "Meta", "아마존": "Amazon",
    "오픈에이아이": "OpenAI", "챗지피티": "ChatGPT", "지피티": "GPT",
    "클로드": "Claude", "제미나이": "Gemini",
    # NVIDIA GPU architectures
    "블랙웰": "Blackwell", "루빈": "Rubin", "호퍼": "Hopper",
    "암페어": "Ampere", "튜링": "Turing", "볼타": "Volta",
    "지포스": "GeForce", "쿼드로": "Quadro", "큐다": "CUDA",
    # General tech terms
    "인공지능": "AI", "머신러닝": "machine learning", "딥러닝": "deep learning",
}

# Number pattern (맥5 -> M5)
NUMBER_PATTERN = re.compile(r'맥(\d+)')


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _normalize_query(text: str) -> str:
    if not text:
        return ""
    cleaned = text.strip().lower()
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = cleaned.strip(".,;:!?")
    return cleaned


# Cache functions removed - fresh search/fetch each time


def _domain_from_url(url: str) -> str:
    try:
        return urlparse(url).netloc.lower()
    except Exception:
        return ""


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self.title_parts: list[str] = []
        self._in_title = False

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in {"p", "br", "li", "div", "h1", "h2", "h3", "h4"}:
            self.parts.append("\n")
        if tag == "title":
            self._in_title = True

    def handle_endtag(self, tag: str) -> None:
        if tag == "title":
            self._in_title = False

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self.title_parts.append(data)
        self.parts.append(data)


def _extract_text_basic(html_text: str) -> tuple[str, str]:
    cleaned = re.sub(r"<script[^>]*>.*?</script>", " ", html_text, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r"<style[^>]*>.*?</style>", " ", cleaned, flags=re.DOTALL | re.IGNORECASE)
    cleaned = html.unescape(cleaned)
    parser = _TextExtractor()
    try:
        parser.feed(cleaned)
    except Exception:
        pass
    title = " ".join(part.strip() for part in parser.title_parts if part.strip())
    text = " ".join(part.strip() for part in parser.parts if part.strip())
    text = re.sub(r"\s+", " ", text).strip()
    return title, text


def _extract_text_trafilatura(html_text: str) -> tuple[str, str] | None:
    try:
        import trafilatura  # type: ignore
        from trafilatura import metadata  # type: ignore
    except Exception:
        return None
    try:
        text = trafilatura.extract(html_text, include_comments=False, include_tables=False)
        if not text:
            return None
        title = ""
        try:
            meta = metadata.extract_metadata(html_text)
            if meta and meta.title:
                title = meta.title
        except Exception:
            title = ""
        return title or "", text.strip()
    except Exception:
        return None


def _extract_text_readability(html_text: str) -> tuple[str, str] | None:
    try:
        from readability import Document  # type: ignore
    except Exception:
        return None
    try:
        doc = Document(html_text)
        title = doc.short_title() or ""
        summary_html = doc.summary()
        title2, text = _extract_text_basic(summary_html)
        if not title:
            title = title2
        if not text:
            return None
        return title, text
    except Exception:
        return None


def _extract_text_and_title(html_text: str) -> tuple[str, str]:
    for extractor in (_extract_text_trafilatura, _extract_text_readability):
        result = extractor(html_text)
        if result and result[1]:
            return result
    return _extract_text_basic(html_text)


# Local index functions removed (CDP search only)


def _dedup_result_items(items: list[dict]) -> list[dict]:
    seen = set()
    deduped = []
    for item in items:
        url = item.get("url", "")
        if not url:
            continue
        norm = _normalize_url_for_dedup(url)
        if norm in seen:
            continue
        seen.add(norm)
        deduped.append(item)
    return deduped


def translate_foreign_words(query: str) -> tuple[str, bool]:
    """Translate Korean foreign words to English (rule-based)"""
    translated = query
    has_foreign = False

    # Handle number patterns first (맥5 -> M5)
    if NUMBER_PATTERN.search(translated):
        translated = NUMBER_PATTERN.sub(r'M\1', translated)
        has_foreign = True

    # Dictionary-based translation
    for kor, eng in FOREIGN_WORD_MAP.items():
        if kor in translated:
            translated = translated.replace(kor, eng)
            has_foreign = True

    # Remove Korean characters (keep only English)
    if has_foreign:
        # Remove Korean characters
        english_only = re.sub(r'[가-힣]+', ' ', translated)
        english_only = re.sub(r'\s+', ' ', english_only).strip()
        return english_only, True

    return query, False

# Tool definitions (format understood by AgentCPM-Explore)
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Search for information on web/local index. Can search multiple queries simultaneously.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of search queries (max 5)"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_url",
            "description": "Fetch webpage content. (Local extraction first, fallback on failure)",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of URLs to fetch (max 15, 5 per portal)"
                    },
                    "purpose": {
                        "type": "string",
                        "description": "Purpose of visit (what information to find)"
                    }
                },
                "required": ["url", "purpose"]
            }
        }
    }
]

SYSTEM_PROMPT = """CRITICAL: Do NOT use <think> tags. Answer directly without internal reasoning. Be concise.

You are a thorough Korean research assistant. Your goal is to gather as much accurate, recent evidence as possible.

Available tools:
1. search(query: list[str]) - Search the web with up to 5 queries in parallel.
   - Returns URL + title + snippet (NO full content)
2. fetch_url(url: list[str], purpose: str) - Fetch full webpage content.

## WORKFLOW (Universal Mode)
1. You MUST call search() at least once for every query.
2. Use search() as many times as needed to gather broad, diverse sources.
3. Use fetch_url() to retrieve full content from the most relevant URLs.
4. Prioritize official/primary sources when available.
5. Do NOT stop early for speed. Favor coverage and evidence.

Tool call format:
<tool_call>
{{"name": "search", "arguments": {{"query": ["query1", "query2", "query3", ...]}}}}
</tool_call>

<tool_call>
{{"name": "fetch_url", "arguments": {{"url": ["https://url1", "https://url2", ...], "purpose": "purpose"}}}}
</tool_call>

## LANGUAGE STRATEGY (Determine before search)
1. Korean topics -> Search in Korean
2. US/Global tech -> Search in English
3. Chinese topics -> Search in Chinese
4. International topics -> Search in English + Korean

## FINAL OUTPUT (Designed for upper LLM to summarize)
- Do NOT write a deep synthesis.
- Provide an evidence-centered response: short bullets of key facts per source.
- Keep it in Korean.

Today's date: {today}. If search results seem outdated, search for more recent information."""


def get_system_prompt():
    today = datetime.now().strftime("%Y-%m-%d")
    return SYSTEM_PROMPT.format(today=today)


URL_REGEX = re.compile(r"https?://[^\s)\]]+")

LOW_QUALITY_DOMAIN_SUFFIXES = {
    "blog.naver.com",
    "m.blog.naver.com",
    "post.naver.com",
    "cafe.naver.com",
    "tistory.com",
    "brunch.co.kr",
    "medium.com",
    "reddit.com",
    "youtube.com",
    "youtu.be",
}


def _normalize_url(url: str) -> str:
    cleaned = url.rstrip(".,;]")
    # Remove trailing ')' only if it looks unbalanced (common in markdown)
    while cleaned.endswith(")") and cleaned.count("(") < cleaned.count(")"):
        cleaned = cleaned[:-1]
    return cleaned


def _normalize_url_for_dedup(url: str) -> str:
    try:
        parsed = urlparse(url)
        scheme = parsed.scheme.lower()
        netloc = parsed.netloc.lower()
        path = parsed.path or ""
        query = parsed.query or ""
        if query:
            params = []
            for part in query.split("&"):
                if "=" in part:
                    k, v = part.split("=", 1)
                else:
                    k, v = part, ""
                k_lower = k.lower()
                if k_lower.startswith("utm_") or k_lower in {"fbclid", "gclid", "ref", "src"}:
                    continue
                params.append(f"{k}={v}" if v else k)
            query = "&".join(params)
        normalized = f"{scheme}://{netloc}{path}"
        if query:
            normalized += f"?{query}"
        return normalized
    except Exception:
        return url


def _dedup_urls(urls: list[str]) -> list[str]:
    seen = set()
    deduped = []
    for u in urls:
        norm = _normalize_url_for_dedup(u)
        if norm in seen:
            continue
        seen.add(norm)
        deduped.append(u)
    return deduped


def _select_diverse_urls(urls: list[str]) -> list[str]:
    # Universal mode: respect model-selected URLs, only deduplicate
    return _dedup_urls(urls)


def _is_low_quality_domain(domain: str) -> bool:
    if not domain:
        return False
    if domain in LOW_QUALITY_DOMAIN_SUFFIXES:
        return True
    for suffix in LOW_QUALITY_DOMAIN_SUFFIXES:
        if domain.endswith("." + suffix):
            return True
    return False


def _filter_fetch_urls(urls: list[str], candidates: list[str], fetched: set[str]) -> list[str]:
    urls = _dedup_urls(urls)
    candidates = _dedup_urls(candidates)

    chosen: list[str] = []
    low: list[str] = []
    seen: set[str] = set()

    for u in urls:
        norm = _normalize_url_for_dedup(_normalize_url(u))
        if norm in seen or norm in fetched:
            continue
        seen.add(norm)
        if _is_low_quality_domain(_domain_from_url(u)):
            low.append(u)
        else:
            chosen.append(u)

    replacements: list[str] = []
    for c in candidates:
        norm = _normalize_url_for_dedup(_normalize_url(c))
        if norm in seen or norm in fetched:
            continue
        if _is_low_quality_domain(_domain_from_url(c)):
            continue
        replacements.append(c)
        seen.add(norm)
        if len(replacements) + len(chosen) >= MAX_PARALLEL:
            break

    final = chosen + replacements

    if len(final) < MAX_PARALLEL:
        for u in low:
            norm = _normalize_url_for_dedup(_normalize_url(u))
            if norm in seen or norm in fetched:
                continue
            final.append(u)
            seen.add(norm)
            if len(final) >= MAX_PARALLEL:
                break

    if len(final) < MAX_PARALLEL:
        for c in candidates:
            norm = _normalize_url_for_dedup(_normalize_url(c))
            if norm in seen or norm in fetched:
                continue
            final.append(c)
            seen.add(norm)
            if len(final) >= MAX_PARALLEL:
                break

    return final


def _extract_urls(text: str) -> list[str]:
    if not text:
        return []
    return [_normalize_url(u) for u in URL_REGEX.findall(text)]


def _is_official_url(url: str) -> bool:
    try:
        host = urlparse(url).netloc.lower()
    except Exception:
        return False
    for domain in OFFICIAL_DOMAINS:
        if host == domain or host.endswith("." + domain):
            return True
    return False


def _select_sources(urls: set[str]) -> list[str]:
    if not urls:
        return []
    official = []
    others = []
    seen = set()
    for u in sorted(urls):
        norm = _normalize_url_for_dedup(_normalize_url(u))
        if norm in seen:
            continue
        seen.add(norm)
        if _is_official_url(u):
            official.append(u)
        else:
            others.append(u)
    ordered = official + others
    return ordered[:MAX_SOURCES]


def _append_sources(answer: str, sources: list[str]) -> str:
    if not sources:
        return answer
    lines = ["", "---", "Sources:"]
    lines.extend([f"- {u}" for u in sources])
    return answer.rstrip() + "\n" + "\n".join(lines)


def _collect_urls_from_logs(log_data: dict) -> set[str]:
    urls: set[str] = set()
    for turn in log_data.get("turns", []):
        for tr in turn.get("tool_results", []):
            result = tr.get("result", "")
            for url in _extract_urls(result):
                urls.add(url)
    return urls


def _save_log(log_data: dict) -> str | None:
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = LOG_DIR / f"search_agent_log_{ts}.json"
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(log_data, f, ensure_ascii=False, indent=2)
        return str(log_path)
    except Exception:
        return None


async def call_smartcrawl(
    keyword: str,
    search_type: str = "web",
    portal: str = "all",
    count: int = 10,
    skip_content: bool = True,  # Default: URL+snippet only (skip crawling)
    _retried: bool = False,
) -> str:
    """Execute CDP-based search (using chrome-devtools MCP)

    Uses Chrome DevTools Protocol for real browser search instead of SmartCrawl HTTP API.
    - Bypass blocking/CAPTCHA
    - Stable search results
    """
    # Cache disabled - fresh search each time
    try:
        # CDP search is synchronous, run via to_thread
        data = await asyncio.to_thread(
            search_with_cdp,
            keyword,
            portal=portal,
            count=count,
            search_type=search_type,
            skip_content=skip_content
        )

        if not data.get("success"):
            error_msg = data.get("error", "Unknown error")
            # Retry logic
            if not _retried and "MCP" in error_msg:
                # On MCP server issue, retry with different portal
                fallback_portal = "brave" if portal == "naver" else "naver"
                return await call_smartcrawl(
                    keyword,
                    search_type=search_type,
                    portal=fallback_portal,
                    count=count,
                    skip_content=skip_content,
                    _retried=True,
                )
            return f"Search failed: {error_msg}"

        results = data.get("data", {}).get("results", [])
        if not results:
            # No results, retry with different portal
            if not _retried:
                fallback_portal = "brave" if portal in ("naver", "google") else "naver"
                return await call_smartcrawl(
                    keyword,
                    search_type=search_type,
                    portal=fallback_portal,
                    count=count,
                    skip_content=skip_content,
                    _retried=True,
                )
            return f"No results for '{keyword}'"

        output = []
        for i, r in enumerate(results[:count], 1):
            title = r.get("title", "No title")
            url = r.get("url", "")
            snippet = (r.get("snippet") or r.get("content") or "")[:300]
            source = r.get("source") or portal
            output.append(f"{i}. [{title}]({url})\n   ({source}) {snippet}")

        mode = "snippet" if skip_content else "full"
        result_text = f"Search results for '{keyword}' ({mode}):\n\n" + "\n\n".join(output)
        return result_text

    except Exception as e:
        # On error, retry with different portal
        if not _retried:
            fallback_portal = "brave" if portal != "brave" else "naver"
            return await call_smartcrawl(
                keyword,
                search_type=search_type,
                portal=fallback_portal,
                count=count,
                skip_content=skip_content,
                _retried=True,
            )
        return f"CDP search error: {e}"


# Fetch cache function removed - fresh fetch each time


def _format_fetched_content(url: str, title: str, content: str, source: str) -> str:
    header = f"URL: {url}"
    if title:
        header += f"\nTitle: {title}"
    header += f"\nSource: {source}"
    return header + "\n\n" + content[:MAX_OUTPUT_CONTENT]


async def _fetch_url_local(url: str) -> tuple[str, str]:
    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
    }
    async with httpx.AsyncClient(timeout=FETCH_LOCAL_TIMEOUT, headers=headers, follow_redirects=True) as client:
        response = await client.get(url)
        if response.status_code >= 400:
            raise ValueError(f"HTTP {response.status_code}")
        html_text = response.text or ""
    title, text = _extract_text_and_title(html_text)
    if len(text) < MIN_LOCAL_CONTENT_LEN:
        raise ValueError("Content too short")
    return title, text


def _simple_clean_html(html_text: str) -> tuple[str, str]:
    """Simple HTML cleanup using BeautifulSoup (Jina replacement)"""
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        # Use basic parser if BeautifulSoup not available
        return _extract_text_basic(html_text)

    soup = BeautifulSoup(html_text, 'html.parser')

    # Remove unnecessary tags
    for tag in soup(['script', 'style', 'nav', 'header', 'footer', 'aside',
                     'noscript', 'svg', 'iframe', 'form', 'button']):
        tag.decompose()

    # Extract title
    title = ""
    title_tag = soup.find('title')
    if title_tag:
        title = title_tag.get_text(strip=True)

    # Extract text
    text = soup.get_text(separator='\n', strip=True)

    # Clean up consecutive newlines
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)

    return title, text.strip()


async def fetch_url_content(url: str) -> str:
    """Fetch URL content (local extraction -> simple_clean fallback)

    Cache disabled - fresh fetch each time
    """
    # Try local extraction (trafilatura/readability)
    try:
        title, content = await _fetch_url_local(url)
        return _format_fetched_content(url, title, content, "local")
    except Exception as e:
        pass  # Fall through to simple_clean

    # simple_clean fallback (BeautifulSoup-based)
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/122.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        }
        async with httpx.AsyncClient(timeout=FETCH_TIMEOUT, headers=headers, follow_redirects=True) as client:
            response = await client.get(url)
            if response.status_code < 400:
                html_text = response.text or ""
                title, content = _simple_clean_html(html_text)
                if content and len(content) >= MIN_LOCAL_CONTENT_LEN:
                    return _format_fetched_content(url, title, content, "simple")
    except Exception as e2:
        return f"Failed to fetch URL: {url} (error: {e2})"

    return f"Failed to fetch URL: {url} (content extraction failed)"


async def search_with_timeout(query: str) -> str:
    """Search with individual timeout"""
    start = datetime.now()
    print(f"[Search Start] '{query}' @ {start.strftime('%H:%M:%S.%f')[:-3]}", file=sys.stderr)
    try:
        # Direct CDP search (local index disabled)
        result = await asyncio.wait_for(
            call_smartcrawl(query),
            timeout=SEARCH_TIMEOUT
        )
        elapsed = (datetime.now() - start).total_seconds()
        print(f"[Search Complete] '{query}' ({elapsed:.2f}s)", file=sys.stderr)
        return result
    except asyncio.TimeoutError:
        elapsed = (datetime.now() - start).total_seconds()
        print(f"[Search Timeout] '{query}' ({elapsed:.2f}s)", file=sys.stderr)
        return f"Search timeout for '{query}' (exceeded {SEARCH_TIMEOUT}s)"
    except Exception as e:
        elapsed = (datetime.now() - start).total_seconds()
        print(f"[Search Error] '{query}' ({elapsed:.2f}s): {e}", file=sys.stderr)
        return f"Search error for '{query}': {e}"


async def fetch_with_timeout(url: str) -> str:
    """URL fetch with individual timeout"""
    # Display shortened URL (domain only)
    try:
        domain = url.split('/')[2][:30]
    except:
        domain = url[:30]

    start = datetime.now()
    print(f"[Fetch Start] {domain} @ {start.strftime('%H:%M:%S.%f')[:-3]}", file=sys.stderr)
    try:
        result = await asyncio.wait_for(
            fetch_url_content(url),
            timeout=FETCH_TIMEOUT
        )
        elapsed = (datetime.now() - start).total_seconds()
        print(f"[Fetch Complete] {domain} ({elapsed:.2f}s)", file=sys.stderr)
        return result
    except asyncio.TimeoutError:
        elapsed = (datetime.now() - start).total_seconds()
        print(f"[Fetch Timeout] {domain} ({elapsed:.2f}s)", file=sys.stderr)
        return f"URL timeout: {url}"
    except Exception as e:
        elapsed = (datetime.now() - start).total_seconds()
        print(f"[Fetch Error] {domain} ({elapsed:.2f}s): {e}", file=sys.stderr)
        return f"URL error: {e}"


def _is_fetch_success(result: str) -> bool:
    if not result:
        return False
    if result.startswith("URL: "):
        return True
    return False


async def execute_tool(tool_name: str, arguments: dict) -> str:
    """Execute tool (sequential + timeout + graceful degradation)"""
    if tool_name == "search":
        queries = arguments.get("query", [])
        if isinstance(queries, str):
            queries = [queries]

        # Sequential call per query (cdp_search internally uses 3 Chrome instances in parallel)
        final_results = []
        for q in queries[:MAX_SEARCH_QUERIES]:
            try:
                result = await search_with_timeout(q)
                final_results.append(result)
            except Exception as e:
                final_results.append(f"Unexpected error: {e}")

        return "\n\n=======\n\n".join(final_results)

    elif tool_name == "fetch_url":
        # Fetch limit based on search depth
        depth_cfg = DEPTH_CONFIG.get(CURRENT_DEPTH, DEPTH_CONFIG["medium"])
        if not depth_cfg["fetch_enabled"]:
            return "[simple mode] Fetch disabled. Generate answer using snippet information."

        urls = arguments.get("url", [])
        if isinstance(urls, str):
            urls = [urls]

        urls = _select_diverse_urls(urls)
        max_fetch = min(depth_cfg["max_fetch"], MAX_PARALLEL)
        urls = urls[:max_fetch]

        if not urls:
            return "No URLs to fetch."

        # Parallel execution (individual timeout + global budget + early termination)
        task_map = {}
        for url in urls[:MAX_PARALLEL]:
            task = asyncio.create_task(fetch_with_timeout(url))
            task_map[task] = url

        final_results = []
        if FETCH_GLOBAL_BUDGET and task_map:
            done, pending = await asyncio.wait(
                task_map.keys(),
                timeout=FETCH_GLOBAL_BUDGET,
            )
            for task in done:
                try:
                    result = task.result()
                except Exception as e:
                    result = f"Unexpected error: {e}"
                final_results.append(result)
            for task in pending:
                url = task_map.get(task, "")
                task.cancel()
                final_results.append(f"URL skipped: {url} (fetch budget exceeded)")
        else:
            results = await asyncio.gather(*task_map.keys(), return_exceptions=True)
            for r in results:
                if isinstance(r, Exception):
                    final_results.append(f"Unexpected error: {r}")
                else:
                    final_results.append(r)

        return "\n\n=======\n\n".join(final_results)

    return f"Unknown tool: {tool_name}"


def parse_tool_calls(text: str) -> list[dict]:
    """Parse tool_call from model output (legacy text-based format)

    For adapters that use function calling API, tool_calls are already parsed
    in the LLMResponse.tool_calls field.
    """
    tool_calls = []

    # Find <tool_call>...</tool_call> pattern
    pattern = r'<tool_call>\s*(\{.*?\})\s*</tool_call>'
    matches = re.findall(pattern, text, re.DOTALL)

    for match in matches:
        try:
            tool_call = json.loads(match)
            tool_calls.append(tool_call)
        except json.JSONDecodeError:
            continue

    return tool_calls


def parse_tool_calls_from_response(response: LLMResponse) -> list[dict]:
    """Parse tool calls from LLMResponse (supports both text and function calling)

    Args:
        response: LLMResponse from adapter

    Returns:
        List of tool call dicts [{"name": "...", "arguments": {...}}, ...]
    """
    # If adapter already parsed tool_calls, use those
    if response.tool_calls:
        return [
            {"name": tc.name, "arguments": tc.arguments}
            for tc in response.tool_calls
        ]

    # Otherwise, try text-based parsing
    return parse_tool_calls(response.content)


async def call_agentcpm(messages: list[dict], max_tokens: int = 10000) -> str:
    """Call LLM via adapter (legacy function name for compatibility)"""
    global LLM_ADAPTER
    start = datetime.now()

    try:
        if LLM_ADAPTER is None:
            # Fallback to direct HTTP call (legacy behavior)
            async with httpx.AsyncClient(timeout=MODEL_TIMEOUT) as client:
                response = await client.post(
                    AGENTCPM_URL,
                    json={
                        "model": "AgentCPM-Explore",
                        "messages": messages,
                        "max_tokens": max_tokens,
                        "temperature": 0.7
                    }
                )
                data = response.json()
                elapsed = (datetime.now() - start).total_seconds()
                print(f"[Model Call] {elapsed:.2f}s (legacy)", file=sys.stderr)
                return data["choices"][0]["message"]["content"]

        # Use adapter
        response = await LLM_ADAPTER.call(messages, tools=TOOLS)
        elapsed = (datetime.now() - start).total_seconds()
        print(f"[Model Call] {elapsed:.2f}s ({LLM_ADAPTER.name})", file=sys.stderr)
        return response.content

    except Exception as e:
        elapsed = (datetime.now() - start).total_seconds()
        print(f"[Model Error] {elapsed:.2f}s: {e}", file=sys.stderr)
        return f"Model call error: {e}"


async def call_llm(messages: list[dict]) -> LLMResponse:
    """Call LLM via adapter (new function with full response)"""
    global LLM_ADAPTER
    start = datetime.now()

    if LLM_ADAPTER is None:
        raise RuntimeError("LLM adapter not initialized. Call init_llm_adapter() first.")

    try:
        response = await LLM_ADAPTER.call(messages, tools=TOOLS)
        elapsed = (datetime.now() - start).total_seconds()
        print(f"[Model Call] {elapsed:.2f}s ({LLM_ADAPTER.name})", file=sys.stderr)
        return response
    except Exception as e:
        elapsed = (datetime.now() - start).total_seconds()
        print(f"[Model Error] {elapsed:.2f}s: {e}", file=sys.stderr)
        raise


async def init_llm_adapter(
    backend: str = "sglang",
    url: str | None = None,
    model: str | None = None,
) -> BaseLLMAdapter:
    """Initialize LLM adapter

    Args:
        backend: One of "sglang", "ollama", "lmstudio", "openai"
        url: Optional custom URL
        model: Optional model name

    Returns:
        Initialized adapter
    """
    global LLM_ADAPTER, LLM_BACKEND, LLM_URL, LLM_MODEL

    LLM_BACKEND = backend
    LLM_URL = url
    LLM_MODEL = model

    kwargs = {"timeout": MODEL_TIMEOUT}
    if url:
        kwargs["url"] = url
    if model:
        kwargs["model"] = model

    LLM_ADAPTER = get_adapter(backend, **kwargs)
    print(f"[LLM] Initialized {LLM_ADAPTER}", file=sys.stderr)
    return LLM_ADAPTER


async def list_available_backends() -> list[dict]:
    """List available LLM backends with their models"""
    backends = await detect_available_backends()
    return backends


def preprocess_query_sync(query: str) -> dict:
    """Analyze user query and generate optimal search terms (rule-based, no LLM needed)"""
    translated, has_foreign = translate_foreign_words(query)

    if has_foreign and translated:
        # Foreign word detected -> generate English queries
        queries = [
            translated,
            f"{translated} latest",
            f"{translated} 2026",
        ]
        query_type = "foreign"
        print(f"[Preprocess] Foreign word detected: '{query}' -> {queries}", file=sys.stderr)
    else:
        # Pure Korean -> use as-is
        queries = [query]
        query_type = "korean"
        print(f"[Preprocess] Pure Korean: '{query}'", file=sys.stderr)

    return {"queries": queries, "type": query_type}


def _has_any_result(text: str) -> bool:
    if not text:
        return False
    return ("http://" in text) or ("https://" in text) or ("URL:" in text)


async def search_agent(query: str, max_turns: int = 10) -> str:
    """Run search agent"""
    # Query preprocessing: Korean foreign words -> English conversion (rule-based, fast)
    preprocessed = preprocess_query_sync(query)
    suggested_queries = preprocessed.get("queries", [query])
    query_type = preprocessed.get("type", "unknown")

    # Add preprocessing result as hint
    if query_type == "foreign" and suggested_queries:
        # Foreign word/product name: provide English query hint (clearer format)
        queries_str = ", ".join(f'"{q}"' for q in suggested_queries)
        hint = f'\n\n[HINT: Search in English. Include in query array when calling search: {queries_str}]'
        user_content = query + hint
    elif query_type == "mixed" and suggested_queries:
        # Mixed: provide English+Korean query hint
        queries_str = ", ".join(f'"{q}"' for q in suggested_queries)
        hint = f'\n\n[HINT: Mixed English+Korean search needed. Include in query array: {queries_str}]'
        user_content = query + hint
    else:
        # Pure Korean: use original
        user_content = query

    messages = [
        {"role": "system", "content": get_system_prompt()},
        {"role": "user", "content": user_content}
    ]
    has_any_results = False
    collected_urls: set[str] = set()
    official_urls: set[str] = set()
    fetched_official_urls: set[str] = set()
    fetch_calls = 0
    fetched_urls: set[str] = set()
    log_data = {
        "query": query,
        "preprocessed": {
            "type": query_type,
            "suggested_queries": suggested_queries,
        },
        "started_at": datetime.now().isoformat(),
        "turns": [],
    }
    auto_search_done = False

    def finalize(answer: str) -> str:
        if fetched_urls:
            sources = _select_sources(fetched_urls)
        else:
            combined_urls = set(collected_urls)
            combined_urls.update(_collect_urls_from_logs(log_data))
            sources = _select_sources(combined_urls)
        final_answer = _append_sources(answer, sources)
        log_payload = {
            **log_data,
            "finished_at": datetime.now().isoformat(),
            "final_answer": final_answer,
            "sources": sources,
            "official_urls": sorted(official_urls),
            "fetched_official_urls": sorted(fetched_official_urls),
            "has_any_results": has_any_results,
        }
        log_path = _save_log(log_payload)
        if log_path:
            print(f"[LOG] {log_path}", file=sys.stderr)
        return final_answer

    for turn in range(max_turns):
        print(f"\n[Turn {turn + 1}] Calling model...", file=sys.stderr)

        response = await call_agentcpm(messages)
        print(f"[Turn {turn + 1}] Response length: {len(response)} chars", file=sys.stderr)
        turn_log = {
            "turn": turn + 1,
            "assistant_response": response,
            "tool_calls": [],
            "tool_results": [],
        }

        # Parse tool_call
        tool_calls = parse_tool_calls(response)

        if not tool_calls:
            if not has_any_results and not auto_search_done:
                auto_search_done = True
                forced_queries = suggested_queries[:MAX_SEARCH_QUERIES] if suggested_queries else [query]
                forced_args = {"query": forced_queries}
                print(f"[Turn {turn + 1}] Auto search execution: {forced_queries}", file=sys.stderr)
                forced_result = await execute_tool("search", forced_args)
                if _has_any_result(forced_result):
                    has_any_results = True
                extracted = _extract_urls(forced_result)
                for url in extracted:
                    collected_urls.add(url)
                    if _is_official_url(url):
                        official_urls.add(url)
                turn_log["tool_calls"].append({"name": "search", "arguments": forced_args})
                turn_log["tool_results"].append(
                    {"name": "search", "arguments": forced_args, "result": forced_result}
                )
                log_data["turns"].append(turn_log)
                messages.append({"role": "assistant", "content": response})
                messages.append(
                    {
                        "role": "user",
                        "content": f"<tool_result name=\"search\">\n{forced_result}\n</tool_result>",
                    }
                )
                continue
            pending_official = [u for u in official_urls if u not in fetched_official_urls]
            remaining = MAX_OFFICIAL_FETCH - len(fetched_official_urls)
            if pending_official and remaining > 0:
                targets = pending_official[:remaining]
                print(f"[Turn {turn + 1}] Forced official URL fetch: {targets}", file=sys.stderr)
                forced_args = {"url": targets, "purpose": "Verify official page content"}
                forced_result = await execute_tool("fetch_url", forced_args)
                if _has_any_result(forced_result):
                    has_any_results = True
                for url in targets:
                    collected_urls.add(url)
                    if _is_official_url(url):
                        official_urls.add(url)
                        fetched_official_urls.add(url)
                extracted = _extract_urls(forced_result)
                for url in extracted:
                    collected_urls.add(url)
                    if _is_official_url(url):
                        official_urls.add(url)
                        fetched_official_urls.add(url)
                turn_log["tool_calls"].append({"name": "fetch_url", "arguments": forced_args})
                turn_log["tool_results"].append(
                    {"name": "fetch_url", "arguments": forced_args, "result": forced_result}
                )
                log_data["turns"].append(turn_log)
                messages.append({"role": "assistant", "content": response})
                messages.append(
                    {
                        "role": "user",
                        "content": f"<tool_result name=\"fetch_url\">\n{forced_result}\n</tool_result>",
                    }
                )
                continue

            if fetch_calls == 0 and collected_urls:
                url_list = sorted(collected_urls)
                targets = _select_diverse_urls(url_list)
                targets = targets[:MAX_PARALLEL]
                if targets:
                    print(f"[Turn {turn + 1}] Forced fetch (before finalize): {len(targets)} URLs", file=sys.stderr)
                    forced_args = {"url": targets, "purpose": "Collect key URL content for final answer"}
                    forced_result = await execute_tool("fetch_url", forced_args)
                    if _has_any_result(forced_result):
                        has_any_results = True
                    for url in targets:
                        collected_urls.add(url)
                        if _is_official_url(url):
                            official_urls.add(url)
                            fetched_official_urls.add(url)
                    extracted = _extract_urls(forced_result)
                    for url in extracted:
                        collected_urls.add(url)
                        if _is_official_url(url):
                            official_urls.add(url)
                            fetched_official_urls.add(url)
                    turn_log["tool_calls"].append({"name": "fetch_url", "arguments": forced_args})
                    turn_log["tool_results"].append(
                        {"name": "fetch_url", "arguments": forced_args, "result": forced_result}
                    )
                    log_data["turns"].append(turn_log)
                    messages.append({"role": "assistant", "content": response})
                    messages.append(
                        {
                            "role": "user",
                            "content": f"<tool_result name=\"fetch_url\">\n{forced_result}\n</tool_result>",
                        }
                    )
                    fetch_calls = 1
                    continue

            # No tool calls = final answer
            log_data["turns"].append(turn_log)
            # Remove <think> tags (including unclosed ones)
            final_answer = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
            final_answer = re.sub(r'<think>.*', '', final_answer, flags=re.DOTALL).strip()
            if final_answer and has_any_results:
                return finalize(final_answer)
            if not has_any_results:
                return finalize("No search results (search failed or blocked).")
            # If no final answer, request one more time
            messages.append({"role": "assistant", "content": response})
            messages.append({"role": "user", "content": "Write only the final answer concisely without tool calls."})
            continue

        # Execute tools
        messages.append({"role": "assistant", "content": response})

        tool_results = []
        for tc in tool_calls:
            tool_name = tc.get("name", "")
            arguments = tc.get("arguments", {})
            print(f"[Turn {turn + 1}] Tool execution: {tool_name}({arguments})", file=sys.stderr)
            turn_log["tool_calls"].append({"name": tool_name, "arguments": arguments})

            if tool_name == "search":
                pass

            if tool_name == "fetch_url":
                fetch_calls += 1
                urls = arguments.get("url", [])
                if isinstance(urls, str):
                    urls = [urls]
                urls = _filter_fetch_urls(urls, list(collected_urls), fetched_urls)
                if not urls:
                    result = "No fetch_url targets. (0 after dedup/filtering)"
                    turn_log["tool_results"].append(
                        {"name": tool_name, "arguments": arguments, "result": result}
                    )
                    tool_results.append(f"<tool_result name=\"{tool_name}\">\n{result}\n</tool_result>")
                    continue
                arguments["url"] = urls

            result = await execute_tool(tool_name, arguments)
            if _has_any_result(result):
                has_any_results = True
            turn_log["tool_results"].append(
                {"name": tool_name, "arguments": arguments, "result": result}
            )

            extracted = _extract_urls(result)
            for url in extracted:
                collected_urls.add(url)
                if _is_official_url(url):
                    official_urls.add(url)
            tool_results.append(f"<tool_result name=\"{tool_name}\">\n{result}\n</tool_result>")

            if tool_name == "fetch_url":
                urls = arguments.get("url", [])
                if isinstance(urls, str):
                    urls = [urls]
                for url in urls:
                    collected_urls.add(url)
                    fetched_urls.add(_normalize_url_for_dedup(_normalize_url(url)))
                    if _is_official_url(url):
                        official_urls.add(url)
                        fetched_official_urls.add(url)

        pending_official = [u for u in official_urls if u not in fetched_official_urls]
        remaining = MAX_OFFICIAL_FETCH - len(fetched_official_urls)
        if pending_official and remaining > 0:
            targets = pending_official[:remaining]
            print(f"[Turn {turn + 1}] Forced official URL fetch: {targets}", file=sys.stderr)
            forced_args = {"url": targets, "purpose": "Verify official page content"}
            forced_result = await execute_tool("fetch_url", forced_args)
            if _has_any_result(forced_result):
                has_any_results = True
            for url in targets:
                collected_urls.add(url)
                if _is_official_url(url):
                    official_urls.add(url)
                    fetched_official_urls.add(url)
            extracted = _extract_urls(forced_result)
            for url in extracted:
                collected_urls.add(url)
                if _is_official_url(url):
                    official_urls.add(url)
                    fetched_official_urls.add(url)
            tool_results.append(f"<tool_result name=\"fetch_url\">\n{forced_result}\n</tool_result>")
            turn_log["tool_calls"].append({"name": "fetch_url", "arguments": forced_args})
            turn_log["tool_results"].append(
                {"name": "fetch_url", "arguments": forced_args, "result": forced_result}
            )

        # Add tool results to messages
        messages.append({"role": "user", "content": "\n\n".join(tool_results)})
        log_data["turns"].append(turn_log)

    return finalize("Maximum turns exceeded. Could not generate final answer.")


async def main():
    global CURRENT_DEPTH
    import argparse
    parser = argparse.ArgumentParser(
        description="LocalWebSearch-CDP: API-key-free Local WebSearch",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python search_agent.py "GPT-5 release date"
  python search_agent.py --llm ollama "latest AI news"
  python search_agent.py --llm lmstudio --model qwen3-8b "query"
  python search_agent.py --list-backends
  python search_agent.py -i  # Interactive mode
        """
    )
    parser.add_argument("query", nargs="?", help="Search query")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive mode")
    parser.add_argument("--depth", "-d", choices=["simple", "medium", "deep"], default="medium",
                        help="Search depth: simple(snippets only), medium(fetch top 5), deep(fetch all)")

    # LLM backend options
    parser.add_argument("--llm", choices=["sglang", "ollama", "lmstudio", "openai"],
                        default="sglang", help="LLM backend (default: sglang)")
    parser.add_argument("--llm-url", type=str, help="Custom LLM API URL")
    parser.add_argument("--model", "-m", type=str, help="Model name (backend-specific)")
    parser.add_argument("--list-backends", action="store_true",
                        help="List available LLM backends and exit")

    args = parser.parse_args()

    # List backends mode
    if args.list_backends:
        print("Detecting available LLM backends...\n")
        backends = await list_available_backends()
        if not backends:
            print("No LLM backends detected.")
            print("\nMake sure one of these is running:")
            print("  - SGLang: http://localhost:30001")
            print("  - Ollama: http://localhost:11434")
            print("  - LM Studio: http://localhost:1234")
            return

        for b in backends:
            status = "✅" if b["status"] == "ready" else "❌"
            print(f"{status} {b['name']}: {b['url']}")
            print(f"   Models: {', '.join(b['models'][:5])}")
            if len(b['models']) > 5:
                print(f"   ... and {len(b['models']) - 5} more")
            print(f"   Tool format: {b['tool_format']}")
            print()
        return

    # Initialize LLM adapter
    try:
        await init_llm_adapter(
            backend=args.llm,
            url=args.llm_url,
            model=args.model,
        )
    except Exception as e:
        print(f"[ERROR] Failed to initialize LLM adapter: {e}", file=sys.stderr)
        print(f"[ERROR] Make sure the {args.llm} backend is running.", file=sys.stderr)
        sys.exit(1)

    CURRENT_DEPTH = args.depth
    print(f"[Search Depth] {CURRENT_DEPTH}: {DEPTH_CONFIG[CURRENT_DEPTH]['description']}", file=sys.stderr)

    if args.interactive:
        print(f"LocalWebSearch-CDP ({args.llm}) - exit: quit")
        print("=" * 50)
        while True:
            try:
                query = input("\nQuery: ").strip()
                if query.lower() in ["quit", "exit", "q"]:
                    break
                if not query:
                    continue

                answer = await search_agent(query)
                print(f"\nAnswer:\n{answer}")
            except KeyboardInterrupt:
                break
    elif args.query:
        answer = await search_agent(args.query)
        print(answer)
    else:
        parser.print_help()


if __name__ == "__main__":
    asyncio.run(main())
