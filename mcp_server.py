#!/usr/bin/env python3
"""
AgentWebSearch MCP Server

CDP 기반 웹 검색을 MCP 도구로 제공합니다.
Chrome DevTools Protocol을 사용하여 API 키 없이 웹 검색을 수행합니다.

Usage:
    # stdio 모드 (Claude Code 등)
    python mcp_server.py

    # SSE 모드 (HTTP 서버)
    python mcp_server.py --sse --port 8902
"""

import asyncio
import json
import re
import sys
from datetime import datetime
from typing import Any
from urllib.parse import urlparse

try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import Tool, TextContent
except ImportError:
    print("Error: mcp 패키지가 필요합니다. pip install mcp 실행하세요.", file=sys.stderr)
    sys.exit(1)

# CDP 검색 모듈
try:
    from cdp_search import search_with_cdp
except ImportError:
    print("Error: cdp_search.py가 필요합니다.", file=sys.stderr)
    sys.exit(1)

# LLM Adapter (agent_search용)
try:
    from llm_adapters import get_adapter, detect_available_backends
    HAS_LLM_ADAPTERS = True
except ImportError:
    HAS_LLM_ADAPTERS = False

# search_agent 함수 (agent_search용)
try:
    import search_agent as sa_module
    HAS_SEARCH_AGENT = True
except ImportError:
    HAS_SEARCH_AGENT = False

# LLM 백엔드 기본 설정
LLM_BACKENDS = {
    "sglang": {
        "url": "http://localhost:30001/v1",
        "model": "AgentCPM-Explore",
        "description": "SGLang + AgentCPM-Explore (검색 특화 모델, 추천)",
        "recommended": True,
    },
    "ollama": {
        "url": "http://localhost:11434",
        "model": "qwen3:8b",
        "description": "Ollama (로컬 LLM, 범용)",
        "recommended": False,
    },
    "lmstudio": {
        "url": "http://localhost:1234/v1",
        "model": "",
        "description": "LM Studio (로컬 LLM, 범용)",
        "recommended": False,
    },
    "openai": {
        "url": "https://api.openai.com/v1",
        "model": "gpt-4o",
        "description": "OpenAI API (유료)",
        "recommended": False,
    },
}

# 설정
SEARCH_TIMEOUT = 90.0
FETCH_TIMEOUT = 5.0
MAX_FETCH_URLS = 10
MAX_CONTENT_LENGTH = 8000
MIN_CONTENT_LENGTH = 200

# 검색 깊이 설정
DEPTH_CONFIG = {
    "simple": {"fetch_enabled": False, "max_fetch": 0, "description": "snippets만 (빠름)"},
    "medium": {"fetch_enabled": True, "max_fetch": 5, "description": "상위 5개 URL fetch (기본)"},
    "deep": {"fetch_enabled": True, "max_fetch": 15, "description": "상위 15개 URL fetch (느림)"},
}

# 저품질 도메인 필터
LOW_QUALITY_DOMAINS = {
    "blog.naver.com", "m.blog.naver.com", "post.naver.com",
    "cafe.naver.com", "tistory.com", "brunch.co.kr",
    "medium.com", "reddit.com", "youtube.com", "youtu.be",
}

# MCP 서버 생성
server = Server("agentwebsearch-mcp")


def _normalize_url(url: str) -> str:
    """URL 정규화"""
    cleaned = url.rstrip(".,;]")
    while cleaned.endswith(")") and cleaned.count("(") < cleaned.count(")"):
        cleaned = cleaned[:-1]
    return cleaned


def _is_low_quality(url: str) -> bool:
    """저품질 도메인 체크"""
    try:
        domain = urlparse(url).netloc.lower()
        if domain in LOW_QUALITY_DOMAINS:
            return True
        for suffix in LOW_QUALITY_DOMAINS:
            if domain.endswith("." + suffix):
                return True
    except:
        pass
    return False


def _dedup_urls(urls: list[str]) -> list[str]:
    """URL 중복 제거"""
    seen = set()
    result = []
    for url in urls:
        norm = _normalize_url(url).lower()
        if norm not in seen:
            seen.add(norm)
            result.append(url)
    return result


def _extract_text_basic(html_text: str) -> tuple[str, str]:
    """기본 HTML 텍스트 추출"""
    # script/style 제거
    cleaned = re.sub(r"<script[^>]*>.*?</script>", " ", html_text, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r"<style[^>]*>.*?</style>", " ", cleaned, flags=re.DOTALL | re.IGNORECASE)

    # title 추출
    title_match = re.search(r"<title[^>]*>(.*?)</title>", cleaned, re.IGNORECASE | re.DOTALL)
    title = title_match.group(1).strip() if title_match else ""

    # HTML 태그 제거
    text = re.sub(r"<[^>]+>", " ", cleaned)
    text = re.sub(r"\s+", " ", text).strip()

    return title, text


def _simple_clean_html(html_text: str) -> tuple[str, str]:
    """BeautifulSoup으로 HTML 정리"""
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        return _extract_text_basic(html_text)

    soup = BeautifulSoup(html_text, 'html.parser')

    # 불필요한 태그 제거
    for tag in soup(['script', 'style', 'nav', 'header', 'footer', 'aside',
                     'noscript', 'svg', 'iframe', 'form', 'button']):
        tag.decompose()

    title = ""
    title_tag = soup.find('title')
    if title_tag:
        title = title_tag.get_text(strip=True)

    text = soup.get_text(separator='\n', strip=True)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)

    return title, text.strip()


async def _fetch_url_content(url: str) -> dict:
    """URL 내용 가져오기"""
    import httpx

    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/122.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
    }

    try:
        async with httpx.AsyncClient(timeout=FETCH_TIMEOUT, headers=headers, follow_redirects=True) as client:
            response = await client.get(url)
            if response.status_code >= 400:
                return {"url": url, "error": f"HTTP {response.status_code}"}

            html_text = response.text or ""
            title, content = _simple_clean_html(html_text)

            if len(content) < MIN_CONTENT_LENGTH:
                return {"url": url, "error": "Content too short"}

            return {
                "url": url,
                "title": title,
                "content": content[:MAX_CONTENT_LENGTH],
            }
    except Exception as e:
        return {"url": url, "error": str(e)}


async def _search_cdp(query: str, portal: str = "all") -> list[dict]:
    """CDP 검색 실행"""
    try:
        data = await asyncio.to_thread(
            search_with_cdp,
            query,
            portal=portal,
            count=10,
            search_type="web",
            skip_content=True
        )

        if not data.get("success"):
            return []

        results = data.get("data", {}).get("results", [])
        return [
            {
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "snippet": (r.get("snippet") or r.get("content") or "")[:300],
                "source": r.get("source", portal),
            }
            for r in results
        ]
    except Exception as e:
        return []


@server.list_tools()
async def list_tools() -> list[Tool]:
    """MCP 도구 목록 반환"""
    tools = [
        Tool(
            name="web_search",
            description="웹 검색을 수행합니다. Chrome DevTools Protocol을 사용하여 Naver, Google, Brave에서 병렬 검색합니다. API 키 불필요.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "검색 쿼리"
                    },
                    "portal": {
                        "type": "string",
                        "enum": ["all", "naver", "google", "brave"],
                        "default": "all",
                        "description": "검색 포털 (all=모두 병렬 검색)"
                    },
                    "count": {
                        "type": "integer",
                        "default": 10,
                        "minimum": 1,
                        "maximum": 20,
                        "description": "결과 개수"
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="fetch_urls",
            description="URL 목록에서 웹페이지 내용을 가져옵니다. HTML을 파싱하여 본문 텍스트를 추출합니다.",
            inputSchema={
                "type": "object",
                "properties": {
                    "urls": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "가져올 URL 목록 (최대 10개)"
                    },
                    "filter_low_quality": {
                        "type": "boolean",
                        "default": True,
                        "description": "저품질 도메인(블로그, SNS 등) 필터링"
                    }
                },
                "required": ["urls"]
            }
        ),
        Tool(
            name="smart_search",
            description="검색 + 주요 URL 내용 가져오기를 한 번에 수행합니다. depth로 검색 깊이를 조절합니다: simple(snippets만), medium(상위5개 fetch), deep(상위15개 fetch).",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "검색 쿼리"
                    },
                    "depth": {
                        "type": "string",
                        "enum": ["simple", "medium", "deep"],
                        "default": "medium",
                        "description": "검색 깊이: simple(snippets만, 빠름), medium(상위 5개 URL fetch, 기본), deep(상위 15개 URL fetch, 느림)"
                    },
                    "portal": {
                        "type": "string",
                        "enum": ["all", "naver", "google", "brave"],
                        "default": "all",
                        "description": "검색 포털"
                    }
                },
                "required": ["query"]
            }
        ),
    ]

    # agent_search 도구 추가 (LLM 어댑터가 있을 때만)
    if HAS_LLM_ADAPTERS and HAS_SEARCH_AGENT:
        tools.append(
            Tool(
                name="agent_search",
                description="""로컬 LLM을 사용한 에이전틱 검색입니다. LLM이 검색 계획을 수립하고, 검색 실행, 결과 분석, 답변 생성까지 자동으로 수행합니다.

**추천**: llm=sglang (AgentCPM-Explore 모델, 검색에 특화되어 학습됨)
**대안**: ollama, lmstudio 등 기존 사용 중인 모델도 선택 가능""",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "검색 쿼리"
                        },
                        "llm": {
                            "type": "string",
                            "enum": ["sglang", "ollama", "lmstudio", "openai"],
                            "default": "sglang",
                            "description": "LLM 백엔드. sglang(추천, AgentCPM-Explore 검색특화), ollama(범용), lmstudio(범용), openai(유료)"
                        },
                        "model": {
                            "type": "string",
                            "default": "",
                            "description": "모델 이름 (비우면 백엔드 기본값 사용). 예: qwen3:8b, gpt-4o"
                        },
                        "depth": {
                            "type": "string",
                            "enum": ["simple", "medium", "deep"],
                            "default": "medium",
                            "description": "검색 깊이: simple(빠름), medium(기본), deep(상세)"
                        }
                    },
                    "required": ["query"]
                }
            )
        )

    return tools


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """MCP 도구 실행"""

    if name == "web_search":
        query = arguments.get("query", "")
        portal = arguments.get("portal", "all")
        count = arguments.get("count", 10)

        if not query:
            return [TextContent(type="text", text="Error: query is required")]

        try:
            results = await asyncio.wait_for(
                _search_cdp(query, portal),
                timeout=SEARCH_TIMEOUT
            )
        except asyncio.TimeoutError:
            return [TextContent(type="text", text=f"Search timeout ({SEARCH_TIMEOUT}s)")]

        if not results:
            return [TextContent(type="text", text=f"No results for '{query}'")]

        # 결과 포맷팅
        output = [f"## Search Results for '{query}'\n"]
        for i, r in enumerate(results[:count], 1):
            output.append(f"{i}. **{r['title']}**")
            output.append(f"   URL: {r['url']}")
            output.append(f"   ({r['source']}) {r['snippet']}\n")

        return [TextContent(type="text", text="\n".join(output))]

    elif name == "fetch_urls":
        urls = arguments.get("urls", [])
        filter_low = arguments.get("filter_low_quality", True)

        if not urls:
            return [TextContent(type="text", text="Error: urls is required")]

        # URL 정리
        urls = _dedup_urls(urls)
        if filter_low:
            urls = [u for u in urls if not _is_low_quality(u)]
        urls = urls[:MAX_FETCH_URLS]

        if not urls:
            return [TextContent(type="text", text="No valid URLs to fetch")]

        # 병렬 fetch
        tasks = [_fetch_url_content(url) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 결과 포맷팅
        output = ["## Fetched Content\n"]
        for r in results:
            if isinstance(r, Exception):
                output.append(f"Error: {r}\n")
            elif "error" in r:
                output.append(f"**{r['url']}**\nError: {r['error']}\n")
            else:
                output.append(f"**{r.get('title', 'No title')}**")
                output.append(f"URL: {r['url']}")
                output.append(f"\n{r['content']}\n")
                output.append("---\n")

        return [TextContent(type="text", text="\n".join(output))]

    elif name == "smart_search":
        query = arguments.get("query", "")
        depth = arguments.get("depth", "medium")
        portal = arguments.get("portal", "all")

        if not query:
            return [TextContent(type="text", text="Error: query is required")]

        # depth 설정 가져오기
        depth_cfg = DEPTH_CONFIG.get(depth, DEPTH_CONFIG["medium"])
        max_fetch = depth_cfg["max_fetch"]

        # 1. 검색
        try:
            results = await asyncio.wait_for(
                _search_cdp(query, portal),
                timeout=SEARCH_TIMEOUT
            )
        except asyncio.TimeoutError:
            return [TextContent(type="text", text=f"Search timeout ({SEARCH_TIMEOUT}s)")]

        if not results:
            return [TextContent(type="text", text=f"No results for '{query}'")]

        output = [f"## Smart Search: '{query}' (depth={depth})\n"]
        output.append("### Search Results\n")
        for i, r in enumerate(results[:10], 1):
            output.append(f"{i}. **{r['title']}**")
            output.append(f"   URL: {r['url']}")
            output.append(f"   ({r['source']}) {r['snippet']}\n")

        # 2. depth에 따라 URL 내용 가져오기
        if depth_cfg["fetch_enabled"] and max_fetch > 0:
            urls_to_fetch = [r['url'] for r in results[:max_fetch] if not _is_low_quality(r['url'])]

            if urls_to_fetch:
                output.append(f"\n### Detailed Content ({len(urls_to_fetch)} URLs)\n")
                tasks = [_fetch_url_content(url) for url in urls_to_fetch]
                fetched = await asyncio.gather(*tasks, return_exceptions=True)

                for r in fetched:
                    if isinstance(r, Exception):
                        continue
                    if not isinstance(r, dict) or "error" in r:
                        continue
                    output.append(f"**{r.get('title', 'No title')}**")
                    output.append(f"URL: {r['url']}")
                    output.append(f"\n{r['content']}\n")
                    output.append("---\n")
        else:
            output.append("\n*[simple mode: snippets only, no URL fetch]*\n")

        return [TextContent(type="text", text="\n".join(output))]

    elif name == "agent_search":
        if not HAS_LLM_ADAPTERS or not HAS_SEARCH_AGENT:
            return [TextContent(type="text", text="Error: agent_search requires llm_adapters and search_agent modules")]

        query = arguments.get("query", "")
        llm = arguments.get("llm", "sglang")
        model = arguments.get("model", "")
        depth = arguments.get("depth", "medium")

        if not query:
            return [TextContent(type="text", text="Error: query is required")]

        # LLM 백엔드 설정
        backend_cfg = LLM_BACKENDS.get(llm, LLM_BACKENDS["sglang"])
        url = backend_cfg["url"]
        model_name = model if model else backend_cfg["model"]

        # 백엔드 사용 가능 여부 확인
        try:
            available = detect_available_backends()
            if llm not in available:
                available_list = ", ".join(available) if available else "none"
                return [TextContent(type="text", text=f"Error: {llm} backend not available. Available: {available_list}")]
        except Exception as e:
            pass  # 확인 실패해도 일단 진행

        # search_agent 모듈 설정 변경
        try:
            # 전역 설정 변경
            sa_module.LLM_BACKEND = llm
            sa_module.LLM_URL = url
            sa_module.LLM_MODEL = model_name
            sa_module.CURRENT_DEPTH = depth

            # 어댑터 초기화
            sa_module.LLM_ADAPTER = get_adapter(llm, url=url, model=model_name)

            # 출력 헤더
            output = [
                f"## Agent Search: '{query}'",
                f"**Backend**: {llm} ({backend_cfg['description']})",
                f"**Model**: {model_name or '(default)'}",
                f"**Depth**: {depth}",
                "",
                "---",
                ""
            ]

            # 에이전트 실행
            result = await sa_module.search_agent(query)
            output.append(result)

            return [TextContent(type="text", text="\n".join(output))]

        except Exception as e:
            return [TextContent(type="text", text=f"Error running agent_search: {e}")]

    return [TextContent(type="text", text=f"Unknown tool: {name}")]


async def run_stdio():
    """stdio 모드로 실행"""
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


async def run_sse(port: int = 8902):
    """SSE 모드로 실행"""
    try:
        from mcp.server.sse import SseServerTransport
        from starlette.applications import Starlette
        from starlette.routing import Route
        import uvicorn
    except ImportError:
        print("Error: SSE 모드에는 starlette, uvicorn이 필요합니다.", file=sys.stderr)
        sys.exit(1)

    sse = SseServerTransport("/messages")

    async def handle_sse(request):
        async with sse.connect_sse(request.scope, request.receive, request._send) as streams:
            await server.run(streams[0], streams[1], server.create_initialization_options())

    async def handle_messages(request):
        await sse.handle_post_message(request.scope, request.receive, request._send)

    app = Starlette(
        routes=[
            Route("/sse", endpoint=handle_sse),
            Route("/messages", endpoint=handle_messages, methods=["POST"]),
        ]
    )

    print(f"SSE server starting on http://127.0.0.1:{port}", file=sys.stderr)
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    server_instance = uvicorn.Server(config)
    await server_instance.serve()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="AgentWebSearch MCP Server")
    parser.add_argument("--sse", action="store_true", help="SSE 모드로 실행")
    parser.add_argument("--port", type=int, default=8902, help="SSE 포트 (기본: 8902)")
    args = parser.parse_args()

    if args.sse:
        asyncio.run(run_sse(args.port))
    else:
        asyncio.run(run_stdio())


if __name__ == "__main__":
    main()
