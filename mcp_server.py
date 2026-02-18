#!/usr/bin/env python3
"""
AgentWebSearch MCP Server

Provides CDP-based web search as MCP tools.
Uses Chrome DevTools Protocol for API-key-free web search.

Usage:
    # stdio mode (Claude Code, etc.)
    python mcp_server.py

    # SSE mode (HTTP server)
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
    print("Error: mcp package required. Run: pip install mcp", file=sys.stderr)
    sys.exit(1)

# CDP search module
try:
    from cdp_search import search_with_cdp
except ImportError:
    print("Error: cdp_search.py required.", file=sys.stderr)
    sys.exit(1)

# LLM Adapter (for agent_search)
try:
    from llm_adapters import get_adapter
    HAS_LLM_ADAPTERS = True
except ImportError:
    HAS_LLM_ADAPTERS = False

# search_agent function (for agent_search)
try:
    import search_agent as sa_module
    HAS_SEARCH_AGENT = True
except ImportError:
    HAS_SEARCH_AGENT = False

# AgentCPM configuration (SGLang only)
AGENTCPM_CONFIG = {
    "url": "http://localhost:30001",
    "model": "AgentCPM-Explore",
    "description": "AgentCPM-Explore 4B model (OpenBMB/THUNLP) - optimized for search tasks",
}

# Configuration
SEARCH_TIMEOUT = 90.0
FETCH_TIMEOUT = 5.0
MAX_FETCH_URLS = 10
MAX_CONTENT_LENGTH = 8000
MIN_CONTENT_LENGTH = 200

# Search depth configuration
DEPTH_CONFIG = {
    "simple": {"fetch_enabled": False, "max_fetch": 0, "description": "snippets only (fast)"},
    "medium": {"fetch_enabled": True, "max_fetch": 5, "description": "fetch top 5 URLs (default)"},
    "deep": {"fetch_enabled": True, "max_fetch": 15, "description": "fetch top 15 URLs (slow)"},
}

# Low quality domain filter
LOW_QUALITY_DOMAINS = {
    "blog.naver.com", "m.blog.naver.com", "post.naver.com",
    "cafe.naver.com", "tistory.com", "brunch.co.kr",
    "medium.com", "reddit.com", "youtube.com", "youtu.be",
}

# Create MCP server
server = Server("agentwebsearch-mcp")


def _normalize_url(url: str) -> str:
    """Normalize URL"""
    cleaned = url.rstrip(".,;]")
    while cleaned.endswith(")") and cleaned.count("(") < cleaned.count(")"):
        cleaned = cleaned[:-1]
    return cleaned


def _is_low_quality(url: str) -> bool:
    """Check if URL is from low quality domain"""
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
    """Deduplicate URLs"""
    seen = set()
    result = []
    for url in urls:
        norm = _normalize_url(url).lower()
        if norm not in seen:
            seen.add(norm)
            result.append(url)
    return result


def _extract_text_basic(html_text: str) -> tuple[str, str]:
    """Basic HTML text extraction"""
    # Remove script/style
    cleaned = re.sub(r"<script[^>]*>.*?</script>", " ", html_text, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r"<style[^>]*>.*?</style>", " ", cleaned, flags=re.DOTALL | re.IGNORECASE)

    # Extract title
    title_match = re.search(r"<title[^>]*>(.*?)</title>", cleaned, re.IGNORECASE | re.DOTALL)
    title = title_match.group(1).strip() if title_match else ""

    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", cleaned)
    text = re.sub(r"\s+", " ", text).strip()

    return title, text


def _simple_clean_html(html_text: str) -> tuple[str, str]:
    """Clean HTML with BeautifulSoup"""
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        return _extract_text_basic(html_text)

    soup = BeautifulSoup(html_text, 'html.parser')

    # Remove unnecessary tags
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
    """Fetch URL content"""
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
    """Execute CDP search"""
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
    """Return MCP tool list"""
    tools = [
        Tool(
            name="web_search",
            description="Perform web search using Chrome DevTools Protocol. Searches Naver, Google, Brave in parallel. No API key required.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query"
                    },
                    "portal": {
                        "type": "string",
                        "enum": ["all", "naver", "google", "brave"],
                        "default": "all",
                        "description": "Search portal (all=parallel search all portals)"
                    },
                    "count": {
                        "type": "integer",
                        "default": 10,
                        "minimum": 1,
                        "maximum": 20,
                        "description": "Number of results"
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="fetch_urls",
            description="Fetch webpage content from URLs. Parses HTML and extracts body text.",
            inputSchema={
                "type": "object",
                "properties": {
                    "urls": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of URLs to fetch (max 10)"
                    },
                    "filter_low_quality": {
                        "type": "boolean",
                        "default": True,
                        "description": "Filter low quality domains (blogs, SNS, etc.)"
                    }
                },
                "required": ["urls"]
            }
        ),
        Tool(
            name="smart_search",
            description="Search + fetch top URLs in one call. Control search depth: simple(snippets only), medium(fetch top 5), deep(fetch top 15).",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query"
                    },
                    "depth": {
                        "type": "string",
                        "enum": ["simple", "medium", "deep"],
                        "default": "medium",
                        "description": "Search depth: simple(snippets only, fast), medium(fetch top 5 URLs, default), deep(fetch top 15 URLs, slow)"
                    },
                    "portal": {
                        "type": "string",
                        "enum": ["all", "naver", "google", "brave"],
                        "default": "all",
                        "description": "Search portal"
                    }
                },
                "required": ["query"]
            }
        ),
    ]

    # Add agentcpm tool (SGLang + AgentCPM-Explore only)
    if HAS_LLM_ADAPTERS and HAS_SEARCH_AGENT:
        tools.append(
            Tool(
                name="agentcpm",
                description="""Agentic search using AgentCPM-Explore model (4B, OpenBMB/THUNLP).
This model is specifically trained for search agent tasks - generates diverse queries and handles tool calling optimally.

**Requires**: SGLang server running with AgentCPM-Explore model on port 30001.
**First run**: Model loading takes ~30-45 seconds.
**Use smart_search instead** if you don't have SGLang/AgentCPM-Explore set up.""",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query"
                        },
                        "depth": {
                            "type": "string",
                            "enum": ["simple", "medium", "deep"],
                            "default": "medium",
                            "description": "Search depth: simple(fast), medium(default), deep(detailed)"
                        },
                        "confirm": {
                            "type": "boolean",
                            "default": False,
                            "description": "Set to true to confirm using AgentCPM-Explore (required if SGLang not running)"
                        }
                    },
                    "required": ["query"]
                }
            )
        )

    return tools


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Execute MCP tool"""

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

        # Format results
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

        # Clean URLs
        urls = _dedup_urls(urls)
        if filter_low:
            urls = [u for u in urls if not _is_low_quality(u)]
        urls = urls[:MAX_FETCH_URLS]

        if not urls:
            return [TextContent(type="text", text="No valid URLs to fetch")]

        # Parallel fetch
        tasks = [_fetch_url_content(url) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Format results
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

        # Get depth config
        depth_cfg = DEPTH_CONFIG.get(depth, DEPTH_CONFIG["medium"])
        max_fetch = depth_cfg["max_fetch"]

        # 1. Search
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

        # 2. Fetch URL content based on depth
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

    elif name == "agentcpm":
        if not HAS_LLM_ADAPTERS or not HAS_SEARCH_AGENT:
            return [TextContent(type="text", text="Error: agentcpm requires llm_adapters and search_agent modules")]

        query = arguments.get("query", "")
        depth = arguments.get("depth", "medium")
        confirm = arguments.get("confirm", False)

        if not query:
            return [TextContent(type="text", text="Error: query is required")]

        # Check if SGLang is running
        sglang_running = False
        try:
            import httpx
            async with httpx.AsyncClient(timeout=3.0) as client:
                resp = await client.get(f"{AGENTCPM_CONFIG['url']}/health")
                sglang_running = resp.status_code == 200
        except Exception:
            pass

        if not sglang_running:
            if not confirm:
                return [TextContent(type="text", text=f"""## AgentCPM-Explore Not Running

SGLang server with AgentCPM-Explore model is not detected on port 30001.

**To use agentcpm:**
1. Start SGLang server: `MODEL_PATH=/path/to/AgentCPM-Explore ./start_sglang.sh`
2. Wait 30-45 seconds for model loading
3. Call agentcpm again with `confirm=true`

**Alternative:** Use `smart_search` instead (no LLM required, works immediately).

Do you want to proceed anyway? Call with `confirm=true` to confirm.""")]
            # User confirmed, but SGLang still not running
            return [TextContent(type="text", text="Error: SGLang server not running. Please start it first with: MODEL_PATH=/path/to/AgentCPM-Explore ./start_sglang.sh")]

        # SGLang is running - proceed with search
        try:
            # Configure search_agent module for SGLang
            sa_module.LLM_BACKEND = "sglang"
            sa_module.LLM_URL = AGENTCPM_CONFIG["url"]
            sa_module.LLM_MODEL = AGENTCPM_CONFIG["model"]
            sa_module.CURRENT_DEPTH = depth

            # Initialize adapter
            sa_module.LLM_ADAPTER = get_adapter("sglang", url=AGENTCPM_CONFIG["url"], model=AGENTCPM_CONFIG["model"])

            # Output header
            output = [
                f"## AgentCPM Search: '{query}'",
                f"**Model**: {AGENTCPM_CONFIG['model']} ({AGENTCPM_CONFIG['description']})",
                f"**Depth**: {depth}",
                "",
                "---",
                ""
            ]

            # Run agent
            result = await sa_module.search_agent(query)
            output.append(result)

            return [TextContent(type="text", text="\n".join(output))]

        except Exception as e:
            return [TextContent(type="text", text=f"Error running agentcpm: {e}")]

    return [TextContent(type="text", text=f"Unknown tool: {name}")]


async def run_stdio():
    """Run in stdio mode"""
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


async def run_sse(port: int = 8902):
    """Run in SSE mode"""
    try:
        from mcp.server.sse import SseServerTransport
        from starlette.applications import Starlette
        from starlette.routing import Route
        import uvicorn
    except ImportError:
        print("Error: SSE mode requires starlette and uvicorn.", file=sys.stderr)
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
