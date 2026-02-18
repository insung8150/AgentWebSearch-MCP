#!/usr/bin/env python3
"""
CDP-based Portal Search Module (v3.1 - 3 Chrome Parallel)

Core Principles:
1. 3 Independent Chrome instances - dedicated instance per portal
2. Session persistence - cookies/login maintained via user-data-dir
3. True parallel - ThreadPoolExecutor for concurrent search
4. Direct CDP - simplified without MCP dependency
"""

import json
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Optional
import httpx

from chrome_launcher import CHROME_INSTANCES, is_chrome_running, start_chrome

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Search result"""
    title: str
    url: str
    snippet: str
    source: str  # naver, google, brave


# ========== Portal Configuration ==========
#
# [Portal Addition Guide - Claude Code can read this and add automatically]
#
# To add a new portal:
# 1. Add new entry to PORTAL_CONFIG below
# 2. Add matching key to CHROME_INSTANCES in chrome_launcher.py
#
# Required information:
# - search_url: Search URL (query appended at end)
# - extract_script: JavaScript to extract search results
#   - Use document.querySelectorAll() to select link elements
#   - Extract title, url, snippet
#   - Exclude portal's own domain (filter self-links)
#   - Return array (max 10 items)
#
# Example - Adding Bing:
# "bing": {
#     "search_url": "https://www.bing.com/search?q=",
#     "extract_script": """
#         const results = [];
#         document.querySelectorAll('#b_results .b_algo').forEach(el => {
#             const titleEl = el.querySelector('h2 a');
#             const snippetEl = el.querySelector('.b_caption p');
#             if (titleEl) {
#                 results.push({
#                     title: titleEl.textContent.trim(),
#                     url: titleEl.href,
#                     snippet: snippetEl ? snippetEl.textContent.trim() : ''
#                 });
#             }
#         });
#         return results.slice(0, 10);
#     """
# },
#
# Example - Adding DuckDuckGo:
# "duckduckgo": {
#     "search_url": "https://duckduckgo.com/?q=",
#     "extract_script": """
#         const results = [];
#         document.querySelectorAll('[data-testid="result"]').forEach(el => {
#             const titleEl = el.querySelector('a[data-testid="result-title-a"]');
#             const snippetEl = el.querySelector('[data-result="snippet"]');
#             if (titleEl) {
#                 results.push({
#                     title: titleEl.textContent.trim(),
#                     url: titleEl.href,
#                     snippet: snippetEl ? snippetEl.textContent.trim() : ''
#                 });
#             }
#         });
#         return results.slice(0, 10);
#     """
# },
#
# Notes:
# - extract_script varies based on portal's HTML structure
# - Selectors may change when portal updates
# - Use indexOf() instead of includes() for CDP compatibility
# - Use ternary operators instead of optional chaining (?.)
#

PORTAL_CONFIG = {
    "naver": {
        "search_url": "https://search.naver.com/search.naver?query=",  # Integrated search
        "extract_script": """
            const results = [];
            document.querySelectorAll('#main_pack a[href^="http"]').forEach(a => {
                const href = a.href;
                const text = a.textContent ? a.textContent.trim() : '';
                if (href &&
                    href.indexOf('naver.com') === -1 &&
                    href.indexOf('javascript') === -1 &&
                    text && text.length > 15 && text.length < 200) {
                    if (!results.find(r => r.url === href)) {
                        results.push({
                            title: text.substring(0, 80),
                            url: href,
                            snippet: ''
                        });
                    }
                }
            });
            return results.slice(0, 10);
        """
    },
    "google": {
        "search_url": "https://www.google.com/search?q=",
        "extract_script": """
            const results = [];
            document.querySelectorAll('#search .g, #rso .g, div[data-hveid]').forEach(el => {
                const titleEl = el.querySelector('h3');
                const linkEl = el.querySelector('a[href^="http"]');
                const snippetEl = el.querySelector('[data-sncf], .VwiC3b');
                if (titleEl && linkEl) {
                    const url = linkEl.href;
                    if (url.indexOf('google.com') === -1) {
                        results.push({
                            title: titleEl.textContent ? titleEl.textContent.trim() : '',
                            url: url,
                            snippet: snippetEl ? (snippetEl.textContent || '').substring(0, 150) : ''
                        });
                    }
                }
            });
            return results.slice(0, 10);
        """
    },
    "brave": {
        "search_url": "https://search.brave.com/search?q=",
        "extract_script": """
            const results = [];
            const seen = new Set();
            document.querySelectorAll('a[href^="http"]').forEach(a => {
                const href = a.href;
                if (href &&
                    href.indexOf('brave.com') === -1 &&
                    href.indexOf('javascript') === -1 &&
                    !seen.has(href)) {
                    const text = a.textContent ? a.textContent.trim() : '';
                    if (text && text.length > 10 && text.length < 150) {
                        seen.add(href);
                        results.push({
                            title: text.substring(0, 80),
                            url: href,
                            snippet: ''
                        });
                    }
                }
            });
            return results.slice(0, 10);
        """
    }
}

# Stealth script - bypass browser automation detection
STEALTH_SCRIPT = """
Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
Object.defineProperty(navigator, 'plugins', { get: () => [1, 2, 3, 4, 5] });
Object.defineProperty(navigator, 'languages', { get: () => ['ko-KR', 'ko', 'en-US', 'en'] });
window.chrome = { runtime: {} };
"""

# CAPTCHA detection script
CAPTCHA_DETECT_SCRIPT = """
    var indicators = [];

    // hCaptcha
    if (document.querySelector('iframe[src*="hcaptcha"]') ||
        document.querySelector('.h-captcha') ||
        document.querySelector('#hcaptcha')) {
        indicators.push('hcaptcha');
    }

    // Cloudflare Turnstile
    if (document.querySelector('iframe[src*="challenges.cloudflare"]') ||
        document.querySelector('.cf-turnstile')) {
        indicators.push('turnstile');
    }

    // reCAPTCHA
    if (document.querySelector('iframe[src*="recaptcha"]') ||
        document.querySelector('.g-recaptcha')) {
        indicators.push('recaptcha');
    }

    // Text-based detection
    var bodyText = document.body ? document.body.innerText.toLowerCase() : '';
    if (bodyText.indexOf('verify you are human') !== -1 ||
        bodyText.indexOf('are you a robot') !== -1 ||
        bodyText.indexOf('security check') !== -1 ||
        bodyText.indexOf('please verify') !== -1 ||
        bodyText.indexOf('captcha') !== -1) {
        indicators.push('text_hint');
    }

    // URL-based detection
    if (window.location.href.indexOf('challenge') !== -1 ||
        window.location.href.indexOf('captcha') !== -1) {
        indicators.push('url_hint');
    }

    return indicators;
"""


# ========== CDP Direct Communication ==========

class CDPClient:
    """CDP Direct Communication Client (Simplified)"""

    def __init__(self, port: int):
        self.port = port
        self.base_url = f"http://localhost:{port}"
        self.ws_url: Optional[str] = None
        self.page_id: Optional[str] = None

    def connect(self) -> bool:
        """Connect to webpage tab (excluding extensions)"""
        try:
            resp = httpx.get(f"{self.base_url}/json/list", timeout=5)
            tabs = resp.json()

            # Find actual webpage tabs starting with http/https
            for tab in tabs:
                url = tab.get("url", "")
                if url.startswith("http://") or url.startswith("https://"):
                    self.ws_url = tab.get("webSocketDebuggerUrl")
                    self.page_id = tab.get("id")
                    return True

            # If no webpage tab, use first tab
            if tabs:
                self.ws_url = tabs[0].get("webSocketDebuggerUrl")
                self.page_id = tabs[0].get("id")
                return True

            return False
        except Exception as e:
            logger.error(f"[CDP:{self.port}] Connection failed: {e}")
            return False

    def navigate(self, url: str, wait_time: float = 3.0) -> bool:
        """Navigate to page (using CDP Protocol with stealth script injection)"""
        try:
            import websocket
            import json as js

            if not self.ws_url:
                self.connect()

            ws = websocket.create_connection(self.ws_url, timeout=10)

            # Inject stealth script (runs before page load)
            ws.send(js.dumps({
                "id": 1,
                "method": "Page.addScriptToEvaluateOnNewDocument",
                "params": {"source": STEALTH_SCRIPT}
            }))
            ws.recv()  # Wait for response

            # Page.navigate
            ws.send(js.dumps({
                "id": 2,
                "method": "Page.navigate",
                "params": {"url": url}
            }))

            # Wait for response
            resp = ws.recv()
            ws.close()

            # Wait for page load
            time.sleep(wait_time)

            # Reconnect (ws_url may change after navigate)
            self.ws_url = None
            self.connect()

            return True

        except Exception as e:
            logger.error(f"[CDP:{self.port}] Navigate failed: {e}")
            return False

    def evaluate(self, script: str) -> any:
        """Execute JavaScript"""
        try:
            import websocket
            import json as js

            if not self.ws_url:
                self.connect()

            ws = websocket.create_connection(self.ws_url, timeout=10)

            # Runtime.evaluate (wrapped in IIFE)
            expression = f"(function() {{{script}}})()"
            ws.send(js.dumps({
                "id": 1,
                "method": "Runtime.evaluate",
                "params": {
                    "expression": expression,
                    "returnByValue": True
                }
            }))

            resp = js.loads(ws.recv())
            ws.close()

            # Error check
            if "error" in resp:
                logger.error(f"[CDP:{self.port}] Evaluate error: {resp['error']}")
                return None

            result = resp.get("result", {}).get("result", {}).get("value")
            logger.debug(f"[CDP:{self.port}] Evaluate result: {type(result)} - {str(result)[:100] if result else 'None'}")
            return result

        except Exception as e:
            logger.error(f"[CDP:{self.port}] Evaluate failed: {e}")
            return None


# ========== Search Functions ==========

def _search_portal(portal: str, keyword: str) -> list[SearchResult]:
    """
    Single portal search (uses independent Chrome instance)
    """
    config = PORTAL_CONFIG.get(portal)
    chrome_config = CHROME_INSTANCES.get(portal)

    if not config or not chrome_config:
        return []

    port = chrome_config["port"]

    # Check if Chrome is running
    if not is_chrome_running(port):
        logger.warning(f"[{portal}] Chrome not running, attempting to start...")
        if not start_chrome(portal):
            logger.error(f"[{portal}] Failed to start Chrome")
            return []
        time.sleep(2)

    try:
        start_time = time.time()

        # CDP client
        client = CDPClient(port)
        if not client.connect():
            return []

        # Navigate to search URL
        import urllib.parse
        search_url = config["search_url"] + urllib.parse.quote(keyword)
        client.navigate(search_url, wait_time=3.5)

        # Extract results
        raw_result = client.evaluate(config["extract_script"])
        results = _parse_results(raw_result, portal)

        elapsed = time.time() - start_time

        # If 0 results, detect CAPTCHA
        if len(results) == 0:
            captcha_indicators = client.evaluate(CAPTCHA_DETECT_SCRIPT)
            if captcha_indicators and len(captcha_indicators) > 0:
                logger.warning(f"[{portal}] CAPTCHA detected: {captcha_indicators} - manual resolution required in browser")
                print(f"\n[{portal.upper()}] CAPTCHA detected! Please solve it in browser (port {port}).", file=__import__('sys').stderr)
            else:
                logger.info(f"[{portal}] 0 results (not CAPTCHA, no search results)")

        logger.info(f"[{portal}] {len(results)} results ({elapsed:.1f}s)")

        return results

    except Exception as e:
        logger.error(f"[{portal}] Search failed: {e}")
        return []


def _parse_results(raw_result: any, source: str) -> list[SearchResult]:
    """Parse results"""
    if not raw_result:
        return []

    try:
        items = raw_result if isinstance(raw_result, list) else []
        return [
            SearchResult(
                title=item.get("title", ""),
                url=item.get("url", ""),
                snippet=item.get("snippet", ""),
                source=source
            )
            for item in items
            if item.get("url")
        ]
    except Exception as e:
        logger.debug(f"[{source}] Parse failed: {e}")
        return []


def search_parallel(
    keyword: str,
    portals: list[str] = None,
) -> list[SearchResult]:
    """
    Parallel search (3 Chrome instances simultaneously)

    Args:
        keyword: Search query
        portals: List of portals to search (default: ["naver", "google", "brave"])

    Returns:
        Search results from all portals
    """
    if portals is None:
        portals = ["naver", "google", "brave"]

    logger.info(f"[CDP] Starting parallel search: {keyword} ({portals})")
    start_time = time.time()

    all_results = []

    # True parallel execution with ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {
            executor.submit(_search_portal, portal, keyword): portal
            for portal in portals
        }

        for future in as_completed(futures):
            portal = futures[future]
            try:
                results = future.result()
                all_results.extend(results)
            except Exception as e:
                logger.error(f"[{portal}] Exception: {e}")

    total_elapsed = time.time() - start_time
    logger.info(f"[CDP] Search complete: {len(all_results)} results ({total_elapsed:.1f}s)")

    return all_results


# ========== SmartCrawl Compatible Interface ==========

def search_with_cdp(
    keyword: str,
    portal: str = "all",
    count: int = 10,
    search_type: str = "news",
    skip_content: bool = True
) -> dict:
    """
    CDP-based search (SmartCrawl compatible interface)
    """
    try:
        if portal == "all":
            portals = ["naver", "google", "brave"]
        else:
            portals = [portal]

        results = search_parallel(keyword, portals)

        data = {
            "results": [
                {
                    "url": r.url,
                    "title": r.title,
                    "snippet": r.snippet,
                    "source": r.source
                }
                for r in results
            ],
            "skip_content": skip_content
        }

        return {
            "success": True,
            "data": data,
            "count": len(results)
        }

    except Exception as e:
        logger.error(f"[CDP] Search error: {e}")
        return {"success": False, "error": str(e)}


# ========== Test ==========

if __name__ == "__main__":
    import sys

    keyword = sys.argv[1] if len(sys.argv) > 1 else "Samsung stock price"
    portal = sys.argv[2] if len(sys.argv) > 2 else "all"

    print(f"\n=== CDP Parallel Search Test (3 Chrome instances) ===")
    print(f"Keyword: {keyword}")
    print(f"Portal: {portal}\n")

    result = search_with_cdp(keyword, portal)

    if result.get("success"):
        print(f"\n[Success] {result['count']} results")
        for item in result["data"]["results"][:5]:
            print(f"  [{item['source']}] {item['title'][:40]}...")
            print(f"       {item['url'][:60]}...")
    else:
        print(f"\n[Failed] {result.get('error')}")
