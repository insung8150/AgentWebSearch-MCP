# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AgentWebSearch-MCP is an API-key-free local web search system using Chrome DevTools Protocol (CDP) for parallel web searches. Includes MCP server for Claude Code integration.

**Optional Model**: [AgentCPM-Explore](https://huggingface.co/openbmb/AgentCPM-Explore) - A 4B model from OpenBMB/THUNLP specifically trained for search agent tasks. Required only for the `agentcpm` tool.

## Commands

### Start Chrome instances (required before search)
```bash
python chrome_launcher.py start   # Start 3 Chrome instances (ports 9222-9224)
python chrome_launcher.py status  # Check status
python chrome_launcher.py stop    # Stop all
```

### Run MCP server
```bash
python mcp_server.py              # stdio mode (Claude Code)
python mcp_server.py --sse --port 8902  # SSE mode (HTTP)
```

### Use MCP tools (recommended)

When running as MCP server, use the MCP tools directly:
- `web_search` - Simple search (no LLM required)
- `smart_search` - Search + fetch URLs (no LLM required) **← Use this for most cases**
- `agentcpm` - Agentic search using AgentCPM-Explore via SGLang

**Do NOT run search_agent.py directly when using MCP.** Just call the MCP tools.

### Run standalone search agent (CLI only)
```bash
# Only for command line usage, not within Claude Code
python search_agent.py "query" --depth deep
python search_agent.py -i                         # Interactive mode
python search_agent.py --llm ollama "query"       # CLI supports multiple backends
```

## Architecture

```
mcp_server.py          MCP server (4 tools)
       |
       +-- web_search      → CDP search only
       +-- fetch_urls      → URL content fetch
       +-- smart_search    → Search + fetch (depth control)
       +-- agentcpm        → SGLang + AgentCPM-Explore + search + fetch
       |
search_agent.py        Standalone agent (CLI, supports multiple backends)
       |
llm_adapters/          Unified LLM interface (for CLI)
       |
cdp_search.py          CDP-based parallel portal search
       |
chrome_launcher.py     Chrome instance lifecycle (ports 9222-9224)
```

### MCP Tools

| Tool | Description | LLM Required |
|------|-------------|--------------|
| `web_search` | Parallel search Naver/Google/Brave | No |
| `fetch_urls` | Fetch webpage content | No |
| `smart_search` | Search + fetch with depth control | No |
| `get_search_status` | Check progress and get partial results | No |
| `cancel_search` | Cancel search and get partial results | No |
| `agentcpm` | Agentic search with AgentCPM-Explore | Yes (SGLang) |

### Partial Results Support

When search takes too long:
1. Call `get_search_status` to see progress and partial results
2. Call `cancel_search` to stop and return what's been collected

This is useful when deep searches or URL fetches are slow.

### agentcpm Tool

The `agentcpm` tool exclusively uses **SGLang + AgentCPM-Explore** model:

| Parameter | Values | Description |
|-----------|--------|-------------|
| `query` | string | Search query (required) |
| `depth` | `simple`/`medium`/`deep` | Search depth |
| `confirm` | boolean | Confirm to proceed if SGLang not running |

**Behavior:**
- If SGLang is running on port 30001: Proceeds with agentic search
- If SGLang is not running: Returns guidance message
- With `confirm=true`: Attempts connection anyway (useful if starting SGLang)

**Alternative**: Use `smart_search` for immediate results without LLM requirements.

## Key Configuration

### mcp_server.py
| Setting | Default | Description |
|---------|---------|-------------|
| `SEARCH_TIMEOUT` | 90s | CDP search timeout |
| `FETCH_TIMEOUT` | 5s | URL fetch timeout |
| `MAX_FETCH_URLS` | 10 | Max URLs per fetch |
| `AGENTCPM_CONFIG` | port 30001 | SGLang server URL |

### search_agent.py (CLI)
| Setting | Default | Description |
|---------|---------|-------------|
| `LLM_BACKEND` | "sglang" | Default backend |
| `CURRENT_DEPTH` | "medium" | Default search depth |

## Depth Settings

| Depth | Fetch | Description |
|-------|-------|-------------|
| `simple` | 0 | Snippets only (fast) |
| `medium` | 5 | Top 5 URLs (default) |
| `deep` | 15 | Top 15 URLs (slow) |

## Chrome Instance Ports

| Portal | Port |
|--------|------|
| Naver | 9222 |
| Google | 9223 |
| Brave | 9224 |

## Adding a New Portal
1. Add entry to `PORTAL_CONFIG` in `cdp_search.py`
2. Add matching entry to `CHROME_INSTANCES` in `chrome_launcher.py`

## Adding a New LLM Backend (CLI only)
1. Create adapter in `llm_adapters/` extending `BaseLLMAdapter`
2. Implement: `call()`, `parse_tool_calls()`, `format_tool_result()`
3. Register in `llm_adapters/__init__.py`

Note: The MCP `agentcpm` tool only supports SGLang. Other backends are for CLI use.

## Dependencies

Core: `httpx`, `beautifulsoup4`, `websocket-client`, `mcp`
Optional: `trafilatura`, `readability-lxml`
For agentcpm: `sglang[all]` (CUDA required)
