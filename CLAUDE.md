# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AgentWebSearch-MCP is an API-key-free local web search system using Chrome DevTools Protocol (CDP) for parallel web searches. Includes MCP server for Claude Code integration.

**Recommended Model**: [AgentCPM-Explore](https://huggingface.co/openbmb/AgentCPM-Explore) - A 4B model from OpenBMB/THUNLP specifically trained for search agent tasks. Generates diverse search queries and handles tool calling optimally.

Supports multi-LLM backends: SGLang (with AgentCPM-Explore), Ollama, LM Studio, OpenAI-compatible.

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

When running as MCP server, use the MCP tools directly. They auto-detect available backends:
- `web_search` - Simple search
- `smart_search` - Search + fetch URLs
- `agent_search` - Full agentic search (auto-detects LLM)

**Do NOT run search_agent.py directly when using MCP.** Just call the MCP tools.

### Run standalone search agent (CLI only)
```bash
# Only for command line usage, not within Claude Code
python search_agent.py "query" --depth deep
python search_agent.py -i                         # Interactive mode
```

## Architecture

```
mcp_server.py          MCP server (4 tools)
       |
       +-- web_search      → CDP search only
       +-- fetch_urls      → URL content fetch
       +-- smart_search    → Search + fetch (depth control)
       +-- agent_search    → LLM + search + fetch (full agent)
       |
search_agent.py        Standalone agent (CLI)
       |
llm_adapters/          Unified LLM interface
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
| `agent_search` | Full agentic search with LLM | Yes |

### agent_search LLM Backends

**Auto-detect**: `agent_search` automatically detects and uses any available backend. No need to specify or start a specific backend manually.

| Backend | Description |
|---------|-------------|
| `auto` | **Default** - Uses first available backend |
| `sglang` | AgentCPM-Explore (if running) |
| `lmstudio` | LM Studio (if running) |
| `ollama` | Ollama (if running) |
| `openai` | Paid API |

**Important**: Just call `agent_search` MCP tool. It will use whatever LLM backend is currently running (including the one running Claude Code itself via LM Studio/Ollama).

## Key Configuration

### mcp_server.py
| Setting | Default | Description |
|---------|---------|-------------|
| `SEARCH_TIMEOUT` | 90s | CDP search timeout |
| `FETCH_TIMEOUT` | 5s | URL fetch timeout |
| `MAX_FETCH_URLS` | 10 | Max URLs per fetch |

### search_agent.py
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

## Adding a New LLM Backend
1. Create adapter in `llm_adapters/` extending `BaseLLMAdapter`
2. Implement: `call()`, `parse_tool_calls()`, `format_tool_result()`
3. Register in `llm_adapters/__init__.py`

## Dependencies

Core: `httpx`, `beautifulsoup4`, `websocket-client`, `mcp`
Optional: `trafilatura`, `readability-lxml`
Backend-specific: `sglang[all]` (CUDA), Ollama, LM Studio
