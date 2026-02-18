# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LocalWebSearch-CDP is an API-key-free local web search system using Chrome DevTools Protocol (CDP) for parallel web searches with multi-LLM backend support (SGLang, Ollama, LM Studio, OpenAI-compatible).

## Commands

### Start Chrome instances (required before search)
```bash
python chrome_launcher.py start   # Start 3 Chrome instances (ports 9222-9224)
python chrome_launcher.py status  # Check status
python chrome_launcher.py stop    # Stop all
```

### Run search agent
```bash
# Default (SGLang backend)
python search_agent.py "query"

# Select backend
python search_agent.py --llm ollama "query"
python search_agent.py --llm lmstudio --model qwen3-8b "query"

# Search depth
python search_agent.py "query" --depth simple   # snippets only
python search_agent.py "query" --depth medium   # fetch top 5 URLs (default)
python search_agent.py "query" --depth deep     # fetch all URLs

# Interactive mode
python search_agent.py -i

# List backends
python search_agent.py --list-backends
```

### Start LLM backends
```bash
# SGLang (recommended)
./start_sglang.sh

# Ollama
ollama serve

# LM Studio - start via GUI
```

### Utilities
```bash
python clear_tabs.py  # Close excess tabs in all Chrome instances
```

## Architecture

```
search_agent.py        Main agent loop (multi-turn tool-use pattern)
       |
llm_adapters/          Unified LLM interface
  base.py              BaseLLMAdapter abstract class, ToolCall, LLMResponse
  sglang_adapter.py    <tool_call> text format parsing
  ollama_adapter.py    Function calling + text fallback
  lmstudio_adapter.py  OpenAI-compatible
  openai_adapter.py    OpenAI/Azure/proxy
       |
cdp_search.py          CDP-based parallel portal search
       |
chrome_launcher.py     Chrome instance lifecycle (ports 9222-9224)
```

### Key Flow
1. Query preprocessing: Korean foreign words → English hints (FOREIGN_WORD_MAP)
2. LLM adapter calls model with tools (search, fetch_url)
3. Tool calls parsed: `<tool_call>JSON</tool_call>` (SGLang) or function_calling
4. CDP search executes parallel queries across Naver/Google/Brave
5. Results appended to conversation, loop until answer or max turns
6. Final answer extracted with sources, logged to `outputs/search_agent_logs/`

### Adding a New Portal
1. Add entry to `PORTAL_CONFIG` in `cdp_search.py` (URL template + JS extraction script)
2. Add matching entry to `CHROME_INSTANCES` in `chrome_launcher.py` (port + profile path)

### Adding a New LLM Backend
1. Create adapter in `llm_adapters/` extending `BaseLLMAdapter`
2. Implement: `call()`, `parse_tool_calls()`, `format_tool_result()`
3. Register in `llm_adapters/__init__.py`

## Key Configuration (search_agent.py)

| Setting | Default | Description |
|---------|---------|-------------|
| `LLM_BACKEND` | "sglang" | Default backend |
| `SEARCH_TIMEOUT` | 90s | CDP search timeout |
| `FETCH_TIMEOUT` | 3s | URL fetch timeout |
| `MAX_SEARCH_QUERIES` | 3 | Queries per search() call |
| `CURRENT_DEPTH` | "medium" | Default search depth |

## Tool Call Formats

### SGLang (text format)
```
<tool_call>
{"name": "search", "arguments": {"query": ["query1", "query2"]}}
</tool_call>
```

### Ollama/LM Studio/OpenAI (function calling)
Standard OpenAI function_calling format with `tool_calls` in response.

## Chrome Instance Ports

| Portal | Port | Profile |
|--------|------|---------|
| Naver | 9222 | /tmp/chrome-naver-profile |
| Google | 9223 | /tmp/chrome-google-profile |
| Brave | 9224 | /tmp/chrome-brave-profile |

## Dependencies

Core: `httpx`, `beautifulsoup4`
Optional: `trafilatura`, `readability-lxml` (enhanced content extraction)
Backend-specific: `sglang[all]` (requires CUDA), Ollama, LM Studio installed separately
