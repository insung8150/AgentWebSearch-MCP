# AgentWebSearch-MCP

**Local WebSearch without API keys + MCP Server**

Need WebSearch for your local LLM (Ollama, LM Studio) or Claude Code?
No API keys needed - **just Chrome**.

## How It Works

> **This tool launches REAL Chrome browsers** - not headless, not simulated.
>
> It opens actual Chrome windows and controls them via CDP (Chrome DevTools Protocol).
> This is why it can bypass bot detection and CAPTCHA - because it IS a real browser.

```
┌─────────────────────────────────────────────────────────┐
│  Your Desktop                                           │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐       │
│  │ Chrome:9222 │ │ Chrome:9223 │ │ Chrome:9224 │       │
│  │   Naver     │ │   Google    │ │   Brave     │       │
│  │  (search)   │ │  (search)   │ │  (search)   │       │
│  └──────┬──────┘ └──────┬──────┘ └──────┬──────┘       │
│         │               │               │               │
│         └───────────────┼───────────────┘               │
│                         │ CDP (WebSocket)               │
│                    ┌────┴────┐                          │
│                    │   MCP   │                          │
│                    │ Server  │                          │
│                    └────┬────┘                          │
│                         │                               │
│                    Claude Code                          │
└─────────────────────────────────────────────────────────┘
```

**Why real browsers?**
- No API keys needed (you're just browsing)
- No rate limits (normal browser behavior)
- CAPTCHA resistant (real browser fingerprint)
- Login sessions persist (cookies saved)
- Korean portals work (Naver requires real browser)

## Key Benefits

| Feature | Tavily/Brave API | AgentWebSearch-MCP |
|---------|------------------|---------------------|
| API Key | **Required** | **Not needed** |
| Cost | Paid/Limited | **Free** |
| Setup | Complex | **Just install Chrome** |
| Korean Portals | Limited | **Naver supported** |
| MCP Support | ❌ | **✅ Built-in** |

## Features

- **No API keys** - Direct Chrome CDP (DevTools Protocol) control
- **MCP Server** - Use as Claude Code / Cursor / LM Studio tool
- **AgentCPM-Explore** - Search-optimized 4B model from OpenBMB/THUNLP (optional)
- **Parallel search** - 3 Chrome instances for Naver/Google/Brave simultaneously
- **Bot detection bypass** - Session persistence + stealth flags
- **CAPTCHA resistant** - Real browser sessions avoid most CAPTCHA challenges

## Installation

### 1. Install dependencies
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Start Chrome (debugging mode)
```bash
python chrome_launcher.py start   # Start 3 Chrome instances
python chrome_launcher.py status  # Check status
python chrome_launcher.py stop    # Stop all
```

## MCP Server

Use web search as an MCP tool in Claude Code, Cursor, LM Studio, or [OpenClaw/Moltbot](https://github.com/nicepkg/openclaw).

### Quick Start
```bash
# Start Chrome first
python chrome_launcher.py start

# Run MCP server (stdio mode)
python mcp_server.py
```

### Claude Code Registration
```bash
claude mcp add agentwebsearch -s user -- python /path/to/mcp_server.py
```

### Claude Code settings.json
```json
{
  "mcpServers": {
    "agentwebsearch": {
      "command": "python",
      "args": ["/path/to/AgentWebSearch-MCP/mcp_server.py"],
      "env": {}
    }
  }
}
```

### OpenClaw/Moltbot Registration
```bash
# Add MCP to OpenClaw
openclaw mcp add agentwebsearch -- python /path/to/mcp_server.py

# Or add to ~/.openclaw/config.json
```

### SSE Mode (HTTP Server)
```bash
python mcp_server.py --sse --port 8902
```

### Available MCP Tools

| Tool | Description | LLM Required |
|------|-------------|--------------|
| `web_search` | Search Naver/Google/Brave in parallel | No |
| `fetch_urls` | Fetch webpage content from URLs | No |
| `smart_search` | Search + auto-fetch with depth control | No |
| `get_search_status` | Check search progress and get partial results | No |
| `cancel_search` | Cancel ongoing search and get partial results | No |
| `agentcpm` | Agentic search with AgentCPM-Explore (SGLang) | Yes |

### Tool Parameters

#### `smart_search`
| Parameter | Values | Description |
|-----------|--------|-------------|
| `query` | string | Search query (required) |
| `depth` | `simple` | Snippets only (fast) |
| | `medium` | Fetch top 5 URLs (default) |
| | `deep` | Fetch top 15 URLs (slow) |
| `portal` | `all`/`naver`/`google`/`brave` | Search portal |

#### `agentcpm`

Uses **AgentCPM-Explore** model (4B, OpenBMB/THUNLP) via SGLang to plan search queries, execute searches, and generate answers.

| Parameter | Values | Description |
|-----------|--------|-------------|
| `query` | string | Search query (required) |
| `depth` | `simple`/`medium`/`deep` | Search depth (default: medium) |
| `confirm` | boolean | Confirm to proceed if SGLang not running |

**Requirements:**
- SGLang server running on port 30001
- AgentCPM-Explore model loaded

**First time setup:** Model loading takes ~30-45 seconds. Use `smart_search` if you don't have SGLang set up.

#### `get_search_status` / `cancel_search`

**Partial results support**: When search takes too long, you can:
1. Call `get_search_status` to check progress and see results collected so far
2. Call `cancel_search` to stop the search and return partial results

| Tool | Description |
|------|-------------|
| `get_search_status` | Returns: status, progress %, elapsed time, partial search results, partial fetched contents |
| `cancel_search` | Cancels ongoing search and returns all partial results collected |

> **Why AgentCPM-Explore?** Trained specifically for search agent tasks by OpenBMB/THUNLP. Generates diverse queries and handles tool calling better than general-purpose models.

### Example Usage
```
"Search latest AI news" → smart_search tool
"Deep search about GPT-5" → smart_search(depth="deep")
"Use AgentCPM for AI news" → agentcpm(query="AI news")
```

## Standalone Agent (CLI)

Run search agent directly from command line.

```bash
# Default: SGLang backend
python search_agent.py "search query"

# Search depth
python search_agent.py "query" --depth simple   # snippets only (fast)
python search_agent.py "query" --depth medium   # fetch top 5 URLs (default)
python search_agent.py "query" --depth deep     # fetch all URLs (slow)

# Interactive mode
python search_agent.py -i

# CLI supports multiple backends (--llm ollama/lmstudio/openai)
python search_agent.py --list-backends
```

## AgentCPM-Explore Setup (for `agentcpm` tool)

**AgentCPM-Explore** is a 4B parameter model from [OpenBMB/THUNLP](https://github.com/OpenBMB/AgentCPM) specifically trained for search agent tasks:
- Automatically generates diverse search queries (Korean/English, multiple perspectives)
- Optimized for tool calling (search, fetch_url)
- Based on Qwen3-4B-Thinking

```bash
# 1. Install SGLang (CUDA required)
pip install sglang[all]

# 2. Download AgentCPM-Explore model (~8GB)
# https://huggingface.co/openbmb/AgentCPM-Explore

# 3. Start server
MODEL_PATH=/path/to/AgentCPM-Explore ./start_sglang.sh
```

**Note:** The `agentcpm` MCP tool exclusively uses SGLang + AgentCPM-Explore. For other LLM backends, use `smart_search` or the CLI (`search_agent.py --llm ollama`).

## Architecture

```
User Query
    |
MCP Server (mcp_server.py)
+-- web_search      → CDP Search only
+-- smart_search    → CDP Search + URL Fetch
+-- agentcpm        → SGLang + AgentCPM-Explore + CDP Search
    |
CDP Search (parallel)
+-- Chrome:9222 → Naver
+-- Chrome:9223 → Google
+-- Chrome:9224 → Brave
    |
Final Answer + Sources
```

## File Structure

```
AgentWebSearch-MCP/
├── mcp_server.py         # MCP server (4 tools)
├── search_agent.py       # Standalone CLI agent
├── cdp_search.py         # CDP parallel search
├── chrome_launcher.py    # Chrome instance manager
├── llm_adapters/         # LLM adapters (CLI use)
│   ├── base.py           # Common interface
│   ├── sglang_adapter.py
│   ├── ollama_adapter.py
│   ├── lmstudio_adapter.py
│   └── openai_adapter.py
└── start_sglang.sh       # SGLang server startup
```

## Performance

| Mode | Time | Tokens |
|------|------|--------|
| simple | ~35s | ~3K |
| medium | ~50s | ~17K |
| deep | ~170s | ~77K |

## Requirements

- Python 3.10+
- Chrome/Chromium
- For `agentcpm` tool: SGLang + AgentCPM-Explore model

## License

MIT License

## Contributing

Portal addition PRs welcome! See `PORTAL_CONFIG` in `cdp_search.py`.

---

Co-Authored-By: inchul <insung8150@users.noreply.github.com>
Co-Authored-By: Claude <noreply@anthropic.com>
