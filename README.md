# AgentWebSearch-MCP

**Local WebSearch without API keys + MCP Server**

Need WebSearch for your local LLM (Ollama, LM Studio) or Claude Code?
No API keys needed - **just Chrome**.

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
- **AgentCPM-Explore** - Search-optimized 4B model from OpenBMB/THUNLP (recommended)
- **Agentic Search** - LLM plans and executes search automatically
- **Parallel search** - 3 Chrome instances for Naver/Google/Brave simultaneously
- **Multi-LLM support** - SGLang, Ollama, LM Studio, OpenAI-compatible APIs
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

Use web search as an MCP tool in Claude Code, Cursor, or LM Studio.

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
| `agent_search` | Agentic search with LLM backend selection | Yes |

### Tool Parameters

#### `smart_search`
| Parameter | Values | Description |
|-----------|--------|-------------|
| `query` | string | Search query (required) |
| `depth` | `simple` | Snippets only (fast) |
| | `medium` | Fetch top 5 URLs (default) |
| | `deep` | Fetch top 15 URLs (slow) |
| `portal` | `all`/`naver`/`google`/`brave` | Search portal |

#### `agent_search`

Uses LLM to plan search queries, execute searches, and generate answers.

| Parameter | Values | Description |
|-----------|--------|-------------|
| `query` | string | Search query (required) |
| `llm` | `sglang` | **Recommended** - [AgentCPM-Explore](https://huggingface.co/openbmb/AgentCPM-Explore) (4B, search-optimized) |
| | `ollama` | Local LLM (general purpose) |
| | `lmstudio` | Local LLM (general purpose) |
| | `openai` | Paid API |
| `model` | string | Model name (empty = backend default) |
| `depth` | `simple`/`medium`/`deep` | Search depth |

> **Why AgentCPM-Explore?** Trained specifically for search agent tasks by OpenBMB/THUNLP. Generates diverse queries and handles tool calling better than general-purpose models.

### Example Usage
```
"Search latest AI news" → web_search tool
"Search AI news with sglang" → agent_search(llm="sglang")
"Deep search about GPT-5" → smart_search(depth="deep")
```

## Standalone Agent (CLI)

Run search agent directly from command line with LLM backend.

```bash
# Default: SGLang backend
python search_agent.py "search query"

# Select LLM backend
python search_agent.py --llm ollama "search query"
python search_agent.py --llm lmstudio --model qwen3-8b "search query"

# Search depth
python search_agent.py "query" --depth simple   # snippets only (fast)
python search_agent.py "query" --depth medium   # fetch top 5 URLs (default)
python search_agent.py "query" --depth deep     # fetch all URLs (slow)

# Interactive mode
python search_agent.py -i

# List available backends
python search_agent.py --list-backends
```

## LLM Backend Setup

### Option A: SGLang + AgentCPM-Explore (Recommended)

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
./start_sglang.sh
```

### Option B: Ollama (Easiest)
```bash
# Install: https://ollama.ai
ollama pull qwen3:8b
ollama serve
```

### Option C: LM Studio
```bash
# Install: https://lmstudio.ai
# Load model and start server in the app
```

## Architecture

```
User Query
    |
MCP Server (mcp_server.py)
+-- web_search      → CDP Search only
+-- smart_search    → CDP Search + URL Fetch
+-- agent_search    → LLM + CDP Search + URL Fetch
    |
LLM Adapter Layer (for agent_search)
+-- SGLang (port 30001) - AgentCPM-Explore (recommended)
+-- Ollama (port 11434) - qwen3:8b etc.
+-- LM Studio (port 1234)
+-- OpenAI (compatible API)
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
├── llm_adapters/         # Multi-LLM support
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
- For `agent_search`: LLM backend (SGLang/Ollama/LM Studio/OpenAI)

## License

MIT License

## Contributing

Portal addition PRs welcome! See `PORTAL_CONFIG` in `cdp_search.py`.

---

Co-Authored-By: inchul <insung8150@users.noreply.github.com>
Co-Authored-By: Claude <noreply@anthropic.com>
