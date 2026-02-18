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
- **Parallel search** - 3 Chrome instances for Naver/Google/Brave simultaneously
- **Extensible portals** - Add your own search portals easily (see `PORTAL_CONFIG` in `cdp_search.py`)
- **Korean support** - Naver integrated search included
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
# Auto-start 3 Chrome instances
python chrome_launcher.py start
```

## MCP Server (Claude Code / Cursor / LM Studio)

Use web search as an MCP tool without any LLM backend setup.

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

| Tool | Description |
|------|-------------|
| `web_search` | Search Naver/Google/Brave in parallel |
| `fetch_urls` | Fetch webpage content from URLs |
| `smart_search` | Search + auto-fetch top results |

### Example Usage in Claude Code
```
"최신 AI 뉴스 검색해줘" → web_search 도구 자동 호출
"이 URL들 내용 가져와" → fetch_urls 도구 호출
```

### 3. Choose and install LLM backend

LocalWebSearch-CDP supports multiple LLM backends:

#### Option A: SGLang + AgentCPM-Explore (Search-optimized, Recommended)
```bash
# 1. Install SGLang (CUDA required)
pip install sglang[all]

# 2. Download AgentCPM-Explore model (~8GB)
# https://huggingface.co/openbmb/AgentCPM-Explore

# 3. Edit MODEL_PATH in start_sglang.sh, then run
./start_sglang.sh
```
> AgentCPM-Explore is a 4B model specialized for search agents.
> Automatically generates diverse search queries (Korean/English, multiple perspectives).

#### Option B: Ollama (Easiest)
```bash
# Install Ollama: https://ollama.ai
ollama pull qwen3:8b
ollama serve
```

#### Option C: LM Studio
```bash
# Install LM Studio: https://lmstudio.ai
# Load model and start server in the app
```

## Usage

```bash
# Single query (default: SGLang)
python search_agent.py "search query"

# Select LLM backend
python search_agent.py --llm ollama "search query"
python search_agent.py --llm lmstudio "search query"
python search_agent.py --llm lmstudio --model qwen3-8b "search query"

# List available backends
python search_agent.py --list-backends

# Search depth settings
python search_agent.py "query" --depth simple   # snippets only (fast)
python search_agent.py "query" --depth medium   # fetch top 5 URLs
python search_agent.py "query" --depth deep     # fetch all URLs (slow)

# Interactive mode
python search_agent.py -i
```

## Architecture

```
User Query
    |
LLM Adapter Layer
+-- SGLang (port 30001) - AgentCPM-Explore (default)
+-- Ollama (port 11434) - qwen3:8b etc.
+-- LM Studio (port 1234) - loaded model
+-- OpenAI (compatible API)
    |
CDP Search (parallel)
+-- Chrome:9222 -> Naver
+-- Chrome:9223 -> Google
+-- Chrome:9224 -> Brave
    |
Final Answer + Sources
```

## File Structure

```
AgentWebSearch-MCP/
+-- mcp_server.py         # MCP server (Claude Code/Cursor)
+-- search_agent.py       # Standalone agent (with LLM)
+-- cdp_search.py         # CDP parallel search
+-- chrome_launcher.py    # Chrome instance manager
+-- llm_adapters/         # Multi-LLM support
|   +-- base.py           # Common interface
|   +-- sglang_adapter.py # SGLang (<tool_call> format)
|   +-- ollama_adapter.py # Ollama (function calling)
|   +-- lmstudio_adapter.py
|   +-- openai_adapter.py
+-- start_sglang.sh       # SGLang server startup
+-- clear_tabs.py         # Tab cleanup utility
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
- LLM backend (one of):
  - SGLang + AgentCPM-Explore (search-optimized, recommended)
  - Ollama + qwen3:8b or higher
  - LM Studio
  - OpenAI compatible API

## License

MIT License

## Contributing

Portal addition PRs welcome! See `PORTAL_CONFIG` comments in `cdp_search.py`.

---

Co-Authored-By: inchul <insung8150@users.noreply.github.com>
Co-Authored-By: Claude <noreply@anthropic.com>
