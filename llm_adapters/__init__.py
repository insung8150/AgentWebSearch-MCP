"""
LLM Adapter Layer for LocalWebSearch-CDP

Provides unified interface for different LLM backends:
- SGLang (AgentCPM-Explore): <tool_call> text format
- Ollama: Function calling API
- LM Studio: OpenAI-compatible API
- OpenAI: Standard OpenAI API
"""

from .base import BaseLLMAdapter, AdapterConfig, LLMResponse
from .sglang_adapter import SGLangAdapter
from .ollama_adapter import OllamaAdapter
from .lmstudio_adapter import LMStudioAdapter
from .openai_adapter import OpenAIAdapter

__all__ = [
    "BaseLLMAdapter",
    "AdapterConfig",
    "LLMResponse",
    "SGLangAdapter",
    "OllamaAdapter",
    "LMStudioAdapter",
    "OpenAIAdapter",
    "get_adapter",
    "detect_available_backends",
]


def get_adapter(backend: str, **kwargs) -> BaseLLMAdapter:
    """Get adapter instance by backend name

    Args:
        backend: One of "sglang", "ollama", "lmstudio", "openai"
        **kwargs: Additional config options (url, model, api_key, etc.)

    Returns:
        Configured adapter instance
    """
    adapters = {
        "sglang": SGLangAdapter,
        "ollama": OllamaAdapter,
        "lmstudio": LMStudioAdapter,
        "openai": OpenAIAdapter,
    }

    if backend not in adapters:
        raise ValueError(f"Unknown backend: {backend}. Available: {list(adapters.keys())}")

    return adapters[backend](**kwargs)


async def detect_available_backends() -> list[dict]:
    """Detect available LLM backends on localhost

    Returns:
        List of available backends with their info:
        [{"name": "sglang", "url": "...", "models": [...], "status": "ready"}, ...]
    """
    import httpx

    backends = []

    # Check SGLang (default port 30001)
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            resp = await client.get("http://localhost:30001/health")
            if resp.status_code == 200:
                backends.append({
                    "name": "sglang",
                    "url": "http://localhost:30001",
                    "models": ["AgentCPM-Explore"],
                    "status": "ready",
                    "tool_format": "text",  # <tool_call> format
                })
    except:
        pass

    # Check Ollama (default port 11434)
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            resp = await client.get("http://localhost:11434/api/tags")
            if resp.status_code == 200:
                data = resp.json()
                models = [m["name"] for m in data.get("models", [])]
                backends.append({
                    "name": "ollama",
                    "url": "http://localhost:11434",
                    "models": models,
                    "status": "ready",
                    "tool_format": "function_calling",
                })
    except:
        pass

    # Check LM Studio (default port 1234)
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            resp = await client.get("http://localhost:1234/v1/models")
            if resp.status_code == 200:
                data = resp.json()
                models = [m["id"] for m in data.get("data", [])]
                backends.append({
                    "name": "lmstudio",
                    "url": "http://localhost:1234",
                    "models": models,
                    "status": "ready",
                    "tool_format": "function_calling",
                })
    except:
        pass

    return backends
