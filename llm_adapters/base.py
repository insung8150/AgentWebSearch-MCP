"""
Base LLM Adapter Interface

Defines the common interface for all LLM backends.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class AdapterConfig:
    """Configuration for LLM adapter"""
    url: str                          # API endpoint URL
    model: str = ""                   # Model name (optional, some backends auto-detect)
    api_key: str = ""                 # API key (optional, for OpenAI/cloud)
    timeout: float = 150.0            # Request timeout in seconds
    temperature: float = 0.7          # Sampling temperature
    max_tokens: int = 10000           # Maximum tokens to generate
    extra: dict = field(default_factory=dict)  # Backend-specific options


@dataclass
class ToolCall:
    """Parsed tool call from LLM response"""
    name: str                         # Tool name (e.g., "search", "fetch_url")
    arguments: dict                   # Tool arguments as dict


@dataclass
class LLMResponse:
    """Unified LLM response format"""
    content: str                      # Text content of the response
    tool_calls: list[ToolCall]        # Parsed tool calls (empty if none)
    raw_response: dict = field(default_factory=dict)  # Raw API response for debugging
    model: str = ""                   # Model name that generated the response
    usage: dict = field(default_factory=dict)  # Token usage stats


class BaseLLMAdapter(ABC):
    """Abstract base class for LLM adapters

    Each adapter must implement:
    - call(): Send messages and get response
    - parse_tool_calls(): Parse tool calls from response
    - format_tool_result(): Format tool result for next turn
    - get_system_prompt(): Get formatted system prompt with tools
    """

    def __init__(self, config: AdapterConfig | None = None, **kwargs):
        """Initialize adapter with config

        Args:
            config: AdapterConfig instance
            **kwargs: Alternative way to pass config options
        """
        if config is None:
            config = AdapterConfig(**kwargs)
        self.config = config

    @property
    @abstractmethod
    def name(self) -> str:
        """Backend name (e.g., 'sglang', 'ollama')"""
        pass

    @property
    @abstractmethod
    def tool_format(self) -> str:
        """Tool call format: 'text' or 'function_calling'"""
        pass

    @abstractmethod
    async def call(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
    ) -> LLMResponse:
        """Send messages to LLM and get response

        Args:
            messages: List of message dicts [{"role": "...", "content": "..."}]
            tools: Tool definitions (OpenAI function format)

        Returns:
            LLMResponse with content and parsed tool calls
        """
        pass

    @abstractmethod
    def parse_tool_calls(self, response: LLMResponse) -> list[ToolCall]:
        """Parse tool calls from LLM response

        For text format (SGLang): Parse <tool_call>...</tool_call> tags
        For function_calling: Extract from tool_calls field

        Args:
            response: LLM response to parse

        Returns:
            List of ToolCall objects
        """
        pass

    @abstractmethod
    def format_tool_result(self, tool_name: str, result: str) -> dict:
        """Format tool execution result for next turn

        Args:
            tool_name: Name of the executed tool
            result: String result of tool execution

        Returns:
            Message dict to append to conversation
        """
        pass

    def format_tools_for_prompt(self, tools: list[dict]) -> str:
        """Format tool definitions for system prompt (text format adapters)

        Args:
            tools: Tool definitions in OpenAI format

        Returns:
            Formatted string for system prompt
        """
        lines = ["Available tools:"]
        for i, tool in enumerate(tools, 1):
            func = tool.get("function", {})
            name = func.get("name", "unknown")
            desc = func.get("description", "")
            params = func.get("parameters", {}).get("properties", {})
            param_strs = []
            for pname, pinfo in params.items():
                ptype = pinfo.get("type", "any")
                pdesc = pinfo.get("description", "")
                param_strs.append(f"{pname}: {ptype} - {pdesc}")
            lines.append(f"{i}. {name}({', '.join(param_strs)})")
            lines.append(f"   {desc}")
        return "\n".join(lines)

    async def health_check(self) -> bool:
        """Check if the backend is healthy and responding

        Returns:
            True if healthy, False otherwise
        """
        try:
            import httpx
            async with httpx.AsyncClient(timeout=2.0) as client:
                if "v1" in self.config.url:
                    # OpenAI-compatible
                    resp = await client.get(f"{self.config.url}/models")
                else:
                    # Custom health endpoint
                    resp = await client.get(f"{self.config.url}/health")
                return resp.status_code == 200
        except:
            return False

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(url={self.config.url!r}, model={self.config.model!r})"
