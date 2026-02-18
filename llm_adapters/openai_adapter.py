"""
OpenAI Adapter

Handles OpenAI-compatible APIs (OpenAI, Azure, local proxies, etc.)
Supports standard function calling format.
"""

import json
import os
import httpx

from .base import BaseLLMAdapter, AdapterConfig, LLMResponse, ToolCall


class OpenAIAdapter(BaseLLMAdapter):
    """Adapter for OpenAI-compatible APIs

    Works with:
    - OpenAI API
    - Azure OpenAI
    - Local proxies (codex-proxy, etc.)
    - Any OpenAI-compatible endpoint
    """

    DEFAULT_URL = "https://api.openai.com/v1/chat/completions"

    def __init__(
        self,
        url: str | None = None,
        model: str = "gpt-4o",
        api_key: str | None = None,
        **kwargs
    ):
        config = AdapterConfig(
            url=url or self.DEFAULT_URL,
            model=model,
            api_key=api_key or os.environ.get("OPENAI_API_KEY", ""),
            **kwargs
        )
        super().__init__(config)

    @property
    def name(self) -> str:
        return "openai"

    @property
    def tool_format(self) -> str:
        return "function_calling"

    async def call(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
    ) -> LLMResponse:
        """Send messages to OpenAI-compatible API"""
        headers = {
            "Content-Type": "application/json",
        }
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"

        # Build request payload
        payload = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }

        # Add tools if provided
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        async with httpx.AsyncClient(timeout=self.config.timeout) as client:
            response = await client.post(
                self.config.url,
                json=payload,
                headers=headers,
            )
            data = response.json()

        # Handle error responses
        if "error" in data:
            error_msg = data["error"].get("message", str(data["error"]))
            raise RuntimeError(f"OpenAI API error: {error_msg}")

        # Extract response
        choice = data.get("choices", [{}])[0]
        message = choice.get("message", {})
        content = message.get("content", "") or ""

        # Check for tool calls
        tool_calls_data = message.get("tool_calls", [])

        llm_response = LLMResponse(
            content=content,
            tool_calls=[],
            raw_response=data,
            model=data.get("model", self.config.model),
            usage=data.get("usage", {}),
        )

        # Parse tool calls
        if tool_calls_data:
            llm_response.tool_calls = self._parse_function_calls(tool_calls_data)

        return llm_response

    def _parse_function_calls(self, tool_calls_data: list) -> list[ToolCall]:
        """Parse OpenAI function calling format"""
        tool_calls = []
        for tc in tool_calls_data:
            func = tc.get("function", {})
            name = func.get("name", "")
            args = func.get("arguments", "{}")
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            tool_calls.append(ToolCall(name=name, arguments=args))
        return tool_calls

    def parse_tool_calls(self, response: LLMResponse) -> list[ToolCall]:
        """Parse tool calls from response (already done in call())"""
        return response.tool_calls

    def format_tool_result(self, tool_name: str, result: str) -> dict:
        """Format tool result for OpenAI (standard format)"""
        return {
            "role": "tool",
            "content": result,
            "name": tool_name,
        }

    @classmethod
    def from_codex_proxy(cls, port: int = 28100, **kwargs) -> "OpenAIAdapter":
        """Create adapter for local codex-proxy

        Args:
            port: Codex proxy port (default 28100)
        """
        return cls(
            url=f"http://localhost:{port}/v1/chat/completions",
            api_key="local",  # Proxy doesn't need real key
            model="codex",
            **kwargs
        )

    @classmethod
    def from_gemini_proxy(cls, port: int = 28101, **kwargs) -> "OpenAIAdapter":
        """Create adapter for local gemini-proxy

        Args:
            port: Gemini proxy port (default 28101)
        """
        return cls(
            url=f"http://localhost:{port}/v1/chat/completions",
            api_key="local",
            model="gemini-2.5-flash",
            **kwargs
        )
