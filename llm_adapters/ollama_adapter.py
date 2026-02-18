"""
Ollama Adapter

Handles Ollama's function calling API for tool use.
Falls back to text-based tool calls if function calling not supported.
"""

import json
import re
import httpx

from .base import BaseLLMAdapter, AdapterConfig, LLMResponse, ToolCall


class OllamaAdapter(BaseLLMAdapter):
    """Adapter for Ollama server

    Ollama supports function calling for compatible models.
    For models without function calling, falls back to text-based parsing.
    """

    DEFAULT_URL = "http://localhost:11434"

    def __init__(self, url: str | None = None, model: str = "qwen2.5:7b", **kwargs):
        config = AdapterConfig(
            url=url or self.DEFAULT_URL,
            model=model,
            **kwargs
        )
        super().__init__(config)
        self._use_function_calling = True  # Will be set based on model support

    @property
    def name(self) -> str:
        return "ollama"

    @property
    def tool_format(self) -> str:
        return "function_calling" if self._use_function_calling else "text"

    async def call(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
    ) -> LLMResponse:
        """Send messages to Ollama server"""
        # Ollama chat API endpoint
        api_url = f"{self.config.url}/api/chat"

        # Build request payload
        payload = {
            "model": self.config.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": self.config.temperature,
                "num_predict": self.config.max_tokens,
            },
        }

        # Add tools if using function calling
        if tools and self._use_function_calling:
            payload["tools"] = tools

        async with httpx.AsyncClient(timeout=self.config.timeout) as client:
            response = await client.post(api_url, json=payload)
            data = response.json()

        # Extract response
        message = data.get("message", {})
        content = message.get("content", "")

        # Check for tool calls in response
        tool_calls_data = message.get("tool_calls", [])

        llm_response = LLMResponse(
            content=content,
            tool_calls=[],
            raw_response=data,
            model=self.config.model,
            usage={
                "prompt_tokens": data.get("prompt_eval_count", 0),
                "completion_tokens": data.get("eval_count", 0),
            },
        )

        # Parse tool calls
        if tool_calls_data:
            # Function calling format
            llm_response.tool_calls = self._parse_function_calls(tool_calls_data)
        else:
            # Try text-based parsing as fallback
            llm_response.tool_calls = self._parse_text_tool_calls(content)

        return llm_response

    def _parse_function_calls(self, tool_calls_data: list) -> list[ToolCall]:
        """Parse Ollama function calling format"""
        tool_calls = []
        for tc in tool_calls_data:
            func = tc.get("function", {})
            name = func.get("name", "")
            # Arguments can be string or dict
            args = func.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            tool_calls.append(ToolCall(name=name, arguments=args))
        return tool_calls

    def _parse_text_tool_calls(self, content: str) -> list[ToolCall]:
        """Parse text-based tool calls (fallback)

        Supports multiple formats:
        - <tool_call>{"name": "...", "arguments": {...}}</tool_call>
        - ```json\n{"tool": "...", "params": {...}}\n```
        """
        tool_calls = []

        # Try <tool_call> format first
        pattern = r'<tool_call>\s*(\{.*?\})\s*</tool_call>'
        matches = re.findall(pattern, content, re.DOTALL)
        for match in matches:
            try:
                data = json.loads(match)
                tool_calls.append(ToolCall(
                    name=data.get("name", data.get("tool", "")),
                    arguments=data.get("arguments", data.get("params", {})),
                ))
            except json.JSONDecodeError:
                continue

        # If no matches, try JSON code blocks
        if not tool_calls:
            pattern = r'```(?:json)?\s*(\{[^`]*\})\s*```'
            matches = re.findall(pattern, content, re.DOTALL)
            for match in matches:
                try:
                    data = json.loads(match)
                    if "name" in data or "tool" in data:
                        tool_calls.append(ToolCall(
                            name=data.get("name", data.get("tool", "")),
                            arguments=data.get("arguments", data.get("params", {})),
                        ))
                except json.JSONDecodeError:
                    continue

        return tool_calls

    def parse_tool_calls(self, response: LLMResponse) -> list[ToolCall]:
        """Parse tool calls from response (already done in call())"""
        return response.tool_calls

    def format_tool_result(self, tool_name: str, result: str) -> dict:
        """Format tool result for Ollama

        For function calling: Use tool role
        For text fallback: Use user role with formatted result
        """
        if self._use_function_calling:
            return {
                "role": "tool",
                "content": result,
            }
        else:
            return {
                "role": "user",
                "content": f"<tool_result name=\"{tool_name}\">\n{result}\n</tool_result>",
            }

    async def list_models(self) -> list[str]:
        """List available models in Ollama"""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{self.config.url}/api/tags")
                if resp.status_code == 200:
                    data = resp.json()
                    return [m["name"] for m in data.get("models", [])]
        except:
            pass
        return []
